import asyncio
from typing import Callable

import pytest

from async_lru import _CacheParameters, alru_cache


async def test_custom_key_function_ignores_argument(
    check_lru: Callable[..., None]
) -> None:
    @alru_cache(key=lambda db, query: query)
    async def fetch_data(db_connection: str, query: str) -> str:
        return f"{db_connection}:{query}"

    result1 = await fetch_data("conn1", "SELECT * FROM users")
    check_lru(fetch_data, hits=0, misses=1, cache=1, tasks=0)

    result2 = await fetch_data("conn2", "SELECT * FROM users")
    check_lru(fetch_data, hits=1, misses=1, cache=1, tasks=0)

    assert result1 == "conn1:SELECT * FROM users"
    assert result2 == result1


async def test_custom_key_function_with_kwargs(
    check_lru: Callable[..., None]
) -> None:
    @alru_cache(key=lambda x, y=0: (x, y))
    async def compute(x: int, y: int = 0) -> int:
        return x + y

    result1 = await compute(1, y=2)
    check_lru(compute, hits=0, misses=1, cache=1, tasks=0)

    result2 = await compute(1, y=2)
    check_lru(compute, hits=1, misses=1, cache=1, tasks=0)

    result3 = await compute(1, y=3)
    check_lru(compute, hits=1, misses=2, cache=2, tasks=0)

    assert result1 == 3
    assert result2 == 3
    assert result3 == 4


async def test_cache_invalidate_with_custom_key(
    check_lru: Callable[..., None]
) -> None:
    @alru_cache(key=lambda db, query: query)
    async def fetch_data(db_connection: str, query: str) -> str:
        return f"{db_connection}:{query}"

    await fetch_data("conn1", "SELECT 1")
    check_lru(fetch_data, hits=0, misses=1, cache=1, tasks=0)

    invalidated = fetch_data.cache_invalidate("conn2", "SELECT 1")
    assert invalidated is True
    check_lru(fetch_data, hits=0, misses=1, cache=0, tasks=0)

    invalidated_again = fetch_data.cache_invalidate("conn3", "SELECT 1")
    assert invalidated_again is False


async def test_cache_parameters_includes_key() -> None:
    def my_key(x: int) -> int:
        return x

    @alru_cache(key=my_key)
    async def func(x: int) -> int:
        return x * 2

    params = func.cache_parameters()
    assert params["key"] is my_key


async def test_cache_parameters_key_is_none_by_default() -> None:
    @alru_cache
    async def func(x: int) -> int:
        return x * 2

    params = func.cache_parameters()
    assert params["key"] is None


async def test_key_function_returns_non_hashable_raises_error() -> None:
    @alru_cache(key=lambda x: [x])
    async def func(x: int) -> int:
        return x * 2

    with pytest.raises(TypeError, match="unhashable type"):
        await func(1)


async def test_key_function_with_multiple_cache_entries(
    check_lru: Callable[..., None]
) -> None:
    @alru_cache(maxsize=2, key=lambda a, b: a)
    async def func(a: int, b: int) -> int:
        return a + b

    await func(1, 100)
    await func(2, 200)
    check_lru(func, hits=0, misses=2, cache=2, tasks=0, maxsize=2)

    await func(1, 999)
    check_lru(func, hits=1, misses=2, cache=2, tasks=0, maxsize=2)

    await func(3, 300)
    check_lru(func, hits=1, misses=3, cache=2, tasks=0, maxsize=2)


async def test_custom_key_with_no_arguments() -> None:
    @alru_cache(key=lambda: "constant")
    async def func() -> str:
        return "result"

    result1 = await func()
    result2 = await func()

    assert result1 == "result"
    assert result2 == "result"
