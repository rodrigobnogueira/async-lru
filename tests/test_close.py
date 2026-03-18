import asyncio
from typing import Any, Callable, cast

import pytest

from async_lru import alru_cache


async def test_cache_close(check_lru: Callable[..., None]) -> None:
    @alru_cache()
    async def coro(val: int) -> int:
        await asyncio.sleep(0.2)
        return val

    assert await coro(0) == 0
    coro.cache_clear()

    assert not coro.cache_parameters()["closed"]

    inputs = [1, 2, 3, 4, 5]

    coros = [coro(v) for v in inputs]

    gather = asyncio.gather(*coros)

    await asyncio.sleep(0.1)

    check_lru(coro, hits=0, misses=5, cache=5, tasks=5)

    close = coro.cache_close()

    check_lru(coro, hits=0, misses=5, cache=5, tasks=5)

    await close

    check_lru(coro, hits=0, misses=5, cache=0, tasks=0)
    assert coro.cache_parameters()["closed"]

    with pytest.raises(asyncio.CancelledError):
        await gather

    check_lru(coro, hits=0, misses=5, cache=0, tasks=0)
    assert coro.cache_parameters()["closed"]

    # double call is no-op
    await coro.cache_close()


async def test_cache_close_wait_bound_method(check_lru: Callable[..., None]) -> None:
    class Foo:
        @alru_cache()
        async def coro(self, val: int) -> int:
            await asyncio.sleep(0.02)
            return val

    foo = Foo()
    inputs = [1, 2, 3]

    coros = [foo.coro(v) for v in inputs]
    gather = asyncio.gather(*coros)

    # Yield to loop to start tasks
    await asyncio.sleep(0)

    # wait=True should allow tasks to finish (no cancellation)
    await foo.coro.cache_close(wait=True)

    results = await gather
    assert results == inputs

    check_lru(foo.coro, hits=0, misses=3, cache=3, tasks=0)
    assert foo.coro.cache_parameters()["closed"]


async def test_cache_close_deprecated_cancel() -> None:
    class MyClass:
        @alru_cache()
        async def coro(self, val: int) -> int:
            return val

    obj = MyClass()
    await obj.coro(1)
    cache_close = cast(Any, obj.coro.cache_close)
    with pytest.warns(DeprecationWarning, match="cancel/return_exceptions"):
        await cache_close(cancel=True)
    with pytest.warns(DeprecationWarning, match="cancel/return_exceptions"):
        await cache_close(return_exceptions=False)
