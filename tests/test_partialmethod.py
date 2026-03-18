import asyncio
from functools import partial, partialmethod
from typing import Any, Callable, cast

from async_lru import _LRUCacheWrapperInstanceMethod, alru_cache


async def test_partialmethod_basic(check_lru: Callable[..., None]) -> None:
    class Obj:
        async def _coro(self, val: int) -> int:
            return val

        coro = alru_cache(partialmethod(_coro, 2))

    obj = Obj()

    coros = [obj.coro() for _ in range(5)]

    check_lru(obj.coro, hits=0, misses=0, cache=0, tasks=0)

    ret = await asyncio.gather(*coros)

    check_lru(obj.coro, hits=4, misses=1, cache=1, tasks=0)

    assert ret == [2, 2, 2, 2, 2]


async def test_partialmethod_partial(check_lru: Callable[..., None]) -> None:
    class Obj:
        def __init__(self) -> None:
            self.coro = alru_cache(partial(self._coro, 2))

        async def __coro(self, val1: int, val2: int) -> int:
            return val1 + val2

        _coro = partialmethod(__coro, 1)

    obj = Obj()

    coros = [obj.coro() for _ in range(5)]

    check_lru(obj.coro, hits=0, misses=0, cache=0, tasks=0)

    ret = await asyncio.gather(*coros)

    check_lru(obj.coro, hits=4, misses=1, cache=1, tasks=0)

    assert ret == [3, 3, 3, 3, 3]


async def test_partialmethod_wrapped_attributes() -> None:
    class MyClass:
        async def _coro(self, val: int) -> int:
            return val

        coro_partial = alru_cache()(partialmethod(_coro, 2))

    obj = MyClass()

    descriptor = MyClass.__dict__["coro_partial"]
    assert descriptor.__get__(None, None) is descriptor
    assert type(MyClass.coro_partial).__name__ == "_LRUCacheWrapperInstanceMethod"

    # Test executing the partial method
    assert await obj.coro_partial() == 2
    assert await obj.coro_partial() == 2

    # Test cache_info
    info = obj.coro_partial.cache_info()
    assert info.hits == 1
    assert info.misses == 1

    # Test cache_clear
    obj.coro_partial.cache_clear()
    info = obj.coro_partial.cache_info()
    assert info.hits == 0
    assert info.misses == 0

    # Test cache_close
    await obj.coro_partial.cache_close()


async def test_wrapper_instance_method_attribute_fallbacks() -> None:
    class WrapperWithMissingAttrs:
        def __getattribute__(self, name: str) -> Any:
            if name in {
                "__module__",
                "__name__",
                "__qualname__",
                "__doc__",
                "__annotations__",
                "__dict__",
            }:
                raise AttributeError(name)
            return super().__getattribute__(name)

        async def __wrapped__(self, *_args: Any, **_kwargs: Any) -> int:
            return 1

    bound = _LRUCacheWrapperInstanceMethod(
        cast(Any, WrapperWithMissingAttrs()),
        object(),
    )

    assert callable(bound.__wrapped__)
    assert await bound.__wrapped__() == 1
