from __future__ import annotations
from contextlib import contextmanager, AbstractContextManager


class Config:
    enable_backprop: bool = True


def no_grid() -> AbstractContextManager[None]:
    return _using_config('enable_backprop', False)


@contextmanager
def _using_config(name: str, value: bool) -> AbstractContextManager[None]:
    old_value: bool = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)
