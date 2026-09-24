"""Python-version support for slotted frozen data containers."""

from __future__ import annotations

import sys
from builtins import zip as _builtin_zip
from dataclasses import dataclass as _stdlib_dataclass
from dataclasses import fields
from enum import Enum
from itertools import zip_longest
from typing import Any

if sys.version_info >= (3, 12):
    from typing import TypeAliasType as TypeAliasType
else:
    from typing_extensions import TypeAliasType as TypeAliasType

if sys.version_info >= (3, 10):
    dataclass = _stdlib_dataclass
else:

    def _inherited_slots(cls: type) -> set[str]:
        return {
            name
            for base in cls.__mro__[1:]
            for name in (
                (base.__slots__,)
                if isinstance(getattr(base, "__slots__", ()), str)
                else getattr(base, "__slots__", ())
            )
        }

    def _install_frozen_pickle(slotted: type, namespace: dict[str, Any]) -> None:
        def getstate(self: object) -> list[object]:
            return [getattr(self, item.name) for item in fields(self)]

        def setstate(self: object, values: list[object]) -> None:
            for item, value in zip(fields(self), values):
                object.__setattr__(self, item.name, value)

        if "__getstate__" not in namespace:
            slotted.__getstate__ = getstate
        if "__setstate__" not in namespace:
            slotted.__setstate__ = setstate

    def _slotted_dataclass(result: type, *, frozen: bool) -> type:
        if "__slots__" in result.__dict__:
            raise TypeError(f"{result.__name__} already specifies __slots__")

        names = tuple(item.name for item in fields(result))
        inherited = _inherited_slots(result)
        namespace = dict(result.__dict__)
        namespace["__slots__"] = tuple(name for name in names if name not in inherited)
        for name in names:
            namespace.pop(name, None)
        namespace.pop("__dict__", None)
        namespace.pop("__weakref__", None)
        slotted = type(result)(result.__name__, result.__bases__, namespace)
        slotted.__qualname__ = result.__qualname__
        if frozen:
            _install_frozen_pickle(slotted, namespace)
        return slotted

    def dataclass(*, slots: bool = False, **options: Any) -> Any:
        """Build stdlib data classes with slots on Python 3.9."""

        def decorate(cls: type) -> type:
            result = _stdlib_dataclass(cls, **options)
            if slots:
                return _slotted_dataclass(result, frozen=bool(options.get("frozen")))
            return result

        return decorate


if sys.version_info >= (3, 11):
    from enum import StrEnum as StrEnum
else:

    class StrEnum(str, Enum):
        """String-valued enum with the standard string representation."""

        __str__ = str.__str__


if sys.version_info >= (3, 10):
    zip = _builtin_zip
else:

    def zip(*iterables: Any, strict: bool = False) -> Any:
        """Preserve strict zip validation on Python 3.9."""
        if not strict:
            return _builtin_zip(*iterables)

        sentinel = object()

        def rows() -> Any:
            for values in zip_longest(*iterables, fillvalue=sentinel):
                if any(value is sentinel for value in values):
                    raise ValueError("zip() arguments have different lengths")
                yield values

        return rows()
