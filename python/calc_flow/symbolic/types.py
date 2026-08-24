"""Strict declaration types and validators for symbolic expressions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from calc_flow.capabilities import PORTABLE_ARROW_TYPES
from calc_flow.symbolic.domains import type_name

type BatchKind = Literal["table", "array"]
type CompileMode = Literal["batch", "stream"]
type LatePolicy = Literal["error", "drop"]
type ScalarLiteral = None | bool | int | float | str

# The initial portable field spelling adds ``timestamp[us, UTC]`` to the
# current portable Arrow names (API note section 2.2).
TABLE_FIELD_TYPES = frozenset((*PORTABLE_ARROW_TYPES, "timestamp[us, UTC]"))
# Array declaration dtypes stay inside the static-input digest v1 coverage:
# bool, the integer types, float32, and float64.
ARRAY_DTYPES = frozenset(
    (
        "bool",
        "float32",
        "float64",
        "int16",
        "int32",
        "int64",
        "int8",
        "uint16",
        "uint32",
        "uint64",
        "uint8",
    )
)


@dataclass(frozen=True, slots=True)
class Field:
    """One exact table field declaration with host-type validation."""

    name: str
    data_type: str
    nullable: bool = True

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError(f"Field.name must be a string; got {type_name(self.name)}")
        if type(self.data_type) is not str:
            raise TypeError(
                f"Field.data_type must be a string; got {type_name(self.data_type)}"
            )
        if type(self.nullable) is not bool:
            raise TypeError(
                f"Field.nullable must be a boolean; got {type_name(self.nullable)}"
            )


def require_str(value: object, path: str, /) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be a string; got {type_name(value)}")
    return value


def require_non_empty_str(value: object, path: str, /) -> str:
    require_str(value, path)
    if not value:  # type: ignore[arg-type]
        raise ValueError(f"{path}: invalid_literal: must be a non-empty string")
    return value  # type: ignore[return-value]


def require_int(value: object, path: str, /) -> int:
    if type(value) is not int:
        raise TypeError(f"{path} must be an integer; got {type_name(value)}")
    return value


def require_positive_int(value: object, path: str, /) -> int:
    require_int(value, path)
    if value <= 0:
        raise ValueError(f"{path}: invalid_literal: must be a positive integer")
    return value


def require_non_negative_int(value: object, path: str, /) -> int:
    require_int(value, path)
    if value < 0:
        raise ValueError(f"{path}: invalid_literal: must be a non-negative integer")
    return value


def require_finite_number(value: object, path: str, /) -> int | float:
    if type(value) not in (int, float):
        raise TypeError(f"{path} must be a finite number; got {type_name(value)}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path}: invalid_literal: must be finite")
    return value


def check_table_field_type(data_type: str, path: str, /) -> None:
    if data_type not in TABLE_FIELD_TYPES:
        raise ValueError(
            f"{path}: unsupported_type: unknown portable Arrow type {data_type!r}"
        )


def check_array_dtype(dtype: str, path: str, /) -> None:
    if dtype not in ARRAY_DTYPES:
        raise ValueError(f"{path}: unsupported_type: unknown array dtype {dtype!r}")
