"""Immutable data declarations for native bounded backward ASOF joins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from calc_flow.join_spec import timedelta_micros

_MAX_SAFE_INTEGER = 9_007_199_254_740_991


def _identity_columns(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a Sequence[str]")
    copied = tuple(values)
    if not copied or any(type(value) is not str or not value for value in copied):
        raise ValueError(f"{field} must contain non-empty exact strings")
    if len(set(copied)) != len(copied):
        raise ValueError(f"{field} must not contain duplicate columns")
    return copied


@dataclass(frozen=True, slots=True)
class AsofStateLimits:
    """Positive aggregate retained-state row and byte limits.

    Values are exact integers up to 9,007,199,254,740,991. The byte limit also
    bounds transient operator workspace separately; it is not a process RSS cap.
    """

    max_state_rows: int
    max_state_bytes: int

    def __post_init__(self) -> None:
        for name in ("max_state_rows", "max_state_bytes"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact int")
            if not 1 <= value <= _MAX_SAFE_INTEGER:
                raise ValueError(f"{name} must be in 1..={_MAX_SAFE_INTEGER}")


@dataclass(frozen=True, slots=True)
class AsofJoinSide:
    """Exact identity columns and output prefix for one ASOF input.

    Key/time/sequence fields must be non-null. Sequence tuples have typed
    lexicographic order. Column sequences are copied to immutable tuples.
    """

    keys: Sequence[str]
    event_time: str
    sequence_by: Sequence[str]
    prefix: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", _identity_columns(self.keys, "keys"))
        object.__setattr__(
            self, "sequence_by", _identity_columns(self.sequence_by, "sequence_by")
        )
        if type(self.event_time) is not str or not self.event_time:
            raise TypeError("event_time must be a non-empty exact string")
        if (
            type(self.prefix) is not str
            or not self.prefix.isascii()
            or not self.prefix.isidentifier()
        ):
            raise ValueError("prefix must be a non-empty ASCII identifier")


@dataclass(frozen=True, slots=True)
class AsofJoinSpec:
    """Backward, inclusive, left-preserving and final-only ASOF settings.

    ``tolerance`` is a finite non-negative integral microsecond distance, not a
    wall-clock waiting timeout. Late rows either fail the job or are dropped and
    counted according to ``late_policy``; accepted duplicate identities fail.
    """

    left: AsofJoinSide
    right: AsofJoinSide
    tolerance: timedelta
    limits: AsofStateLimits
    late_policy: Literal["error", "drop"] = "error"

    def __post_init__(self) -> None:
        for name in ("left", "right"):
            if type(getattr(self, name)) is not AsofJoinSide:
                raise TypeError(f"{name} must be a calc_flow.AsofJoinSide")
        if type(self.limits) is not AsofStateLimits:
            raise TypeError("limits must be a calc_flow.AsofStateLimits")
        timedelta_micros(self.tolerance, "tolerance")
        if len(self.left.keys) != len(self.right.keys):
            raise ValueError("left.keys and right.keys must have equal length")
        if self.left.prefix == self.right.prefix:
            raise ValueError("prefixes must be distinct")
        if type(self.late_policy) is not str or self.late_policy not in (
            "error",
            "drop",
        ):
            raise ValueError("late_policy must be 'error' or 'drop'")


def _side_wire(side: AsofJoinSide) -> dict[str, object]:
    return {
        "keys": list(side.keys),
        "event_time": side.event_time,
        "sequence_by": list(side.sequence_by),
        "prefix": side.prefix,
    }


def _asof_wire_spec(spec: AsofJoinSpec) -> dict[str, object]:
    return {
        "left": _side_wire(spec.left),
        "right": _side_wire(spec.right),
        "tolerance_micros": timedelta_micros(spec.tolerance, "tolerance"),
        "limits": {
            "max_state_rows": spec.limits.max_state_rows,
            "max_state_bytes": spec.limits.max_state_bytes,
        },
        "late_policy": spec.late_policy,
    }
