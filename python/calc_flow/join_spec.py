"""The stream-join wire specification shared by every surface.

``calc_flow.pipeline`` (project JSON), ``calc_flow.symbolic`` (node
attrs), and the lowering path each encode the same bounded inner join
contract. The value containers, validators, and the single wire-spec
builder live here so the contract is defined once and the symbolic layer
no longer imports runtime-side helpers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta

STREAM_JOIN_MAX_SAFE_JSON_INTEGER = 9_007_199_254_740_991


@dataclass(frozen=True, slots=True)
class JoinTimeBounds:
    """Inclusive non-negative event-time distances around a left row."""

    before: timedelta
    after: timedelta

    def __post_init__(self) -> None:
        timedelta_micros(self.before, "before")
        timedelta_micros(self.after, "after")


@dataclass(frozen=True, slots=True)
class JoinStateLimits:
    """Required logical state and per-input match limits."""

    max_state_rows_per_side: int
    max_state_bytes_per_side: int
    max_matches_per_input_batch: int

    def __post_init__(self) -> None:
        for field_name in (
            "max_state_rows_per_side",
            "max_state_bytes_per_side",
            "max_matches_per_input_batch",
        ):
            value = getattr(self, field_name)
            if value.__class__ is not int:
                raise TypeError(f"{field_name} must be an exact int")
            if not 1 <= value <= STREAM_JOIN_MAX_SAFE_JSON_INTEGER:
                raise ValueError(
                    f"{field_name} must be in 1..={STREAM_JOIN_MAX_SAFE_JSON_INTEGER}"
                )


@dataclass(frozen=True, slots=True)
class JoinSideWire:
    """One join side's wire-level keys, event time, and output prefix."""

    keys: tuple[str, ...]
    event_time: str
    prefix: str


def require_equal_key_counts(
    left_keys: Sequence[str], right_keys: Sequence[str]
) -> None:
    if len(left_keys) != len(right_keys):
        raise ValueError("left_keys and right_keys must have equal length")


def require_event_time_columns(left_event_time: str, right_event_time: str) -> None:
    for field_name, value in (
        ("left_event_time", left_event_time),
        ("right_event_time", right_event_time),
    ):
        if not isinstance(value, str) or not value:
            raise TypeError(f"{field_name} must be a non-empty string")


def require_join_bounds(bounds: JoinTimeBounds) -> None:
    if not isinstance(bounds, JoinTimeBounds):
        raise TypeError("bounds must be a calc_flow.JoinTimeBounds")


def require_join_limits(limits: JoinStateLimits) -> None:
    if not isinstance(limits, JoinStateLimits):
        raise TypeError("limits must be a calc_flow.JoinStateLimits")


def _portable_identifier(value: object) -> bool:
    return isinstance(value, str) and value.isidentifier() and value.isascii()


def require_distinct_prefixes(left_prefix: str, right_prefix: str) -> None:
    if not _portable_identifier(left_prefix) or not _portable_identifier(right_prefix):
        raise ValueError("prefixes must be distinct portable identifiers")
    if left_prefix == right_prefix:
        raise ValueError("prefixes must be distinct portable identifiers")


def timedelta_micros(value: timedelta, field_name: str) -> int:
    # Exact-type checks: bool is an int subclass and timedelta subclasses
    # must not satisfy the wire contract, so compare classes directly.
    if value.__class__ is not timedelta:
        raise TypeError(f"{field_name} must be an exact datetime.timedelta")
    micros = (
        value.days * 86_400_000_000 + value.seconds * 1_000_000 + value.microseconds
    )
    if not 0 <= micros <= STREAM_JOIN_MAX_SAFE_JSON_INTEGER:
        raise ValueError(
            f"{field_name} must resolve to "
            f"0..={STREAM_JOIN_MAX_SAFE_JSON_INTEGER} microseconds"
        )
    return micros


def bounds_wire(before_micros: int, after_micros: int) -> dict[str, int]:
    """Build the wire ``bounds`` object from resolved microsecond distances."""

    return {"before_micros": before_micros, "after_micros": after_micros}


def limits_wire(
    max_state_rows_per_side: int,
    max_state_bytes_per_side: int,
    max_matches_per_input_batch: int,
) -> dict[str, int]:
    """Build the wire ``limits`` object from the three resolved limits."""

    return {
        "max_state_rows_per_side": max_state_rows_per_side,
        "max_state_bytes_per_side": max_state_bytes_per_side,
        "max_matches_per_input_batch": max_matches_per_input_batch,
    }


def join_wire_spec(
    left: JoinSideWire,
    right: JoinSideWire,
    bounds: Mapping[str, int],
    limits: Mapping[str, int],
    *,
    join_type: str = "inner",
) -> dict[str, object]:
    """Build the canonical ``stream_join`` operator wire specification."""

    return {
        "join_type": join_type,
        "left_keys": list(left.keys),
        "right_keys": list(right.keys),
        "left_event_time": left.event_time,
        "right_event_time": right.event_time,
        "bounds": dict(bounds),
        "limits": dict(limits),
        "left_prefix": left.prefix,
        "right_prefix": right.prefix,
    }
