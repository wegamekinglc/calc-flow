"""Exact integer transport for ASOF diagnostics over the Studio JSON API."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictBool, StrictStr

_U64_MAX = 2**64 - 1
_I64_MIN = -(2**63)
_I64_MAX = 2**63 - 1


def _unsigned_range(value: str) -> str:
    if int(value) > _U64_MAX:
        raise ValueError("ASOF counter exceeds u64")
    return value


def _signed_range(value: str) -> str:
    if not _I64_MIN <= int(value) <= _I64_MAX:
        raise ValueError("ASOF watermark exceeds i64")
    return value


type UnsignedDecimal = Annotated[
    StrictStr,
    Field(pattern=r"^(0|[1-9][0-9]*)$", max_length=20),
    AfterValidator(_unsigned_range),
]
type SignedDecimal = Annotated[
    StrictStr,
    Field(pattern=r"^(0|-?[1-9][0-9]*)$", max_length=20),
    AfterValidator(_signed_range),
]


class StreamAsofJoinSideMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted_rows: UnsignedDecimal
    late_rows: UnsignedDecimal
    duplicate_rows: UnsignedDecimal
    watermark_micros: SignedDecimal | None
    idle: StrictBool
    ended: StrictBool


class StreamAsofJoinMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: StrictStr = Field(min_length=1)
    left: StreamAsofJoinSideMetrics
    right: StreamAsofJoinSideMetrics
    pending_left_rows: UnsignedDecimal
    retained_right_rows: UnsignedDecimal
    identity_only_rows: UnsignedDecimal
    state_rows: UnsignedDecimal
    state_bytes: UnsignedDecimal
    emitted_left_rows: UnsignedDecimal
    matched_rows: UnsignedDecimal
    unmatched_rows: UnsignedDecimal
    evicted_right_rows: UnsignedDecimal
    state_limit_failures: UnsignedDecimal
    workspace_limit_failures: UnsignedDecimal
    output_limit_failures: UnsignedDecimal
    output_watermark_micros: SignedDecimal | None


def _integer(value: object, field: str, *, signed: bool = False) -> str:
    if type(value) is not int:
        raise TypeError(f"ASOF {field} must be an exact integer")
    lower, upper = (_I64_MIN, _I64_MAX) if signed else (0, _U64_MAX)
    if not lower <= value <= upper:
        raise ValueError(f"ASOF {field} exceeds its native integer range")
    return str(value)


def _native_fields(raw: object) -> dict[str, object]:
    if type(raw) is not dict:
        raise TypeError("ASOF metrics must be a data mapping")
    result = {}
    for field, value in raw.items():
        if field in {"idle", "ended", "node_id"}:
            result[field] = value
        elif field in {"left", "right"}:
            result[field] = _native_fields(value)
        elif field in {"watermark_micros", "output_watermark_micros"}:
            result[field] = (
                None if value is None else _integer(value, field, signed=True)
            )
        else:
            result[field] = _integer(value, field)
    return result


def stream_asof_progress(raw: object) -> tuple[dict[str, object], ...] | None:
    """Normalize native status or validate an already serialized worker event."""
    if raw is None:
        return None
    if type(raw) is dict:
        if any(type(node_id) is not str for node_id in raw):
            raise TypeError("ASOF node identifiers must be strings")
        entries = [
            {**_native_fields(metrics), "node_id": node_id}
            for node_id, metrics in sorted(raw.items())
        ]
    elif type(raw) in {tuple, list}:
        entries = raw
    else:
        raise TypeError("ASOF progress must be a native mapping or worker list")
    return tuple(
        StreamAsofJoinMetrics.model_validate(entry).model_dump(mode="json")
        for entry in entries
    )
