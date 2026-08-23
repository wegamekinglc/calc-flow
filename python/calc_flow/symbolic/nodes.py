"""Canonical declaration values and immutable node identity.

Implements the frozen ``calc_flow.symbolic.declaration.v1`` byte format from
``.codex/artifacts/specs/symbolic-computation-contract.md`` (D2), including
the versioned primitive reference, the normalized attribute map, and the
deterministic node digest. Node construction is declaration-only: no data
object, callable, or execution path is accepted anywhere in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from calc_flow.symbolic.domains import type_name

ENCODING_VERSION = "calc_flow.symbolic.declaration.v1"

_MAGIC = ENCODING_VERSION.encode("ascii") + b"\x00"
_NODE_TAG = 0x20
_INT64_MIN = -(2**63)
_UINT64_MAX = 2**64 - 1
_CANONICAL_NAN = struct.unpack(">d", b"\x7f\xf8\x00\x00\x00\x00\x00\x00")[0]


def _u64(value: int, /) -> bytes:
    return value.to_bytes(8, "big")


def _text(value: str, /) -> bytes:
    encoded = value.encode("utf-8")
    return _u64(len(encoded)) + encoded


@dataclass(frozen=True, slots=True)
class CNull:
    """The canonical ``null`` value (tag ``0x00``)."""


@dataclass(frozen=True, slots=True)
class CBool:
    """A canonical boolean value; ``false`` is ``0x01`` and ``true`` ``0x02``."""

    value: bool


@dataclass(frozen=True, slots=True)
class CInt:
    """A canonical integer in the portable JSON range (tag ``0x03``)."""

    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int:
            raise TypeError(
                f"canonical integers must be Python int; got {type_name(self.value)}"
            )
        if not _INT64_MIN <= self.value <= _UINT64_MAX:
            raise ValueError(
                "canonical integers must fit the portable JSON range"
                f" [-2^63, 2^64 - 1]; got {self.value}"
            )


@dataclass(frozen=True, slots=True)
class CFloat:
    """A canonical IEEE binary64 value (tag ``0x04``).

    Every NaN is canonicalized to the single quiet-NaN bit pattern; every
    other value keeps its exact bits, so signed zero and infinity sign are
    retained.
    """

    value: float

    def __post_init__(self) -> None:
        if type(self.value) is not float:
            raise TypeError(
                f"canonical floats must be Python float; got {type_name(self.value)}"
            )
        if math.isnan(self.value):
            object.__setattr__(self, "value", _CANONICAL_NAN)


@dataclass(frozen=True, slots=True)
class CStr:
    """A canonical UTF-8 string value (tag ``0x05``)."""

    value: str


@dataclass(frozen=True, slots=True)
class CBytes:
    """A canonical byte string value (tag ``0x06``).

    The declaration language itself rejects bytes; the tag exists so the
    encoder is byte-exact against the v1 golden vectors.
    """

    value: bytes


@dataclass(frozen=True, slots=True)
class CEnum:
    """A canonical enum reference by family and variant (tag ``0x07``)."""

    family: str
    variant: str


@dataclass(frozen=True, slots=True)
class CSeq:
    """A canonical sequence retaining declaration order (tag ``0x08``)."""

    items: tuple[CValue, ...]


@dataclass(frozen=True, slots=True)
class CMap:
    """A canonical map with keys sorted by raw UTF-8 bytes (tag ``0x09``)."""

    entries: tuple[tuple[str, CValue], ...]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, CValue], /) -> CMap:
        entries = tuple(
            sorted(
                mapping.items(),
                key=lambda entry: entry[0].encode("utf-8"),
            )
        )
        return cls(entries)

    def get(self, key: str, /) -> CValue | None:
        for name, value in self.entries:
            if name == key:
                return value
        return None


@dataclass(frozen=True, slots=True)
class CShape:
    """A canonical shape with known or symbolic dimensions (tag ``0x0a``).

    Known dimensions must be non-negative so encoding never reaches a
    low-level unsigned-conversion failure.
    """

    dims: tuple[CValue, ...]

    def __post_init__(self) -> None:
        for dimension in self.dims:
            if isinstance(dimension, CInt) and dimension.value < 0:
                raise ValueError(
                    "canonical shape dimensions must be non-negative known"
                    f" sizes or symbolic identifiers; got {dimension.value}"
                )


@dataclass(frozen=True, slots=True)
class CDType:
    """A canonical Arrow/provider dtype spelling (tag ``0x0b``)."""

    name: str


CValue = (
    CNull
    | CBool
    | CInt
    | CFloat
    | CStr
    | CBytes
    | CEnum
    | CSeq
    | CMap
    | CShape
    | CDType
)


def encode_value(value: CValue, /) -> bytes:
    """Encode one canonical declaration value to its exact v1 bytes."""

    if isinstance(value, CNull):
        return b"\x00"
    if isinstance(value, CBool):
        return b"\x02" if value.value else b"\x01"
    if isinstance(value, CInt):
        if value.value < 0:
            return b"\x03\x01" + _u64(-value.value)
        return b"\x03\x00" + _u64(value.value)
    if isinstance(value, CFloat):
        return b"\x04" + struct.pack(">d", value.value)
    if isinstance(value, CStr):
        return b"\x05" + _text(value.value)
    if isinstance(value, CBytes):
        return b"\x06" + _u64(len(value.value)) + value.value
    if isinstance(value, CEnum):
        return b"\x07" + _text(value.family) + _text(value.variant)
    if isinstance(value, CSeq):
        return (
            b"\x08"
            + _u64(len(value.items))
            + b"".join(encode_value(item) for item in value.items)
        )
    if isinstance(value, CMap):
        return (
            b"\x09"
            + _u64(len(value.entries))
            + b"".join(_text(key) + encode_value(item) for key, item in value.entries)
        )
    if isinstance(value, CShape):
        dimensions = b"".join(_encode_dimension(dim) for dim in value.dims)
        return b"\x0a" + _u64(len(value.dims)) + dimensions
    if isinstance(value, CDType):
        return b"\x0b" + _text(value.name)
    raise TypeError(f"unsupported canonical value; got {type_name(value)}")


def _encode_dimension(dimension: CValue, /) -> bytes:
    if isinstance(dimension, CInt):
        if dimension.value < 0:
            raise ValueError(
                "canonical shape dimensions must be non-negative known"
                f" sizes or symbolic identifiers; got {dimension.value}"
            )
        return b"\x00" + _u64(dimension.value)
    if isinstance(dimension, CStr):
        return b"\x01" + _text(dimension.value)
    raise TypeError(
        f"canonical dimensions must be CInt or CStr; got {type_name(dimension)}"
    )


@dataclass(frozen=True, slots=True)
class OpRef:
    """The versioned primitive identity of one declaration node."""

    name: str
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError(f"OpRef.name must be a string; got {type_name(self.name)}")
        if type(self.version) is not int:
            raise TypeError(
                f"OpRef.version must be an integer; got {type_name(self.version)}"
            )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Node:
    """One immutable declaration node with its precomputed canonical bytes.

    ``node_bytes`` is the exact ``NODE_BYTES`` encoding and ``digest`` the
    lowercase hexadecimal SHA-256 of ``MAGIC || 0x01 || BYTES(node_bytes)``.
    Structural identity is ``digest`` plus exact ``node_bytes`` equality.
    """

    op: OpRef
    args: tuple[Node, ...]
    attrs: CMap
    node_bytes: bytes
    digest: str

    def attr(self, key: str, /) -> CValue | None:
        return self.attrs.get(key)

    def __repr__(self) -> str:
        return f"Node({self.op.name}@{self.op.version}, digest={self.digest})"


@dataclass(frozen=True, slots=True)
class PrimitiveSpec:
    """Catalog entry naming the accepted attributes and their defaults."""

    allowed: frozenset[str]
    defaults: CMap


_NO_ATTRS: dict[str, PrimitiveSpec] = {
    name: PrimitiveSpec(frozenset(), CMap(()))
    for name in (
        "add",
        "and",
        "coalesce",
        "eq",
        "exp",
        "filter",
        "ge",
        "gt",
        "le",
        "log",
        "lt",
        "matmul",
        "mul",
        "ne",
        "neg",
        "not",
        "or",
        "sqrt",
        "sub",
        "truediv",
        "where",
        "abs",
    )
}

_PRIMITIVES: dict[str, PrimitiveSpec] = {
    "table_input": PrimitiveSpec(
        frozenset({"name", "schema", "entity_by", "event_time", "sequence_by"}),
        CMap.from_mapping(
            {
                "entity_by": CSeq(()),
                "event_time": CNull(),
                "sequence_by": CSeq(()),
            }
        ),
    ),
    "parameter": PrimitiveSpec(
        frozenset(
            {"name", "kind", "mutability", "schema", "backend", "dtype", "shape"}
        ),
        CMap(()),
    ),
    "column_ref": PrimitiveSpec(frozenset({"name"}), CMap(())),
    "literal": PrimitiveSpec(frozenset({"value"}), CMap(())),
    "clip": PrimitiveSpec(frozenset({"lower", "upper"}), CMap(())),
    "cast": PrimitiveSpec(frozenset({"data_type"}), CMap(())),
    "lag": PrimitiveSpec(
        frozenset({"periods"}),
        CMap.from_mapping({"periods": CInt(1)}),
    ),
    "delta": PrimitiveSpec(
        frozenset({"periods"}),
        CMap.from_mapping({"periods": CInt(1)}),
    ),
    **{
        name: PrimitiveSpec(
            frozenset({"frame", "min_periods"}),
            CMap.from_mapping({"min_periods": CInt(1)}),
        )
        for name in ("count", "sum", "mean", "min", "max")
    },
    **{
        name: PrimitiveSpec(
            frozenset({"frame", "min_periods", "ddof"}),
            CMap.from_mapping({"min_periods": CInt(1), "ddof": CInt(1)}),
        )
        for name in ("variance", "stddev", "covariance", "correlation")
    },
    **{
        name: PrimitiveSpec(
            frozenset(
                {
                    "grouping",
                    "direction",
                    "tie_method",
                    "null_placement",
                    "min_samples",
                }
            ),
            CMap.from_mapping(
                {
                    "direction": CEnum("direction", "ascending"),
                    "tie_method": CEnum("rank_tie_method", "average"),
                    "null_placement": CEnum("null_placement", "exclude"),
                    "min_samples": CInt(1),
                }
            ),
        )
        for name in ("rank", "percentile")
    },
    "demean": PrimitiveSpec(
        frozenset({"grouping", "min_samples"}),
        CMap.from_mapping({"min_samples": CInt(1)}),
    ),
    "zscore": PrimitiveSpec(
        frozenset({"grouping", "min_samples", "ddof"}),
        CMap.from_mapping({"min_samples": CInt(1), "ddof": CInt(0)}),
    ),
    "winsorize": PrimitiveSpec(
        frozenset({"grouping", "min_samples", "lower", "upper"}),
        CMap.from_mapping({"min_samples": CInt(1)}),
    ),
    "project": PrimitiveSpec(frozenset({"columns"}), CMap(())),
    "attach_columns": PrimitiveSpec(frozenset({"names"}), CMap(())),
    "from_columns": PrimitiveSpec(frozenset({"columns", "backend"}), CMap(())),
    "window_tumbling": PrimitiveSpec(
        frozenset({"event_time", "size_micros", "group_by"}),
        CMap.from_mapping({"group_by": CSeq(())}),
    ),
    "window_hopping": PrimitiveSpec(
        frozenset({"event_time", "size_micros", "slide_micros", "group_by"}),
        CMap.from_mapping({"group_by": CSeq(())}),
    ),
    **_NO_ATTRS,
}


def build(
    name: str,
    args: Sequence[Node],
    attrs: Mapping[str, CValue],
    /,
    *,
    version: int = 1,
) -> Node:
    """Build one normalized node with materialized defaults and its digest."""

    spec = _PRIMITIVES.get(name)
    if spec is None:
        raise ValueError(f"unknown symbolic primitive {name!r}")
    unknown = frozenset(attrs) - spec.allowed
    if unknown:
        raise ValueError(
            f"primitive {name!r} does not accept attributes {sorted(unknown)}"
        )
    merged: dict[str, CValue] = dict(spec.defaults.entries)
    merged.update(attrs)
    attr_map = CMap.from_mapping(merged)
    children = tuple(args)
    body = (
        bytes((_NODE_TAG,))
        + _text(name)
        + _text(str(version))
        + _u64(len(children))
        + b"".join(bytes.fromhex(child.digest) for child in children)
        + encode_value(attr_map)
    )
    digest = hashlib.sha256(_MAGIC + b"\x01" + _u64(len(body)) + body).hexdigest()
    return Node(OpRef(name, version), children, attr_map, body, digest)


def literal_value(value: object, /) -> CValue:
    """Convert one strict JSON scalar to its canonical value.

    Non-finite floats and integers outside the portable JSON range are
    rejected before any node is built.
    """

    if value is None:
        return CNull()
    if type(value) is bool:
        return CBool(value)
    if type(value) is int:
        if not _INT64_MIN <= value <= _UINT64_MAX:
            raise ValueError(
                "calc_flow.symbolic.literal.value: invalid_literal: integer"
                " literals must fit the portable JSON range [-2^63, 2^64 - 1]"
            )
        return CInt(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(
                "calc_flow.symbolic.literal.value: invalid_literal:"
                " floating-point literals must be finite"
            )
        return CFloat(value)
    if type(value) is str:
        return CStr(value)
    raise ValueError(
        "calc_flow.symbolic.literal.value: invalid_literal: declaration"
        f" literals must be strict JSON scalars; got {type_name(value)}"
    )


def format_value(value: CValue, /) -> str:
    """Render one canonical value deterministically for ``explain`` output."""

    if isinstance(value, CNull):
        return "null"
    if isinstance(value, CBool):
        return "true" if value.value else "false"
    if isinstance(value, CInt):
        return str(value.value)
    if isinstance(value, CFloat):
        return repr(value.value)
    if isinstance(value, CStr):
        return json.dumps(value.value)
    if isinstance(value, CDType):
        return value.name
    if isinstance(value, CBytes):
        return f"bytes({value.value.hex()})"
    if isinstance(value, CEnum):
        return f"{value.family}.{value.variant}"
    if isinstance(value, CSeq):
        return "[" + ", ".join(format_value(item) for item in value.items) + "]"
    if isinstance(value, CMap):
        return (
            "{"
            + ", ".join(f"{key}={format_value(item)}" for key, item in value.entries)
            + "}"
        )
    if isinstance(value, CShape):
        return "shape(" + ", ".join(format_value(dim) for dim in value.dims) + ")"
    raise TypeError(f"unsupported canonical value; got {type_name(value)}")


def explain_node(node: Node, /) -> str:
    """Render the complete declaration tree deterministically."""

    def header(current: Node) -> str:
        attrs = ", ".join(
            f"{key}={format_value(item)}" for key, item in current.attrs.entries
        )
        rendered = f"[{attrs}]" if attrs else ""
        return (
            f"{current.op.name}@{current.op.version}{rendered} digest={current.digest}"
        )

    lines = [header(node)]

    def walk(current: Node, depth: int) -> None:
        for index, child in enumerate(current.args):
            lines.append("  " * depth + f"[{index}] {header(child)}")
            walk(child, depth + 1)

    walk(node, 1)
    return "\n".join(lines)
