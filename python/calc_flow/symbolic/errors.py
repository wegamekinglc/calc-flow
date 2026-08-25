"""Frozen symbolic issue codes and the lowering ``CompileError`` boundary.

The codes are the frozen vocabulary of
``.codex/artifacts/api-notes/symbolic-computation-engine.md`` section 4. Every
engine-originated symbolic failure begins with a stable path followed by the
code, matching ``{path}: {code}: {message}``.
"""

from __future__ import annotations

from typing import NoReturn

from calc_flow.errors import CompileError

CAPABILITY_MISMATCH = "capability_mismatch"
DUPLICATE_NAME = "duplicate_name"
INVALID_LITERAL = "invalid_literal"
ORDERING_REQUIRED = "ordering_required"
SCHEMA_MISMATCH = "schema_mismatch"
UNBOUNDED_STATE = "unbounded_state"
UNKNOWN_PRIMITIVE_VERSION = "unknown_primitive_version"
UNRESOLVED_TYPE = "unresolved_type"
UNSUPPORTED_MODE = "unsupported_mode"
UNSUPPORTED_TYPE = "unsupported_type"

ISSUE_CODES: tuple[str, ...] = (
    CAPABILITY_MISMATCH,
    DUPLICATE_NAME,
    INVALID_LITERAL,
    ORDERING_REQUIRED,
    SCHEMA_MISMATCH,
    UNBOUNDED_STATE,
    UNKNOWN_PRIMITIVE_VERSION,
    UNRESOLVED_TYPE,
    UNSUPPORTED_MODE,
    UNSUPPORTED_TYPE,
)

__all__ = [
    "CAPABILITY_MISMATCH",
    "DUPLICATE_NAME",
    "INVALID_LITERAL",
    "ISSUE_CODES",
    "ORDERING_REQUIRED",
    "SCHEMA_MISMATCH",
    "UNBOUNDED_STATE",
    "UNKNOWN_PRIMITIVE_VERSION",
    "UNRESOLVED_TYPE",
    "UNSUPPORTED_MODE",
    "UNSUPPORTED_TYPE",
    "raise_compile",
]


def raise_compile(path: str, code: str, message: str, /) -> NoReturn:
    """Raise one ``CompileError`` in the frozen ``{path}: {code}: {message}`` shape."""

    raise CompileError(f"{path}: {code}: {message}")
