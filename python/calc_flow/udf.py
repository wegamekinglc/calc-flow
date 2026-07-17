from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _validate_scalar_udf_registration(
    input_types: Sequence[str], function: Any
) -> tuple[str, ...]:
    if isinstance(input_types, (str, bytes)) or not isinstance(input_types, Sequence):
        raise TypeError("input_types must be a sequence of Arrow type names")
    copied_types = tuple(input_types)
    if len(copied_types) > 64:
        raise ValueError("input_types must contain at most 64 entries")
    if not all(isinstance(value, str) for value in copied_types):
        raise TypeError("input_types must contain only Arrow type names")
    if not callable(function):
        raise TypeError("function must be callable")
    return copied_types
