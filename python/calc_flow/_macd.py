from __future__ import annotations

from typing import Generic, TypeVar

from calc_flow._compat import dataclass

T = TypeVar("T")
_MACD_PARAMETER_NAMES = (
    "fast_span",
    "slow_span",
    "signal_span",
    "min_periods",
    "signal_min_periods",
)


@dataclass(frozen=True, slots=True)
class MacdOutputs(Generic[T]):
    """Aligned line, signal and histogram values from one MACD state owner."""

    line: T
    signal: T
    histogram: T


def _macd_parameters(
    values: tuple[int, int, int, int, int], prefix: str
) -> tuple[int, int, int, int, int]:
    for name, value in zip(_MACD_PARAMETER_NAMES, values):
        if type(value) is not int:
            raise TypeError(f"{prefix}.{name} must be an integer")
        if not 1 <= value <= 2**64 - 1:
            raise ValueError(f"{prefix}.{name}: must be a positive u64")
    if values[0] >= values[1]:
        raise ValueError(f"{prefix}.fast_span: must be less than slow_span")
    return values
