"""TA-Lib adapter for appends preceded by a bounded warm-history context."""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass

import numpy as np

from benchmarks.rolling_indicator_comparison import TaLibMethod


@dataclass(frozen=True, slots=True)
class BoundedHistoryTaLibMethod(TaLibMethod):
    """Recalculate bounded context and return only the appended rows."""

    history_rows: int = 0

    def __post_init__(self) -> None:
        if type(self.history_rows) is not int or not 0 <= self.history_rows < len(
            self.prices
        ):
            raise ValueError("TA-Lib history must leave at least one appended row")

    def run(self) -> np.ndarray:
        return TaLibMethod.run(self)[self.history_rows :]


def timed_ta_lib_append(method: BoundedHistoryTaLibMethod, iterations: int) -> float:
    """Time only complete bounded-context TA-Lib calls, with output validation."""
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        started = time.perf_counter_ns()
        for _iteration in range(iterations):
            result = method.run()
        seconds = (time.perf_counter_ns() - started) / 1_000_000_000
    finally:
        if gc_was_enabled:
            gc.enable()
    if len(result) != len(method.prices) - method.history_rows:
        raise RuntimeError("TA-Lib benchmark produced an invalid row count")
    return seconds / iterations
