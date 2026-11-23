"""Public eager rolling mean on the native Rust array fixture."""

from __future__ import annotations

from statistics import median
from time import perf_counter_ns

import numpy as np

import calc_flow.array as cfa

ROWS = 1024
CALLS = 10


def validate(result):
    if result.shape != (ROWS,) or result.dtype != "float64":
        raise ValueError("public rolling mean shape or dtype differs")
    arrow = result.to_arrow()
    if len(arrow) != ROWS or arrow.null_count:
        raise ValueError("public rolling mean validity differs")
    if not (arrow.to_numpy().view(np.uint64) == np.float64(1.0).view(np.uint64)).all():
        raise ValueError("public rolling mean Float64 bits differ")


class RollingCountCase:
    """Own one prepared immutable input and validate each fresh public result."""

    def __init__(self, case, _root):
        if case["rows"] != ROWS:
            raise ValueError("public rolling fixture requires 1,024 rows")
        self.input = cfa.from_numpy(np.ones(ROWS, dtype=np.float64))

    def sample(self):
        calls = []
        for _ in range(CALLS):
            start = perf_counter_ns()
            result = cfa.rolling_mean(self.input, window=20)
            calls.append(perf_counter_ns() - start)
            validate(result)
            del result
        return {
            "seconds": median(calls) / 1e9,
            "calls_ns": calls,
            "correctness": {
                "passed": True,
                "rows": ROWS,
                "null_count": 0,
                "calls": CALLS,
            },
        }

    def finish(self):
        return {"state": "completed"}

    def close(self):
        pass
