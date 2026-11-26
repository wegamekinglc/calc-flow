"""Report operator cells for the benchmark suite's isolated worker protocol."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np

from benchmarks.report_operator_cases import FINANCE_BACKEND, STREAM_BACKEND
from benchmarks.report_operator_comparison import NativeCase, start_finance
from benchmarks.report_operator_inputs import OperatorWorkload, columns
from benchmarks.report_operator_oracles import oracle


class ReportOperatorCase:
    """Own one backend, its input, warm result, and fixture for a worker round."""

    def __init__(self, case: dict, root: Path) -> None:
        self.workload = OperatorWorkload(
            case["rows"], case["entities"], case["chunk_ticks"]
        )
        self.root = root
        self.samples = 0
        self.finance = None
        self.loop = None
        self.warm = None
        name = case["scenario"]
        backend = case["backend"]
        if backend == FINANCE_BACKEND:
            python = os.environ.get("FINANCE_PYTHON_PYTHON")
            if not python:
                raise ValueError(
                    "FINANCE_PYTHON_PYTHON must name the Python 3.9 worker"
                )
            data = columns(self.workload.rows, self.workload.entities)
            expected = oracle(name, data, self.workload.entities)
            self.finance = start_finance(
                Path(python), name, self.workload, root, expected
            )
        else:
            self.native = NativeCase(name, self.workload)
            if backend == STREAM_BACKEND:
                self.loop = asyncio.new_event_loop()

    def sample(self) -> dict:
        correctness = {"passed": True, "rows": self.workload.rows}
        if self.finance is not None:
            return {
                "seconds": self.finance.sample(1),
                "correctness": correctness,
                "finance_python": self.finance.identity,
            }
        if self.loop is None:
            output, seconds = self.native.batch()
        else:
            output, seconds = self.loop.run_until_complete(
                self.native.stream(self.root / f"sample-{self.samples}")
            )
        self.samples += 1
        values = self.native.validate(output)
        if self.warm is None:
            self.warm = values
        else:
            np.testing.assert_array_equal(values, self.warm)
        return {"seconds": seconds, "correctness": correctness}

    def finish(self) -> dict:
        return {"state": "completed"}

    def close(self) -> None:
        if self.finance is not None:
            self.finance.close()
        if self.loop is not None:
            try:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            finally:
                self.loop.close()
