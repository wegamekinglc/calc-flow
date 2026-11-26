"""Opt-in native Batch, Streaming, and Finance-Python operator comparisons."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa

import calc_flow
from benchmarks.engine_stream import BASE_MICROS, run_stream, stream_events
from benchmarks.native_release_identity import file_hash, release_identity
from benchmarks.report_operator_cases import (
    CROSS_SECTION,
    NO_FINANCE_EQUIVALENT,
    OPERATORS,
    REPORT_INDICATOR_PREFIX,
    SCHEDULED_CHUNK_TICKS,
    SCHEDULED_ENTITIES,
    SCHEDULED_ROWS,
    validate_finance_identity,
)
from benchmarks.report_operator_inputs import OperatorWorkload, columns
from benchmarks.report_operator_oracles import oracle
from benchmarks.rolling_indicator_comparison import FinancePythonWorker
from calc_flow import (
    Batch,
    Program,
    Runtime,
    _native,
    cs,
    exact_time,
    row,
    rows,
    table_input,
    ts,
)

_FLOAT64 = pa.float64()
_NATIVE_VALUE_TYPES = {
    **dict.fromkeys(
        ("argmax20", "argmin20", "rank20", "count20", "unique_count20"), pa.uint64()
    ),
    **dict.fromkeys(
        (
            "all_true20",
            "any_true20",
            "cs_top16",
            "cs_bottom16",
            "cs_top_quantile25",
            "cs_bottom_quantile25",
        ),
        pa.bool_(),
    ),
    "count_positive20": pa.int64(),
}


def _table(data: dict[str, np.ndarray]) -> pa.Table:
    event_time = pa.array(
        BASE_MICROS + 1_000_000 * data["tick"], type=pa.timestamp("us", "UTC")
    )
    keys = {
        "event_time": event_time,
        "sequence": pa.array(data["sequence"]),
        "symbol": pa.array(data["symbol"]),
    }
    keys.update(
        {
            name: pa.array(data[name], from_pandas=name == "fill")
            for name in ("x", "y", "unit", "positive", "signed", "flag", "fill")
        }
    )
    schema = pa.schema(
        [
            pa.field(name, value.type, nullable=name == "fill")
            for name, value in keys.items()
        ]
    )
    return pa.Table.from_arrays(list(keys.values()), schema=schema)


def _program(table: pa.Table, name: str):
    source = table_input(
        "input",
        schema=table.schema,
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    x, y, unit, positive, signed, flag = (
        source[key] for key in ("x", "y", "unit", "positive", "signed", "flag")
    )
    frame = rows(20)
    group = exact_time(source["event_time"])
    value = {
        "average": lambda: ts.average(x),
        "ma20": lambda: ts.mean(x, window=frame),
        "ema20": lambda: ts.ema(x, span=20),
        "macd_12_26": lambda: ts.macd(x),
        "rsi20": lambda: ts.rsi(x, period=20),
        "correlation20": lambda: ts.correlation(x, y, window=frame),
        "decay20": lambda: ts.decay(x, window=frame),
        "max20": lambda: ts.max(x, window=frame),
        "argmax20": lambda: ts.argmax(x, window=frame),
        "min20": lambda: ts.min(x, window=frame),
        "argmin20": lambda: ts.argmin(x, window=frame),
        "rank20": lambda: ts.rank(x, window=frame),
        "quantile20": lambda: ts.quantile(x, window=frame),
        "count20": lambda: ts.count(x, window=frame),
        "unique_count20": lambda: ts.unique_count(x, window=frame),
        "all_true20": lambda: ts.all_true(flag, window=frame),
        "any_true20": lambda: ts.any_true(flag, window=frame),
        "sum20": lambda: ts.sum(x, window=frame),
        "variance20": lambda: ts.variance(x, window=frame),
        "stddev20": lambda: ts.stddev(x, window=frame),
        "count_positive20": lambda: ts.count_positive(signed, window=frame),
        "mean_positive20": lambda: ts.mean_positive(signed, window=frame),
        "lag1": lambda: ts.lag(x),
        "delta1": lambda: ts.delta(x),
        "return_simple1": lambda: x / ts.lag(x) - 1.0,
        "return_log1": lambda: row.log(x / ts.lag(x)),
        "cs_rank": lambda: cs.rank(x, group=group),
        "cs_top16": lambda: cs.top(x, group=group, count=16),
        "cs_bottom16": lambda: cs.bottom(x, group=group, count=16),
        "cs_top_quantile25": lambda: cs.top_quantile(x, group=group, fraction=0.25),
        "cs_bottom_quantile25": lambda: cs.bottom_quantile(
            x, group=group, fraction=0.25
        ),
        "cs_mean": lambda: cs.mean(x, group=group),
        "cs_demean": lambda: cs.demean(x, group=group),
        "cs_percentile": lambda: cs.percentile(x, group=group),
        "cs_zscore": lambda: cs.zscore(x, group=group),
        "cs_residual": lambda: cs.residual(x, y, group=group),
        "cs_mean_fill": lambda: cs.mean_fill(source["fill"], group=group),
        "last": lambda: ts.last(x),
        "add": lambda: x + y,
        "subtract": lambda: x - y,
        "multiply": lambda: x * y,
        "divide": lambda: x / y,
        "power": lambda: row.pow(positive, 2.5),
        "negate": lambda: -signed,
        "sign": lambda: row.sign(signed),
        "maximum": lambda: row.maximum(x, y),
        "minimum": lambda: row.minimum(x, y),
        "sqrt": lambda: row.sqrt(positive),
        "exp": lambda: row.exp(signed),
        "log": lambda: row.log(positive),
        "abs": lambda: row.abs(signed),
        "acos": lambda: row.acos(unit),
        "acosh": lambda: row.acosh(positive),
        "asin": lambda: row.asin(unit),
        "asinh": lambda: row.asinh(signed),
        "norminv": lambda: row.norminv(unit),
        "ceil": lambda: row.ceil(signed),
        "floor": lambda: row.floor(signed),
        "round": lambda: row.round(signed),
        "where": lambda: row.where(flag, x, y),
    }[name]()
    return Program(
        "report-operator",
        mode="batch",
        inputs=(source,),
        outputs=(("result", source.select("sequence", value=value)),),
    )


def validate_output(
    output: pa.Table, expected: np.ndarray, *, value_type: pa.DataType = _FLOAT64
) -> np.ndarray:
    if output.column_names != ["sequence", "value"]:
        raise AssertionError(f"unexpected output columns: {output.column_names}")
    for name, expected_type in (("sequence", pa.uint64()), ("value", value_type)):
        actual_type = output.schema.field(name).type
        if actual_type != expected_type:
            raise AssertionError(
                f"{name} type must be {expected_type}, got {actual_type}"
            )
    ordered = output.sort_by("sequence")
    np.testing.assert_array_equal(
        ordered["sequence"].to_numpy(),
        np.arange(len(expected), dtype=np.uint64),
        err_msg="sequence must identify each input row exactly once",
    )
    values = ordered["value"].cast(_FLOAT64).to_numpy(zero_copy_only=False)
    np.testing.assert_allclose(values, expected, rtol=1e-7, atol=1e-8, equal_nan=True)
    return values


class NativeCase:
    """Own prebuilt inputs, a reusable Batch plan, and the untimed oracle."""

    def __init__(self, name: str, workload: OperatorWorkload) -> None:
        data = columns(workload.rows, workload.entities)
        table = _table(data)
        program = _program(table, name)
        self.batch_plan = program.compile_batch(Runtime())
        self.stream_program = program.for_mode("stream")
        self.batch_input = {"input": Batch.from_pyarrow(table)}
        self.events = stream_events(
            table, workload.entities, batch_rows=workload.chunk_rows
        )
        self.expected_rows = workload.rows
        self.expected = oracle(name, data, workload.entities)
        self.value_type = _NATIVE_VALUE_TYPES.get(name, _FLOAT64)

    def batch(self) -> tuple[pa.Table, float]:
        started = time.perf_counter_ns()
        result = (
            self.batch_plan.execute(self.batch_input).outputs["output"].to_pyarrow()
        )
        return result, (time.perf_counter_ns() - started) / 1e9

    async def stream(self, root: Path) -> tuple[pa.Table, float]:
        return await run_stream(
            self.stream_program.compile_stream(Runtime()),
            {"input": self.events},
            root,
            self.expected_rows,
        )

    def validate(self, output: pa.Table) -> np.ndarray:
        return validate_output(output, self.expected, value_type=self.value_type)


WORKLOADS = {
    "64k": OperatorWorkload(SCHEDULED_ROWS, SCHEDULED_ENTITIES, SCHEDULED_CHUNK_TICKS),
    "762k": OperatorWorkload(762_048, 64, 100),
    "3000-stocks": OperatorWorkload(762_000, 3_000, 10),
}


def _verified_finance(
    worker: FinancePythonWorker, warm_path: Path, operator: str, expected: np.ndarray
) -> None:
    identity = worker.identity
    validate_finance_identity(identity, operator, len(expected))
    values = np.load(warm_path, allow_pickle=False)
    if hashlib.sha256(values.tobytes()).hexdigest() != identity["sha256"]:
        raise ValueError(
            "Finance-Python warm output checksum differs from worker reply"
        )
    np.testing.assert_allclose(values, expected, rtol=1e-7, atol=1e-8, equal_nan=True)


def start_finance(
    python: Path,
    name: str,
    workload: OperatorWorkload,
    root: Path,
    expected: np.ndarray,
) -> FinancePythonWorker:
    """Start a legacy worker and prove its aligned warm result."""
    root.mkdir(parents=True, exist_ok=True)
    warm_path = root / "finance-warm.npy"
    worker = FinancePythonWorker(
        python,
        rows=workload.rows,
        entities=workload.entities,
        window=20,
        warm_output=warm_path,
        indicator=REPORT_INDICATOR_PREFIX + name,
    )
    try:
        _verified_finance(worker, warm_path, name, expected)
    except BaseException:
        worker.close()
        raise
    return worker


def collect_case(
    name: str,
    workload: OperatorWorkload,
    root: Path,
    *,
    samples: int,
    finance_python: Path | None = None,
) -> dict:
    """Validate both native engines then collect rotating single-call samples."""
    if samples < 0:
        raise ValueError("samples must be nonnegative")
    native = NativeCase(name, workload)
    root.mkdir(parents=True, exist_ok=True)
    warm_batch, _ = native.batch()
    warm_stream, _ = asyncio.run(native.stream(root / "warm-stream"))
    warm = {
        "batch": native.validate(warm_batch),
        "stream": native.validate(warm_stream),
    }
    unsupported = NO_FINANCE_EQUIVALENT.get(name)
    finance_worker = (
        start_finance(finance_python, name, workload, root, native.expected)
        if finance_python is not None and unsupported is None
        else nullcontext()
    )
    observations = {"batch": [], "stream": [], "finance": []}
    with finance_worker as worker:
        engines = ("batch", "stream")
        if worker is not None:
            engines = (*engines, "finance")
        for index in range(samples):
            offset = index % len(engines)
            for engine in engines[offset:] + engines[:offset]:
                if engine == "finance":
                    assert worker is not None
                    seconds = worker.sample(1)
                else:
                    output, seconds = (
                        native.batch()
                        if engine == "batch"
                        else asyncio.run(native.stream(root / f"stream-{index}"))
                    )
                    np.testing.assert_array_equal(native.validate(output), warm[engine])
                observations[engine].append(seconds)
        finance = (
            {"status": "unsupported", "reason": unsupported}
            if unsupported is not None
            else {"status": "not_requested"}
            if worker is None
            else {
                "status": "passed",
                "identity": worker.identity,
                "seconds": observations["finance"],
            }
        )
    medians = {
        engine: median(seconds) for engine, seconds in observations.items() if seconds
    }
    return {
        "operator": name,
        "correctness": "passed",
        "batch_seconds": observations["batch"],
        "stream_seconds": observations["stream"],
        "finance": finance,
        "median_seconds": medians,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=WORKLOADS, default="64k")
    parser.add_argument("--case", action="append", choices=OPERATORS)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--finance-python", type=Path, default=os.environ.get("FINANCE_PYTHON_PYTHON")
    )
    parser.add_argument(
        "--build",
        type=Path,
        help="Verify the installed package using benchmark-suite release.json",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    names = args.case or (
        CROSS_SECTION if args.workload == "3000-stocks" else OPERATORS
    )
    if len(set(names)) != len(names):
        parser.error("each operator may be selected once")
    workload = WORKLOADS[args.workload]
    build = (
        release_identity(args.build)
        if args.build is not None
        else {
            "version": calc_flow.__version__,
            "native_sha256": file_hash(Path(_native.__file__)),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with TemporaryDirectory(
        prefix="report-operators-", dir=args.output.parent
    ) as temporary:
        for name in names:
            results.append(
                collect_case(
                    name,
                    workload,
                    Path(temporary) / name,
                    samples=0 if args.validate_only else args.samples,
                    finance_python=args.finance_python,
                )
            )
    args.output.write_text(
        json.dumps(
            {
                "contract": "report-operator-comparison-v1",
                "workload": {
                    **asdict(workload),
                    "chunks": (workload.rows - 1) // workload.chunk_rows + 1,
                },
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "build": build,
                "driver_sha256": {
                    path.name: file_hash(path)
                    for path in sorted(
                        Path(__file__).parent.glob("report_operator_*.py")
                    )
                },
                "samples_per_engine": 0 if args.validate_only else args.samples,
                "timing_scopes": {
                    "batch": "compiled native expression execution to Arrow output",
                    "stream": "ready native runner from enqueue through Arrow output",
                    "finance": "transform on prepared frame with a fresh expression",
                },
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
