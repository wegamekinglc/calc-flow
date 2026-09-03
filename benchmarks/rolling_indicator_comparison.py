"""Compare rolling indicators across Calc Flow and reference libraries.

The comparison uses the same deterministic multi-symbol quote shape and a
20-row moving average or composed ``SMA(5) - SMA(20)`` spread for four
implementations:

* Calc Flow's current native incremental rolling operator, declared as
  ``ts.mean``;
* Calc Flow's earlier DataFusion SQL window-function path;
* Finance-Python 0.9.10's public ``MA(...).transform(...)`` operator, built
  from the repository commit frozen by Calc Flow's finance reference tests;
  and
* TA-Lib Python 0.7.1's public ``SMA`` wrapper over the bundled C library.

Plan compilation, input construction, worker startup, warm-up, and correctness
normalization are outside the timed boundary. Every timed call creates and
materializes one complete rolling output. Finance-Python runs in an isolated
legacy interpreter and is driven synchronously between Calc Flow samples.
TA-Lib's per-symbol slicing, contiguous input ownership, calls, and output
assembly are all inside its timed boundary.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess  # nosec B404
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import talib

from calc_flow import Batch, PipelineBuilder, Runtime
from calc_flow.symbolic import FeatureSet, Field, Program, rows, table_input, ts

FINANCE_PYTHON_COMMIT = "3e33d3e70c3458b4c6dcf76b88df6148229b402c"
TA_LIB_PYTHON_VERSION = "0.7.1"
TA_LIB_PYTHON_TAG = "v0.7.1"
TA_LIB_PYTHON_COMMIT = "a9ff1b47b3ddbd57274116645d688c0ed677338b"
INDICATOR_ROLLING_MEAN = "rolling_mean"
INDICATOR_DUAL_SMA_SPREAD = "dual_sma_spread"
DEFAULT_ROW_SCALES = (10, 100, 1_000, 10_000, 100_000, 1_000_000)
DEFAULT_ENTITIES = 64
DEFAULT_FAST_WINDOW = 5
DEFAULT_WINDOW = 20
_OUTPUT_COLUMNS = {
    INDICATOR_ROLLING_MEAN: "moving_average",
    INDICATOR_DUAL_SMA_SPREAD: "dual_sma_spread",
}
_CORRECTNESS_RTOL = 1e-10
_CORRECTNESS_ATOL = 1e-10
_SCHEMA = pa.schema(
    (
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("sequence", pa.uint64(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
    )
)


@dataclass(frozen=True, slots=True)
class RollingWorkload:
    """One immutable, timestamp-major quote workload."""

    batch: Batch
    prices: np.ndarray
    rows: int
    entities: int


@dataclass(frozen=True, slots=True)
class CalcFlowMethod:
    """Compiled Calc Flow plan with untimed output-order normalization."""

    plan: Any
    batch: Batch
    output_column: str

    def run(self) -> Any:
        """Execute and materialize one native result batch."""
        return self.plan.execute({"input": self.batch})

    def values(self, result: Any) -> np.ndarray:
        """Return rolling values in original sequence order."""
        table = result.outputs["output"].to_pyarrow()
        sequence = table["sequence"].to_numpy(zero_copy_only=False)
        values = table[self.output_column].to_numpy(zero_copy_only=False)
        ordered = np.empty(len(values), dtype=np.float64)
        ordered[sequence] = values
        return ordered

    def execute(self) -> np.ndarray:
        """Execute once and return normalized values for correctness checks."""
        return self.values(self.run())


@dataclass(frozen=True, slots=True)
class TaLibMethod:
    """End-to-end multi-symbol adapter around TA-Lib's single-series SMA."""

    prices: np.ndarray
    entities: int
    window: int
    fast_window: int | None = None

    def run(self) -> np.ndarray:
        """Calculate and restore timestamp-major order without mutating input."""
        output = np.empty(len(self.prices), dtype=np.float64)
        active_entities = min(self.entities, len(self.prices))
        for entity in range(active_entities):
            entity_values = np.array(
                self.prices[entity :: self.entities],
                dtype=np.float64,
                copy=True,
                order="C",
            )
            slow = talib.SMA(entity_values, timeperiod=self.window)
            output[entity :: self.entities] = (
                slow
                if self.fast_window is None
                else talib.SMA(entity_values, timeperiod=self.fast_window) - slow
            )
        return output


@dataclass(frozen=True, slots=True)
class MethodResult:
    """Robust summary for one method and row scale."""

    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    throughput_rows_per_second: float
    samples: tuple[float, ...]


class FinancePythonWorker:
    """Long-lived isolated worker for the legacy Finance-Python runtime."""

    def __init__(
        self,
        python: Path,
        *,
        rows: int,
        entities: int,
        window: int,
        warm_output: Path,
        indicator: str = INDICATOR_ROLLING_MEAN,
        fast_window: int = DEFAULT_FAST_WINDOW,
    ) -> None:
        runner = Path(__file__).with_name("finance_python_rolling_runner.py")
        self._process = subprocess.Popen(  # nosec B603  # nosemgrep
            [
                str(python),
                str(runner),
                "--rows",
                str(rows),
                "--entities",
                str(entities),
                "--window",
                str(window),
                "--indicator",
                indicator,
                "--fast-window",
                str(fast_window),
                "--warm-output",
                str(warm_output),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            ready = self._read_reply()
        except Exception:
            self.close()
            raise
        if (
            ready.get("event") != "ready"
            or ready.get("rows") != rows
            or ready.get("indicator") != indicator
        ):
            self.close()
            raise RuntimeError(f"invalid Finance-Python ready reply: {ready!r}")
        self.identity = ready

    def _read_reply(self) -> dict[str, object]:
        if self._process.stdout is None:
            raise RuntimeError("Finance-Python worker stdout is unavailable")
        line = self._process.stdout.readline()
        if line:
            return json.loads(line)
        stderr = ""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read()
        raise RuntimeError(
            f"Finance-Python worker exited with {self._process.poll()}: {stderr}"
        )

    def sample(self, iterations: int) -> float:
        """Measure one fresh public Finance-Python MA transform."""
        if self._process.stdin is None:
            raise RuntimeError("Finance-Python worker stdin is unavailable")
        self._process.stdin.write(
            json.dumps({"command": "run", "iterations": iterations}) + "\n"
        )
        self._process.stdin.flush()
        reply = self._read_reply()
        if reply.get("event") != "sample" or reply.get("iterations") != iterations:
            raise RuntimeError(f"invalid Finance-Python sample reply: {reply!r}")
        return float(reply["seconds"])

    def close(self) -> None:
        """Stop the worker and reap the subprocess."""
        if self._process.poll() is not None:
            return
        if self._process.stdin is not None:
            try:
                self._process.stdin.write('{"command":"stop"}\n')
                self._process.stdin.flush()
            except BrokenPipeError:
                pass
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=10)

    def __enter__(self) -> FinancePythonWorker:
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


def rolling_workload(*, rows: int, entities: int) -> RollingWorkload:
    """Build deterministic quotes without version-dependent random numbers."""
    sequence = np.arange(rows, dtype=np.uint64)
    entity_index = sequence % entities
    positions = sequence // entities
    symbol_names = np.asarray([f"S{index:03d}" for index in range(entities)])
    prices = (
        100.0
        + ((sequence * np.uint64(17)) % np.uint64(1_000)).astype(np.float64) / 100.0
        + entity_index.astype(np.float64) / 100.0
    )
    base_micros = np.int64(1_767_225_600_000_000)
    table = pa.table(
        {
            "event_time": pa.array(
                base_micros + positions.astype(np.int64) * 1_000_000,
                type=pa.timestamp("us", tz="UTC"),
            ),
            "sequence": sequence,
            "symbol": symbol_names[entity_index],
            "price": prices,
        },
        schema=_SCHEMA,
    )
    return RollingWorkload(
        batch=Batch.from_pyarrow(table),
        prices=prices,
        rows=rows,
        entities=entities,
    )


def expected_rolling_mean(
    prices: np.ndarray, *, entities: int, window: int
) -> np.ndarray:
    """Compute a stable direct-window oracle without long cumulative drift."""
    expected = np.empty(len(prices), dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    for entity in range(min(entities, len(prices))):
        entity_values = prices[entity::entities]
        warmup = min(window - 1, len(entity_values))
        if warmup:
            expected[entity : entity + warmup * entities : entities] = np.cumsum(
                entity_values[:warmup], dtype=np.float64
            ) / np.arange(1, warmup + 1)
        if len(entity_values) >= window:
            expected[entity + (window - 1) * entities :: entities] = np.convolve(
                entity_values,
                kernel,
                mode="valid",
            )
    return expected


def expected_dual_sma_spread(
    prices: np.ndarray,
    *,
    entities: int,
    fast_window: int,
    slow_window: int,
) -> np.ndarray:
    """Compose partial-window fast and slow rolling means."""
    fast = expected_rolling_mean(
        prices,
        entities=entities,
        window=fast_window,
    )
    slow = expected_rolling_mean(
        prices,
        entities=entities,
        window=slow_window,
    )
    return fast - slow


def ta_lib_expected_rolling_mean(
    prices: np.ndarray, *, entities: int, window: int
) -> np.ndarray:
    """Return the oracle with TA-Lib's full-window-only warm-up semantics."""
    expected = expected_rolling_mean(prices, entities=entities, window=window)
    for entity in range(min(entities, len(prices))):
        warmup = min(window - 1, len(prices[entity::entities]))
        expected[entity : entity + warmup * entities : entities] = np.nan
    return expected


def ta_lib_expected_dual_sma_spread(
    prices: np.ndarray,
    *,
    entities: int,
    fast_window: int,
    slow_window: int,
) -> np.ndarray:
    """Compose dual SMA values with TA-Lib's slow-window warm-up."""
    expected = expected_dual_sma_spread(
        prices,
        entities=entities,
        fast_window=fast_window,
        slow_window=slow_window,
    )
    for entity in range(min(entities, len(prices))):
        warmup = min(slow_window - 1, len(prices[entity::entities]))
        expected[entity : entity + warmup * entities : entities] = np.nan
    return expected


def ta_lib_valid_rows(*, rows: int, entities: int, window: int) -> int:
    """Count non-warm-up TA-Lib outputs for the timestamp-major workload."""
    return sum(
        max(0, len(range(entity, rows, entities)) - window + 1)
        for entity in range(min(rows, entities))
    )


def _native_program(
    window: int,
    *,
    indicator: str,
    fast_window: int,
) -> Program:
    quotes = table_input(
        "quotes",
        schema=(
            Field("event_time", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("price", "float64", nullable=False),
        ),
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    slow = ts.mean(quotes["price"], window=rows(window))
    value = (
        slow
        if indicator == INDICATOR_ROLLING_MEAN
        else ts.mean(quotes["price"], window=rows(fast_window)) - slow
    )
    indicators = quotes.with_columns(FeatureSet(((_OUTPUT_COLUMNS[indicator], value),)))
    return Program(
        f"incremental-{indicator.replace('_', '-')}",
        inputs=(quotes,),
        outputs=(("indicators", indicators),),
    )


def _sql_window_average(window: int) -> str:
    preceding = window - 1
    return f"""avg(price) OVER (
    PARTITION BY symbol
    ORDER BY event_time, sequence
    ROWS BETWEEN {preceding} PRECEDING AND CURRENT ROW
  )"""


def _sql_query(window: int, *, indicator: str, fast_window: int) -> str:
    slow = _sql_window_average(window)
    value = (
        slow
        if indicator == INDICATOR_ROLLING_MEAN
        else f"({_sql_window_average(fast_window)}) - ({slow})"
    )
    return f"""
SELECT
  event_time,
  sequence,
  symbol,
  price,
  {value} AS {_OUTPUT_COLUMNS[indicator]}
FROM input
"""


def build_calc_flow_methods(
    workload: RollingWorkload,
    *,
    window: int,
    indicator: str = INDICATOR_ROLLING_MEAN,
    fast_window: int = DEFAULT_FAST_WINDOW,
) -> dict[str, CalcFlowMethod]:
    """Compile equivalent native-incremental and SQL-window plans."""
    native = _native_program(
        window,
        indicator=indicator,
        fast_window=fast_window,
    ).compile_batch(Runtime())
    sql = (
        PipelineBuilder(f"sql-window-{indicator.replace('_', '-')}")
        .sql(
            _OUTPUT_COLUMNS[indicator],
            _sql_query(
                window,
                indicator=indicator,
                fast_window=fast_window,
            ),
        )
        .compile_batch()
    )
    output_column = _OUTPUT_COLUMNS[indicator]
    return {
        "incremental": CalcFlowMethod(native, workload.batch, output_column),
        "sql_window": CalcFlowMethod(sql, workload.batch, output_column),
    }


def iterations_per_sample(rows: int) -> int:
    """Amortize sub-millisecond cases without multiplying large workloads."""
    return max(1, min(50, 50_000 // rows))


def ta_lib_iterations_per_sample(rows: int) -> int:
    """Amortize TA-Lib's substantially shorter direct-call measurements."""
    return max(1, min(200, 5_000_000 // rows))


def _verified_finance_python_root(root: Path) -> str:
    head = subprocess.run(  # nosec B603  # nosemgrep
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(  # nosec B603  # nosemgrep
        ["git", "-C", str(root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != FINANCE_PYTHON_COMMIT or status:
        raise RuntimeError(
            "Finance-Python source must be the clean frozen commit "
            f"{FINANCE_PYTHON_COMMIT}; observed head={head!r}, status={status!r}"
        )
    return head


def _timed_calc_flow(method: CalcFlowMethod, iterations: int) -> float:
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
    if result.outputs["output"].num_rows == 0:
        raise RuntimeError("rolling benchmark unexpectedly produced no rows")
    return seconds / iterations


def _timed_ta_lib(method: TaLibMethod, iterations: int) -> float:
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
    if len(result) != len(method.prices):
        raise RuntimeError("TA-Lib benchmark produced an invalid row count")
    return seconds / iterations


def _summarize(samples: list[float], rows: int) -> MethodResult:
    median = statistics.median(samples)
    return MethodResult(
        median_seconds=median,
        minimum_seconds=min(samples),
        maximum_seconds=max(samples),
        throughput_rows_per_second=rows / median,
        samples=tuple(samples),
    )


def _measure_scale(
    *,
    rows: int,
    entities: int,
    window: int,
    indicator: str,
    fast_window: int,
    rounds: int,
    finance_python: Path,
    temporary_root: Path,
) -> tuple[dict[str, MethodResult], dict[str, object]]:
    workload = rolling_workload(rows=rows, entities=entities)
    methods = build_calc_flow_methods(
        workload,
        window=window,
        indicator=indicator,
        fast_window=fast_window,
    )
    is_composite = indicator == INDICATOR_DUAL_SMA_SPREAD
    ta_lib_method = TaLibMethod(
        workload.prices,
        entities=entities,
        window=window,
        fast_window=fast_window if is_composite else None,
    )
    expected = (
        expected_dual_sma_spread(
            workload.prices,
            entities=entities,
            fast_window=fast_window,
            slow_window=window,
        )
        if is_composite
        else expected_rolling_mean(
            workload.prices,
            entities=entities,
            window=window,
        )
    )
    ta_lib_expected = (
        ta_lib_expected_dual_sma_spread(
            workload.prices,
            entities=entities,
            fast_window=fast_window,
            slow_window=window,
        )
        if is_composite
        else ta_lib_expected_rolling_mean(
            workload.prices,
            entities=entities,
            window=window,
        )
    )
    warm_outputs = {name: method.execute() for name, method in methods.items()}
    ta_lib_output = ta_lib_method.run()
    warm_output_path = temporary_root / f"finance-python-{rows}.npy"
    with FinancePythonWorker(
        finance_python,
        rows=rows,
        entities=entities,
        window=window,
        warm_output=warm_output_path,
        indicator=indicator,
        fast_window=fast_window,
    ) as finance_worker:
        finance_output = np.load(warm_output_path, allow_pickle=False)
        for name, actual in (*warm_outputs.items(), ("finance_python", finance_output)):
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=_CORRECTNESS_RTOL,
                atol=_CORRECTNESS_ATOL,
                err_msg=f"{name} rolling output differs from the oracle",
            )
        np.testing.assert_allclose(
            ta_lib_output,
            ta_lib_expected,
            rtol=_CORRECTNESS_RTOL,
            atol=_CORRECTNESS_ATOL,
            err_msg="TA-Lib rolling output differs from its warm-up-aware oracle",
        )

        iterations = iterations_per_sample(rows)
        ta_lib_iterations = ta_lib_iterations_per_sample(rows)
        samples = {name: [] for name in (*methods, "finance_python", "ta_lib")}
        names = tuple(samples)
        for round_index in range(rounds):
            offset = round_index % len(names)
            order = (*names[offset:], *names[:offset])
            for name in order:
                gc.collect()
                if name == "finance_python":
                    seconds = finance_worker.sample(iterations)
                elif name == "ta_lib":
                    seconds = _timed_ta_lib(ta_lib_method, ta_lib_iterations)
                else:
                    seconds = _timed_calc_flow(methods[name], iterations)
                samples[name].append(seconds)
        identity = finance_worker.identity

    return (
        {name: _summarize(values, rows) for name, values in samples.items()},
        identity,
    )


def _git_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(  # nosec B603  # nosemgrep
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(  # nosec B603  # nosemgrep
            ["git", "-C", str(root), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": head, "dirty": dirty}


def _parse_scales(raw: str) -> tuple[int, ...]:
    scales = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not scales or any(value <= 0 for value in scales):
        raise argparse.ArgumentTypeError(
            "row scales must be positive comma-separated ints"
        )
    return scales


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=_parse_scales,
        default=DEFAULT_ROW_SCALES,
        help=("comma-separated row scales (default: 10,100,1000,10000,100000,1000000)"),
    )
    parser.add_argument("--entities", type=int, default=DEFAULT_ENTITIES)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument(
        "--indicator",
        choices=(INDICATOR_ROLLING_MEAN, INDICATOR_DUAL_SMA_SPREAD),
        default=INDICATOR_ROLLING_MEAN,
    )
    parser.add_argument("--fast-window", type=int, default=DEFAULT_FAST_WINDOW)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument(
        "--finance-python-python",
        type=Path,
        default=os.environ.get("FINANCE_PYTHON_PYTHON"),
        required=os.environ.get("FINANCE_PYTHON_PYTHON") is None,
    )
    parser.add_argument(
        "--finance-python-root",
        type=Path,
        default=os.environ.get("FINANCE_PYTHON_ROOT"),
        required=os.environ.get("FINANCE_PYTHON_ROOT") is None,
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.entities <= 0 or args.window <= 0 or args.fast_window <= 0:
        parser.error("entities, window, and fast-window must be positive")
    if args.indicator == INDICATOR_DUAL_SMA_SPREAD and args.fast_window >= args.window:
        parser.error("fast-window must be smaller than window for dual SMA spread")
    if args.rounds < 4 or args.rounds % 4 != 0:
        parser.error("rounds must be a positive multiple of 4 and at least 4")
    return args


def _markdown_table(results: list[dict[str, object]]) -> str:
    lines = [
        "| Rows | Native incremental | SQL window | Finance-Python | TA-Lib | "
        "TA-Lib valid rows | SQL/native | Finance/native | TA-Lib/native |",
        "| ----: | -----------------: | ---------: | -------------: | "
        "-----: | ----------------: | ---------: | -------------: | "
        "------------: |",
    ]
    for row in results:
        methods = row["methods"]
        incremental = methods["incremental"]["median_seconds"]
        sql_window = methods["sql_window"]["median_seconds"]
        finance_python = methods["finance_python"]["median_seconds"]
        ta_lib = methods["ta_lib"]["median_seconds"]
        lines.append(
            f"| {row['rows']:,} | {incremental * 1_000:.3f} ms | "
            f"{sql_window * 1_000:.3f} ms | {finance_python * 1_000:.3f} ms | "
            f"{ta_lib * 1_000:.3f} ms | {row['ta_lib_valid_rows']:,} | "
            f"{sql_window / incremental:.3f}x | "
            f"{finance_python / incremental:.3f}x | "
            f"{ta_lib / incremental:.3f}x |"
        )
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    if talib.__version__ != TA_LIB_PYTHON_VERSION:
        raise RuntimeError(
            f"TA-Lib Python {TA_LIB_PYTHON_VERSION} is required; "
            f"observed {talib.__version__}"
        )
    finance_commit = _verified_finance_python_root(args.finance_python_root)
    scale_results: list[dict[str, object]] = []
    finance_identity: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="calc-flow-rolling-") as temporary:
        temporary_root = Path(temporary)
        for row_count in args.rows:
            summaries, finance_identity = _measure_scale(
                rows=row_count,
                entities=args.entities,
                window=args.window,
                indicator=args.indicator,
                fast_window=args.fast_window,
                rounds=args.rounds,
                finance_python=args.finance_python_python,
                temporary_root=temporary_root,
            )
            scale_results.append(
                {
                    "rows": row_count,
                    "entities": args.entities,
                    "ta_lib_valid_rows": ta_lib_valid_rows(
                        rows=row_count,
                        entities=args.entities,
                        window=args.window,
                    ),
                    "iterations_per_sample": {
                        "incremental": iterations_per_sample(row_count),
                        "sql_window": iterations_per_sample(row_count),
                        "finance_python": iterations_per_sample(row_count),
                        "ta_lib": ta_lib_iterations_per_sample(row_count),
                    },
                    "methods": {
                        name: {
                            "median_seconds": summary.median_seconds,
                            "minimum_seconds": summary.minimum_seconds,
                            "maximum_seconds": summary.maximum_seconds,
                            "throughput_rows_per_second": (
                                summary.throughput_rows_per_second
                            ),
                            "samples": summary.samples,
                        }
                        for name, summary in summaries.items()
                    },
                }
            )
            print(f"completed {row_count:,} rows", file=sys.stderr)

    report = {
        "contract": "rolling-indicator-comparison-v3",
        "calc_flow": _git_identity(),
        "calc_flow_python": platform.python_version(),
        "finance_python": {
            "commit": finance_commit,
            **finance_identity,
        },
        "ta_lib_python": {
            "package_version": talib.__version__,
            "c_library_version": talib.__ta_version__.decode(),
            "release_tag": TA_LIB_PYTHON_TAG,
            "release_commit": TA_LIB_PYTHON_COMMIT,
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "workload": {
            "entities": args.entities,
            "indicator": args.indicator,
            "formula": (
                f"SMA({args.fast_window}) - SMA({args.window})"
                if args.indicator == INDICATOR_DUAL_SMA_SPREAD
                else f"SMA({args.window})"
            ),
            "fast_window": (
                args.fast_window
                if args.indicator == INDICATOR_DUAL_SMA_SPREAD
                else None
            ),
            "window": args.window,
            "rounds": args.rounds,
            "row_scales": args.rows,
            "correctness_tolerance": {
                "rtol": _CORRECTNESS_RTOL,
                "atol": _CORRECTNESS_ATOL,
            },
            "partial_window_methods": (
                "incremental",
                "sql_window",
                "finance_python",
            ),
            "full_window_only_methods": ("ta_lib",),
        },
        "results": scale_results,
    }
    print(f"Indicator: {report['workload']['formula']}")
    print(_markdown_table(scale_results))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
