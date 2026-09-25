"""Correctness-gated engine adapters with explicit, materialized timing scopes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess  # nosec B404
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa

from benchmarks.engine_stream import (
    dimension_events,
    run_stream,
    stream_dimension,
    stream_events,
    stream_plan,
)
from benchmarks.rolling_indicator_comparison import (
    TaLibMethod,
    ta_lib_expected_dual_sma_spread,
    ta_lib_expected_rolling_mean,
)
from benchmarks.warm_stream import BASE, _segment
from calc_flow import Batch, PipelineBuilder
from scripts.benchmark_suite.catalog import BATCH_ROWS, CAPABILITIES, THREADS


@dataclass(frozen=True, slots=True)
class Workload:
    table: pa.Table
    dimension: pa.Table
    entities: int


def workload(count: int) -> Workload:
    if type(count) is not int or count <= 0:
        raise ValueError("rows must be a positive integer")
    entities = min(64, max(1, count // 40))
    table = _segment(0, count, entities)
    sequence = table["sequence"].to_numpy()
    # Exact binary fractions keep long-running performance fixtures separate
    # from decimal accumulation stress tests. Every engine sees these bytes.
    prices = 100 + (sequence % 257) / 8 + (sequence % entities) / 8
    table = table.set_column(
        table.column_names.index("price"), table.schema.field("price"), pa.array(prices)
    )
    table = pa.Table.from_batches(table.to_batches(max_chunksize=BATCH_ROWS))
    dimension = pa.table(
        {
            "symbol": [f"S{i:03d}" for i in range(entities)],
            "factor": np.arange(entities, dtype=np.float64) + 1,
        }
    )
    return Workload(table, dimension, entities)


def _rolling_expected(data: Workload, scenario: str) -> np.ndarray:
    prices = data.table["price"].to_numpy()
    if scenario == "sma20":
        return ta_lib_expected_rolling_mean(prices, entities=data.entities, window=20)
    return ta_lib_expected_dual_sma_spread(
        prices, entities=data.entities, fast_window=5, slow_window=20
    )


def _average_expected(prices: np.ndarray, entities: int) -> np.ndarray:
    result = np.empty(len(prices), dtype=np.float64)
    for entity in range(entities):
        series = prices[entity::entities]
        result[entity::entities] = np.cumsum(series) / np.arange(1, len(series) + 1)
    return result


def _argmax_expected(prices: np.ndarray, entities: int, window: int) -> np.ndarray:
    # The exact-eighth fixture repeats every 257 entity ticks. Precompute the
    # complete-window answers for one cycle, then handle only the warm prefix.
    from numpy.lib.stride_tricks import sliding_window_view

    result = np.empty(len(prices), dtype=np.float64)
    for entity in range(entities):
        series = prices[entity::entities]
        if not len(series):
            continue
        cycles = np.arange(-(window - 1), 257, dtype=np.int64)
        periodic = 100 + ((entity + cycles * entities) % 257) / 8 + entity / 8
        lookup = window - 1 - np.argmax(sliding_window_view(periodic, window), axis=1)
        positions = np.arange(len(series))
        values = lookup[positions % 257].astype(np.float64)
        for index in range(min(len(series), window - 1)):
            values[index] = float(index - np.argmax(series[: index + 1]))
        result[entity::entities] = values
    return result


def _cross_section_mean(prices: np.ndarray, entities: int) -> np.ndarray:
    starts = np.arange(0, len(prices), entities)
    means = np.add.reduceat(prices, starts) / np.diff(np.append(starts, len(prices)))
    return np.repeat(means, np.diff(np.append(starts, len(prices))))


def _window_sum_expected(data: Workload) -> pa.Table:
    prices = data.table["price"].to_numpy()
    sequence = data.table["sequence"].to_numpy()
    groups = (
        sequence // data.entities // 10
    ) * data.entities + sequence % data.entities
    keys, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=prices)
    starts = [BASE + timedelta(seconds=int(key // data.entities * 10)) for key in keys]
    symbols = [f"S{int(key % data.entities):03d}" for key in keys]
    return pa.table(
        {
            "symbol": pa.array(symbols),
            "window_start": pa.array(starts, type=pa.timestamp("us", tz="UTC")),
            "value": pa.array(sums),
        }
    )


def expected_output(data: Workload, scenario: str) -> pa.Table:
    table = data.table
    prices = table["price"].to_numpy()
    sequence = table["sequence"].to_numpy()
    if scenario in ("sma20", "dual_sma"):
        return table.append_column("value", pa.array(_rolling_expected(data, scenario)))
    if scenario == "average":
        return pa.table(
            {
                "sequence": table["sequence"],
                "value": _average_expected(prices, data.entities),
            }
        )
    if scenario in ("argmax64", "argmax256"):
        return pa.table(
            {
                "sequence": table["sequence"],
                "value": _argmax_expected(prices, data.entities, int(scenario[6:])),
            }
        )
    if scenario == "unique64":
        count = np.minimum(sequence // data.entities + 1, 64).astype(np.float64)
        return pa.table({"sequence": table["sequence"], "value": count})
    if scenario == "cs_mean":
        return pa.table(
            {
                "sequence": table["sequence"],
                "value": _cross_section_mean(prices, data.entities),
            }
        )
    if scenario == "window_sum":
        return _window_sum_expected(data)
    if scenario == "asof_join":
        return pa.table({"sequence": table["sequence"], "value": prices})
    if scenario == "group_by":
        sums = [
            float(np.sum(prices[index :: data.entities]))
            for index in range(data.entities)
        ]
        return pa.table({"symbol": data.dimension["symbol"], "value": sums})
    if scenario == "filter":
        mask = sequence % 4 == 0
        return pa.table({"sequence": sequence[mask], "value": prices[mask]})
    values = (
        prices * 2 + 1
        if scenario == "projection"
        else prices * (sequence % data.entities + 1)
    )
    return pa.table({"sequence": sequence, "value": values})


def sql_query(scenario: str) -> str:
    # Closed, literal query catalog: scenario names never become SQL fragments.
    queries = {
        "projection": "SELECT sequence, price * 2 + 1 AS value FROM input",
        "filter": "SELECT sequence, price AS value FROM input WHERE sequence % 4 = 0",
        "group_by": "SELECT symbol, SUM(price) AS value FROM input GROUP BY symbol",
        "join": (
            "SELECT sequence, price * factor AS value "
            "FROM input JOIN dimension USING (symbol)"
        ),
        "sma20": (
            "SELECT event_time, sequence, symbol, price, "
            "CASE WHEN COUNT(price) OVER slow = 20 "
            "THEN AVG(price) OVER slow END AS value FROM input "
            "WINDOW slow AS (PARTITION BY symbol ORDER BY event_time, sequence "
            "ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)"
        ),
        "dual_sma": (
            "SELECT event_time, sequence, symbol, price, "
            "CASE WHEN COUNT(price) OVER slow = 20 "
            "THEN AVG(price) OVER fast - AVG(price) OVER slow END AS value FROM input "
            "WINDOW slow AS (PARTITION BY symbol ORDER BY event_time, sequence "
            "ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), "
            "fast AS (PARTITION BY symbol ORDER BY event_time, sequence "
            "ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)"
        ),
    }
    try:
        return queries[scenario]
    except KeyError as error:
        raise ValueError("unsupported SQL benchmark scenario") from error


def _calc_flow(data: Workload, scenario: str):
    aliases = ("input", "dimension") if scenario == "join" else ("input",)
    plan = (
        PipelineBuilder("suite-sql")
        .with_datafusion_config(target_partitions=THREADS, batch_size=8192)
        .sql("query", sql_query(scenario), aliases=aliases)
        .compile_batch()
    )
    inputs = {"input": Batch.from_pyarrow(data.table)}
    if scenario == "join":
        inputs["dimension"] = Batch.from_pyarrow(data.dimension)
    return lambda: plan.execute(inputs).outputs["output"].to_pyarrow()


def _datafusion(data: Workload, scenario: str):
    from datafusion import SessionConfig, SessionContext

    batches = data.table.to_batches(max_chunksize=BATCH_ROWS)
    dimension = data.dimension.to_batches()
    query = sql_query(scenario)

    def execute():
        context = SessionContext(
            SessionConfig().with_target_partitions(THREADS).with_batch_size(8192)
        )
        context.register_record_batches("input", [batches])
        if scenario == "join":
            context.register_record_batches("dimension", [dimension])
        return pa.Table.from_batches(context.sql(query).collect())

    return execute


def _polars_plan(data: Workload, scenario: str):
    import polars as pl

    frame = pl.from_arrow(data.table).lazy()
    value = pl.col("price")
    if scenario == "projection":
        return frame.select("sequence", (value * 2 + 1).alias("value"))
    if scenario == "filter":
        return frame.filter(pl.col("sequence") % 4 == 0).select(
            "sequence", value.alias("value")
        )
    if scenario == "group_by":
        return frame.group_by("symbol").agg(value.sum().alias("value"))
    if scenario == "join":
        return frame.join(pl.from_arrow(data.dimension).lazy(), on="symbol").select(
            "sequence", (value * pl.col("factor")).alias("value")
        )
    slow = value.rolling_mean(20, min_samples=20).over(
        "symbol", order_by=["event_time", "sequence"]
    )
    indicator = (
        slow
        if scenario == "sma20"
        else value.rolling_mean(5, min_samples=5).over(
            "symbol", order_by=["event_time", "sequence"]
        )
        - slow
    )
    return frame.with_columns(indicator.alias("value"))


def _polars(data: Workload, scenario: str):
    plan = _polars_plan(data, scenario)
    return lambda: plan.collect(engine="streaming").to_arrow()


def _ta_lib(data: Workload, scenario: str):
    method = TaLibMethod(
        data.table["price"].to_numpy(),
        data.entities,
        20,
        5 if scenario == "dual_sma" else None,
    )
    return lambda: data.table.append_column("value", pa.array(method.run()))


class _FinanceCase:
    """Own the pinned Finance-Python interpreter and its timed transforms."""

    def __init__(self, data: Workload, scenario: str, root: Path) -> None:
        python = os.environ.get("FINANCE_PYTHON_PYTHON")
        if not python:
            raise RuntimeError("FINANCE_PYTHON_PYTHON must name the Python 3.9 worker")
        root.mkdir(parents=True, exist_ok=True)
        self.output = root / "finance-warm.npy"
        runner = Path(__file__).with_name("finance_python_rolling_runner.py")
        self.process = subprocess.Popen(  # nosec B603  # nosemgrep
            [
                python,
                str(runner),
                "--rows",
                str(data.table.num_rows),
                "--entities",
                str(data.entities),
                "--window",
                "20",
                "--warm-output",
                str(self.output),
                "--suite-scenario",
                scenario,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.identity = self._reply()
            if (
                self.identity.get("event") != "ready"
                or self.identity.get("suite_scenario") != scenario
                or self.identity.get("rows") != data.table.num_rows
                or self.identity.get("finance_python_version") != "0.9.10"
                or not str(self.identity.get("python_version", "")).startswith("3.9.")
            ):
                raise RuntimeError("Finance-Python prepared a different benchmark case")
            self.values = np.load(self.output, allow_pickle=False)
            if (
                hashlib.sha256(self.values.tobytes()).hexdigest()
                != self.identity["sha256"]
            ):
                raise RuntimeError("Finance-Python warm output checksum differs")
        except BaseException:
            self.close()
            raise

    def _reply(self) -> dict:
        line = self.process.stdout.readline()
        if line:
            return json.loads(line)
        error = self.process.stderr.read()
        raise RuntimeError(f"Finance-Python worker exited: {error}")

    def sample(self) -> float:
        self.process.stdin.write('{"command":"run","iterations":1}\n')
        self.process.stdin.flush()
        reply = self._reply()
        if (
            reply.get("event") != "sample"
            or reply.get("sha256") != self.identity["sha256"]
            or reply.get("rows") != len(self.values)
        ):
            raise RuntimeError("Finance-Python timed output differs from warm output")
        return float(reply["seconds"])

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.stdin.write('{"command":"stop"}\n')
            self.process.stdin.flush()
        except BrokenPipeError:
            pass
        try:
            self.process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()


class EngineCase:
    """Own compiled plans, immutable inputs and a worker-local event loop."""

    def __init__(self, case: dict, root: Path) -> None:
        backend, scenario = case["backend"], case["scenario"]
        if scenario not in CAPABILITIES[backend]:
            raise ValueError("unsupported engine/workload combination")
        self.case, self.root, self.count = case, root, 0
        self.data = workload(case["rows"])
        self.expected = expected_output(self.data, scenario)
        self.loop = None
        self.finance = None
        if backend == "calc-flow-stream":
            self.events = stream_events(
                self.data.table,
                self.data.entities,
                close_windows=scenario == "window_sum",
            )
            # The join's dimension side is complete at the stream origin, so
            # its events are enqueued before the quote batches.
            self.streams = (
                {
                    "right": dimension_events(stream_dimension(self.data.dimension)),
                    "left": self.events,
                }
                if scenario == "join"
                else (
                    {"reference.input": self.events, "quotes.input": self.events}
                    if scenario == "asof_join"
                    else {"input": self.events}
                )
            )
            self.loop = asyncio.new_event_loop()
        elif backend == "finance-python":
            self.finance = _FinanceCase(self.data, scenario, root)
            try:
                values = self.finance.values
                result = (
                    self.data.table.append_column("value", pa.array(values))
                    if scenario in ("sma20", "dual_sma")
                    else pa.table(
                        {"sequence": self.data.table["sequence"], "value": values}
                    )
                )
                self.finance_correctness = self.validate(result)
            except BaseException:
                self.finance.close()
                raise
        else:
            factory = {
                "calc-flow-sql": _calc_flow,
                "datafusion": _datafusion,
                "polars": _polars,
                "ta-lib": _ta_lib,
            }[backend]
            self.calculate = factory(self.data, scenario)

    def _stream(self):
        self.count += 1
        # Each single-use plan starts with empty rolling state. Compilation and
        # runner startup both precede the adapter's ready-to-Arrow timer.
        plan = stream_plan(self.case["scenario"], self.data.table, self.data.dimension)
        return self.loop.run_until_complete(
            run_stream(
                plan,
                self.streams,
                self.root / f"sample-{self.count}",
                self.expected.num_rows,
            )
        )

    def validate(self, result: pa.Table) -> dict:
        if set(result.column_names) != set(self.expected.column_names):
            raise ValueError("engine output columns differ from the oracle")
        if result.num_rows != self.expected.num_rows:
            raise ValueError("engine output row count differs from the oracle")
        key = (
            [("window_start", "ascending"), ("symbol", "ascending")]
            if self.case["scenario"] == "window_sum"
            else "symbol"
            if self.case["scenario"] == "group_by"
            else "sequence"
        )
        expected = self.expected.sort_by(key)
        result = result.sort_by(key)
        for name in expected.column_names:
            if name != "value" and not result[name].cast(expected[name].type).equals(
                expected[name]
            ):
                raise ValueError(f"engine output payload differs: {name}")
        actual = result["value"].to_numpy()
        reference = expected["value"].to_numpy()
        np.testing.assert_allclose(
            actual, reference, rtol=1e-10, atol=1e-10, equal_nan=True
        )
        finite = np.isfinite(reference)
        error = (
            float(np.max(np.abs(actual[finite] - reference[finite])))
            if finite.any()
            else 0.0
        )
        return {
            "passed": True,
            "rows": result.num_rows,
            "finite_rows": int(finite.sum()),
            "max_abs_error": error,
        }

    def sample(self) -> dict:
        if self.finance is not None:
            return {
                "seconds": self.finance.sample(),
                "correctness": self.finance_correctness,
                "finance_python": self.finance.identity,
            }
        if self.loop is not None:
            result, seconds = self._stream()
        else:
            started = time.perf_counter_ns()
            result = self.calculate()
            seconds = (time.perf_counter_ns() - started) / 1e9
        return {"seconds": seconds, "correctness": self.validate(result)}

    def finish(self) -> dict:
        return {"state": "completed"}

    def close(self) -> None:
        if self.finance is not None:
            self.finance.close()
        if self.loop is not None:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
