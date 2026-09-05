"""Correctness-gated engine adapters with explicit, materialized timing scopes."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from benchmarks.engine_stream import run_stream, stream_events, stream_plan
from benchmarks.rolling_indicator_comparison import (
    TaLibMethod,
    ta_lib_expected_dual_sma_spread,
    ta_lib_expected_rolling_mean,
)
from benchmarks.warm_stream import _segment
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


def expected_output(data: Workload, scenario: str) -> pa.Table:
    table = data.table
    prices = table["price"].to_numpy()
    sequence = table["sequence"].to_numpy()
    if scenario in ("sma20", "dual_sma"):
        return table.append_column("value", pa.array(_rolling_expected(data, scenario)))
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


def _mean_sql(window: int) -> str:
    frame = (
        "PARTITION BY symbol ORDER BY event_time, sequence "
        f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
    )
    return (
        f"CASE WHEN COUNT(price) OVER ({frame}) = {window} "
        f"THEN AVG(price) OVER ({frame}) END"
    )


def sql_query(scenario: str) -> str:
    queries = {
        "projection": "SELECT sequence, price * 2 + 1 AS value FROM input",
        "filter": "SELECT sequence, price AS value FROM input WHERE sequence % 4 = 0",
        "group_by": "SELECT symbol, SUM(price) AS value FROM input GROUP BY symbol",
        "join": (
            "SELECT sequence, price * factor AS value "
            "FROM input JOIN dimension USING (symbol)"
        ),
    }
    if scenario in queries:
        return queries[scenario]
    slow = _mean_sql(20)
    value = slow if scenario == "sma20" else f"({_mean_sql(5)}) - ({slow})"
    return f"SELECT event_time, sequence, symbol, price, {value} AS value FROM input"


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
    return lambda: plan.collect().to_arrow()


def _ta_lib(data: Workload, scenario: str):
    method = TaLibMethod(
        data.table["price"].to_numpy(),
        data.entities,
        20,
        5 if scenario == "dual_sma" else None,
    )
    return lambda: data.table.append_column("value", pa.array(method.run()))


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
        if backend == "calc-flow-stream":
            self.plan = stream_plan(scenario)
            self.events = stream_events(self.data.table, self.data.entities)
            self.loop = asyncio.new_event_loop()
            self.calculate = self._stream
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
        return self.loop.run_until_complete(
            run_stream(self.plan, self.events, self.root / f"sample-{self.count}")
        )

    def validate(self, result: pa.Table) -> dict:
        if set(result.column_names) != set(self.expected.column_names):
            raise ValueError("engine output columns differ from the oracle")
        if result.num_rows != self.expected.num_rows:
            raise ValueError("engine output row count differs from the oracle")
        key = "symbol" if self.case["scenario"] == "group_by" else "sequence"
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
        started = time.perf_counter_ns()
        result = self.calculate()
        seconds = (time.perf_counter_ns() - started) / 1e9
        return {"seconds": seconds, "correctness": self.validate(result)}

    def close(self) -> None:
        if self.loop is not None:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()
