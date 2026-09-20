"""Ready StreamingRunner adapter: empty state, real tasks and finalization."""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from pathlib import Path

import pyarrow as pa

from benchmarks.warm_stream import BASE, BASE_MICROS, _InteractiveSource
from calc_flow import (
    Batch,
    Cursor,
    Data,
    EdgeBudget,
    JoinStateLimits,
    JoinTimeBounds,
    ManagedCheckpointRuntime,
    Runtime,
    SinkBinding,
    SourceBinding,
    SourceProvidedWatermarks,
    StreamingRunner,
    StreamRuntimeConfig,
    Watermark,
)
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    lit,
    rows,
    table_input,
    ts,
    window,
)
from calc_flow.symbolic import (
    table as tables,
)
from scripts.benchmark_suite.catalog import BATCH_ROWS


def stream_dimension(dimension: pa.Table) -> pa.Table:
    """Return the dimension table temporalized at the stream origin."""

    rows = dimension.num_rows
    # The stream port validates an exact schema, and the comparison workload
    # builds the dimension table with inferred nullable fields; rebuild every
    # column as required in the declared input order.
    schema = pa.schema(
        (
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("factor", pa.float64(), nullable=False),
            pa.field("sequence", pa.uint64(), nullable=False),
            pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        )
    )
    return pa.Table.from_arrays(
        (
            dimension["symbol"],
            dimension["factor"],
            pa.array(range(rows), type=pa.uint64()),
            pa.array([BASE] * rows, type=pa.timestamp("us", tz="UTC")),
        ),
        schema=schema,
    )


def _join_span(table: pa.Table) -> timedelta:
    """Bound the join so an origin row matches every quote row inclusively."""

    span = int(table["event_time"][-1].value) - int(BASE_MICROS) + 1
    return timedelta(microseconds=span)


def _join_limits(rows: int) -> JoinStateLimits:
    """Declare room for one fully retained side plus one input batch."""

    capacity = rows + BATCH_ROWS
    return JoinStateLimits(capacity, max(capacity << 10, 1 << 30), capacity)


def _join_stream_output(table: pa.Table, dimension: pa.Table, quotes):
    """Return the bounded temporal join output and its program inputs."""

    if table is None or dimension is None:
        raise ValueError("join stream plans require the workload tables")
    factors = table_input(
        "dimension",
        schema=(
            Field("symbol", "string", nullable=False),
            Field("factor", "float64", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("event_time", "timestamp[us, UTC]", nullable=False),
        ),
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    # The dimension side completes at the stream origin and the inclusive
    # `before` bound spans the whole workload, so every quote row matches
    # exactly its symbol's factor row — the stream equivalent of the
    # suite's shared `join` query in engine_comparison.sql_query.
    joined = tables.stream_join(
        quotes,
        factors,
        left_keys=("symbol",),
        right_keys=("symbol",),
        left_event_time="event_time",
        right_event_time="event_time",
        bounds=JoinTimeBounds(_join_span(table), timedelta()),
        limits=_join_limits(table.num_rows),
        left_prefix="quote",
        right_prefix="dimension",
    )
    output = joined.with_columns(
        FeatureSet(
            (
                ("sequence", joined["quote__sequence"]),
                ("value", joined["quote__price"] * joined["dimension__factor"]),
            )
        )
    ).select("sequence", "value")
    return output, (quotes, factors)


def stream_plan(
    scenario: str, table: pa.Table | None = None, dimension: pa.Table | None = None
):
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
    inputs = (quotes,)
    if scenario in ("sma20", "dual_sma"):
        slow = ts.mean(quotes["price"], window=rows(20), min_periods=20)
        value = (
            slow
            if scenario == "sma20"
            else ts.mean(quotes["price"], window=rows(5), min_periods=5) - slow
        )
        output = quotes.with_columns(FeatureSet((("value", value),)))
    elif scenario == "projection":
        output = quotes.with_columns(
            FeatureSet((("value", quotes["price"] * lit(2.0) + lit(1.0)),))
        ).select("sequence", "value")
    elif scenario == "filter":
        # The row-local expression language has no modulo primitive, so the
        # filter scenario runs through the native stream SQL stage; the
        # predicate is row-local, making per-batch execution equivalent to the
        # suite's shared `filter` query in engine_comparison.sql_query.
        output = quotes.sql(
            "SELECT sequence, price AS value FROM input WHERE sequence % 4 = 0"
        )
    elif scenario == "group_by":
        if table is None:
            raise ValueError("group_by stream plans require the workload table")
        output = window.tumbling(
            quotes,
            event_time="event_time",
            # One epoch-anchored window ending exactly at the final watermark
            # holds every row, so the emitted per-symbol sums equal the whole
            # input GROUP BY aggregate.
            size_micros=int(table["event_time"][-1].value) + 1,
            group_by=("symbol",),
            aggregates=(window.sum("price", output="value"),),
        ).select("symbol", "value")
    elif scenario == "join":
        output, inputs = _join_stream_output(table, dimension, quotes)
    else:
        raise ValueError("unsupported stream benchmark scenario")
    return Program(
        "suite-stream", inputs=inputs, outputs=(("result", output),)
    ).compile_stream(Runtime())


def stream_events(table: pa.Table, entities: int) -> tuple:
    # Never finalize half an entity tick before the next data batch.
    size = max(entities, BATCH_ROWS // entities * entities)
    events = []
    for start in range(0, table.num_rows, size):
        part = table.slice(start, size)
        end = start + part.num_rows
        events.append(
            Data(
                Batch.from_pyarrow(part), Cursor(end.to_bytes(8, "big"), {"rows": end})
            )
        )
        micros = part["event_time"][-1].value - int(BASE_MICROS) + 1
        events.append(Watermark(BASE + timedelta(microseconds=micros)))
    return (*events, None)


def dimension_events(dimension: pa.Table) -> tuple:
    """Seed the dimension side at the origin, then advance and end it."""

    rows = dimension.num_rows
    return (
        Data(
            Batch.from_pyarrow(dimension),
            Cursor(rows.to_bytes(8, "big"), {"rows": rows}),
        ),
        Watermark(BASE + timedelta(microseconds=1)),
        None,
    )


class _ReadySource(_InteractiveSource):
    def __init__(self) -> None:
        super().__init__(max_batch_rows=BATCH_ROWS)
        self.ready = asyncio.Event()

    async def next(self) -> Data | Watermark | None:
        # The first poll proves that the runtime's startup data gate opened.
        self.ready.set()
        return await super().next()


class _CollectSink:
    def __init__(self, expected_rows: int) -> None:
        self.expected_rows = expected_rows
        self.rows = 0
        self.tables: list[pa.Table] = []
        self.opened = asyncio.Event()
        self.complete = asyncio.Event()

    async def open(self) -> None:
        self.opened.set()

    async def write(self, batch: Batch) -> None:
        table = batch.to_pyarrow()
        self.tables.append(table)
        self.rows += table.num_rows
        if self.rows >= self.expected_rows:
            self.complete.set()

    async def close(self) -> None:
        return None


def _validated_timed_streams(plan, streams: dict[str, tuple]) -> dict[str, tuple]:
    """Return each binding's timed events after checking the EOF contract."""

    if set(streams) != set(plan.source_binding_ids):
        raise ValueError("stream events must cover every plan source binding")
    if any(not events or events[-1] is not None for events in streams.values()):
        raise ValueError("every stream input must end with an EOF marker")
    return {name: events[:-1] for name, events in streams.items()}


def _require_ready_sources(
    sources: dict[str, _ReadySource], sink: _CollectSink
) -> None:
    """Fail unless every source passed the startup gate with empty state."""

    if (
        any(not source.opened.is_set() for source in sources.values())
        or not sink.opened.is_set()
        or sink.rows
    ):
        raise RuntimeError("stream must be ready with empty state before timing")


async def _measure_ready(
    sources: dict[str, _ReadySource], sink: _CollectSink, streams: dict[str, tuple]
) -> tuple[pa.Table, float]:
    await asyncio.wait_for(
        asyncio.gather(*(source.ready.wait() for source in sources.values())),
        timeout=30,
    )
    _require_ready_sources(sources, sink)
    started = time.perf_counter_ns()
    for name, events in streams.items():
        for event in events:
            await sources[name].push(event)
    await asyncio.wait_for(sink.complete.wait(), timeout=600)
    if sink.rows != sink.expected_rows:
        raise RuntimeError("stream output row count differs from the timed workload")
    table = pa.concat_tables(sink.tables)
    return table, (time.perf_counter_ns() - started) / 1e9


async def run_stream(
    plan, streams: dict[str, tuple], root: Path, expected_rows: int
) -> tuple[pa.Table, float]:
    timed = _validated_timed_streams(plan, streams)
    sources = {name: _ReadySource() for name in streams}
    sink = _CollectSink(expected_rows)
    job = await StreamingRunner(
        plan,
        {
            name: SourceBinding(source, watermark_policy=SourceProvidedWatermarks())
            for name, source in sources.items()
        },
        {"output": [SinkBinding.ordinary("suite", sink)]},
        ManagedCheckpointRuntime(root),
        config=StreamRuntimeConfig(
            checkpoint_interval=timedelta(hours=24),
            edge_budget=EdgeBudget(max_rows=BATCH_ROWS, max_bytes=64 << 20),
        ),
    ).start_async()
    try:
        table, seconds = await _measure_ready(sources, sink, timed)
        # Complete and verify the job, but do not time EOF/shutdown bookkeeping.
        for source in sources.values():
            await source.push(None)
        outcome = await asyncio.wait_for(job.wait_async(), timeout=600)
        if outcome.state != "completed":
            raise RuntimeError(f"stream failed: {outcome.errors}")
        if sink.rows != expected_rows:
            raise RuntimeError("stream output row count changed after the timed result")
        return table, seconds
    finally:
        await job.cancel_async()
