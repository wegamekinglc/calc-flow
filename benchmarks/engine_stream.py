"""Ready StreamingRunner adapter: empty state, real tasks and finalization."""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from benchmarks.engine_lifecycle import interleaved_events, run_with_completion
from benchmarks.warm_stream import BASE, BASE_MICROS, _InteractiveSource
from calc_flow import (
    AsofStateLimits,
    Batch,
    Cursor,
    Data,
    EdgeBudget,
    JoinStateLimits,
    JoinTimeBounds,
    ManagedCheckpointRuntime,
    NativeWatermarkCapability,
    ReplayPositioning,
    Runtime,
    SinkBinding,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    SourceProvidedWatermarks,
    StreamingRunner,
    StreamRuntimeConfig,
    Watermark,
)
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    cs,
    exact_time,
    lit,
    rows,
    table_input,
    ts,
    window,
)
from calc_flow.symbolic import (
    table as tables,
)
from scripts.benchmark_suite.catalog import BATCH_ROWS, stream_dimensions

QUOTE_FIELDS = (
    Field("event_time", "timestamp[us, UTC]", nullable=False),
    Field("sequence", "uint64", nullable=False),
    Field("symbol", "string", nullable=False),
    Field("price", "float64", nullable=False),
)


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


def _join_limits(dimension_rows: int, batch_rows: int | None = None) -> JoinStateLimits:
    """Bound the sealed static dimension and one in-flight quote batch."""

    capacity = dimension_rows + (BATCH_ROWS if batch_rows is None else batch_rows)
    return JoinStateLimits(capacity, max(capacity << 10, 1 << 30), capacity)


def _join_stream_output(table: pa.Table, dimension: pa.Table, quotes, batch_rows: int):
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
    joined = tables.stream_join(
        quotes,
        factors,
        left_keys=("symbol",),
        right_keys=("symbol",),
        left_event_time="event_time",
        right_event_time="event_time",
        bounds=JoinTimeBounds(_join_span(table), timedelta()),
        limits=_join_limits(dimension.num_rows, batch_rows),
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


def _scalar_stream_output(scenario: str, quotes):
    """Build row-local and scalar indicator benchmark outputs."""

    if scenario in ("sma20", "dual_sma"):
        slow = ts.mean(quotes["price"], window=rows(20), min_periods=20)
        value = (
            slow
            if scenario == "sma20"
            else ts.mean(quotes["price"], window=rows(5), min_periods=5) - slow
        )
        output = quotes.with_columns(FeatureSet((("value", value),)))
    elif scenario in ("average", "argmax64", "argmax256", "unique64"):
        price = quotes["price"]
        value = {
            "average": lambda: ts.average(price),
            "argmax64": lambda: ts.argmax(price, window=rows(64)),
            "argmax256": lambda: ts.argmax(price, window=rows(256)),
            "unique64": lambda: ts.unique_count(price, window=rows(64)),
        }[scenario]()
        output = quotes.select("sequence", value=value)
    elif scenario == "cs_mean":
        output = quotes.select(
            "sequence",
            value=cs.mean(quotes["price"], group=exact_time(quotes["event_time"])),
        )
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
    else:
        raise ValueError("unsupported stream benchmark scenario")
    return output


def stream_plan(
    scenario: str,
    table: pa.Table | None = None,
    dimension: pa.Table | None = None,
    *,
    batch_rows: int | None = None,
):
    batch_rows = BATCH_ROWS if batch_rows is None else batch_rows
    quotes = table_input(
        "quotes",
        schema=QUOTE_FIELDS,
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    inputs = (quotes,)
    if scenario == "window_sum":
        output = window.tumbling(
            quotes,
            event_time="event_time",
            size_micros=10_000_000,
            group_by=("symbol",),
            aggregates=(window.sum("price", output="value"),),
        ).select("symbol", "window_start", "value")
    elif scenario == "asof_join":
        if table is None:
            raise ValueError("asof_join stream plans require the workload table")
        reference = table_input(
            "reference",
            schema=QUOTE_FIELDS,
            entity_by=("symbol",),
            event_time="event_time",
            sequence_by=("sequence",),
        )
        joined = tables.stream_asof_join(
            quotes,
            reference,
            tolerance=timedelta(),
            limits=AsofStateLimits(2 * batch_rows + 128, 1 << 30),
        )
        output = joined.select(
            sequence=joined["left__sequence"], value=joined["right__price"]
        )
        inputs = (quotes, reference)
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
        output, inputs = _join_stream_output(table, dimension, quotes, batch_rows)
    elif scenario == "interval_join":
        reference = table_input(
            "reference",
            schema=QUOTE_FIELDS,
            entity_by=("symbol",),
            event_time="event_time",
            sequence_by=("sequence",),
        )
        joined = tables.stream_join(
            quotes,
            reference,
            left_keys=("symbol",),
            right_keys=("symbol",),
            left_event_time="event_time",
            right_event_time="event_time",
            bounds=JoinTimeBounds(timedelta(seconds=5), timedelta(seconds=5)),
            limits=JoinStateLimits(2 * batch_rows + 768, 1 << 30, 11 * batch_rows),
            left_prefix="left",
            right_prefix="right",
        )
        output = joined.select(
            sequence=joined["left__sequence"],
            right_sequence=joined["right__sequence"],
            value=joined["left__price"] * joined["right__price"],
        )
        inputs = (quotes, reference)
    else:
        output = _scalar_stream_output(scenario, quotes)
    return Program(
        "suite-stream", engine="streaming", inputs=inputs, outputs=(("result", output),)
    ).compile_stream(Runtime())


def stream_events(
    table: pa.Table,
    entities: int,
    *,
    close_windows: bool = False,
    batch_rows: int | None = None,
    checkpoint_split: bool = False,
) -> tuple:
    batch_rows = BATCH_ROWS if batch_rows is None else batch_rows
    # Never finalize half an entity tick before the next data batch.
    size = max(entities, batch_rows // entities * entities)
    starts = list(range(0, table.num_rows, size))
    if checkpoint_split and len(starts) == 1 and table.num_rows > 1:
        prefix = max(1, table.num_rows // 2 // entities * entities)
        if prefix == 1:
            prefix = table.num_rows // 2
        starts = [0, prefix]
    events = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else table.num_rows
        part = table.slice(start, end - start)
        end = start + part.num_rows
        events.append(
            Data(
                Batch.from_pyarrow(part),
                Cursor(
                    end.to_bytes(8, "big"),
                    {"rows": end, "event_index": len(events) + 1},
                ),
            )
        )
        micros = int(pc.max(part["event_time"]).value) + 1
        if end < table.num_rows:
            micros = min(
                micros, int(pc.min(table.slice(end, size)["event_time"]).value)
            )
        events.append(
            Watermark(BASE + timedelta(microseconds=micros - int(BASE_MICROS)))
        )
    if close_windows:
        # Close the final ten-second tumbling window before awaiting its output.
        final_tick = (table.num_rows - 1) // entities
        events.append(Watermark(BASE + timedelta(seconds=(final_tick // 10 + 1) * 10)))
    return (*events, None)


def dimension_events(dimension: pa.Table, quotes: pa.Table) -> tuple:
    """Seed the static dimension and seal the quote time range."""

    rows = dimension.num_rows
    return (
        Data(
            Batch.from_pyarrow(dimension),
            Cursor(rows.to_bytes(8, "big"), {"rows": rows, "event_index": 1}),
        ),
        Watermark(BASE + _join_span(quotes)),
        None,
    )


class _ReadySource(_InteractiveSource):
    def __init__(self, *, max_batch_rows: int | None = None) -> None:
        super().__init__(
            max_batch_rows=BATCH_ROWS if max_batch_rows is None else max_batch_rows
        )
        self._events = asyncio.Queue(maxsize=1)
        self.ready = asyncio.Event()

    async def next(self) -> Data | Watermark | None:
        # The first poll proves that the runtime's startup data gate opened.
        self.ready.set()
        return await super().next()


class _ReplaySource(_ReadySource):
    """Gate an immutable event log with exact next-data cursor recovery."""

    def __init__(self, events: tuple, *, batch_rows: int) -> None:
        super().__init__(max_batch_rows=batch_rows)
        self.events = events
        self.position = 0
        self._pushed = 0
        if not events or events[-1] is not None:
            raise ValueError("replay event log must end with EOF")
        if any(
            event.batch.num_rows > batch_rows
            for event in events
            if isinstance(event, Data)
        ):
            raise ValueError("replay event log exceeds declared batch rows")

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            SourceDeliveryCapability.LOSSLESS,
            self._max_batch_rows,
            32 << 20,
            native_watermarks=NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self.position = 0
        if cursor is not None:
            position = cursor.payload.get("event_index")
            if type(position) is not int or not 1 <= position < len(self.events):
                raise ValueError("invalid replay cursor position")
            previous = self.events[position - 1]
            if (
                not isinstance(previous, Data)
                or previous.cursor.order != cursor.order
                or dict(previous.cursor.payload) != dict(cursor.payload)
            ):
                raise ValueError(
                    "replay cursor does not identify the recorded data position"
                )
            self.position = position
        self._pushed = self.position
        self.opened.set()

    async def push(self, event: Data | Watermark | None) -> None:
        expected = self.events[self._pushed]
        if event is not expected and event != expected:
            raise ValueError("replay feed differs from the immutable event log")
        self._pushed += 1
        await super().push(event)

    async def next(self) -> Data | Watermark | None:
        event = await super().next()
        self.position += 1
        return event


class _CollectSink:
    def __init__(self, expected_rows: int) -> None:
        self.expected_rows = expected_rows
        self.rows = 0
        self.tables: list[pa.Table] = []
        self.opened = asyncio.Event()
        self.complete = asyncio.Event()
        self.progress = asyncio.Event()

    async def open(self) -> None:
        self.opened.set()

    async def write(self, batch: Batch) -> None:
        table = batch.to_pyarrow()
        self.tables.append(table)
        self.rows += table.num_rows
        self.progress.set()
        if self.rows >= self.expected_rows:
            self.complete.set()

    async def wait_for_rows(self, rows: int) -> None:
        while self.rows < rows:
            self.progress.clear()
            await self.progress.wait()

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
    sources: dict[str, _ReadySource],
    sink: _CollectSink,
    streams: dict[str, tuple],
    job,
    *,
    static_join: bool = False,
) -> tuple[pa.Table, float]:
    await asyncio.wait_for(
        asyncio.gather(*(source.ready.wait() for source in sources.values())),
        timeout=30,
    )
    _require_ready_sources(sources, sink)
    pending = streams
    if static_join:
        for event in streams["right"]:
            await sources["right"].push(event)
        delta = streams["right"][-1].at - BASE
        watermark = BASE_MICROS + delta // timedelta(microseconds=1)
        await asyncio.wait_for(
            _wait_static_dimension_progress(job, watermark),
            timeout=600,
        )
        pending = {"left": streams["left"]}
    started = time.perf_counter_ns()
    if (
        set(streams) == {"reference.input", "quotes.input"}
        and len(streams["quotes.input"]) > 2
    ):
        await _feed_asof_chunks(sources, streams, sink)
        pending = {}
    for name, event in interleaved_events(pending):
        await sources[name].push(event)
    await asyncio.wait_for(sink.complete.wait(), timeout=600)
    if sink.rows != sink.expected_rows:
        raise RuntimeError("stream output row count differs from the timed workload")
    table = pa.concat_tables(sink.tables)
    seconds = (time.perf_counter_ns() - started) / 1e9
    if static_join:
        await asyncio.wait_for(
            _wait_static_quote_progress(job, sink.expected_rows), timeout=600
        )
    return table, seconds


async def _feed_asof_chunks(
    sources: dict, streams: dict[str, tuple], sink: _CollectSink
) -> None:
    rows = 0
    for start in range(0, len(streams["quotes.input"]), 2):
        chunk = {name: events[start : start + 2] for name, events in streams.items()}
        for name, event in interleaved_events(chunk):
            await sources[name].push(event)
        rows += chunk["quotes.input"][0].batch.num_rows
        await asyncio.wait_for(sink.wait_for_rows(rows), timeout=600)


def _static_join_status(job) -> dict:
    statuses = tuple(job.status()["stream_joins"].values())
    if len(statuses) != 1:
        raise RuntimeError("static Join requires exactly one Join status")
    return statuses[0]


def _require_static_left_counters(status: dict) -> None:
    if status["left"]["retained_rows"] or status["left"]["evicted_rows"]:
        raise RuntimeError("static Join retained or evicted quote rows")


def _require_static_join_no_left_state(job) -> dict:
    status = _static_join_status(job)
    _require_static_left_counters(status)
    return status


async def _wait_static_quote_progress(job, expected_rows: int) -> dict:
    while True:
        status = _static_join_status(job)
        if status["emitted_match_rows"] >= expected_rows:
            _require_static_left_counters(status)
            return status
        await asyncio.sleep(0.001)


async def _wait_static_dimension_progress(job, watermark: int) -> None:
    while True:
        status = _require_static_join_no_left_state(job)
        accepted = status["right"]["watermark_micros"]
        if accepted is not None and accepted >= watermark:
            return
        await asyncio.sleep(0.001)


async def run_stream(
    plan,
    streams: dict[str, tuple],
    root: Path,
    expected_rows: int,
    *,
    static_join: bool = False,
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

    async def measure_and_end():
        result = await _measure_ready(
            sources, sink, timed, job, static_join=static_join
        )
        for source in sources.values():
            await source.push(None)
        return result

    try:
        table, seconds = await run_with_completion(measure_and_end(), job.wait_async())
        if sink.rows != expected_rows:
            raise RuntimeError("stream output row count changed after the timed result")
        return table, seconds
    finally:
        await job.cancel_async()


async def _with_running_job(operation, completion):
    """Cancel pending benchmark work as soon as its owned runtime terminates."""

    running = asyncio.ensure_future(operation)
    finished = asyncio.ensure_future(completion)
    try:
        done, _ = await asyncio.wait(
            (running, finished), return_when=asyncio.FIRST_COMPLETED
        )
        if finished in done:
            outcome = await finished
            raise RuntimeError(
                f"stream terminated before checkpoint cut: {outcome.state}: "
                f"{outcome.errors}"
            )
        return await running
    finally:
        for task in (running, finished):
            if not task.done():
                task.cancel()
        await asyncio.gather(running, finished, return_exceptions=True)


async def _interval_observation(job, streams: dict[str, tuple]) -> dict:
    """Await accepted eviction frontiers after the throughput timer stopped."""

    targets = {
        name: int(BASE_MICROS) + (events[-2].at - BASE) // timedelta(microseconds=1)
        for name, events in streams.items()
    }
    while True:
        observation = job.status()["stream_joins"]
        statuses = tuple(observation.values())
        if len(statuses) != 1:
            raise RuntimeError("interval Join requires exactly one Join status")
        if all(
            statuses[0][side]["watermark_micros"] is not None
            and statuses[0][side]["watermark_micros"] >= target
            for side, target in targets.items()
        ):
            return observation
        await asyncio.sleep(0.001)


async def _start_replay_job(
    plan,
    streams: dict[str, tuple],
    root: Path,
    expected_rows: int,
    case: dict,
):
    batch_rows = case["batch_rows"]
    checkpoint = case["checkpoint_interval_millis"] is not None
    _validated_timed_streams(plan, streams)
    sources = {
        name: _ReplaySource(events, batch_rows=batch_rows)
        for name, events in streams.items()
    }
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
            checkpoint_interval=timedelta(milliseconds=100)
            if checkpoint
            else timedelta(hours=24),
            edge_budget=EdgeBudget(max_rows=batch_rows, max_bytes=64 << 20),
        ),
    ).start_async()
    return sources, sink, job


async def _start_variant_job(plan_factory, streams, root, expected_rows, case):
    plan = plan_factory()
    dimensions = stream_dimensions(
        case["scenario"],
        case["batch_rows"],
        case["checkpoint_interval_millis"] is not None,
    )
    dimensions["source_bindings"] = sorted(plan.source_binding_ids)
    timed = _validated_timed_streams(plan, streams)
    sources, sink, job = await _start_replay_job(
        plan, streams, root, expected_rows, case
    )
    return dimensions, timed, sources, sink, job


async def _run_throughput_stream(plan_factory, streams, root, expected_rows, case):
    dimensions, timed, sources, sink, job = await _start_variant_job(
        plan_factory, streams, root, expected_rows, case
    )

    async def measure_and_end():
        result = await _measure_ready(
            sources, sink, timed, job, static_join=case["scenario"] == "join"
        )
        observation = (
            await asyncio.wait_for(_interval_observation(job, streams), 600)
            if case["scenario"] == "interval_join"
            else {}
        )
        lookup = (
            _require_static_join_no_left_state(job)
            if case["scenario"] == "join"
            else None
        )
        for source in sources.values():
            await source.push(None)
        return (
            *result,
            {
                **dimensions,
                "nonterminal_epochs": [],
                "recovery": "not-requested",
                "interval_join": observation,
                **({"lookup_join": lookup} if lookup is not None else {}),
            },
        )

    try:
        result = await run_with_completion(measure_and_end(), job.wait_async())
        if sink.rows != expected_rows:
            raise RuntimeError("stream output row count changed after the timed result")
        return result
    finally:
        await job.cancel_async()


async def _checkpoint_cut(job, sources, sink, streams, case):
    prefix_outputs = _checkpoint_prefix_rows(streams, case)[1]
    await asyncio.wait_for(
        asyncio.gather(*(source.ready.wait() for source in sources.values())), 30
    )
    _require_ready_sources(sources, sink)
    prefix = await _checkpoint_pending_prefix(job, sources, streams, case)
    started = time.perf_counter_ns()
    for name, event in interleaved_events(prefix):
        await sources[name].push(event)
    await asyncio.wait_for(sink.wait_for_rows(prefix_outputs), 600)
    await asyncio.sleep(0.1)
    epoch = await job.trigger_checkpoint_async()
    _require_checkpoint_ack(job, epoch)
    if case["scenario"] == "join":
        await asyncio.wait_for(_wait_static_quote_progress(job, prefix_outputs), 600)
    return started, epoch, tuple(sink.tables), sink.rows


async def _checkpoint_pending_prefix(job, sources, streams, case):
    prefix = {name: events[:2] for name, events in streams.items()}
    if case["scenario"] == "join":
        for event in prefix.pop("right"):
            await sources["right"].push(event)
        delta = streams["right"][1].at - BASE
        await asyncio.wait_for(
            _wait_static_dimension_progress(
                job, BASE_MICROS + delta // timedelta(microseconds=1)
            ),
            600,
        )
    return prefix


def _require_checkpoint_ack(job, epoch: int) -> None:
    completed = job.status()["checkpoint"]["last_completed_epoch"]
    if type(completed) is not int or completed < epoch:
        raise RuntimeError("checkpoint acknowledgement lacks durable publication")


def _checkpoint_prefix_rows(streams, case):
    primary = (
        "quotes.input"
        if "quotes.input" in streams
        else "left"
        if "left" in streams
        else "input"
    )
    prefix_rows = streams[primary][0].batch.num_rows
    prefix_outputs = (
        sum(max(0, prefix_rows - abs(offset) * 64) for offset in range(-5, 6))
        if case["scenario"] == "interval_join"
        else prefix_rows
    )
    return prefix_rows, prefix_outputs


async def _recovery_pending_events(sources: dict) -> dict:
    pending = {}
    for name, source in sources.items():
        events = source.events[source.position : -1]
        while events and not isinstance(events[0], Data):
            await source.push(events[0])
            events = events[1:]
        pending[name] = events
    return pending


async def _run_checkpoint_stream(plan_factory, streams, root, expected_rows, case):
    dimensions, _, sources, sink, job = await _start_variant_job(
        plan_factory, streams, root, expected_rows, case
    )
    prefix_rows, _ = _checkpoint_prefix_rows(streams, case)
    if not 0 < prefix_rows < case["rows"]:
        await job.cancel_async()
        raise ValueError("checkpoint workload requires a nonterminal input prefix")

    try:
        started, epoch, prefix_tables, delivered = await _with_running_job(
            _checkpoint_cut(job, sources, sink, streams, case), job.wait_async()
        )
    finally:
        await job.cancel_async()

    restored_plan = plan_factory()
    resumed_sources, resumed_sink, resumed = await _start_replay_job(
        restored_plan, streams, root, expected_rows - delivered, case
    )

    async def complete_recovery():
        await asyncio.wait_for(
            asyncio.gather(
                *(source.ready.wait() for source in resumed_sources.values())
            ),
            30,
        )
        _require_ready_sources(resumed_sources, resumed_sink)
        pending = await _recovery_pending_events(resumed_sources)
        await _measure_ready(resumed_sources, resumed_sink, pending, resumed)
        table = pa.concat_tables((*prefix_tables, *resumed_sink.tables))
        seconds = (time.perf_counter_ns() - started) / 1e9
        observation = (
            await asyncio.wait_for(_interval_observation(resumed, streams), 600)
            if case["scenario"] == "interval_join"
            else {}
        )
        lookup = (
            await asyncio.wait_for(
                _wait_static_quote_progress(resumed, expected_rows), 600
            )
            if case["scenario"] == "join"
            else None
        )
        for source in resumed_sources.values():
            await source.push(None)
        return (
            table,
            seconds,
            {
                **dimensions,
                "nonterminal_epochs": [epoch],
                "rows_before_checkpoint": prefix_rows,
                "recovery": "verified",
                "interval_join": observation,
                **({"lookup_join": lookup} if lookup is not None else {}),
            },
        )

    try:
        result = await run_with_completion(complete_recovery(), resumed.wait_async())
        if resumed_sink.rows + delivered != expected_rows:
            raise RuntimeError(
                "recovery output row count differs from the timed workload"
            )
        return result
    finally:
        await resumed.cancel_async()


async def run_variant_stream(
    plan_factory,
    streams: dict[str, tuple],
    root: Path,
    expected_rows: int,
    *,
    case: dict,
) -> tuple[pa.Table, float, dict]:
    """Run replay-backed throughput or an explicitly paced recovery lifecycle."""

    run = (
        _run_checkpoint_stream
        if case["checkpoint_interval_millis"] is not None
        else _run_throughput_stream
    )
    return await run(plan_factory, streams, root, expected_rows, case)
