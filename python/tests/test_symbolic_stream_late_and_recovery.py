"""Late-row, recovery-rejection, and join checkpoint ownership coverage.

The cross-cutting test matrix promises bounded out-of-order arrival, late
error/drop policies, and restore rejection before sources resume at the
public Python surface. These tests drive those paths end to end through
``StreamingRunner`` and inspect one physical checkpoint manifest directly to
prove a shared symbolic stream join owns exactly one durable state entry.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from calc_flow import (
    Batch,
    Cursor,
    Data,
    Idle,
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
    StreamingRuntimeError,
    Watermark,
)
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    rows,
    table,
    table_input,
    ts,
)
from calc_flow.symbolic.lower import lower_program_document

BASE = datetime(2026, 1, 1, tzinfo=UTC)
_BASE_MICROS = int(BASE.timestamp() * 1_000_000)


def _quotes() -> object:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )


def _rolling_program(window: int) -> Program:
    quotes = _quotes()
    signals = quotes.with_columns(
        FeatureSet([("avg", ts.mean(quotes["x"], window=rows(window)))])
    )
    return Program("late-policy", inputs=[quotes], outputs=[("signals", signals)])


def _row(ts_micros: int, value: float) -> pa.Table:
    stamp = BASE + timedelta(microseconds=ts_micros)
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("x", pa.float64(), nullable=True),
        ]
    )
    return pa.table(
        {
            "ts": pa.array([stamp], type=pa.timestamp("us", tz="UTC")),
            "symbol": ["a"],
            "seq": pa.array([ts_micros], type=pa.uint64()),
            "x": pa.array([value], type=pa.float64()),
        },
        schema=schema,
    )


class _ScriptedSource:
    """Emits scripted rows and watermarks, optionally pausing at one index."""

    def __init__(
        self,
        events: list[tuple[str, object]],
        pause_at: int | None = None,
        opened_offsets: list[int] | None = None,
    ) -> None:
        self._events = events
        self._pause_at = pause_at
        self._opened_offsets = opened_offsets
        self._index = 0
        self.paused = asyncio.Event()

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=1,
            max_batch_bytes=16 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._index = 0 if cursor is None else int(cursor.payload["index"])
        if self._opened_offsets is not None:
            self._opened_offsets.append(self._index)
        self.paused.clear()

    async def next(self) -> Data | Watermark | Idle | None:
        if self._pause_at is not None and self._index == self._pause_at:
            self.paused.set()
            await asyncio.sleep(0)
            return Idle()
        if self._index >= len(self._events):
            return None
        kind, payload = self._events[self._index]
        self._index += 1
        if kind == "watermark":
            return Watermark(BASE + timedelta(microseconds=int(payload)))
        return Data(
            Batch.from_pyarrow(payload),
            Cursor(self._index.to_bytes(8, "big"), {"index": self._index}),
        )

    async def close(self) -> None:
        return None


class _CollectSink:
    def __init__(self) -> None:
        self.tables: list[pa.Table] = []

    async def open(self) -> None:
        return None

    async def write(self, batch: Batch) -> None:
        table = batch.to_pyarrow()
        if table.num_rows:
            self.tables.append(table)

    async def close(self) -> None:
        return None


def _collected(sink: _CollectSink) -> list[tuple[int, float, float | None]]:
    merged = pa.concat_tables(sink.tables)
    return list(
        zip(
            (
                micros - _BASE_MICROS
                for micros in merged["ts"].cast(pa.int64()).to_pylist()
            ),
            merged["x"].to_pylist(),
            merged["avg"].to_pylist(),
            strict=True,
        )
    )


def _runner(
    plan: object,
    source: _ScriptedSource,
    sinks: dict[str, list[SinkBinding]],
    state_root: Path,
) -> StreamingRunner:
    return StreamingRunner(
        plan,
        {"input": SourceBinding(source, watermark_policy=SourceProvidedWatermarks())},
        sinks,
        ManagedCheckpointRuntime(state_root),
    )


async def _run_to_completion(runner: StreamingRunner) -> object:
    job = await runner.start_async()
    return await job.wait_async()


def test_out_of_order_rows_within_lateness_emit_in_canonical_order(
    tmp_path: Path,
) -> None:
    # Rows arrive 12, 10, 11 before any watermark exists, so none can be
    # classified late; allowed lateness of 5s lets the closing watermark at
    # 18 finalize all three, which must emit in (event time, entity,
    # sequence) order with window means over the accepted history.
    events = [
        ("row", _row(12, 3.0)),
        ("row", _row(10, 1.0)),
        ("row", _row(11, 2.0)),
        ("watermark", 18_000_000),
    ]
    sink = _CollectSink()
    plan = _rolling_program(3).compile_stream(
        Runtime(), allowed_lateness_micros=5_000_000, late_policy="drop"
    )

    asyncio.run(
        _run_to_completion(
            _runner(
                plan,
                _ScriptedSource(events),
                {"output": [SinkBinding.ordinary("archive", sink)]},
                tmp_path,
            )
        )
    )

    assert _collected(sink) == [
        (10, 1.0, 1.0),
        (11, 2.0, 1.5),
        (12, 3.0, 2.0),
    ]


def test_too_late_row_is_dropped_and_surviving_rows_match_reference(
    tmp_path: Path,
) -> None:
    # The watermark at 20s closes rows 10..12 (emitted in order). The
    # follow-up row at 10 is beyond the 5s allowed lateness (closing 15s)
    # and is dropped instead of duplicating the buffered identity.
    events = [
        ("row", _row(10, 1.0)),
        ("row", _row(11, 2.0)),
        ("row", _row(12, 3.0)),
        ("watermark", 20_000_000),
        ("row", _row(10, 99.0)),
    ]
    sink = _CollectSink()
    plan = _rolling_program(3).compile_stream(
        Runtime(), allowed_lateness_micros=5_000_000, late_policy="drop"
    )

    outcome = asyncio.run(
        _run_to_completion(
            _runner(
                plan,
                _ScriptedSource(events),
                {"output": [SinkBinding.ordinary("archive", sink)]},
                tmp_path,
            )
        )
    )

    assert outcome.state == "completed"
    assert outcome.errors == ()
    assert _collected(sink) == [
        (10, 1.0, 1.0),
        (11, 2.0, 1.5),
        (12, 3.0, 2.0),
    ]


def test_too_late_row_fails_the_job_under_the_error_policy(tmp_path: Path) -> None:
    events = [
        ("row", _row(10, 1.0)),
        ("row", _row(11, 2.0)),
        ("row", _row(12, 3.0)),
        ("watermark", 20_000_000),
        ("row", _row(10, 99.0)),
    ]
    sink = _CollectSink()
    plan = _rolling_program(3).compile_stream(
        Runtime(), allowed_lateness_micros=5_000_000, late_policy="error"
    )

    outcome = asyncio.run(
        _run_to_completion(
            _runner(
                plan,
                _ScriptedSource(events),
                {"output": [SinkBinding.ordinary("archive", sink)]},
                tmp_path,
            )
        )
    )

    assert outcome.state == "failed"
    assert outcome.errors[0].category == "operator"


def test_recovery_with_a_different_window_is_rejected_before_sources_resume(
    tmp_path: Path,
) -> None:
    opened: list[int] = []

    async def exercise() -> None:
        events = [("row", _row(10, 1.0)), ("row", _row(11, 2.0))]
        first_source = _ScriptedSource(events, pause_at=2, opened_offsets=opened)
        first = await _runner(
            _rolling_program(3).compile_stream(Runtime()),
            first_source,
            {"output": [SinkBinding.ordinary("archive", _CollectSink())]},
            tmp_path,
        ).start_async()
        await asyncio.wait_for(first_source.paused.wait(), timeout=30)
        assert await first.trigger_checkpoint_async() == 1
        assert (await first.cancel_async()).state == "cancelled"

        second_source = _ScriptedSource([("row", _row(12, 3.0))], opened_offsets=opened)
        second = _runner(
            _rolling_program(4).compile_stream(Runtime()),
            second_source,
            {"output": [SinkBinding.ordinary("archive", _CollectSink())]},
            tmp_path,
        )
        with pytest.raises(StreamingRuntimeError, match="checkpoint lineage"):
            await second.start_async()

    asyncio.run(exercise())

    assert opened == [0], "the mismatched recovery must not resume sources"


def _join_program() -> Program:
    left = table_input(
        "left_events",
        schema=[
            Field("key", "int64", nullable=False),
            Field("left_ts", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("left_value", "float64", nullable=False),
        ],
        entity_by=["key"],
        event_time="left_ts",
        sequence_by=["sequence"],
    )
    right = table_input(
        "right_events",
        schema=[
            Field("key", "int64", nullable=False),
            Field("right_ts", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("right_value", "float64", nullable=False),
        ],
        entity_by=["key"],
        event_time="right_ts",
        sequence_by=["sequence"],
    )
    joined = table.stream_join(
        left,
        right,
        left_keys=["key"],
        right_keys=["key"],
        left_event_time="left_ts",
        right_event_time="right_ts",
        bounds=JoinTimeBounds(timedelta(seconds=5), timedelta(seconds=2)),
        limits=JoinStateLimits(1_000, 16 * 1024 * 1024, 10_000),
        left_prefix="left",
        right_prefix="right",
    )
    settled = table.filter(joined, joined["left__left_value"] > 0.0)
    amounts = table.project(
        joined.with_columns(FeatureSet([("double", joined["left__left_value"] * 2.0)])),
        ["left__key", "double"],
    )
    return Program(
        "join-fanout",
        inputs=[left, right],
        outputs=[("settled", settled), ("amounts", amounts)],
    )


def _side_table(prefix: str) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("key", pa.int64(), nullable=False),
            pa.field(f"{prefix}_ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("sequence", pa.uint64(), nullable=False),
            pa.field(f"{prefix}_value", pa.float64(), nullable=False),
        ]
    )
    return pa.table(
        {
            "key": pa.array([1, 2], type=pa.int64()),
            f"{prefix}_ts": pa.array(
                [
                    BASE + timedelta(seconds=10),
                    BASE + timedelta(seconds=20),
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "sequence": pa.array([1, 1], type=pa.uint64()),
            f"{prefix}_value": pa.array([100.0, 50.0], type=pa.float64()),
        },
        schema=schema,
    )


def test_one_join_digest_owns_one_physical_checkpoint_entry(tmp_path: Path) -> None:
    program = _join_program()
    document = lower_program_document(program, Runtime(), "stream")
    join_nodes = [
        node["id"]
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "stream_join"
    ]
    assert len(join_nodes) == 1

    async def exercise() -> None:
        left_source = _ScriptedSource(
            [("row", _side_table("left").slice(0, 1))], pause_at=1
        )
        right_source = _ScriptedSource(
            [("row", _side_table("right").slice(0, 1))], pause_at=1
        )
        runner = StreamingRunner(
            program.compile_stream(Runtime()),
            {
                "left": SourceBinding(
                    left_source, watermark_policy=SourceProvidedWatermarks()
                ),
                "right": SourceBinding(
                    right_source, watermark_policy=SourceProvidedWatermarks()
                ),
            },
            {
                "settled.output": [
                    SinkBinding.ordinary("settled_archive", _CollectSink())
                ],
                "amounts.output": [
                    SinkBinding.ordinary("amounts_archive", _CollectSink())
                ],
            },
            ManagedCheckpointRuntime(tmp_path),
        )
        job = await runner.start_async()
        await asyncio.wait_for(left_source.paused.wait(), timeout=30)
        await asyncio.wait_for(right_source.paused.wait(), timeout=30)
        assert await job.trigger_checkpoint_async() == 1
        assert (await job.cancel_async()).state == "cancelled"

    asyncio.run(exercise())

    manifests = sorted((tmp_path / "manifests").glob("manifest-*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    operators = manifest["operators"]
    join_id = join_nodes[0]
    assert list(operators).count(join_id) == 1
    join_entry = operators[join_id]
    assert join_entry["inline_metadata"]["layout_version"] == 1
    segment_owners = {
        operator_id: len(entry["segments"])
        for operator_id, entry in operators.items()
        if entry["segments"]
    }
    assert segment_owners == {join_id: 2}
    segment_ids = {segment["segment_id"] for segment in join_entry["segments"]}
    assert segment_ids == {"left-delta-1", "right-delta-1"}
    # Fan-out branches and the join's output frontier record no durable
    # segment of their own.
    assert all(
        not entry["segments"]
        for operator_id, entry in operators.items()
        if operator_id != join_id
    )
