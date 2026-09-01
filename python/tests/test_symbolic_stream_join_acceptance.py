from __future__ import annotations

import asyncio
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
    Watermark,
)
from calc_flow.symbolic import FeatureSet, Field, Program, table, table_input


def _program() -> Program:
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
    output = joined.with_columns(
        FeatureSet(
            [
                (
                    "spread",
                    joined["left__left_value"] - joined["right__right_value"],
                )
            ]
        )
    )
    return Program(
        "symbolic-join-acceptance",
        inputs=[left, right],
        outputs=[("matches", output)],
    )


def _left_table() -> pa.Table:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    schema = pa.schema(
        [
            pa.field("key", pa.int64(), nullable=False),
            pa.field("left_ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("sequence", pa.uint64(), nullable=False),
            pa.field("left_value", pa.float64(), nullable=False),
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array([1, 2, 1, 3], type=pa.int64()),
            pa.array(
                [
                    base + timedelta(seconds=10),
                    base + timedelta(seconds=20),
                    base + timedelta(seconds=30),
                    base + timedelta(seconds=40),
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            pa.array([1, 1, 2, 1], type=pa.uint64()),
            pa.array([100.0, 50.0, 80.0, 25.0], type=pa.float64()),
        ],
        schema=schema,
    )


def _right_table() -> pa.Table:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    schema = pa.schema(
        [
            pa.field("key", pa.int64(), nullable=False),
            pa.field("right_ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("sequence", pa.uint64(), nullable=False),
            pa.field("right_value", pa.float64(), nullable=False),
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array([1, 2, 1, 3], type=pa.int64()),
            pa.array(
                [
                    base + timedelta(seconds=12),
                    base + timedelta(seconds=16),
                    base + timedelta(seconds=20),
                    base + timedelta(seconds=50),
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            pa.array([1, 1, 2, 1], type=pa.uint64()),
            pa.array([90.0, 45.0, 70.0, 20.0], type=pa.float64()),
        ],
        schema=schema,
    )


def _reference(left: pa.Table, right: pa.Table) -> list[tuple[int, int, int, float]]:
    matches: list[tuple[int, int, int, float]] = []
    for left_row in left.to_pylist():
        for right_row in right.to_pylist():
            if left_row["key"] != right_row["key"]:
                continue
            delta = right_row["right_ts"] - left_row["left_ts"]
            if -timedelta(seconds=5) <= delta <= timedelta(seconds=2):
                matches.append(
                    (
                        left_row["key"],
                        left_row["sequence"],
                        right_row["sequence"],
                        left_row["left_value"] - right_row["right_value"],
                    )
                )
    return sorted(matches)


class _SegmentedSource:
    def __init__(self, value: pa.Table, segment: int) -> None:
        self._value = value
        self._segment = segment
        self._offset = 0
        self._watermark_emitted = False

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=10_000,
            max_batch_bytes=16 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._offset = 0
        self._watermark_emitted = False

    async def next(self) -> Data | Watermark | None:
        if self._offset < self._value.num_rows:
            end = min(self._offset + self._segment, self._value.num_rows)
            value = self._value.slice(self._offset, end - self._offset)
            self._offset = end
            return Data(
                Batch.from_pyarrow(value),
                Cursor(end.to_bytes(8, "big"), {"offset": end}),
            )
        if not self._watermark_emitted:
            self._watermark_emitted = True
            return Watermark(datetime(2030, 1, 1, tzinfo=UTC))
        return None

    async def close(self) -> None:
        return None


class _ReplayPauseSource:
    def __init__(
        self,
        value: pa.Table,
        pause_at: int | None,
        opened_offsets: list[int],
    ) -> None:
        self._value = value
        self._pause_at = pause_at
        self._opened_offsets = opened_offsets
        self._offset = 0
        self._watermark_emitted = False
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
        self._offset = 0 if cursor is None else int(cursor.payload["offset"])
        self._opened_offsets.append(self._offset)
        self._watermark_emitted = False

    async def next(self) -> Data | Watermark | Idle | None:
        if self._pause_at == self._offset:
            self.paused.set()
            await asyncio.sleep(0)
            return Idle()
        if self._offset < self._value.num_rows:
            value = self._value.slice(self._offset, 1)
            self._offset += 1
            return Data(
                Batch.from_pyarrow(value),
                Cursor(
                    self._offset.to_bytes(8, "big"),
                    {"offset": self._offset},
                ),
            )
        if not self._watermark_emitted:
            self._watermark_emitted = True
            return Watermark(datetime(2030, 1, 1, tzinfo=UTC))
        return None

    async def close(self) -> None:
        return None


class _CollectSink:
    def __init__(self) -> None:
        self.tables: list[pa.Table] = []

    async def open(self) -> None:
        return None

    async def write(self, batch: Batch) -> None:
        value = batch.to_pyarrow()
        if value.num_rows:
            self.tables.append(value)

    async def close(self) -> None:
        return None


def _rows(tables: list[pa.Table]) -> list[tuple[int, int, int, float]]:
    output = pa.concat_tables(tables).to_pylist()
    return sorted(
        (
            row["left__key"],
            row["left__sequence"],
            row["right__sequence"],
            row["spread"],
        )
        for row in output
    )


@pytest.mark.parametrize(("left_segment", "right_segment"), ((1, 2), (3, 1)))
def test_symbolic_stream_join_matches_reference_across_segmentations(
    tmp_path: Path, left_segment: int, right_segment: int
) -> None:
    plan = _program().compile_stream(Runtime())
    assert plan.source_binding_ids == ("left", "right")
    assert plan.sink_binding_ids == ("output",)
    sink = _CollectSink()

    async def exercise() -> None:
        job = await StreamingRunner(
            plan,
            {
                "left": SourceBinding(
                    _SegmentedSource(_left_table(), left_segment),
                    watermark_policy=SourceProvidedWatermarks(),
                ),
                "right": SourceBinding(
                    _SegmentedSource(_right_table(), right_segment),
                    watermark_policy=SourceProvidedWatermarks(),
                ),
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        assert (await job.wait_async()).state == "completed"

    asyncio.run(exercise())

    assert _rows(sink.tables) == _reference(_left_table(), _right_table())


def test_symbolic_stream_join_restores_buffered_state(tmp_path: Path) -> None:
    left = _left_table().slice(0, 1)
    right = _right_table().slice(0, 1)
    sink = _CollectSink()
    left_offsets: list[int] = []
    right_offsets: list[int] = []

    def runner(
        left_source: _ReplayPauseSource,
        right_source: _ReplayPauseSource,
    ) -> StreamingRunner:
        return StreamingRunner(
            _program().compile_stream(Runtime()),
            {
                "left": SourceBinding(
                    left_source,
                    watermark_policy=SourceProvidedWatermarks(),
                ),
                "right": SourceBinding(
                    right_source,
                    watermark_policy=SourceProvidedWatermarks(),
                ),
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
        )

    async def exercise() -> None:
        first_left = _ReplayPauseSource(left, 1, left_offsets)
        first_right = _ReplayPauseSource(right, 0, right_offsets)
        first = await runner(first_left, first_right).start_async()
        await asyncio.wait_for(first_left.paused.wait(), timeout=2)
        await asyncio.wait_for(first_right.paused.wait(), timeout=2)
        assert await first.trigger_checkpoint_async() == 1
        assert (await first.cancel_async()).state == "cancelled"
        assert sink.tables == []

        second_left = _ReplayPauseSource(left, None, left_offsets)
        second_right = _ReplayPauseSource(right, None, right_offsets)
        second = await runner(second_left, second_right).start_async()
        assert (await second.wait_async()).state == "completed"

    asyncio.run(exercise())

    assert _rows(sink.tables) == _reference(left, right)
    assert left_offsets == [0, 1]
    assert right_offsets == [0, 0]
