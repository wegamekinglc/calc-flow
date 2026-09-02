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


def _input(name: str, value_name: str):
    return table_input(
        name,
        schema=[
            Field("key", "int64", nullable=False),
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field(value_name, "float64", nullable=False),
        ],
        entity_by=["key"],
        event_time="ts",
        sequence_by=["sequence"],
    )


def _program() -> Program:
    left = _input("left_events", "left_value")
    middle = _input("middle_events", "middle_value")
    right = _input("right_events", "right_value")
    limits = JoinStateLimits(1_000, 16 * 1024 * 1024, 10_000)
    bounds = JoinTimeBounds(timedelta(seconds=5), timedelta(seconds=2))
    first = table.stream_join(
        left,
        middle,
        left_keys=["key"],
        right_keys=["key"],
        left_event_time="ts",
        right_event_time="ts",
        bounds=bounds,
        limits=limits,
        left_prefix="left",
        right_prefix="middle",
        output_entity_by=["left__key"],
        output_event_time="left__ts",
        output_sequence_by=["left__sequence", "middle__sequence"],
    )
    second = table.stream_join(
        first,
        right,
        left_keys=["left__key"],
        right_keys=["key"],
        left_event_time="left__ts",
        right_event_time="ts",
        bounds=bounds,
        limits=limits,
        left_prefix="matched",
        right_prefix="right",
    )
    output = second.with_columns(
        FeatureSet(
            [
                (
                    "total",
                    second["matched__left__left_value"]
                    - second["matched__middle__middle_value"]
                    + second["right__right_value"],
                )
            ]
        )
    )
    return Program(
        "symbolic-relational-dag-acceptance",
        inputs=[left, middle, right],
        outputs=[("matches", output)],
    )


def _table(value_name: str, rows: list[tuple[int, int, int, float]]) -> pa.Table:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    schema = pa.schema(
        [
            pa.field("key", pa.int64(), nullable=False),
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("sequence", pa.uint64(), nullable=False),
            pa.field(value_name, pa.float64(), nullable=False),
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array([row[0] for row in rows], type=pa.int64()),
            pa.array(
                [base + timedelta(seconds=row[1]) for row in rows],
                type=pa.timestamp("us", tz="UTC"),
            ),
            pa.array([row[2] for row in rows], type=pa.uint64()),
            pa.array([row[3] for row in rows], type=pa.float64()),
        ],
        schema=schema,
    )


def _left_table() -> pa.Table:
    return _table("left_value", [(1, 10, 1, 100.0), (1, 20, 2, 80.0), (2, 30, 1, 50.0)])


def _middle_table() -> pa.Table:
    return _table(
        "middle_value", [(1, 11, 1, 90.0), (1, 19, 2, 70.0), (2, 31, 1, 40.0)]
    )


def _right_table() -> pa.Table:
    return _table("right_value", [(1, 12, 1, 5.0), (1, 22, 2, 6.0), (2, 32, 1, 7.0)])


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
                Cursor(self._offset.to_bytes(8, "big"), {"offset": self._offset}),
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


def _rows(tables: list[pa.Table]) -> list[tuple[int, int, int, int, float]]:
    output = pa.concat_tables(tables).select(
        [
            "matched__left__key",
            "matched__left__sequence",
            "matched__middle__sequence",
            "right__sequence",
            "total",
        ]
    )
    return sorted(
        (
            row["matched__left__key"],
            row["matched__left__sequence"],
            row["matched__middle__sequence"],
            row["right__sequence"],
            row["total"],
        )
        for row in output.to_pylist()
    )


def _expected() -> list[tuple[int, int, int, int, float]]:
    return [(1, 1, 1, 1, 15.0), (1, 2, 2, 2, 16.0), (2, 1, 1, 1, 17.0)]


@pytest.mark.parametrize("segments", [(1, 2, 1), (3, 1, 2)])
def test_nested_symbolic_joins_match_reference_across_segmentations(
    tmp_path: Path, segments: tuple[int, int, int]
) -> None:
    plan = _program().compile_stream(Runtime())
    assert plan.source_binding_ids == (
        "left_events.input",
        "middle_events.input",
        "right_events.input",
    )
    assert plan.sink_binding_ids == ("output",)
    sink = _CollectSink()

    async def exercise() -> None:
        job = await StreamingRunner(
            plan,
            {
                "left_events.input": SourceBinding(
                    _SegmentedSource(_left_table(), segments[0]),
                    watermark_policy=SourceProvidedWatermarks(),
                ),
                "middle_events.input": SourceBinding(
                    _SegmentedSource(_middle_table(), segments[1]),
                    watermark_policy=SourceProvidedWatermarks(),
                ),
                "right_events.input": SourceBinding(
                    _SegmentedSource(_right_table(), segments[2]),
                    watermark_policy=SourceProvidedWatermarks(),
                ),
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        assert (await job.wait_async()).state == "completed"

    asyncio.run(exercise())

    assert _rows(sink.tables) == _expected()


def test_nested_symbolic_joins_restore_each_state_owner(tmp_path: Path) -> None:
    left = _left_table().slice(0, 1)
    middle = _middle_table().slice(0, 1)
    right = _right_table().slice(0, 1)
    sink = _CollectSink()
    left_offsets: list[int] = []
    middle_offsets: list[int] = []
    right_offsets: list[int] = []

    def runner(
        left_source: _ReplayPauseSource,
        middle_source: _ReplayPauseSource,
        right_source: _ReplayPauseSource,
    ) -> StreamingRunner:
        return StreamingRunner(
            _program().compile_stream(Runtime()),
            {
                "left_events.input": SourceBinding(
                    left_source,
                    watermark_policy=SourceProvidedWatermarks(),
                ),
                "middle_events.input": SourceBinding(
                    middle_source,
                    watermark_policy=SourceProvidedWatermarks(),
                ),
                "right_events.input": SourceBinding(
                    right_source,
                    watermark_policy=SourceProvidedWatermarks(),
                ),
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
        )

    async def exercise() -> None:
        first_left = _ReplayPauseSource(left, 1, left_offsets)
        first_middle = _ReplayPauseSource(middle, 1, middle_offsets)
        first_right = _ReplayPauseSource(right, 0, right_offsets)
        first = await runner(first_left, first_middle, first_right).start_async()
        await asyncio.wait_for(first_left.paused.wait(), timeout=2)
        await asyncio.wait_for(first_middle.paused.wait(), timeout=2)
        await asyncio.wait_for(first_right.paused.wait(), timeout=2)
        assert await first.trigger_checkpoint_async() == 1
        assert (await first.cancel_async()).state == "cancelled"
        assert sink.tables == []

        second = await runner(
            _ReplayPauseSource(left, None, left_offsets),
            _ReplayPauseSource(middle, None, middle_offsets),
            _ReplayPauseSource(right, None, right_offsets),
        ).start_async()
        assert (await second.wait_async()).state == "completed"

    asyncio.run(exercise())

    assert _rows(sink.tables) == [_expected()[0]]
    assert left_offsets == [0, 1]
    assert middle_offsets == [0, 1]
    assert right_offsets == [0, 0]
