"""High-cardinality acceptance coverage for the symbolic temporal catalog.

The SCE-07 state gate and the cross-cutting test matrix promise correctness
with ``one / many / high-cardinality`` active entities and groups. The shipped
fixtures top out near eight entities, so these tests drive thousands of
interleaved entities through rolling row windows (shared numeric state plus
EWMA layout v2 accumulators) and thousands of complete cross-section groups,
comparing every output row against an independently derived Python oracle and
against the batch engine across stream segmentation and checkpoint recovery.
"""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from calc_flow import (
    Batch,
    Cursor,
    Data,
    DisabledWatermarks,
    Idle,
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
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    cs,
    exact_time,
    rows,
    table_input,
    ts,
)

BATCH_ENTITIES = 2_000
STREAM_ENTITIES = 800
ROWS_PER_ENTITY = 5
WINDOW = 3
EWMA_SPAN = 3
EWMA_MIN_PERIODS = 2
CROSS_SECTION_GROUPS = 1_200
BASE = datetime(2026, 1, 1, tzinfo=UTC)


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


def _rolling_program() -> Program:
    quotes = _quotes()
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("n3", ts.count(quotes["x"], window=rows(WINDOW), min_periods=1)),
                ("avg3", ts.mean(quotes["x"], window=rows(WINDOW), min_periods=1)),
                (
                    "ema3",
                    ts.ewma(quotes["x"], span=EWMA_SPAN, min_periods=EWMA_MIN_PERIODS),
                ),
            ]
        )
    )
    return Program(
        "high-cardinality-rolling", inputs=[quotes], outputs=[("signals", signals)]
    )


def _cross_section_program() -> Program:
    quotes = _quotes()
    group = exact_time(quotes["ts"])
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("rank", cs.rank(quotes["x"], group=group)),
                ("zscore", cs.zscore(quotes["x"], group=group, ddof=0)),
            ]
        )
    )
    return Program(
        "high-cardinality-cross-section",
        inputs=[quotes],
        outputs=[("signals", signals)],
    )


def _rolling_rows(entity_count: int) -> list[tuple[datetime, str, int, float | None]]:
    rows_ = []
    for row_index in range(ROWS_PER_ENTITY):
        for entity in range(entity_count):
            # Distinct timestamps per entity keep (event time, sequence)
            # identities unique while arrival stays round-robin interleaved.
            stamp = BASE + timedelta(
                microseconds=1_000_000 * (2 * row_index + 1) + 1_000 * (entity % 977)
            )
            value = 1.0 + entity + row_index / 4.0
            if (entity * ROWS_PER_ENTITY + row_index) % 13 == 0:
                value = None
            rows_.append((stamp, f"sym-{entity:05d}", row_index + 1, value))
    return rows_


def _cross_section_rows(group_count: int) -> list[tuple[datetime, str, int, float]]:
    rows_ = []
    for group in range(group_count):
        stamp = BASE + timedelta(microseconds=1_000_000 * group)
        for member in range(3):
            symbol = f"mem-{group:05d}-{member}"
            rows_.append((stamp, symbol, member + 1, 0.5 + member + (group % 7) / 8.0))
    return rows_


def _table(rows_: list[tuple]) -> pa.Table:
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
            "ts": pa.array(
                [row[0] for row in rows_], type=pa.timestamp("us", tz="UTC")
            ),
            "symbol": [row[1] for row in rows_],
            "seq": pa.array([row[2] for row in rows_], type=pa.uint64()),
            "x": pa.array([row[3] for row in rows_], type=pa.float64()),
        },
        schema=schema,
    )


def _rolling_oracle(
    rows_: list[tuple],
) -> list[tuple[str, int, int | None, float | None, float | None]]:
    # Independent reference: per-entity row windows slide over every row
    # (validity gates the aggregates) while EWMA keeps one unbounded
    # valid-sample accumulator per entity (unadjusted recurrence, first valid
    # sample seeds, null/NaN inputs are ignored).
    ordered = sorted(rows_, key=lambda row: (row[0], row[1], row[2]))
    by_entity: dict[str, list[tuple]] = {}
    for row in ordered:
        by_entity.setdefault(row[1], []).append(row)
    alpha = 2.0 / (EWMA_SPAN + 1.0)
    results: dict[int, tuple[int | None, float | None, float | None]] = {}
    for entity_rows in by_entity.values():
        window: list[float | None] = []
        valid_seen = 0
        accumulator: float | None = None
        for row in entity_rows:
            window.append(row[3])
            if len(window) > WINDOW:
                window.pop(0)
            samples = [
                value for value in window if value is not None and not math.isnan(value)
            ]
            # The engine gates every aggregate on the valid-sample
            # minimum-period rule, so an all-null window reads null.
            valid = len(samples)
            count = valid if valid else None
            mean = sum(samples) / valid if valid else None
            value = row[3]
            if value is not None and not math.isnan(value):
                valid_seen += 1
                accumulator = (
                    value
                    if accumulator is None
                    else accumulator + alpha * (value - accumulator)
                )
            ema = accumulator if valid_seen >= EWMA_MIN_PERIODS else None
            results[id(row)] = (count, mean, ema)
    return [(row[1], row[2], *results[id(row)]) for row in ordered]


def _cross_section_oracle(
    rows_: list[tuple],
) -> dict[str, tuple[float, float]]:
    by_group: dict[datetime, list[tuple]] = {}
    for row in rows_:
        by_group.setdefault(row[0], []).append(row)
    results: dict[str, tuple[float, float]] = {}
    for members in by_group.values():
        ordered = sorted(members, key=lambda row: row[3])
        values = [row[3] for row in ordered]
        count = len(values)
        mean = sum(values) / count
        variance = sum((value - mean) ** 2 for value in values) / count
        stddev = math.sqrt(variance)
        for index, row in enumerate(ordered):
            rank = float(index + 1)
            zscore = (row[3] - mean) / stddev if stddev > 0.0 else None
            results[row[1]] = (rank, zscore)
    return results


def _execute_batch(program: Program, table: pa.Table) -> pa.Table:
    result = program.compile_batch(Runtime()).execute(
        {"input": Batch.from_pyarrow(table)}
    )
    return result.outputs["output"].to_pyarrow()


def _assert_optional_floats(
    actual: list[float | None], expected: list[float | None]
) -> None:
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert observed is None
        else:
            assert observed == pytest.approx(reference, rel=1e-9, abs=1e-12)


def test_high_cardinality_rolling_matches_independent_reference() -> None:
    rows_ = _rolling_rows(BATCH_ENTITIES)
    output = _execute_batch(_rolling_program(), _table(rows_))

    expected = _rolling_oracle(rows_)
    assert output.num_rows == len(expected)
    assert len({row[0] for row in expected}) == BATCH_ENTITIES
    _assert_optional_floats(output["n3"].to_pylist(), [row[2] for row in expected])
    _assert_optional_floats(output["avg3"].to_pylist(), [row[3] for row in expected])
    _assert_optional_floats(output["ema3"].to_pylist(), [row[4] for row in expected])


def test_high_cardinality_cross_section_matches_independent_reference() -> None:
    rows_ = _cross_section_rows(CROSS_SECTION_GROUPS)
    output = _execute_batch(_cross_section_program(), _table(rows_))

    expected = _cross_section_oracle(rows_)
    assert output.num_rows == len(rows_)
    assert len(expected) == len(rows_)
    assert len({row[0] for row in rows_}) == CROSS_SECTION_GROUPS
    symbols = output["symbol"].to_pylist()
    _assert_optional_floats(
        output["rank"].to_pylist(), [expected[symbol][0] for symbol in symbols]
    )
    _assert_optional_floats(
        output["zscore"].to_pylist(), [expected[symbol][1] for symbol in symbols]
    )


class _SegmentedSource:
    def __init__(self, table: pa.Table, segment: int, *, watermarks: bool) -> None:
        self._table = table
        self._segment = segment
        self._watermarks = watermarks
        self._offset = 0
        self._watermark_emitted = False

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=10_000,
            max_batch_bytes=16 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.EMITS_NATIVE
            if self._watermarks
            else NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._offset = 0
        self._watermark_emitted = False

    async def next(self) -> Data | Watermark | None:
        if self._offset < self._table.num_rows:
            end = min(self._offset + self._segment, self._table.num_rows)
            chunk = self._table.slice(self._offset, end - self._offset)
            self._offset = end
            return Data(
                Batch.from_pyarrow(chunk),
                Cursor(end.to_bytes(8, "big"), {"offset": end}),
            )
        if self._watermarks and not self._watermark_emitted:
            self._watermark_emitted = True
            return Watermark(BASE + timedelta(days=365))
        return None

    async def close(self) -> None:
        return None


class _ReplayPauseSource:
    def __init__(
        self,
        table: pa.Table,
        pause_at: int | None,
        opened_offsets: list[int],
    ) -> None:
        self._table = table
        self._pause_at = pause_at
        self._opened_offsets = opened_offsets
        self._offset = 0
        self.paused = asyncio.Event()

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=1,
            max_batch_bytes=16 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._offset = 0 if cursor is None else int(cursor.payload["offset"])
        self._opened_offsets.append(self._offset)

    async def next(self) -> Data | Idle | None:
        if self._pause_at == self._offset:
            self.paused.set()
            await asyncio.sleep(0)
            return Idle()
        if self._offset >= self._table.num_rows:
            return None
        chunk = self._table.slice(self._offset, 1)
        self._offset += 1
        return Data(
            Batch.from_pyarrow(chunk),
            Cursor(self._offset.to_bytes(8, "big"), {"offset": self._offset}),
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


@pytest.mark.parametrize("segmentation", (50, 10_000))
def test_high_cardinality_rolling_stream_matches_batch_across_segmentation(
    tmp_path: Path, segmentation: int
) -> None:
    table = _table(_rolling_rows(STREAM_ENTITIES))
    plan = _rolling_program().compile_stream(Runtime())
    sink = _CollectSink()

    async def exercise() -> None:
        job = await StreamingRunner(
            plan,
            {
                "input": SourceBinding(
                    _SegmentedSource(table, segmentation, watermarks=False),
                    watermark_policy=DisabledWatermarks(),
                )
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        assert (await job.wait_async()).state == "completed"

    asyncio.run(exercise())

    stream_output = pa.concat_tables(sink.tables)
    batch_output = _execute_batch(_rolling_program(), table)
    assert stream_output.num_rows == batch_output.num_rows
    for column in ("n3", "avg3", "ema3"):
        _assert_optional_floats(
            stream_output[column].to_pylist(), batch_output[column].to_pylist()
        )


def test_high_cardinality_cross_section_stream_matches_batch(tmp_path: Path) -> None:
    table = _table(_cross_section_rows(CROSS_SECTION_GROUPS))
    plan = _cross_section_program().compile_stream(Runtime())
    sink = _CollectSink()

    async def exercise() -> None:
        job = await StreamingRunner(
            plan,
            {
                "input": SourceBinding(
                    _SegmentedSource(table, 60, watermarks=True),
                    watermark_policy=SourceProvidedWatermarks(),
                )
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        assert (await job.wait_async()).state == "completed"

    asyncio.run(exercise())

    stream_output = pa.concat_tables(sink.tables)
    batch_output = _execute_batch(_cross_section_program(), table)
    assert stream_output.num_rows == batch_output.num_rows
    _assert_optional_floats(
        stream_output["rank"].to_pylist(), batch_output["rank"].to_pylist()
    )
    _assert_optional_floats(
        stream_output["zscore"].to_pylist(), batch_output["zscore"].to_pylist()
    )


def test_high_cardinality_rolling_checkpoint_recovery_matches_batch(
    tmp_path: Path,
) -> None:
    table = _table(_rolling_rows(STREAM_ENTITIES))
    expected = _execute_batch(_rolling_program(), table)
    sink = _CollectSink()
    opened_offsets: list[int] = []
    pause_at = table.num_rows // 2

    async def runner(source: _ReplayPauseSource) -> StreamingRunner:
        return StreamingRunner(
            _rolling_program().compile_stream(Runtime()),
            {"input": SourceBinding(source, watermark_policy=DisabledWatermarks())},
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
        )

    async def exercise() -> None:
        first_source = _ReplayPauseSource(table, pause_at, opened_offsets)
        first = await (await runner(first_source)).start_async()
        await asyncio.wait_for(first_source.paused.wait(), timeout=30)
        assert await first.trigger_checkpoint_async() == 1
        assert (await first.cancel_async()).state == "cancelled"

        second_source = _ReplayPauseSource(table, None, opened_offsets)
        second = await (await runner(second_source)).start_async()
        assert (await second.wait_async()).state == "completed"

    asyncio.run(exercise())

    recovered = pa.concat_tables(sink.tables)
    assert recovered.num_rows == expected.num_rows
    for column in ("n3", "avg3", "ema3"):
        _assert_optional_floats(
            recovered[column].to_pylist(), expected[column].to_pylist()
        )
    assert opened_offsets == [0, pause_at]
