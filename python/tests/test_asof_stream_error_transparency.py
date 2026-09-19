"""ASOF dimension-lookup regressions from the DAL-287 repro.

Both access paths — the convenience ``.stream()`` feed and the explicit
``StreamingRunner`` bindings — must admit 10,000-row fact batches under the
limits ``docs/asof-join-guide.md`` recommends, and a workspace-limit failure
must surface the underlying operator message and reason code in its error
text instead of an opaque ``operator ... execution failed``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

import calc_flow as cf
from calc_flow.symbolic import Program, table_input

BASE = datetime(2026, 1, 1, tzinfo=UTC)
BASE_MICROS = int(BASE.timestamp() * 1_000_000)
RECOMMENDED = cf.AsofStateLimits(100_000, 64 * 1024 * 1024)
TOLERANCE = timedelta(seconds=10_000_000)
ENTITIES = 64
BATCH_ROWS = 10_000

SCHEMA = pa.schema(
    [
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("sequence", pa.uint64(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
    ]
)
DIM_SCHEMA = pa.schema(
    [
        pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("sequence", pa.uint64(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("factor", pa.float64(), nullable=False),
    ]
)


def fact_table(rows: int) -> pa.Table:
    return pa.table(
        {
            "event_time": pa.array(
                [BASE_MICROS + row * 1_000 for row in range(rows)],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "sequence": pa.array(list(range(rows)), type=pa.uint64()),
            "symbol": [f"S{row % ENTITIES:03d}" for row in range(rows)],
            "price": [100.0 + (row % 257) / 8 for row in range(rows)],
        },
        schema=SCHEMA,
    )


def dim_table() -> pa.Table:
    return pa.table(
        {
            "event_time": pa.array(
                [BASE_MICROS] * ENTITIES, type=pa.timestamp("us", tz="UTC")
            ),
            "sequence": pa.array([0] * ENTITIES, type=pa.uint64()),
            "symbol": [f"S{index:03d}" for index in range(ENTITIES)],
            "factor": [float(index + 1) for index in range(ENTITIES)],
        },
        schema=DIM_SCHEMA,
    )


def _source(name: str, schema: pa.Schema) -> cf.TableExpr:
    return table_input(
        name,
        schema=schema,
        entity_by=["symbol"],
        event_time="event_time",
        sequence_by=("sequence",),
    )


def _joined(limits: cf.AsofStateLimits) -> cf.TableExpr:
    return _source("facts", SCHEMA).stream_asof_join(
        _source("dims", DIM_SCHEMA),
        tolerance=TOLERANCE,
        limits=limits,
        keys=(["symbol"], ["symbol"]),
        prefixes=("fact", "dim"),
    )


def _chunks(total: int) -> list[pa.Table]:
    return [chunk for chunk in fact_table(total).to_batches(max_chunksize=BATCH_ROWS)]


class _Feed:
    def __init__(self, tables: list[pa.Table]) -> None:
        self.tables = iter(tables)

    def __aiter__(self):
        return self

    async def __anext__(self) -> pa.Table:
        try:
            return next(self.tables)
        except StopIteration:
            raise StopAsyncIteration from None


class _InteractiveSource:
    def __init__(self, *, max_batch_rows: int) -> None:
        self._events: asyncio.Queue[cf.Data | cf.Watermark | None] = asyncio.Queue()
        self._max_batch_rows = max_batch_rows

    def capabilities(self) -> cf.SourceCapabilities:
        return cf.SourceCapabilities(
            cf.ReplayPositioning.UNSUPPORTED,
            cf.SourceDeliveryCapability.LOSSY,
            max_batch_rows=self._max_batch_rows,
            max_batch_bytes=32 * 1024 * 1024,
            native_watermarks=cf.NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: cf.Cursor | None) -> None:
        if cursor is not None:
            raise RuntimeError("regression source does not support replay")

    async def next(self) -> cf.Data | cf.Watermark | None:
        return await self._events.get()

    async def close(self) -> None:
        return None

    async def push(self, event: cf.Data | cf.Watermark | None) -> None:
        await self._events.put(event)


class _CountingSink:
    def __init__(self) -> None:
        self.rows = 0

    async def open(self) -> None:
        return None

    async def write(self, batch) -> None:
        self.rows += batch.to_pyarrow().num_rows

    async def close(self) -> None:
        return None


async def _drain_convenience(limits: cf.AsofStateLimits, chunks: list[pa.Table]) -> int:
    result = _joined(limits).stream(
        {"facts": _Feed(chunks), "dims": _Feed([dim_table()])}
    )
    rows = 0
    async with asyncio.timeout(120), result:
        async for table in result:
            rows += table.num_rows
    return rows


async def _drive_explicit(
    limits: cf.AsofStateLimits, chunks: list[pa.Table], checkpoints: Path
) -> tuple[str, str | None, int]:
    plan = Program(
        "dal288-asof",
        inputs=(_source("facts", SCHEMA), _source("dims", DIM_SCHEMA)),
        outputs=(("result", _joined(limits)),),
    ).compile_stream(cf.Runtime())
    fact_source = _InteractiveSource(max_batch_rows=BATCH_ROWS)
    dim_source = _InteractiveSource(max_batch_rows=BATCH_ROWS)
    sink = _CountingSink()
    job = await cf.StreamingRunner(
        plan,
        {
            "facts.input": cf.SourceBinding(
                fact_source, watermark_policy=cf.SourceProvidedWatermarks()
            ),
            "dims.input": cf.SourceBinding(
                dim_source, watermark_policy=cf.SourceProvidedWatermarks()
            ),
        },
        {"output": [cf.SinkBinding.ordinary("out", sink)]},
        cf.ManagedCheckpointRuntime(checkpoints),
        config=cf.StreamRuntimeConfig(checkpoint_interval=timedelta(hours=24)),
    ).start_async()
    try:
        await dim_source.push(
            cf.Data(
                cf.Batch.from_pyarrow(dim_table()),
                cf.Cursor(ENTITIES.to_bytes(8, "big"), {"rows": ENTITIES}),
            )
        )
        await dim_source.push(cf.Watermark(BASE + timedelta(microseconds=1)))
        rows = 0
        for chunk in chunks:
            rows += chunk.num_rows
            await fact_source.push(
                cf.Data(
                    cf.Batch.from_pyarrow(chunk),
                    cf.Cursor(rows.to_bytes(8, "big"), {"rows": rows}),
                )
            )
            last = datetime.fromtimestamp(
                chunk["event_time"][-1].value / 1_000_000, tz=UTC
            )
            await fact_source.push(cf.Watermark(last + timedelta(microseconds=1)))
        await fact_source.push(None)
        await dim_source.push(None)
        outcome = await asyncio.wait_for(job.wait_async(), timeout=120)
        failure = outcome.errors[0].message if outcome.errors else None
        return outcome.state, failure, sink.rows
    finally:
        await job.cancel_async()


def test_convenience_path_admits_ten_thousand_row_batches_under_recommended_limits():
    chunks = _chunks(2 * BATCH_ROWS)
    assert asyncio.run(_drain_convenience(RECOMMENDED, chunks)) == 2 * BATCH_ROWS


def test_convenience_path_workspace_failure_surfaces_reason_detail():
    chunks = _chunks(BATCH_ROWS)
    with pytest.raises(cf.StreamingRuntimeError) as raised:
        asyncio.run(_drain_convenience(cf.AsofStateLimits(100_000, 64 * 1024), chunks))
    message = str(raised.value)
    assert "ASOF aggregate workspace exceeds max_state_bytes" in message
    assert "asof_workspace_limit_exceeded" in message
    assert raised.value.reason_code == "asof_workspace_limit_exceeded"


def test_explicit_path_admits_ten_thousand_row_batches_under_recommended_limits(
    tmp_path,
):
    state, failure, rows = asyncio.run(
        _drive_explicit(RECOMMENDED, _chunks(2 * BATCH_ROWS), tmp_path / "checkpoints")
    )
    assert state == "completed"
    assert failure is None
    assert rows == 2 * BATCH_ROWS


def test_explicit_path_workspace_failure_surfaces_reason_detail(tmp_path):
    state, failure, _ = asyncio.run(
        _drive_explicit(
            cf.AsofStateLimits(100_000, 64 * 1024),
            _chunks(BATCH_ROWS),
            tmp_path / "checkpoints",
        )
    )
    assert state == "failed"
    assert failure is not None
    assert "ASOF aggregate workspace exceeds max_state_bytes" in failure
    assert "asof_workspace_limit_exceeded" in failure
