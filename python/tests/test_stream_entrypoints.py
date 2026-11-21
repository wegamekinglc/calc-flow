"""Explicit progress policies work through both public stream entry points."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pytest

import calc_flow as cf


def _data() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("price", pa.float64(), nullable=False),
        ]
    )
    return pa.table(
        {"symbol": ["a", "a"], "ts": [1, 2], "price": [10.0, 12.0]}, schema=schema
    )


async def _read_previous(results) -> list[float | None]:
    values = []
    async for result in results:
        table = result.table if isinstance(result, cf.StreamOutput) else result
        values.extend(table["previous"].to_pylist())
        if len(values) == 2:
            return values
    raise AssertionError("stream ended before yielding its declared rows")


@pytest.mark.parametrize("named", [False, True], ids=["table", "program"])
def test_stream_entrypoints_forward_explicit_progress_before_eof(named):
    data = _data()
    source = cf.table_input(
        "quotes",
        schema=data.schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )
    output = source.select(previous=cf.ts.lag(source["price"]))
    owner = (
        cf.Program("lag", engine="streaming", outputs={"lag": output})
        if named
        else output
    )
    policy = cf.SourceProvidedWatermarks()

    async def run():
        paused, closed = asyncio.Event(), asyncio.Event()

        async def feed():
            try:
                yield data
                yield cf.Watermark(
                    datetime(1970, 1, 1, microsecond=3, tzinfo=timezone.utc)
                )
                paused.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

        inputs = {"quotes": feed()} if named else feed()
        watermarks = {"quotes": policy} if named else policy
        results = owner.stream(inputs, watermarks=watermarks)

        async def consume():
            async with results:
                await paused.wait()
                assert await _read_previous(results) == [None, 10.0]
                assert not closed.is_set()

        await asyncio.wait_for(consume(), 5)
        assert closed.is_set()
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


class _PausedSource:
    def __init__(self, data: pa.Table) -> None:
        self._data = data
        self._sent = False
        self.paused = asyncio.Event()

    def capabilities(self) -> cf.SourceCapabilities:
        return cf.SourceCapabilities(
            cf.ReplayPositioning.UNSUPPORTED,
            cf.SourceDeliveryCapability.LOSSY,
            max_batch_rows=self._data.num_rows,
            max_batch_bytes=1 << 20,
            schema=self._data.schema,
            native_watermarks=cf.NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: cf.Cursor | None) -> None:
        self._sent = False

    async def next(self) -> cf.Data | None:
        if self._sent:
            self.paused.set()
            await asyncio.Event().wait()
        self._sent = True
        return cf.Data(cf.Batch.from_pyarrow(self._data), cf.Cursor(b"1", {}))

    async def close(self) -> None:
        pass


class _RowSink:
    def __init__(self) -> None:
        self.previous: list[float | None] = []
        self.written = asyncio.Event()

    async def open(self) -> None:
        pass

    async def write(self, batch: cf.Batch) -> None:
        self.previous.extend(batch.to_pyarrow()["previous"].to_pylist())
        if self.previous:
            self.written.set()

    async def close(self) -> None:
        pass


def test_explicit_runner_accepts_per_batch_generated_watermarks():
    data = _data()
    source = cf.table_input(
        "quotes",
        schema=data.schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )
    program = cf.Program(
        "lag",
        engine="streaming",
        outputs={"lag": source.select(previous=cf.ts.lag(source["price"]))},
    )
    policy = cf.BoundedOutOfOrderness("ts", timedelta(microseconds=1), timedelta(0))

    async def run():
        plan = program.compile_stream(cf.Runtime())
        feed, sink = _PausedSource(data), _RowSink()
        job = await cf.StreamingRunner(
            plan,
            {
                plan.source_binding_ids[0]: cf.SourceBinding(
                    feed, watermark_policy=policy
                )
            },
            {plan.sink_binding_ids[0]: [cf.SinkBinding.ordinary("rows", sink)]},
            config=cf.StreamRuntimeConfig(checkpointing=False),
        ).start_async()
        try:
            await asyncio.wait_for(sink.written.wait(), 5)
            # The paused source never ends; only the per-batch watermark finalizes ts 1.
            assert sink.previous == [None]
            assert job.status()["watermark_micros"] == 1
        finally:
            await job.cancel_async()
        assert (await job.wait_async()).state == "cancelled"
        assert job.status()["task_count"] == 0

    asyncio.run(run())
