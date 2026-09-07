"""Aggregate UTC minute bars with one shared native event-window state owner."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from tempfile import TemporaryDirectory

import pyarrow as pa

from calc_flow import (
    Batch,
    Cursor,
    Data,
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
from calc_flow.symbolic import FeatureSet, Field, Program, table_input, window

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def minute_program() -> Program:
    trades = table_input(
        "trades",
        schema=[
            Field("ts", "timestamp[us, UTC]"),
            Field("symbol", "string"),
            Field("trade_id", "uint64", nullable=False),
            Field("quantity", "int64"),
            Field("price", "float64"),
        ],
    )
    minute = window.tumbling(
        trades,
        event_time="ts",
        size_micros=60_000_000,
        group_by=["symbol"],
        aggregates=[
            window.count("trade_id", output="trade_count"),
            window.sum("quantity", output="volume"),
            window.min("price", output="low"),
            window.max("price", output="high"),
            window.avg("price", output="avg_price"),
        ],
    )
    summary = minute.with_columns(
        FeatureSet([("price_range", minute["high"] - minute["low"])])
    )
    return Program(
        "minute-bars",
        inputs=[trades],
        outputs=[("minute", minute), ("summary", summary)],
    )


def trade_data() -> pa.Table:
    return pa.table(
        {
            "ts": [
                BASE + timedelta(seconds=seconds) if seconds is not None else None
                for seconds in (5, 35, 59, 15, 60, 95, None)
            ],
            "symbol": ["A", "A", "A", "B", "A", "A", "A"],
            "trade_id": [1, 2, 3, 4, 5, 6, 7],
            "quantity": [10, None, 30, None, 7, 13, 999],
            "price": [100.0, 102.0, None, None, 110.0, 90.0, 999.0],
        },
        schema=pa.schema(
            [
                pa.field("ts", pa.timestamp("us", tz="UTC")),
                pa.field("symbol", pa.string()),
                pa.field("trade_id", pa.uint64(), nullable=False),
                pa.field("quantity", pa.int64()),
                pa.field("price", pa.float64()),
            ]
        ),
    )


class TradeSource:
    """Emit bounded example data followed by two explicit minute watermarks."""

    def __init__(self) -> None:
        self._events = (
            Data(Batch.from_pyarrow(trade_data()), Cursor(b"\x01", {"index": 1})),
            Watermark(BASE + timedelta(minutes=1)),
            Watermark(BASE + timedelta(minutes=2)),
        )
        self._index = 0

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=7,
            max_batch_bytes=65536,
            native_watermarks=NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._index = 0 if cursor is None else int(cursor.payload["index"])

    async def next(self) -> Data | Watermark | None:
        if self._index == len(self._events):
            return None
        event = self._events[self._index]
        self._index += 1
        return event

    async def close(self) -> None:
        pass


class CollectSink:
    """Collect the final window results in their emitted order."""

    def __init__(self) -> None:
        self.tables: list[pa.Table] = []

    async def open(self) -> None:
        pass

    async def write(self, batch: Batch) -> None:
        self.tables.append(batch.to_pyarrow())

    async def close(self) -> None:
        pass


def expected_minutes() -> pa.Table:
    first_end = BASE + timedelta(minutes=1)
    return pa.table(
        {
            "window_start": [BASE, BASE, first_end],
            "window_end": [first_end, first_end, BASE + timedelta(minutes=2)],
            "symbol": ["A", "B", "A"],
            "trade_count": [3, 1, 2],
            "volume": [40, None, 20],
            "low": [100.0, None, 90.0],
            "high": [102.0, None, 110.0],
            "avg_price": [101.0, None, 100.0],
        },
        schema=pa.schema(
            [
                pa.field("window_start", pa.timestamp("us", tz="UTC"), False),
                pa.field("window_end", pa.timestamp("us", tz="UTC"), False),
                pa.field("symbol", pa.string()),
                pa.field("trade_count", pa.uint64(), False),
                pa.field("volume", pa.int64()),
                pa.field("low", pa.float64()),
                pa.field("high", pa.float64()),
                pa.field("avg_price", pa.float64()),
            ]
        ),
    )


async def main() -> None:
    runtime = Runtime()
    program = minute_program()
    if "window state_stages 1 shared_outputs 2" not in program.explain(
        runtime, mode="stream"
    ):
        raise RuntimeError("expected one shared window state for both outputs")
    plan = program.compile_stream(runtime)
    minute_sink, summary_sink = CollectSink(), CollectSink()
    with TemporaryDirectory(prefix="calc-flow-symbolic-window-") as directory:
        runner = StreamingRunner(
            plan,
            {
                plan.source_binding_ids[0]: SourceBinding(
                    TradeSource(), watermark_policy=SourceProvidedWatermarks()
                )
            },
            {
                "minute.output": [SinkBinding.ordinary("minute-bars", minute_sink)],
                "summary.output": [SinkBinding.ordinary("price-ranges", summary_sink)],
            },
            ManagedCheckpointRuntime(directory),
        )
        job = await runner.start_async()
        try:
            outcome = await asyncio.wait_for(job.wait_async(), timeout=30)
            if outcome.state != "completed":
                raise RuntimeError(outcome)
        finally:
            if job.status()["state"] == "running":
                await job.cancel_async()

    minute = pa.concat_tables(minute_sink.tables)
    summary = pa.concat_tables(summary_sink.tables)
    expected = expected_minutes()
    if minute.schema != expected.schema or not minute.equals(expected):
        raise RuntimeError(f"unexpected minute result: {minute}")
    expected_summary = expected.append_column(
        "price_range", pa.array([2.0, None, 20.0], type=pa.float64())
    )
    if summary.schema != expected_summary.schema or not summary.equals(
        expected_summary
    ):
        raise RuntimeError(f"unexpected summary result: {summary}")
    print("minute rows:", minute.num_rows)
    print("volumes:", minute["volume"].to_pylist())
    print("average prices:", minute["avg_price"].to_pylist())
    print("price ranges:", summary["price_range"].to_pylist())


if __name__ == "__main__":
    asyncio.run(main())
