"""Checkpoint, recover, and terminally restart a symbolic rolling stream."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa

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
    StreamingRunner,
)
from calc_flow.symbolic import FeatureSet, Field, Program, row, rows, table_input, ts


class ReplaySource:
    def __init__(
        self,
        prices: tuple[float, ...],
        pause_at: int | None,
        opened_at: list[int],
    ) -> None:
        self._prices = prices
        self._pause_at = pause_at
        self._offset = 0
        self._opened_at = opened_at
        self.paused = asyncio.Event()

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=1,
            max_batch_bytes=4096,
            native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._offset = 0 if cursor is None else int(cursor.payload["offset"])
        self._opened_at.append(self._offset)

    async def next(self) -> Data | Idle | None:
        if self._pause_at == self._offset:
            self.paused.set()
            await asyncio.sleep(0)
            return Idle()
        if self._offset == len(self._prices):
            return None
        index = self._offset
        self._offset += 1
        schema = pa.schema(
            (
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("price", pa.float64(), nullable=False),
            )
        )
        table = pa.table(
            {
                "ts": [self._offset * 1_000_000],
                "symbol": ["AAA"],
                "seq": [self._offset],
                "price": [self._prices[index]],
            },
            schema=schema,
        )
        return Data(
            Batch.from_pyarrow(table),
            Cursor(
                self._offset.to_bytes(8, "big"),
                {"offset": self._offset},
            ),
        )

    async def close(self) -> None:
        pass


class CollectSink:
    def __init__(self) -> None:
        self.average_gains: list[float | None] = []

    async def open(self) -> None:
        pass

    async def write(self, batch: Batch) -> None:
        self.average_gains.extend(batch.to_pyarrow()["average_gain"].to_pylist())

    async def close(self) -> None:
        pass


def symbolic_program() -> Program:
    quotes = table_input(
        "quotes",
        schema=(
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("price", "float64", nullable=False),
        ),
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("seq",),
    )
    change = ts.delta(quotes["price"])
    gain = row.clip(change, lower=0.0, upper=1.0e100)
    output = quotes.with_columns(
        FeatureSet(
            (
                ("change", change),
                ("average_gain", ts.mean(gain, window=rows(3))),
            )
        )
    )
    return Program(
        "symbolic-streaming-recovery",
        inputs=(quotes,),
        outputs=(("signals", output),),
    )


def require(condition: bool, message: str) -> None:
    """Keep example verification active under regular and optimized Python."""
    if not condition:
        raise RuntimeError(message)


def _runner(
    checkpoint_root: Path,
    source: ReplaySource,
    sink: CollectSink,
) -> StreamingRunner:
    return StreamingRunner(
        symbolic_program().compile_stream(Runtime()),
        {
            "input": SourceBinding(
                source,
                watermark_policy=DisabledWatermarks(),
            )
        },
        {"output": [SinkBinding.ordinary("collector", sink)]},
        ManagedCheckpointRuntime(checkpoint_root),
    )


def _matches(actual: list[float | None], expected: list[float | None]) -> bool:
    return len(actual) == len(expected) and all(
        left is right
        or (
            left is not None
            and right is not None
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
        )
        for left, right in zip(actual, expected, strict=True)
    )


async def recover_midstream(checkpoint_root: Path) -> None:
    prices = (100.0, 110.0, 105.0, 115.0, 100.0, 105.0)
    expected = [None, 10.0, 5.0, 20.0 / 3.0, 10.0 / 3.0, 5.0]
    opened_at: list[int] = []
    sink = CollectSink()

    first_source = ReplaySource(prices, 3, opened_at)
    first = await _runner(checkpoint_root, first_source, sink).start_async()
    await first_source.paused.wait()
    checkpoint_epoch = await first.trigger_checkpoint_async()
    require(checkpoint_epoch == 1, f"unexpected checkpoint: {checkpoint_epoch}")
    require((await first.cancel_async()).state == "cancelled", "cancel failed")

    second_source = ReplaySource(prices, None, opened_at)
    second = await _runner(checkpoint_root, second_source, sink).start_async()
    completed = await second.wait_async()
    require(completed.state == "completed", f"unexpected outcome: {completed.state}")
    require(_matches(sink.average_gains, expected), "recovered values differ")
    require(opened_at == [0, 3], f"unexpected recovery cursors: {opened_at}")

    before_restart = list(sink.average_gains)
    terminal_source = ReplaySource(prices, None, opened_at)
    terminal = await _runner(checkpoint_root, terminal_source, sink).start_async()
    terminal_outcome = await terminal.wait_async()
    require(terminal_outcome.state == "completed", "terminal restart failed")
    require(sink.average_gains == before_restart, "terminal restart duplicated output")
    require(opened_at == [0, 3], "terminal restart reopened the source")

    print("recovered rolling values:", sink.average_gains)
    print("source cursors:", opened_at, "terminal epoch:", completed.completed_epoch)


async def main() -> None:
    program = symbolic_program()
    print(program.explain(Runtime(), mode="stream"))
    with TemporaryDirectory(prefix="calc-flow-symbolic-recovery-") as directory:
        await recover_midstream(Path(directory))


if __name__ == "__main__":
    asyncio.run(main())
