"""Restart a finite stream from its managed terminal checkpoint."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa

from calc_flow import (
    Batch,
    Cursor,
    Data,
    DisabledWatermarks,
    ManagedCheckpointRuntime,
    NativeWatermarkCapability,
    PipelineBuilder,
    ReplayPositioning,
    SinkBinding,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    StreamingRunner,
)


class ReplaySource:
    def __init__(self, values: tuple[int, ...]) -> None:
        self._values = values
        self._offset = 0
        self.opened_at: list[int] = []

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=1,
            max_batch_bytes=1024,
            native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._offset = 0 if cursor is None else int(cursor.payload["offset"])
        self.opened_at.append(self._offset)

    async def next(self) -> Data | None:
        if self._offset == len(self._values):
            return None
        value = self._values[self._offset]
        self._offset += 1
        return Data(
            Batch.from_pyarrow(pa.table({"value": [value]})),
            Cursor(self._offset.to_bytes(8, "big"), {"offset": self._offset}),
        )

    async def close(self) -> None:
        pass


class CollectSink:
    def __init__(self) -> None:
        self.values: list[int] = []

    async def open(self) -> None:
        pass

    async def write(self, batch: Batch) -> None:
        self.values.extend(batch.to_pyarrow()["result"].to_pylist())

    async def close(self) -> None:
        pass


async def run_once(checkpoint_root: Path) -> tuple[list[int], list[int], int]:
    plan = (
        PipelineBuilder("streaming-recovery-example")
        .expression("calculate", "result = value * 2")
        .compile_stream()
    )
    source = ReplaySource((10, 20, 30))
    sink = CollectSink()
    runner = StreamingRunner(
        plan,
        {
            "input": SourceBinding(
                source,
                watermark_policy=DisabledWatermarks(),
            )
        },
        {"output": [SinkBinding.ordinary("collector", sink)]},
        ManagedCheckpointRuntime(checkpoint_root),
    )
    outcome = await (await runner.start_async()).wait_async()
    assert outcome.state == "completed"
    assert outcome.completed_epoch is not None
    return sink.values, source.opened_at, outcome.completed_epoch


async def main() -> None:
    with TemporaryDirectory(prefix="calc-flow-recovery-") as directory:
        checkpoint_root = Path(directory)
        first_values, first_open, first_epoch = await run_once(checkpoint_root)
        second_values, second_open, second_epoch = await run_once(checkpoint_root)

        assert first_values == [20, 40, 60]
        assert first_open == [0]
        assert second_values == []
        # A terminal manifest records the source as ended. Recovery therefore
        # does not reopen it merely to seek to its final cursor.
        assert second_open == []
        assert second_epoch == first_epoch

        print("first run:", first_values, "opened at", first_open[0])
        print("restart:", second_values, "source reopened:", bool(second_open))
        print("recovered terminal epoch:", second_epoch)


if __name__ == "__main__":
    asyncio.run(main())
