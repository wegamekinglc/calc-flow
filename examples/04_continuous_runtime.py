"""Run the source-driven continuous API with managed checkpoints."""

from __future__ import annotations

import asyncio
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
    def __init__(self, values: list[int]) -> None:
        self._values = tuple(values)
        self._offset = 0

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


async def main() -> None:
    with TemporaryDirectory(prefix="calc-flow-continuous-") as directory:
        plan = (
            PipelineBuilder("continuous-example")
            .expression("calculate", "result = value + 1")
            .compile_stream()
        )
        source = ReplaySource([1, 2, 3])
        sink = CollectSink()
        runner = StreamingRunner(
            plan,
            {
                "input": SourceBinding(
                    source,
                    watermark_policy=DisabledWatermarks(),
                )
            },
            {"output": [SinkBinding.ordinary("console", sink)]},
            ManagedCheckpointRuntime(directory),
        )
        job = await runner.start_async()
        print("started job:", job.id, job.status()["state"])
        outcome = await job.wait_async()

        assert outcome.state == "completed"
        assert outcome.completed_epoch is not None
        assert sink.values == [2, 3, 4]
        print("terminal state:", outcome.state)
        print("completed epoch:", outcome.completed_epoch)
        print("results:", sink.values)


if __name__ == "__main__":
    asyncio.run(main())
