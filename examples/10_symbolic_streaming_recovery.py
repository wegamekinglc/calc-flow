"""Compile a symbolic stream and recover its terminal checkpoint."""

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
    ReplayPositioning,
    Runtime,
    SinkBinding,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    StreamingRunner,
)
from calc_flow.symbolic import FeatureSet, Field, Program, table_input


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
        schema = pa.schema((pa.field("value", pa.int64(), nullable=False),))
        batch = Batch.from_pyarrow(pa.table({"value": [value]}, schema=schema))
        return Data(
            batch,
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


def symbolic_program() -> Program:
    rows = table_input(
        "rows",
        schema=(Field("value", "int64", nullable=False),),
    )
    output = rows.with_columns(FeatureSet((("result", rows["value"] * 2),)))
    return Program(
        "symbolic-streaming-recovery",
        inputs=(rows,),
        outputs=(("signals", output),),
    )


def require(condition: bool, message: str) -> None:
    """Keep example verification active under regular and optimized Python."""
    if not condition:
        raise RuntimeError(message)


async def run_once(checkpoint_root: Path) -> tuple[list[int], list[int], int]:
    source = ReplaySource((10, 20, 30))
    sink = CollectSink()
    plan = symbolic_program().compile_stream(Runtime())
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
    require(outcome.state == "completed", f"unexpected outcome: {outcome.state}")
    require(outcome.completed_epoch is not None, "terminal epoch is missing")
    return sink.values, source.opened_at, outcome.completed_epoch


async def main() -> None:
    program = symbolic_program()
    print(program.explain(Runtime(), mode="stream"))
    with TemporaryDirectory(prefix="calc-flow-symbolic-recovery-") as directory:
        root = Path(directory)
        first_values, first_open, first_epoch = await run_once(root)
        second_values, second_open, second_epoch = await run_once(root)

        require(first_values == [20, 40, 60], f"unexpected output: {first_values}")
        require(first_open == [0], f"unexpected initial cursor: {first_open}")
        require(second_values == [], f"restart duplicated output: {second_values}")
        require(second_open == [], f"terminal source reopened: {second_open}")
        require(second_epoch == first_epoch, "terminal epoch changed during recovery")

        print("first run:", first_values, "epoch:", first_epoch)
        print("restart output:", second_values, "source reopened:", bool(second_open))


if __name__ == "__main__":
    asyncio.run(main())
