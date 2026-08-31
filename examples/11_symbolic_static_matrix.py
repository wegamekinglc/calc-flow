"""Run one symbolic NumPy static-weight matrix in batch and stream modes."""

from __future__ import annotations

import asyncio
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa

from calc_flow import (
    Batch,
    ConfigError,
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
    register_numpy,
)
from calc_flow.symbolic import Field, Program, linalg, parameter, table, table_input


class SegmentedSource:
    def __init__(self, value: pa.Table, rows_per_batch: int) -> None:
        self._value = value
        self._rows_per_batch = rows_per_batch
        self._offset = 0

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=self._rows_per_batch,
            max_batch_bytes=1024 * 1024,
            native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._offset = 0

    async def next(self) -> Data | None:
        if self._offset == self._value.num_rows:
            return None
        end = min(self._offset + self._rows_per_batch, self._value.num_rows)
        value = self._value.slice(self._offset, end - self._offset)
        self._offset = end
        return Data(Batch.from_pyarrow(value), Cursor(end.to_bytes(8, "big"), {}))

    async def close(self) -> None:
        pass


class CollectSink:
    def __init__(self) -> None:
        self.batches: list[Batch] = []

    async def open(self) -> None:
        pass

    async def write(self, batch: Batch) -> None:
        self.batches.append(batch)

    async def close(self) -> None:
        pass


def symbolic_matrix_program() -> Program:
    source = table_input(
        "prices",
        schema=(Field("return", "float64"), Field("volatility", "float64")),
    )
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(2, 1),
    )
    matrix = linalg.from_columns(
        source,
        columns=("return", "volatility"),
        backend="numpy",
    )
    output = table.attach_columns(
        source,
        linalg.matmul(matrix, weights),
        names=("risk_score",),
    )
    return Program(
        "symbolic-static-matrix",
        inputs=(source, weights),
        outputs=(("signals", output),),
    )


def require(condition: bool, message: str) -> None:
    """Keep example verification active under regular and optimized Python."""
    if not condition:
        raise RuntimeError(message)


async def stream_scores(
    program: Program,
    runtime: Runtime,
    input_table: pa.Table,
    weights: Batch,
) -> tuple[list[float], list[int]]:
    sink = CollectSink()
    with TemporaryDirectory(prefix="calc-flow-symbolic-matrix-") as directory:
        runner = StreamingRunner(
            program.compile_stream(runtime),
            {
                "input": SourceBinding(
                    SegmentedSource(input_table, 2),
                    watermark_policy=DisabledWatermarks(),
                )
            },
            {"output": [SinkBinding.ordinary("collector", sink)]},
            ManagedCheckpointRuntime(directory),
            static_inputs={"weights": weights},
        )
        outcome = await (await runner.start_async()).wait_async()
        require(outcome.state == "completed", f"unexpected outcome: {outcome.state}")
    output = pa.concat_tables([batch.to_pyarrow() for batch in sink.batches])
    placements = [batch.metadata["copy_bytes"]["weights"] for batch in sink.batches]
    return output["risk_score"].to_pylist(), placements


async def main() -> None:
    program = symbolic_matrix_program()
    unsupported = Runtime()
    try:
        program.compile_stream(unsupported)
    except ConfigError as error:
        require(
            "missing_provider" in str(error),
            f"unexpected provider error: {error}",
        )
        print("expected final provider gate:", error)
    else:
        raise RuntimeError("matrix compilation unexpectedly succeeded without NumPy")

    runtime = Runtime()
    register_numpy(runtime)
    print(program.explain(runtime, mode="stream"))
    input_table = pa.table(
        {
            "return": pa.array([0.02, -0.01, 0.04], type=pa.float64()),
            "volatility": pa.array([0.10, 0.20, 0.15], type=pa.float64()),
        }
    )
    weights = Batch.from_array(
        np.array([[100.0], [-10.0]], dtype=np.float64),
        backend="numpy",
    )
    batch_result = await program.compile_batch(runtime).execute_async(
        {"input": Batch.from_pyarrow(input_table), "weights": weights}
    )
    batch_scores = batch_result.outputs["output"].to_pyarrow()["risk_score"].to_pylist()
    streamed_scores, placements = await stream_scores(
        program,
        runtime,
        input_table,
        weights,
    )

    require(streamed_scores == batch_scores, "stream result differs from batch result")
    require(placements == [16, 0], f"weights were not placed once: {placements}")
    print("batch/stream risk scores:", batch_scores)
    print("static-weight placement bytes by micro-batch:", placements)


if __name__ == "__main__":
    asyncio.run(main())
