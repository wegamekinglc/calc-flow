from __future__ import annotations

import asyncio
from pathlib import Path

import pyarrow as pa
import pytest

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
from calc_flow.symbolic import FeatureSet, Field, Program, row, table, table_input

_ROWS = 700
_SEGMENTATIONS = (1, 7, 1000)


def _program() -> Program:
    quotes = table_input(
        "quotes",
        schema=[
            Field("x", "float64", nullable=False),
            Field("y", "float64", nullable=False),
        ],
    )
    shared = quotes["x"] * quotes["y"]
    signals = table.filter(
        quotes.with_columns(
            FeatureSet(
                [
                    ("product", shared),
                    ("shifted", shared + 1.0),
                    ("picked", row.where(quotes["x"] > 2.0, quotes["y"], 0.0)),
                    ("clipped", row.clip(quotes["x"], lower=1.0, upper=4.0)),
                    ("root", row.sqrt(row.abs(quotes["x"]))),
                    ("cast_x", row.cast(quotes["x"], "float32")),
                ]
            )
        ),
        quotes["x"] >= 0.5,
    )
    return Program("features", inputs=[quotes], outputs=[("signals", signals)])


def _input_table() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("x", pa.float64(), nullable=False),
            pa.field("y", pa.float64(), nullable=False),
        ]
    )
    return pa.table(
        {
            "x": pa.array(
                [(index % 25) * 0.37 - 1.0 for index in range(_ROWS)],
                type=pa.float64(),
            ),
            "y": pa.array(
                [(index % 13) * 0.53 + 0.25 for index in range(_ROWS)],
                type=pa.float64(),
            ),
        },
        schema=schema,
    )


def _expected() -> dict[str, list[float]]:
    table = _input_table().to_pydict()
    expected: dict[str, list[float]] = {
        "x": [],
        "y": [],
        "product": [],
        "shifted": [],
        "picked": [],
        "clipped": [],
        "root": [],
        "cast_x": [],
    }
    for x, y in zip(table["x"], table["y"], strict=True):
        if x < 0.5:
            continue
        expected["x"].append(x)
        expected["y"].append(y)
        expected["product"].append(x * y)
        expected["shifted"].append(x * y + 1.0)
        expected["picked"].append(y if x > 2.0 else 0.0)
        expected["clipped"].append(min(max(x, 1.0), 4.0))
        expected["root"].append(abs(x) ** 0.5)
        expected["cast_x"].append(x)
    return expected


class _SegmentedSource:
    def __init__(self, table: pa.Table, segment: int) -> None:
        self._table = table
        self._segment = segment
        self._offset = 0

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=10_000,
            max_batch_bytes=16 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._offset = 0

    async def next(self) -> Data | None:
        if self._offset >= self._table.num_rows:
            return None
        end = min(self._offset + self._segment, self._table.num_rows)
        chunk = self._table.slice(self._offset, end - self._offset)
        self._offset = end
        order = end.to_bytes(8, "big")
        return Data(Batch.from_pyarrow(chunk), Cursor(order, {"offset": end}))

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


def test_batch_executes_fused_features() -> None:
    plan = _program().compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_input_table())})
    output = result.outputs["output"].to_pyarrow().to_pydict()
    expected = _expected()
    for name, values in expected.items():
        assert output[name] == pytest.approx(values), name


@pytest.mark.parametrize("segmentation", _SEGMENTATIONS)
def test_stream_matches_batch_across_segmentation(
    tmp_path: Path, segmentation: int
) -> None:
    plan = _program().compile_stream(Runtime())
    sink = _CollectSink()

    async def exercise() -> None:
        job = await StreamingRunner(
            plan,
            {
                "input": SourceBinding(
                    _SegmentedSource(_input_table(), segmentation),
                    watermark_policy=DisabledWatermarks(),
                )
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
        ).start_async()
        outcome = await job.wait_async()
        assert outcome.state == "completed"

    asyncio.run(exercise())

    stream_output = pa.concat_tables(sink.tables).to_pydict()
    batch_result = (
        _program()
        .compile_batch(Runtime())
        .execute({"input": Batch.from_pyarrow(_input_table())})
    )
    batch_output = batch_result.outputs["output"].to_pyarrow().to_pydict()
    assert stream_output == batch_output


def test_execution_never_calls_symbolic_python(tmp_path: Path) -> None:
    import calc_flow.symbolic.analyzer as analyzer_module
    import calc_flow.symbolic.expr as expr_module
    import calc_flow.symbolic.lower as lower_module
    import calc_flow.symbolic.nodes as nodes_module
    import calc_flow.symbolic.ops as ops_module
    import calc_flow.symbolic.optimizer as optimizer_module
    import calc_flow.symbolic.program as program_module

    plan = _program().compile_batch(Runtime())
    stream_plan = _program().compile_stream(Runtime())
    sink = _CollectSink()
    blocked: list[str] = []

    def block(name: str):
        def _raise(*args: object, **kwargs: object) -> None:
            blocked.append(name)
            raise AssertionError(f"symbolic Python called at runtime: {name}")

        return _raise

    modules = (
        analyzer_module,
        expr_module,
        lower_module,
        nodes_module,
        ops_module,
        optimizer_module,
        program_module,
    )
    originals: list[tuple[object, str, object]] = []
    for module in modules:
        for attribute in dir(module):
            value = getattr(module, attribute)
            if callable(value) and not attribute.startswith("__"):
                originals.append((module, attribute, value))
                setattr(module, attribute, block(f"{module.__name__}.{attribute}"))
    try:
        plan.execute({"input": Batch.from_pyarrow(_input_table())})

        async def exercise() -> None:
            job = await StreamingRunner(
                stream_plan,
                {
                    "input": SourceBinding(
                        _SegmentedSource(_input_table(), 7),
                        watermark_policy=DisabledWatermarks(),
                    )
                },
                {"output": [SinkBinding.ordinary("archive", sink)]},
                ManagedCheckpointRuntime(tmp_path),
            ).start_async()
            outcome = await job.wait_async()
            assert outcome.state == "completed"

        asyncio.run(exercise())
    finally:
        for module, attribute, value in originals:
            setattr(module, attribute, value)
    assert blocked == []
