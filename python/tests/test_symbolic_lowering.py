from __future__ import annotations

import pyarrow as pa
import pytest

from calc_flow import Batch, Runtime
from calc_flow.errors import CompileError
from calc_flow.pipeline import BatchExecutionPlan, StreamExecutionPlan
from calc_flow.symbolic import FeatureSet, Field, Program, table_input


def _quotes() -> object:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("x", "float64", nullable=False),
            Field("y", "float64"),
        ],
    )


def _quotes_batch() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("x", pa.float64(), nullable=False),
            pa.field("y", pa.float64(), nullable=True),
        ]
    )
    return pa.table(
        {
            "ts": pa.array([1_700_000_000_000_000], type=pa.timestamp("us", tz="UTC")),
            "x": pa.array([1.5], type=pa.float64()),
            "y": pa.array([2.5], type=pa.float64()),
        },
        schema=schema,
    )


def test_compile_batch_lowers_passthrough_program() -> None:
    quotes = _quotes()
    program = Program("p", inputs=[quotes], outputs=[("signals", quotes)])

    plan = program.compile_batch(Runtime())

    assert isinstance(plan, BatchExecutionPlan)
    result = plan.execute({"input": Batch.from_pyarrow(_quotes_batch())})
    output = result.outputs["output"].to_pyarrow()
    assert output.schema.names == ["ts", "x", "y"]
    assert output.column("x").to_pylist() == [1.5]
    assert output.column("y").to_pylist() == [2.5]


def test_compile_batch_is_deterministic_across_builds() -> None:
    def build() -> BatchExecutionPlan:
        quotes = _quotes()
        program = Program("p", inputs=[quotes], outputs=[("signals", quotes)])
        return program.compile_batch(Runtime())

    assert build().fingerprint == build().fingerprint


def test_compile_batch_requires_a_runtime() -> None:
    quotes = _quotes()
    program = Program("p", inputs=[quotes], outputs=[("signals", quotes)])

    with pytest.raises(TypeError, match="explicit calc_flow Runtime"):
        program.compile_batch(object())


def test_compile_batch_raises_compile_error_from_analysis_issues() -> None:
    quotes = table_input(
        "quotes",
        schema=[Field("x", "float64", nullable=False)],
    )
    other = table_input(
        "trades",
        schema=[Field("y", "float64", nullable=False)],
    )
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[("signals", quotes.with_columns(FeatureSet([("score", other["y"])])))],
    )

    with pytest.raises(CompileError) as excinfo:
        program.compile_batch(Runtime())

    message = str(excinfo.value)
    assert "unresolved_type" in message
    assert "inputs.trades" in message


def test_compile_stream_lowers_passthrough_program() -> None:
    quotes = _quotes()
    program = Program("p", inputs=[quotes], outputs=[("signals", quotes)])

    plan = program.compile_stream(Runtime())

    assert isinstance(plan, StreamExecutionPlan)


def test_compile_stream_validates_lateness_arguments() -> None:
    quotes = _quotes()
    program = Program("p", inputs=[quotes], outputs=[("signals", quotes)])

    with pytest.raises(TypeError):
        program.compile_stream(Runtime(), allowed_lateness_micros=True)
    with pytest.raises(ValueError):
        program.compile_stream(Runtime(), allowed_lateness_micros=-1)
    with pytest.raises(ValueError):
        program.compile_stream(Runtime(), late_policy="retry")
