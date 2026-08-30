from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

import calc_flow.array as array_module
from calc_flow import (
    Batch,
    CompileError,
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
    register_jax,
    register_numpy,
)
from calc_flow.symbolic import (
    ArrayExpr,
    FeatureSet,
    Field,
    Program,
    linalg,
    parameter,
    table,
    table_input,
)


class _SegmentedSource:
    def __init__(self, value: pa.Table, segment: int) -> None:
        self._value = value
        self._segment = segment
        self._offset = 0

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=100,
            max_batch_bytes=1024 * 1024,
            native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: Cursor | None) -> None:
        self._offset = 0

    async def next(self) -> Data | None:
        if self._offset >= self._value.num_rows:
            return None
        end = min(self._offset + self._segment, self._value.num_rows)
        value = self._value.slice(self._offset, end - self._offset)
        self._offset = end
        return Data(Batch.from_pyarrow(value), Cursor(end.to_bytes(8, "big"), {}))

    async def close(self) -> None:
        return None


class _CollectSink:
    def __init__(self) -> None:
        self.batches: list[Batch] = []

    async def open(self) -> None:
        return None

    async def write(self, batch: Batch) -> None:
        self.batches.append(batch)

    async def close(self) -> None:
        return None


def test_symbolic_matrix_batch_attaches_selected_columns_in_order() -> None:
    source = table_input(
        "input",
        schema=[
            Field("x", "float64", nullable=True),
            Field("y", "float64", nullable=True),
        ],
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
        columns=("y", "x"),
        backend="numpy",
    )
    output = table.attach_columns(
        source,
        linalg.matmul(matrix * 2.0 + 1.0, weights),
        names=("score",),
    )
    program = Program(
        "symbolic-matrix",
        inputs=(source, weights),
        outputs=(("signals", output),),
    )
    runtime = Runtime()
    register_numpy(runtime)

    plan = program.compile_batch(runtime)
    result = plan.execute(
        {
            "input": Batch.from_pyarrow(
                pa.table(
                    {
                        "x": pa.array([1.0, 2.0], type=pa.float64()),
                        "y": pa.array([10.0, 20.0], type=pa.float64()),
                    }
                )
            ),
            "weights": Batch.from_array(
                np.array([[1.0], [10.0]], dtype=np.float64),
                backend="numpy",
            ),
        }
    )

    assert result.outputs["output"].to_pyarrow().to_pydict() == {
        "x": [1.0, 2.0],
        "y": [10.0, 20.0],
        "score": [51.0, 91.0],
    }


def test_symbolic_matrix_batch_consumes_the_lowered_table_segment() -> None:
    source = table_input(
        "input",
        schema=[
            Field("x", "float64", nullable=True),
            Field("y", "float64", nullable=True),
        ],
    )
    features = source.with_columns(FeatureSet((("total", source["x"] + source["y"]),)))
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(2, 1),
    )
    matrix = linalg.from_columns(
        features,
        columns=("total", "x"),
        backend="numpy",
    )
    output = table.attach_columns(
        features,
        linalg.matmul(matrix, weights),
        names=("score",),
    )
    program = Program(
        "symbolic-matrix-upstream",
        inputs=(source, weights),
        outputs=(("signals", output),),
    )
    runtime = Runtime()
    register_numpy(runtime)

    result = program.compile_batch(runtime).execute(
        {
            "input": Batch.from_pyarrow(
                pa.table(
                    {
                        "x": pa.array([1.0, 2.0], type=pa.float64()),
                        "y": pa.array([10.0, 20.0], type=pa.float64()),
                    }
                )
            ),
            "weights": Batch.from_array(
                np.array([[1.0], [10.0]], dtype=np.float64),
                backend="numpy",
            ),
        }
    )

    assert result.outputs["output"].to_pyarrow().to_pydict() == {
        "x": [1.0, 2.0],
        "y": [10.0, 20.0],
        "total": [11.0, 22.0],
        "score": [21.0, 42.0],
    }


@pytest.mark.parametrize("mode", ("batch", "stream"))
def test_symbolic_matrix_rejects_columns_from_a_different_attached_table(
    mode: str,
) -> None:
    source = table_input(
        "input",
        schema=[Field("x", "float64", nullable=True)],
    )
    left = source.with_columns(FeatureSet((("z", source["x"] * 2.0),)))
    right = source.with_columns(FeatureSet((("z", source["x"] * 3.0),)))
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(1, 1),
    )
    left_matrix = linalg.from_columns(left, columns=("z",), backend="numpy")
    right_matrix = linalg.from_columns(right, columns=("z",), backend="numpy")
    program = Program(
        "symbolic-matrix-distinct-sources",
        inputs=(source, weights),
        outputs=(
            (
                "signals",
                table.attach_columns(
                    left,
                    linalg.matmul(left_matrix + right_matrix, weights),
                    names=("score",),
                ),
            ),
        ),
    )
    runtime = Runtime()
    register_numpy(runtime)

    with pytest.raises(
        CompileError,
        match="attached matrix columns must come from the attached table",
    ):
        getattr(program, f"compile_{mode}")(runtime)


@pytest.mark.parametrize(
    "case",
    (
        "zero-matmul",
        "two-matmuls",
        "rhs-expression",
        "weights-after-matmul",
        "weights-in-matmul-left",
    ),
)
@pytest.mark.parametrize("mode", ("batch", "stream"))
def test_symbolic_matrix_requires_one_matmul_with_identity_weights_rhs(
    case: str,
    mode: str,
) -> None:
    source = table_input(
        "input",
        schema=[Field("x", "float64", nullable=True)],
    )
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(1, 1),
    )
    matrix = linalg.from_columns(source, columns=("x",), backend="numpy")
    if case == "zero-matmul":
        scores = matrix * weights
    elif case == "two-matmuls":
        scores = linalg.matmul(matrix, weights) + linalg.matmul(matrix, weights)
    elif case == "rhs-expression":
        scores = linalg.matmul(matrix, ArrayExpr(weights._node) + 0.0)
    elif case == "weights-after-matmul":
        scores = linalg.matmul(matrix, weights) + weights
    else:
        scores = linalg.matmul(matrix + weights, weights)
    program = Program(
        "symbolic-matrix-frozen-shape",
        inputs=(source, weights),
        outputs=(
            (
                "signals",
                table.attach_columns(source, scores, names=("score",)),
            ),
        ),
    )
    runtime = Runtime()
    register_numpy(runtime)

    with pytest.raises(
        CompileError,
        match=(
            "exactly one matmul whose right operand is the static 'weights' parameter"
        ),
    ):
        getattr(program, f"compile_{mode}")(runtime)


@pytest.mark.parametrize("segmentation", (1, 2, 10))
def test_symbolic_matrix_stream_uses_latched_weights_per_microbatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    segmentation: int,
) -> None:
    source = table_input(
        "input",
        schema=[
            Field("x", "float64", nullable=True),
            Field("y", "float64", nullable=True),
        ],
    )
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(2, 1),
    )
    features = source.with_columns(FeatureSet((("total", source["x"] + source["y"]),)))
    matrix = linalg.from_columns(
        features,
        columns=("total", "x"),
        backend="numpy",
    )
    output = table.attach_columns(
        features,
        linalg.matmul(matrix, weights),
        names=("score",),
    )
    program = Program(
        "symbolic-matrix-stream",
        inputs=(source, weights),
        outputs=(("signals", output),),
    )
    runtime = Runtime()
    register_numpy(runtime)
    weight_objects: list[int] = []
    original_call = array_module._SymbolicMatrixProvider.__call__

    def tracked_call(
        provider: object,
        inputs: dict[str, Batch],
        options: dict[str, object],
    ) -> dict[str, Batch]:
        weight_objects.append(id(inputs["weights"].array))
        return original_call(provider, inputs, options)

    monkeypatch.setattr(array_module._SymbolicMatrixProvider, "__call__", tracked_call)
    plan = program.compile_stream(runtime)
    assert plan.static_input_ids == ("weights",)
    assert plan.source_binding_ids == ("input",)
    sink = _CollectSink()
    input_table = pa.table(
        {
            "x": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            "y": pa.array([10.0, 20.0, 30.0], type=pa.float64()),
        }
    )

    async def exercise() -> None:
        job = await StreamingRunner(
            plan,
            {
                "input": SourceBinding(
                    _SegmentedSource(input_table, segmentation),
                    watermark_policy=DisabledWatermarks(),
                )
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
            static_inputs={
                "weights": Batch.from_array(
                    np.array([[1.0], [10.0]], dtype=np.float64),
                    backend="numpy",
                )
            },
        ).start_async()
        outcome = await job.wait_async()
        assert outcome.state == "completed"

    asyncio.run(exercise())

    output_table = pa.concat_tables(
        [batch.to_pyarrow() for batch in sink.batches]
    ).to_pydict()
    assert output_table == {
        "x": [1.0, 2.0, 3.0],
        "y": [10.0, 20.0, 30.0],
        "total": [11.0, 22.0, 33.0],
        "score": [21.0, 42.0, 63.0],
    }
    expected_calls = (input_table.num_rows + segmentation - 1) // segmentation
    assert [batch.metadata["provider_calls"] for batch in sink.batches] == [
        1
    ] * expected_calls
    assert len(weight_objects) == expected_calls
    assert len(set(weight_objects)) == 1
    assert [batch.metadata["copy_bytes"]["weights"] for batch in sink.batches] == [
        16,
        *([0] * (expected_calls - 1)),
    ]


def test_symbolic_matrix_numpy_uses_safe_common_dtype() -> None:
    source = table_input(
        "input",
        schema=[Field("x", "float32", nullable=True)],
    )
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(1, 1),
    )
    matrix = linalg.from_columns(source, columns=("x",), backend="numpy")
    program = Program(
        "symbolic-matrix-dtype",
        inputs=(source, weights),
        outputs=(
            (
                "signals",
                table.attach_columns(
                    source,
                    linalg.matmul(matrix, weights),
                    names=("score",),
                ),
            ),
        ),
    )
    runtime = Runtime()
    register_numpy(runtime)

    result = program.compile_batch(runtime).execute(
        {
            "input": Batch.from_pyarrow(
                pa.table({"x": pa.array([1.5, 2.5], type=pa.float32())})
            ),
            "weights": Batch.from_array(
                np.array([[2.0]], dtype=np.float64), backend="numpy"
            ),
        }
    )
    output = result.outputs["output"].to_pyarrow()

    assert output.schema.field("score").type == pa.float64()
    assert output["score"].to_pylist() == [3.0, 5.0]


def test_symbolic_matrix_float32_literal_dtype_matches_analysis_and_execution() -> None:
    source = table_input(
        "input",
        schema=[Field("x", "float32", nullable=True)],
    )
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float32",
        shape=(1, 1),
    )
    matrix = linalg.from_columns(source, columns=("x",), backend="numpy")
    program = Program(
        "symbolic-matrix-float32-literal",
        inputs=(source, weights),
        outputs=(
            (
                "signals",
                table.attach_columns(
                    source,
                    linalg.matmul(matrix * 2.0, weights),
                    names=("score",),
                ),
            ),
        ),
    )
    runtime = Runtime()
    register_numpy(runtime)

    assert "field score float32" in program.explain(runtime, mode="batch")
    output = (
        program.compile_batch(runtime)
        .execute(
            {
                "input": Batch.from_pyarrow(
                    pa.table({"x": pa.array([1.5, 2.5], type=pa.float32())})
                ),
                "weights": Batch.from_array(
                    np.array([[3.0]], dtype=np.float32),
                    backend="numpy",
                ),
            }
        )
        .outputs["output"]
        .to_pyarrow()
    )

    assert output.schema.field("score").type == pa.float32()
    assert output["score"].to_pylist() == [9.0, 15.0]


def test_symbolic_matrix_rejects_provider_output_dtype_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = table_input(
        "input",
        schema=[Field("x", "float32", nullable=True)],
    )
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float32",
        shape=(1, 1),
    )
    matrix = linalg.from_columns(source, columns=("x",), backend="numpy")
    program = Program(
        "symbolic-matrix-dtype-drift",
        inputs=(source, weights),
        outputs=(
            (
                "signals",
                table.attach_columns(
                    source,
                    linalg.matmul(matrix, weights),
                    names=("score",),
                ),
            ),
        ),
    )
    runtime = Runtime()
    register_numpy(runtime)
    plan = program.compile_batch(runtime)
    monkeypatch.setattr(
        np,
        "matmul",
        lambda left, _right: np.ones((left.shape[0], 1), dtype=np.float64),
    )

    with pytest.raises(
        Exception,
        match="output.dtype: expected float32, received float64",
    ):
        plan.execute(
            {
                "input": Batch.from_pyarrow(
                    pa.table({"x": pa.array([1.0, 2.0], type=pa.float32())})
                ),
                "weights": Batch.from_array(
                    np.ones((1, 1), dtype=np.float32),
                    backend="numpy",
                ),
            }
        )


def test_symbolic_matrix_jax_stream_matches_batch_on_cpu(tmp_path: Path) -> None:
    jnp = pytest.importorskip("jax.numpy")
    source = table_input(
        "input",
        schema=[Field("x", "float32", nullable=True)],
    )
    weights = parameter(
        "weights",
        kind="array",
        backend="jax",
        dtype="float32",
        shape=(1, 1),
    )
    matrix = linalg.from_columns(source, columns=("x",), backend="jax")
    program = Program(
        "symbolic-matrix-jax-stream",
        inputs=(source, weights),
        outputs=(
            (
                "signals",
                table.attach_columns(
                    source,
                    linalg.matmul(matrix, weights),
                    names=("score",),
                ),
            ),
        ),
    )
    runtime = Runtime()
    register_jax(runtime)
    plan = program.compile_stream(runtime)
    input_table = pa.table({"x": pa.array([1.5, 2.5, 3.5], type=pa.float32())})
    weight_batch = Batch.from_array(
        jnp.asarray([[2.0]], dtype=jnp.float32), backend="jax"
    )
    sink = _CollectSink()

    async def exercise() -> None:
        job = await StreamingRunner(
            plan,
            {
                "input": SourceBinding(
                    _SegmentedSource(input_table, 2),
                    watermark_policy=DisabledWatermarks(),
                )
            },
            {"output": [SinkBinding.ordinary("archive", sink)]},
            ManagedCheckpointRuntime(tmp_path),
            static_inputs={"weights": weight_batch},
        ).start_async()
        outcome = await job.wait_async()
        assert outcome.state == "completed"

    asyncio.run(exercise())

    stream_scores = pa.concat_tables([batch.to_pyarrow() for batch in sink.batches])[
        "score"
    ].to_pylist()
    batch_scores = (
        program.compile_batch(runtime)
        .execute(
            {
                "input": Batch.from_pyarrow(input_table),
                "weights": weight_batch,
            }
        )
        .outputs["output"]
        .to_pyarrow()["score"]
        .to_pylist()
    )
    assert stream_scores == pytest.approx(batch_scores, rel=1e-6, abs=1e-6)


def test_symbolic_matrix_jax_rejects_float64_when_x64_is_disabled() -> None:
    jax = pytest.importorskip("jax")
    if jax.config.x64_enabled:
        pytest.skip("this process has JAX x64 enabled")
    source = table_input(
        "input",
        schema=[Field("x", "float64", nullable=True)],
    )
    weights = parameter(
        "weights",
        kind="array",
        backend="jax",
        dtype="float64",
        shape=(1, 1),
    )
    matrix = linalg.from_columns(source, columns=("x",), backend="jax")
    program = Program(
        "symbolic-matrix-jax-x64-disabled",
        inputs=(source, weights),
        outputs=(
            (
                "signals",
                table.attach_columns(
                    source,
                    linalg.matmul(matrix, weights),
                    names=("score",),
                ),
            ),
        ),
    )
    runtime = Runtime()
    register_jax(runtime)

    with pytest.raises(CompileError, match="no provable result dtype"):
        program.compile_batch(runtime)


def test_symbolic_matrix_jax_float64_with_x64_enabled() -> None:
    script = """
import jax
import jax.numpy as jnp
import pyarrow as pa

from calc_flow import Batch, Runtime, register_jax
from calc_flow.symbolic import Field, Program, linalg, parameter, table, table_input

assert jax.config.x64_enabled
source = table_input(
    "input",
    schema=[Field("x", "float64", nullable=True)],
)
weights = parameter(
    "weights",
    kind="array",
    backend="jax",
    dtype="float64",
    shape=(1, 1),
)
matrix = linalg.from_columns(source, columns=("x",), backend="jax")
program = Program(
    "symbolic-matrix-jax-x64",
    inputs=(source, weights),
    outputs=(("signals", table.attach_columns(
        source,
        linalg.matmul(matrix, weights),
        names=("score",),
    )),),
)
runtime = Runtime()
register_jax(runtime)
output = program.compile_batch(runtime).execute({
    "input": Batch.from_pyarrow(
        pa.table({"x": pa.array([1.5, 2.5], type=pa.float64())})
    ),
    "weights": Batch.from_array(
        jnp.asarray([[2.0]], dtype=jnp.float64),
        backend="jax",
    ),
}).outputs["output"].to_pyarrow()
assert output.schema.field("score").type == pa.float64()
assert output["score"].to_pylist() == [3.0, 5.0]
"""
    # Test-owned interpreter and fixed source; shell execution remains disabled.
    completed = subprocess.run(  # noqa: E501  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env={**os.environ, "JAX_ENABLE_X64": "true", "JAX_PLATFORMS": "cpu"},
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_symbolic_matrix_rejects_wrong_provider_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = table_input(
        "input",
        schema=[Field("x", "float64", nullable=True)],
    )
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(1, 1),
    )
    matrix = linalg.from_columns(source, columns=("x",), backend="numpy")
    program = Program(
        "symbolic-matrix-row-count",
        inputs=(source, weights),
        outputs=(
            (
                "signals",
                table.attach_columns(
                    source,
                    linalg.matmul(matrix, weights),
                    names=("score",),
                ),
            ),
        ),
    )
    runtime = Runtime()
    register_numpy(runtime)
    plan = program.compile_batch(runtime)
    monkeypatch.setattr(np, "matmul", lambda _left, _right: np.ones((1, 1)))

    with pytest.raises(Exception, match="output.rows: expected 2, received 1"):
        plan.execute(
            {
                "input": Batch.from_pyarrow(
                    pa.table({"x": pa.array([1.0, 2.0], type=pa.float64())})
                ),
                "weights": Batch.from_array(
                    np.ones((1, 1), dtype=np.float64), backend="numpy"
                ),
            }
        )
