from __future__ import annotations

import asyncio
import json
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
from calc_flow.symbolic.nodes import CBool, build


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


def _symbolic_matrix_project(options: dict[str, object]) -> str:
    return json.dumps(
        {
            "data_sources": [
                {
                    "data": [],
                    "format": "inline_json",
                    "id": "source_1",
                    "input": "input",
                },
                {
                    "data": [],
                    "format": "inline_json",
                    "id": "source_2",
                    "input": "weights",
                },
            ],
            "format_version": 3,
            "id": "symbolic-matrix-provider-defense",
            "name": "symbolic-matrix-provider-defense",
            "runtime": {"mode": "batch", "options": {}},
            "graph": {
                "edges": [],
                "name": "symbolic-matrix-provider-defense",
                "nodes": [
                    {
                        "id": "matrix",
                        "input_ports": [
                            {"kind": "table", "name": "input", "required": True},
                            {"kind": "array", "name": "weights", "required": True},
                        ],
                        "operator": {
                            "kind": "external",
                            "name": "symbolic_matrix",
                            "options": options,
                            "provider": "numpy",
                            "version": "1",
                        },
                        "output_ports": [
                            {"kind": "table", "name": "output", "required": True}
                        ],
                    }
                ],
            },
        }
    )


def _invalid_symbolic_matrix_options(case: str) -> dict[str, object]:
    options: dict[str, object] = {
        "columns": ["x"],
        "expression": {
            "left": {"op": "input"},
            "op": "matmul",
            "right": {"op": "weights"},
        },
        "names": ["score"],
    }
    if case == "option-keys":
        options.pop("names")
    elif case == "columns-container":
        options["columns"] = "x"
    elif case == "columns-empty":
        options["columns"] = []
    elif case == "columns-type":
        options["columns"] = [1]
    elif case == "columns-empty-name":
        options["columns"] = [""]
    elif case == "columns-duplicate":
        options["columns"] = ["x", "x"]
    elif case == "tree-not-mapping":
        options["expression"] = None
    elif case == "tree-operation":
        options["expression"] = {"op": "unsupported"}
    elif case == "leaf-shape":
        options["expression"] = {"extra": True, "op": "input"}
    elif case == "literal-shape":
        options["expression"] = {"op": "literal"}
    elif case == "literal-type":
        options["expression"] = {"op": "literal", "value": "secret"}
    elif case == "unary-shape":
        options["expression"] = {
            "extra": True,
            "op": "neg",
            "value": {"op": "input"},
        }
    elif case == "binary-shape":
        options["expression"] = {
            "extra": True,
            "left": {"op": "input"},
            "op": "add",
            "right": {"op": "weights"},
        }
    else:
        expression: dict[str, object] = {"op": "input"}
        for _ in range(25):
            expression = {"op": "neg", "value": expression}
        options["expression"] = expression
    return options


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


@pytest.mark.parametrize("mode", ("batch", "stream"))
def test_symbolic_matrix_unary_not_bool_scalar_matches_analysis_and_provider(
    mode: str,
    tmp_path: Path,
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
    bool_scalar = ArrayExpr(build("literal", (), {"value": CBool(True)}))
    score = (linalg.matmul(matrix, weights) > 0.0) == ~bool_scalar
    program = Program(
        "symbolic-matrix-unary-not-bool-scalar",
        inputs=(source, weights),
        outputs=(
            (
                "signals",
                table.attach_columns(source, score, names=("score",)),
            ),
        ),
    )
    runtime = Runtime()
    register_numpy(runtime)
    input_table = pa.table({"x": pa.array([-1.0, 2.0], type=pa.float64())})
    static_inputs = {
        "weights": Batch.from_array(
            np.array([[2.0]], dtype=np.float64),
            backend="numpy",
        )
    }

    assert "field score bool" in program.explain(runtime, mode=mode)
    if mode == "batch":
        output = (
            program.compile_batch(runtime)
            .execute({"input": Batch.from_pyarrow(input_table), **static_inputs})
            .outputs["output"]
            .to_pyarrow()
        )
    else:
        sink = _CollectSink()

        async def exercise() -> None:
            job = await StreamingRunner(
                program.compile_stream(runtime),
                {
                    "input": SourceBinding(
                        _SegmentedSource(input_table, 2),
                        watermark_policy=DisabledWatermarks(),
                    )
                },
                {"output": [SinkBinding.ordinary("archive", sink)]},
                ManagedCheckpointRuntime(tmp_path),
                static_inputs=static_inputs,
            ).start_async()
            outcome = await job.wait_async()
            assert outcome.state == "completed"

        asyncio.run(exercise())
        output = pa.concat_tables([batch.to_pyarrow() for batch in sink.batches])

    assert output.to_pydict() == {
        "x": [-1.0, 2.0],
        "score": [True, False],
    }


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


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("option-keys", "expected columns, expression, and names"),
        ("columns-container", "columns: expected unique non-empty strings"),
        ("columns-empty", "columns: expected unique non-empty strings"),
        ("columns-type", "columns: expected unique non-empty strings"),
        ("columns-empty-name", "columns: expected unique non-empty strings"),
        ("columns-duplicate", "columns: expected unique non-empty strings"),
        ("tree-not-mapping", "node must be a mapping"),
        ("tree-operation", "unsupported node 'unsupported'"),
        ("leaf-shape", "unsupported node 'input'"),
        ("literal-shape", "unsupported node 'literal'"),
        ("literal-type", "literal must be finite"),
        ("unary-shape", "unsupported node 'neg'"),
        ("binary-shape", "unsupported node 'add'"),
        ("tree-depth", "depth limit exceeded"),
    ),
)
def test_symbolic_matrix_public_compile_rejects_malformed_provider_options(
    case: str,
    message: str,
) -> None:
    runtime = Runtime()
    register_numpy(runtime)

    with pytest.raises(ConfigError, match=message):
        runtime.compile_batch_project(
            _symbolic_matrix_project(_invalid_symbolic_matrix_options(case))
        )


def test_symbolic_matrix_public_execution_rejects_unresolved_output_dtype() -> None:
    runtime = Runtime()
    register_numpy(runtime)
    plan = runtime.compile_batch_project(
        _symbolic_matrix_project(
            {
                "columns": ["x"],
                "expression": {"op": "literal", "value": True},
                "names": ["score"],
            }
        )
    )

    with pytest.raises(Exception, match="output.dtype: unresolved dtype"):
        plan.execute(
            {
                "input": Batch.from_pyarrow(pa.table({"x": [1.0, 2.0]})),
                "weights": Batch.from_array(
                    np.ones((1, 1), dtype=np.float64), backend="numpy"
                ),
            }
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("input-kind", r'input" \(matrix.input\) expects a Table batch'),
        ("weights-backend", "weights.backend: expected numpy"),
    ),
)
def test_symbolic_matrix_public_execution_rejects_malformed_inputs(
    case: str,
    message: str,
) -> None:
    runtime = Runtime()
    register_numpy(runtime)
    plan = runtime.compile_batch_project(
        _symbolic_matrix_project(
            {
                "columns": ["x"],
                "expression": {
                    "left": {"op": "input"},
                    "op": "matmul",
                    "right": {"op": "weights"},
                },
                "names": ["score"],
            }
        )
    )
    table_batch = Batch.from_pyarrow(pa.table({"x": [1.0, 2.0]}))
    array_batch = Batch.from_array(np.ones((1, 1), dtype=np.float64), backend="numpy")
    if case == "input-kind":
        inputs = {"input": array_batch, "weights": array_batch}
    else:
        jnp = pytest.importorskip("jax.numpy")
        inputs = {
            "input": table_batch,
            "weights": Batch.from_array(jnp.ones((1, 1)), backend="jax"),
        }

    with pytest.raises(Exception, match=message):
        plan.execute(inputs)


@pytest.mark.parametrize(
    ("shape", "message"),
    (
        ((2,), "output.rank: expected rank two; received 1"),
        ((2, 2), "output.width: expected 1, received 2"),
    ),
)
def test_symbolic_matrix_public_execution_rejects_provider_output_shape(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, ...],
    message: str,
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
        "symbolic-matrix-provider-shape",
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
    monkeypatch.setattr(np, "matmul", lambda _left, _right: np.ones(shape))

    with pytest.raises(Exception, match=message):
        plan.execute(
            {
                "input": Batch.from_pyarrow(pa.table({"x": [1.0, 2.0]})),
                "weights": Batch.from_array(
                    np.ones((1, 1), dtype=np.float64), backend="numpy"
                ),
            }
        )
