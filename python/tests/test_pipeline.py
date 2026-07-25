from __future__ import annotations

import gc
import json
import threading
import time
from types import MappingProxyType

import pyarrow as pa
import pytest

from calc_flow import (
    Batch,
    CompileError,
    ConfigError,
    ExecutionError,
    PipelineBuilder,
    RunResult,
    Runtime,
    project_json_schema,
    validate_project_json,
)


def _batch(**columns: list[int]) -> Batch:
    return Batch.from_pyarrow(pa.table(columns), metadata={"nested": {"value": 1}})


def test_python_builder_compiles_through_rust() -> None:
    plan = PipelineBuilder("totals").expression("calc", "total = a + b").compile()

    result = plan.execute({"input": _batch(a=[1], b=[2])})

    assert isinstance(result, RunResult)
    assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [3]
    assert result.metadata["pipeline_name"] == "totals"
    assert result.metadata["pipeline_fingerprint"]
    assert result.metadata["run_id"]
    assert result.node_timings["calc"]["input_rows"] == {"input": 1}
    assert result.node_timings["calc"]["output_rows"] == {"output": 1}
    assert result.datafusion_metrics[0]["node_id"] == "calc"
    assert result.datafusion_metrics[0]["output_rows"] == 1


def test_builder_is_functional_and_project_values_are_defensive() -> None:
    original = PipelineBuilder("flow")
    first = original.expression("first", "b = a + 1")
    second = first.expression("second", "c = b * 2").connect("first", "second")

    assert original.project["pipeline"]["nodes"] == []
    assert [node["id"] for node in first.project["pipeline"]["nodes"]] == ["first"]
    assert [node["id"] for node in second.project["pipeline"]["nodes"]] == [
        "first",
        "second",
    ]
    assert second.project["data_sources"] == [
        {"data": [], "format": "inline_json", "id": "source_1", "input": "input"}
    ]

    returned = second.project
    returned["pipeline"]["nodes"].clear()
    returned["data_sources"][0]["data"].append({"executable": "never"})
    assert len(second.project["pipeline"]["nodes"]) == 2
    assert second.project["data_sources"][0]["data"] == []

    output = second.compile().execute({"input": _batch(a=[2])}).outputs["output"]
    assert output.to_pyarrow()["c"].to_pylist() == [6]


def test_table_matmul_builder_is_functional_and_defensive() -> None:
    columns = ["quantity", "unit_price"]
    original = PipelineBuilder("matrix")
    builder = original.table_matmul(
        "multiply",
        backend="numpy",
        columns=columns,
    )
    columns[0] = "mutated"

    assert original.project["pipeline"]["nodes"] == []
    assert builder.project["data_sources"] == [
        {
            "data": [],
            "format": "inline_json",
            "id": "source_1",
            "input": "table",
        },
        {
            "data": [],
            "format": "inline_json",
            "id": "source_2",
            "input": "weights",
        },
    ]
    assert builder.project["pipeline"]["nodes"] == [
        {
            "id": "multiply",
            "input_ports": [
                {
                    "kind": "table",
                    "name": "table",
                    "required": True,
                    "schema": [],
                },
                {
                    "kind": "array",
                    "name": "weights",
                    "required": True,
                    "schema": [],
                },
            ],
            "operator": {
                "kind": "external",
                "name": "table_matmul",
                "options": {"columns": ["quantity", "unit_price"]},
                "provider": "numpy",
                "version": "1",
            },
            "output_ports": [
                {
                    "kind": "array",
                    "name": "output",
                    "required": True,
                    "schema": [],
                }
            ],
        }
    ]


@pytest.mark.parametrize("backend", ["", "numpy ", "pandas", "JAX"])
def test_table_matmul_rejects_unknown_backends(backend: str) -> None:
    with pytest.raises(ValueError, match="backend must be 'numpy' or 'jax'"):
        PipelineBuilder("matrix").table_matmul(
            "multiply",
            backend=backend,  # type: ignore[arg-type]
            columns=("a",),
        )


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ((), "at least one"),
        (("a", "a"), "unique"),
        (("",), "non-empty strings"),
        (("a", 1), "non-empty strings"),
        ("a", "sequence of column names"),
    ],
)
def test_table_matmul_rejects_invalid_columns(
    columns: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        PipelineBuilder("matrix").table_matmul(
            "multiply",
            backend="numpy",
            columns=columns,  # type: ignore[arg-type]
        )


def test_builder_derives_exact_qualified_inputs_for_sql() -> None:
    builder = PipelineBuilder("join").sql(
        "query",
        "SELECT left.a + right.b AS total FROM left CROSS JOIN right",
        aliases=("left", "right"),
    )

    assert [source["input"] for source in builder.project["data_sources"]] == [
        "left",
        "right",
    ]
    result = builder.compile().execute(
        MappingProxyType({"left": _batch(a=[2]), "right": _batch(b=[3])})
    )
    assert result.outputs["output"].to_pyarrow()["total"].to_pylist() == [5]


def test_duplicate_external_port_names_are_qualified_deterministically() -> None:
    builder = (
        PipelineBuilder("branches")
        .expression("zeta", "result = value + 1")
        .expression("alpha", "result = value + 2")
    )

    assert [source["input"] for source in builder.project["data_sources"]] == [
        "alpha.input",
        "zeta.input",
    ]
    result = builder.compile().execute(
        {"alpha.input": _batch(value=[1]), "zeta.input": _batch(value=[1])}
    )
    assert set(result.outputs) == {"alpha.output", "zeta.output"}


def test_builder_accepts_task_19_udf_reference_tuple_shape() -> None:
    project = (
        PipelineBuilder("udfs")
        .expression("calc", "result = double(value)", udfs=(("python", "double", "1"),))
        .project
    )

    assert project["pipeline"]["nodes"][0]["operator"]["udfs"] == [
        {
            "kind": "data_fusion_scalar",
            "name": "double",
            "provider": "python",
            "version": "1",
        }
    ]


def test_runtime_compiles_strict_json_and_plan_outlives_runtime() -> None:
    runtime = Runtime()
    document = json.dumps(
        PipelineBuilder("lifetime").expression("calc", "b = a + 1").project
    )
    plan = runtime.compile_project(document)
    del runtime
    gc.collect()

    assert plan.execute({"input": _batch(a=[4])}).outputs["output"].to_pyarrow()[
        "b"
    ].to_pylist() == [5]

    with pytest.raises(ConfigError):
        Runtime().compile_project('{"format_version":2,"format_version":2}')


def test_execution_plan_exposes_immutable_rust_identity() -> None:
    plan = PipelineBuilder("identity").expression("calc", "b = a + 1").compile()

    assert plan.name == "identity"
    assert plan.fingerprint
    with pytest.raises((AttributeError, TypeError)):
        plan.name = "changed"  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        plan.fingerprint = "changed"  # type: ignore[misc]


def test_runtime_plan_and_result_are_visible_to_cyclic_gc() -> None:
    runtime = Runtime()
    plan = PipelineBuilder("tracked").expression("calc", "b = a + 1").compile(runtime)
    result = plan.execute({"input": _batch(a=[1])})

    assert gc.is_tracked(runtime._inner)
    assert gc.is_tracked(plan._inner)
    assert gc.is_tracked(result)


def test_project_json_helpers_are_canonical_strict_and_defaulted() -> None:
    schema = json.loads(project_json_schema())
    assert schema["title"] == "Calc Flow Project V2"
    assert schema["properties"]["format_version"]["const"] == 2

    minimal = {
        "format_version": 2,
        "id": "demo",
        "name": "Demo",
        "pipeline": {
            "name": "demo",
            "nodes": [
                {
                    "id": "calc",
                    "operator": {"kind": "expression", "expression": "b = a + 1"},
                }
            ],
        },
    }
    canonical = validate_project_json(json.dumps(minimal))
    assert canonical == json.dumps(
        json.loads(canonical), sort_keys=True, separators=(",", ":")
    )
    validated = json.loads(canonical)
    assert validated["description"] == ""
    assert validated["run_options"]["timeout_seconds"] == 30
    assert validated["pipeline"]["datafusion"]["target_partitions"] == 1

    with pytest.raises(ConfigError):
        validate_project_json(json.dumps({**minimal, "format_version": 1}))
    with pytest.raises(ConfigError):
        validate_project_json(json.dumps({**minimal, "callable": "os.system"}))


def test_compile_and_execution_failures_keep_declared_exception_types() -> None:
    cyclic = (
        PipelineBuilder("cycle")
        .expression("left", "b = a + 1")
        .expression("right", "c = b + 1")
        .connect("left", "right")
        .connect("right", "left")
    )
    with pytest.raises(CompileError, match="cycle"):
        cyclic.compile()

    plan = (
        PipelineBuilder("missing_column")
        .expression("calc", "result = absent + 1")
        .compile()
    )
    with pytest.raises(ExecutionError, match="absent"):
        plan.execute({"input": _batch(value=[1])})


def test_execute_validates_inputs_without_mutating_the_caller() -> None:
    plan = PipelineBuilder("inputs").expression("calc", "b = a + 1").compile()
    batch = _batch(a=[1])
    inputs = {"input": batch}

    plan.execute(inputs)
    assert inputs == {"input": batch}

    with pytest.raises(ConfigError, match="missing required graph input"):
        plan.execute({})
    with pytest.raises(ConfigError, match="unknown graph inputs"):
        plan.execute({"input": batch, "extra": batch})
    with pytest.raises(TypeError, match="Batch"):
        plan.execute({"input": object()})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="strings"):
        plan.execute({1: batch})  # type: ignore[dict-item]


def test_run_result_properties_return_defensive_values() -> None:
    plan = PipelineBuilder("copies").expression("calc", "b = a + 1").compile()
    result = plan.execute({"input": _batch(a=[1])})

    outputs = result.outputs
    metadata = result.metadata
    timings = result.node_timings
    metrics = result.datafusion_metrics
    outputs.clear()
    metadata.clear()
    timings["calc"]["input_rows"].clear()
    metrics.clear()

    assert set(result.outputs) == {"output"}
    assert result.metadata["pipeline_name"] == "copies"
    assert result.node_timings["calc"]["input_rows"] == {"input": 1}
    assert result.datafusion_metrics


def test_blocking_execute_releases_the_gil() -> None:
    plan = (
        PipelineBuilder("gil")
        .sql(
            "query",
            "SELECT sum(left.value * right.value) AS total FROM left CROSS JOIN right",
            aliases=("left", "right"),
        )
        .compile()
    )
    progressed = threading.Event()
    ready = threading.Event()
    values = _batch(value=list(range(5_000)))

    def heartbeat() -> None:
        ready.set()
        time.sleep(0.005)
        progressed.set()

    thread = threading.Thread(target=heartbeat)
    thread.start()
    ready.wait()
    plan.execute({"left": values, "right": values})
    assert progressed.is_set()
    thread.join()
