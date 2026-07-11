from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from pydantic import ValidationError

from calc_flow.batch import Batch
from calc_flow.config import (
    CONFIG_FORMAT_VERSION,
    ArrowFieldConfig,
    DataFusionConfig,
    DataSourceConfig,
    EdgeConfig,
    NodeConfig,
    PipelineConfig,
    PortConfig,
    ProjectConfig,
    UdfReferenceConfig,
    compile_project,
    validate_project,
)
from calc_flow.udf import UdfRegistry


def _registry() -> UdfRegistry:
    registry = UdfRegistry()

    @registry.datafusion_scalar(
        name="double_value",
        version="1",
        input_fields=[pa.int64()],
        return_field=pa.int64(),
    )
    def double_value(values):
        return pc.multiply(values, 2)

    return registry


def _project() -> ProjectConfig:
    return ProjectConfig(
        id="branching",
        name="Branching project",
        pipeline=PipelineConfig(
            id="main",
            name="Main pipeline",
            nodes=(
                NodeConfig(
                    id="calculate",
                    kind="expression",
                    expression="doubled = double_value(value)",
                    udfs=(UdfReferenceConfig(name="double_value", version="1"),),
                ),
                NodeConfig(
                    id="left",
                    kind="expression",
                    expression="left_value = doubled + 1",
                    position={"x": 240, "y": -80},
                ),
                NodeConfig(
                    id="right",
                    kind="expression",
                    expression="right_value = doubled * 10",
                    position={"x": 240, "y": 80},
                ),
            ),
            edges=(
                EdgeConfig(source_node="calculate", target_node="left"),
                EdgeConfig(source_node="calculate", target_node="right"),
            ),
        ),
    )


def test_project_config_compiles_branching_registered_udf_pipeline() -> None:
    plan = compile_project(_project(), udf_registry=_registry())

    result = plan.execute({"input": Batch.table(pa.table({"value": [2, 4]}))})

    assert set(result.outputs) == {"left.output", "right.output"}
    assert result.outputs["left.output"].table_payload["left_value"].to_pylist() == [
        5,
        9,
    ]
    assert result.outputs["right.output"].table_payload["right_value"].to_pylist() == [
        40,
        80,
    ]


def test_project_config_round_trips_canonical_json_data() -> None:
    project = _project()
    data = project.model_dump(mode="json", by_alias=True)

    restored = ProjectConfig.model_validate(data)

    assert restored == project
    assert data["format_version"] == CONFIG_FORMAT_VERSION
    assert data["pipeline"]["nodes"][0]["udfs"] == [
        {"name": "double_value", "version": "1"}
    ]


def test_port_config_uses_schema_alias_and_builds_arrow_schema() -> None:
    port = PortConfig.model_validate(
        {
            "name": "input",
            "kind": "table",
            "schema": [{"name": "value", "type": "int64", "nullable": False}],
        }
    )

    assert port.model_dump(mode="json") == {
        "name": "input",
        "kind": "table",
        "required": True,
        "schema": [{"name": "value", "type": "int64", "nullable": False}],
    }
    assert port.arrow_schema() == pa.schema(
        [pa.field("value", pa.int64(), nullable=False)]
    )


def test_config_rejects_unknown_fields_and_executable_values() -> None:
    data = _project().model_dump(mode="json")
    data["inline_code"] = "import os"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ProjectConfig.model_validate(data)

    data = _project().model_dump(mode="json")
    data["data_sources"] = [
        {
            "id": "source",
            "format": "inline_json",
            "data": object(),
        }
    ]
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(data)


@pytest.mark.parametrize(
    "node",
    [
        {
            "id": "bad",
            "kind": "expression",
            "expression": "a + 1",
            "backend": "numpy",
        },
        {"id": "bad", "kind": "sql", "query": "delete from input", "inputs": ["input"]},
        {"id": "bad", "kind": "array_expression", "expression": "x + 1"},
    ],
)
def test_node_config_rejects_inconsistent_modes(node: dict) -> None:
    with pytest.raises(ValidationError):
        NodeConfig.model_validate(node)


def test_node_config_rejects_table_backend_selector() -> None:
    with pytest.raises(ValidationError, match="backend"):
        NodeConfig(
            id="calculate",
            kind="expression",
            expression="a + 1",
            backend="jax",
        )


def test_node_config_validates_declared_ports() -> None:
    with pytest.raises(ValidationError, match="ports must be"):
        NodeConfig(
            id="calculate",
            kind="expression",
            expression="a + 1",
            input_ports=(PortConfig(name="wrong", kind="table"),),
        )

    with pytest.raises(ValidationError, match="table batches"):
        NodeConfig(
            id="calculate",
            kind="expression",
            expression="a + 1",
            input_ports=(PortConfig(name="input", kind="array"),),
        )


def test_arrow_field_config_rejects_unsupported_type() -> None:
    with pytest.raises(ValidationError, match="unsupported Arrow type"):
        ArrowFieldConfig(name="value", type="object")


def test_datafusion_config_enforces_local_execution_bounds() -> None:
    with pytest.raises(ValidationError):
        DataFusionConfig(target_partitions=0)
    with pytest.raises(ValidationError):
        DataFusionConfig(batch_size=2_000_000)


def test_validation_report_contains_graph_contract() -> None:
    report = validate_project(_project(), udf_registry=_registry())

    assert report.valid
    assert report.fingerprint
    assert report.graph_inputs == ("input",)
    assert report.graph_outputs == ("left.output", "right.output")
    assert report.warnings[0].code == "no_sample_data"


def test_validation_report_returns_compile_errors() -> None:
    project = _project().model_copy(
        update={
            "pipeline": _project().pipeline.model_copy(
                update={
                    "edges": (
                        EdgeConfig(source_node="calculate", target_node="left"),
                        EdgeConfig(source_node="left", target_node="calculate"),
                    )
                }
            )
        }
    )

    report = validate_project(project, udf_registry=_registry())

    assert not report.valid
    assert "cycle" in report.errors[0].message


def test_port_config_rejects_array_and_duplicate_schemas() -> None:
    field = ArrowFieldConfig(name="value", type="int64")

    with pytest.raises(ValidationError, match="only table ports"):
        PortConfig(name="input", kind="array", fields=(field,))
    with pytest.raises(ValidationError, match="field names must be unique"):
        PortConfig(name="input", kind="table", fields=(field, field))

    assert PortConfig(name="input", kind="table").arrow_schema() is None


def test_node_config_rejects_duplicate_declared_ports() -> None:
    port = PortConfig(name="input", kind="table")

    with pytest.raises(ValidationError, match="port names must be unique"):
        NodeConfig(
            id="calculate",
            kind="expression",
            expression="result = value + 1",
            input_ports=(port, port),
        )


def test_pipeline_and_project_config_reject_duplicate_identities() -> None:
    node = NodeConfig(
        id="calculate",
        kind="expression",
        expression="result = value + 1",
    )
    with pytest.raises(ValidationError, match="at least one node"):
        PipelineConfig(id="main", name="Main", nodes=())
    with pytest.raises(ValidationError, match="node IDs must be unique"):
        PipelineConfig(id="main", name="Main", nodes=(node, node))

    source = {
        "id": "source",
        "input_name": "input",
        "format": "inline_json",
        "data": [{"value": 1}],
    }
    project_data = {
        "id": "duplicates",
        "name": "Duplicates",
        "pipeline": {"id": "main", "name": "Main", "nodes": [node]},
        "data_sources": [source, source],
    }
    with pytest.raises(ValidationError, match="source IDs must be unique"):
        ProjectConfig.model_validate(project_data)

    project_data["data_sources"][1] = {**source, "id": "other"}
    with pytest.raises(ValidationError, match="input names must be unique"):
        ProjectConfig.model_validate(project_data)


def test_validation_report_rejects_saved_source_input_mismatch() -> None:
    project = _project().model_copy(
        update={
            "data_sources": (
                DataSourceConfig(
                    id="source",
                    input_name="wrong",
                    format="inline_json",
                    data=[{"value": 1}],
                ),
            )
        }
    )

    report = validate_project(project, udf_registry=_registry())

    assert not report.valid
    assert report.errors[0].code == "source_input_mismatch"


def test_array_project_config_compiles_and_executes() -> None:
    import numpy as np

    project = ProjectConfig(
        id="array-project",
        name="Array project",
        pipeline=PipelineConfig(
            id="main",
            name="Array pipeline",
            nodes=(
                NodeConfig(
                    id="calculate",
                    kind="array_expression",
                    expression="x * 2",
                    backend="numpy",
                ),
            ),
        ),
    )

    result = compile_project(project).execute(
        {"input": Batch.array(np.asarray([1, 2, 3]))}
    )

    assert result.output.array_payload.tolist() == [2, 4, 6]


@pytest.mark.parametrize(
    "node",
    (
        {"id": "bad", "kind": "expression"},
        {"id": "bad", "kind": "sql", "query": "select 1"},
        {
            "id": "bad",
            "kind": "sql",
            "query": "select * from input",
            "inputs": ["input"],
            "expression": "value + 1",
        },
        {
            "id": "bad",
            "kind": "array_expression",
            "expression": "x + 1",
            "backend": "numpy",
            "select": ["x"],
        },
    ),
)
def test_node_config_rejects_missing_or_conflicting_calculation_modes(
    node: dict,
) -> None:
    with pytest.raises(ValidationError):
        NodeConfig.model_validate(node)


def test_node_config_resolves_declared_input_and_output_schemas() -> None:
    input_port = PortConfig(
        name="input",
        kind="table",
        fields=(ArrowFieldConfig(name="value", type="int64"),),
    )
    output_port = PortConfig(
        name="output",
        kind="table",
        fields=(ArrowFieldConfig(name="result", type="int64"),),
    )
    node = NodeConfig(
        id="calculate",
        kind="expression",
        expression="result = value + 1",
        input_ports=(input_port,),
        output_ports=(output_port,),
    )

    assert node.schema_for("input") == pa.schema([("value", pa.int64())])
    assert node.schema_for("output", output=True) == pa.schema([("result", pa.int64())])
    assert node.schema_for("missing") is None
