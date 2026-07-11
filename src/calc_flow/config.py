from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

from calc_flow.batch import BatchKind, JSONValue
from calc_flow.engine.datafusion import (
    DataFusionConfig as RuntimeDataFusionConfig,
)
from calc_flow.engine.datafusion import validate_datafusion_query
from calc_flow.operator import (
    ArrayExpressionOperator,
    ExpressionOperator,
    SqlOperator,
)
from calc_flow.pipeline import ExecutionPlan, Pipeline
from calc_flow.udf import (
    UdfReference as RuntimeUdfReference,
)
from calc_flow.udf import UdfRegistry, UdfRegistrySnapshot

CONFIG_FORMAT_VERSION = "1"
MAX_PREVIEW_BYTES = 10 * 1024 * 1024
MAX_PREVIEW_ROWS = 100_000
MAX_PREVIEW_SECONDS = 30.0
SUPPORTED_ARROW_TYPES = (
    "bool",
    "date32",
    "date64",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "large_string",
    "string",
    "time32[s]",
    "time64[us]",
    "timestamp[ms]",
    "timestamp[us]",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
)

_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{0,63}$"
_PORT_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
        allow_inf_nan=False,
    )


class NodeKind(StrEnum):
    EXPRESSION = "expression"
    SQL = "sql"
    ARRAY_EXPRESSION = "array_expression"


class InputFormat(StrEnum):
    INLINE_JSON = "inline_json"
    CSV = "csv"
    JSON = "json"
    ARROW_IPC = "arrow_ipc"


class PositionConfig(StrictModel):
    x: float = 0
    y: float = 0


class ArrowFieldConfig(StrictModel):
    name: Annotated[str, Field(pattern=_PORT_PATTERN)]
    type: str
    nullable: bool = True

    @model_validator(mode="after")
    def validate_type(self) -> ArrowFieldConfig:
        arrow_type(self.type)
        return self

    def to_arrow(self) -> pa.Field:
        return pa.field(self.name, arrow_type(self.type), nullable=self.nullable)


class PortConfig(StrictModel):
    name: Annotated[str, Field(pattern=_PORT_PATTERN)]
    kind: BatchKind
    required: bool = True
    fields: tuple[ArrowFieldConfig, ...] | None = Field(
        default=None,
        validation_alias="schema",
        serialization_alias="schema",
    )

    @model_validator(mode="after")
    def validate_schema_kind(self) -> PortConfig:
        if self.fields is not None and self.kind is not BatchKind.TABLE:
            msg = "only table ports may declare an Arrow schema"
            raise ValueError(msg)
        if self.fields is not None:
            names = [field.name for field in self.fields]
            if len(names) != len(set(names)):
                msg = "Arrow schema field names must be unique"
                raise ValueError(msg)
        return self

    def arrow_schema(self) -> pa.Schema | None:
        if self.fields is None:
            return None
        return pa.schema([field.to_arrow() for field in self.fields])


class UdfReferenceConfig(StrictModel):
    name: str
    version: str

    @model_validator(mode="after")
    def validate_reference(self) -> UdfReferenceConfig:
        RuntimeUdfReference(self.name, self.version)
        return self

    def to_runtime(self) -> RuntimeUdfReference:
        return RuntimeUdfReference(self.name, self.version)


class DataFusionConfig(StrictModel):
    batch_size: Annotated[int, Field(gt=0, le=1_000_000)] = 8192
    target_partitions: Annotated[int, Field(gt=0, le=256)] = 1
    repartition_aggregations: bool = True
    repartition_joins: bool = True
    repartition_sorts: bool = True
    repartition_windows: bool = True

    def to_runtime(self) -> RuntimeDataFusionConfig:
        return RuntimeDataFusionConfig(**self.model_dump())


class NodeConfig(StrictModel):
    id: Annotated[str, Field(pattern=_ID_PATTERN)]
    kind: NodeKind
    expression: str | None = None
    select: tuple[str, ...] = ()
    filter_expression: str | None = None
    query: str | None = None
    inputs: tuple[Annotated[str, Field(pattern=_PORT_PATTERN)], ...] = ()
    backend: Literal["numpy", "jax"] | None = None
    udfs: tuple[UdfReferenceConfig, ...] = ()
    input_ports: tuple[PortConfig, ...] = ()
    output_ports: tuple[PortConfig, ...] = ()
    position: PositionConfig = PositionConfig()

    @model_validator(mode="after")
    def validate_node_mode(self) -> NodeConfig:
        if self.kind is NodeKind.EXPRESSION:
            if (self.expression is None) == (not self.select):
                msg = "expression nodes require exactly one expression or select list"
                raise ValueError(msg)
            if self.query is not None or self.inputs or self.backend is not None:
                msg = "expression nodes cannot configure SQL inputs, query, or backend"
                raise ValueError(msg)
            expected_inputs = {"input"}
            expected_kind = BatchKind.TABLE
        elif self.kind is NodeKind.SQL:
            if not self.query or not self.inputs:
                msg = "SQL nodes require a query and at least one input alias"
                raise ValueError(msg)
            if self.expression is not None or self.select or self.backend is not None:
                msg = "SQL nodes cannot configure expression, select, or backend"
                raise ValueError(msg)
            validate_datafusion_query(self.query)
            expected_inputs = set(self.inputs)
            expected_kind = BatchKind.TABLE
        else:
            if not self.expression or self.backend is None:
                msg = "array expression nodes require an expression and backend"
                raise ValueError(msg)
            if self.select or self.query is not None or self.inputs:
                msg = "array expression nodes cannot configure table query options"
                raise ValueError(msg)
            expected_inputs = {"input"}
            expected_kind = BatchKind.ARRAY

        self._validate_ports(
            self.input_ports,
            expected=expected_inputs,
            expected_kind=expected_kind,
            direction="input",
        )
        self._validate_ports(
            self.output_ports,
            expected={"output"},
            expected_kind=expected_kind,
            direction="output",
        )
        return self

    @staticmethod
    def _validate_ports(
        ports: tuple[PortConfig, ...],
        *,
        expected: set[str],
        expected_kind: BatchKind,
        direction: str,
    ) -> None:
        if not ports:
            return
        names = [port.name for port in ports]
        if len(names) != len(set(names)):
            msg = f"node {direction} port names must be unique"
            raise ValueError(msg)
        if set(names) != expected:
            msg = f"node {direction} ports must be {sorted(expected)}"
            raise ValueError(msg)
        if any(port.kind is not expected_kind for port in ports):
            msg = f"node {direction} ports must use {expected_kind.value} batches"
            raise ValueError(msg)

    def schema_for(self, port_name: str, *, output: bool = False) -> pa.Schema | None:
        ports = self.output_ports if output else self.input_ports
        for port in ports:
            if port.name == port_name:
                return port.arrow_schema()
        return None


class EdgeConfig(StrictModel):
    source_node: Annotated[str, Field(pattern=_ID_PATTERN)]
    target_node: Annotated[str, Field(pattern=_ID_PATTERN)]
    source_port: Annotated[str, Field(pattern=_PORT_PATTERN)] = "output"
    target_port: Annotated[str, Field(pattern=_PORT_PATTERN)] = "input"


class PipelineConfig(StrictModel):
    id: Annotated[str, Field(pattern=_ID_PATTERN)]
    name: Annotated[str, Field(min_length=1, max_length=120)]
    nodes: tuple[NodeConfig, ...]
    edges: tuple[EdgeConfig, ...] = ()
    datafusion: DataFusionConfig = DataFusionConfig()

    @model_validator(mode="after")
    def validate_graph_identity(self) -> PipelineConfig:
        if not self.nodes:
            msg = "pipeline requires at least one node"
            raise ValueError(msg)
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            msg = "pipeline node IDs must be unique"
            raise ValueError(msg)
        return self


class DataSourceConfig(StrictModel):
    id: Annotated[str, Field(pattern=_ID_PATTERN)]
    input_name: Annotated[str, Field(min_length=1, max_length=160)] = "input"
    format: InputFormat
    data: JSONValue
    source_id: str | None = None


class RunOptions(StrictModel):
    max_input_bytes: Annotated[int, Field(gt=0, le=MAX_PREVIEW_BYTES)] = (
        MAX_PREVIEW_BYTES
    )
    max_rows: Annotated[int, Field(gt=0, le=MAX_PREVIEW_ROWS)] = MAX_PREVIEW_ROWS
    timeout_seconds: Annotated[float, Field(gt=0, le=MAX_PREVIEW_SECONDS)] = (
        MAX_PREVIEW_SECONDS
    )
    memory_limit_mb: Annotated[int, Field(ge=64, le=2048)] = 512
    output_rows: Annotated[int, Field(gt=0, le=10_000)] = 1000


class ProjectConfig(StrictModel):
    format_version: Literal[CONFIG_FORMAT_VERSION] = CONFIG_FORMAT_VERSION
    id: Annotated[str, Field(pattern=_ID_PATTERN)]
    name: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=2000)] = ""
    pipeline: PipelineConfig
    data_sources: tuple[DataSourceConfig, ...] = ()
    run_options: RunOptions = RunOptions()

    @model_validator(mode="after")
    def validate_source_identity(self) -> ProjectConfig:
        source_ids = [source.id for source in self.data_sources]
        if len(source_ids) != len(set(source_ids)):
            msg = "project data source IDs must be unique"
            raise ValueError(msg)
        input_names = [source.input_name for source in self.data_sources]
        if len(input_names) != len(set(input_names)):
            msg = "project data source input names must be unique"
            raise ValueError(msg)
        return self


class ValidationIssue(StrictModel):
    code: str
    message: str
    path: str | None = None


class ValidationReport(StrictModel):
    valid: bool
    errors: tuple[ValidationIssue, ...] = ()
    warnings: tuple[ValidationIssue, ...] = ()
    fingerprint: str | None = None
    graph_inputs: tuple[str, ...] = ()
    graph_outputs: tuple[str, ...] = ()


def arrow_type(name: str) -> pa.DataType:
    if not isinstance(name, str) or name not in SUPPORTED_ARROW_TYPES:
        msg = f"unsupported Arrow type {name!r}"
        raise ValueError(msg)
    try:
        return pa.type_for_alias(name)
    except ValueError as error:
        msg = f"unsupported Arrow type {name!r}"
        raise ValueError(msg) from error


def _runtime_references(
    references: tuple[UdfReferenceConfig, ...],
) -> tuple[RuntimeUdfReference, ...]:
    return tuple(reference.to_runtime() for reference in references)


def compile_project(
    project: ProjectConfig,
    *,
    udf_registry: UdfRegistry | UdfRegistrySnapshot | None = None,
) -> ExecutionPlan:
    """Compile a data-only project configuration into a validated runtime plan."""
    pipeline = Pipeline(
        project.pipeline.name,
        datafusion_config=project.pipeline.datafusion.to_runtime(),
        udf_registry=udf_registry,
    )
    for node in project.pipeline.nodes:
        references = _runtime_references(node.udfs)
        if node.kind is NodeKind.EXPRESSION:
            operator = ExpressionOperator(
                node.id,
                node.expression,
                select=node.select or None,
                filter_expression=node.filter_expression,
                udfs=references,
                input_schema=node.schema_for("input"),
                output_schema=node.schema_for("output", output=True),
            )
        elif node.kind is NodeKind.SQL:
            operator = SqlOperator(
                node.id,
                node.query or "",
                inputs=node.inputs,
                udfs=references,
                input_schemas={
                    name: schema
                    for name in node.inputs
                    if (schema := node.schema_for(name)) is not None
                },
                output_schema=node.schema_for("output", output=True),
            )
        else:
            operator = ArrayExpressionOperator(
                node.id,
                node.expression or "",
                backend=node.backend or "numpy",
                udfs=references,
            )
        pipeline.add_node(node.id, operator)

    for edge in project.pipeline.edges:
        pipeline.connect(
            edge.source_node,
            edge.target_node,
            source_port=edge.source_port,
            target_port=edge.target_port,
        )
    return pipeline.compile()


def validate_project(
    project: ProjectConfig,
    *,
    udf_registry: UdfRegistry | UdfRegistrySnapshot | None = None,
) -> ValidationReport:
    try:
        plan = compile_project(project, udf_registry=udf_registry)
    except Exception as error:
        return ValidationReport(
            valid=False,
            errors=(
                ValidationIssue(
                    code=type(error).__name__,
                    message=str(error),
                ),
            ),
        )
    if project.data_sources:
        source_inputs = {source.input_name for source in project.data_sources}
        if source_inputs != set(plan.graph_inputs):
            return ValidationReport(
                valid=False,
                errors=(
                    ValidationIssue(
                        code="source_input_mismatch",
                        message=(
                            f"saved source inputs must be {sorted(plan.graph_inputs)}; "
                            f"configured {sorted(source_inputs)}"
                        ),
                        path="data_sources",
                    ),
                ),
            )
    warnings = ()
    if not project.data_sources:
        warnings = (
            ValidationIssue(
                code="no_sample_data",
                message="project has no saved sample data source",
            ),
        )
    return ValidationReport(
        valid=True,
        warnings=warnings,
        fingerprint=plan.fingerprint,
        graph_inputs=tuple(plan.graph_inputs),
        graph_outputs=tuple(plan.graph_outputs),
    )


def project_json_schema() -> Mapping[str, Any]:
    return ProjectConfig.model_json_schema()
