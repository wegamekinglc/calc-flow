from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from calc_flow import ProjectDocument
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictBool,
    StrictInt,
    field_validator,
)

type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)
_MAX_JSON_DEPTH = 32


def _copy_json_value(value: object, *, depth: int = 0) -> JSONValue:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError(
            f"transport value exceeds the maximum JSON depth of {_MAX_JSON_DEPTH}"
        )
    if value is None or type(value) in {bool, int, str}:
        return value  # type: ignore[return-value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("transport JSON numbers must be finite")
        return value
    if type(value) is list:
        return [_copy_json_value(item, depth=depth + 1) for item in value]
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise ValueError("transport JSON object keys must be strings")
        return {
            key: _copy_json_value(item, depth=depth + 1) for key, item in value.items()
        }
    raise ValueError(
        f"transport contains a non-JSON value of type {type(value).__name__}"
    )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _camel_case(name: str) -> str:
    head, *tail = name.split("_")
    return head + "".join(part.capitalize() for part in tail)


class CapabilityModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_camel_case,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )


class RuntimeSessionScopeResponse(CapabilityModel):
    kind: Literal["runtimeSession"]
    session_id: str
    revision: int = Field(ge=0)


class OperatorCapabilityResponse(CapabilityModel):
    kind: str
    input_kinds: tuple[Literal["table", "array"], ...]
    output_kinds: tuple[Literal["table", "array"], ...]
    requires_datafusion: bool


class UdfCapabilityResponse(CapabilityModel):
    provider: str
    name: str
    version: str
    kind: Literal["data_fusion_scalar"]
    input_types: tuple[str, ...]
    return_type: str
    volatility: str


class ProviderPortResponse(CapabilityModel):
    name: str
    kind: Literal["table", "array"]
    required: bool


class ProviderOptionResponse(CapabilityModel):
    name: str
    value_type: Literal["string", "integer", "number", "boolean"]
    required: bool = False


class ProviderOptionsSchemaResponse(CapabilityModel):
    fields: tuple[ProviderOptionResponse, ...] = ()
    additional_properties: Literal[False] = False


class ProviderCapabilityResponse(CapabilityModel):
    provider: str
    name: str
    version: str
    input_ports: tuple[ProviderPortResponse, ...]
    output_ports: tuple[ProviderPortResponse, ...]
    options_schema: ProviderOptionsSchemaResponse | None


class ConnectorAxesResponse(CapabilityModel):
    delivery: Literal["best_effort", "at_least_once", "exactly_once"]
    replay: Literal["replayable_exact", "unreplayable"]
    watermark: Literal["native", "generated_only"]
    transaction: Literal[
        "none", "pre_commit_commit", "ledger_idempotent", "retry_deduplicated"
    ]
    snapshot: bool
    polling: bool
    cdc: bool
    lookup: bool


class ConnectorCapabilityResponse(CapabilityModel):
    provider: str
    name: str
    version: str
    kind: Literal["source", "sink", "both"]
    capabilities: ConnectorAxesResponse
    formats: tuple[str, ...]
    options_schema: dict[str, object]


class RuntimeCapabilitiesResponse(CapabilityModel):
    scope: RuntimeSessionScopeResponse
    package_version: str
    project_format_versions: tuple[int, ...]
    batch_kinds: tuple[Literal["table", "array"], ...]
    portable_arrow_types: tuple[str, ...]
    operators: tuple[OperatorCapabilityResponse, ...]
    udfs: tuple[UdfCapabilityResponse, ...]
    providers: tuple[ProviderCapabilityResponse, ...]
    connectors: tuple[ConnectorCapabilityResponse, ...]


class SerializedWorkerRegistration(CapabilityModel):
    reconstruction: Literal["serialized"]
    registration_kind: Literal["provider", "dataFusionScalar"]
    provider: str
    name: str
    version: str


class LazyBuiltinWorkerRegistration(CapabilityModel):
    reconstruction: Literal["lazyBuiltin"]
    registration_kind: Literal["provider", "dataFusionScalar"]
    provider: str
    name: str
    version: str


class UnavailableWorkerRegistration(CapabilityModel):
    reconstruction: Literal["unavailable"]
    registration_kind: Literal["provider", "dataFusionScalar"]
    provider: str
    name: str
    version: str
    reason_code: Literal["serializationFailed"]


type WorkerRegistrationCapability = Annotated[
    SerializedWorkerRegistration
    | LazyBuiltinWorkerRegistration
    | UnavailableWorkerRegistration,
    Field(discriminator="reconstruction"),
]


class PreviewLimit(CapabilityModel):
    default: int
    minimum: int
    maximum: int


class PreviewLimitsResponse(CapabilityModel):
    max_input_bytes: PreviewLimit
    max_rows: PreviewLimit
    timeout_seconds: PreviewLimit
    memory_limit_mb: PreviewLimit
    output_rows: PreviewLimit


class PreviewCapabilitiesResponse(CapabilityModel):
    input_batch_kinds: tuple[Literal["table", "array"], ...]
    request_input_formats: tuple[Literal["arrow_ipc", "columns", "records"], ...]
    project_input_formats: tuple[
        Literal["arrow_ipc", "csv", "inline_json", "json"], ...
    ]
    worker_registrations: tuple[WorkerRegistrationCapability, ...]
    limits: PreviewLimitsResponse


class CapabilitiesResponse(CapabilityModel):
    schema_version: Literal[1]
    runtime: RuntimeCapabilitiesResponse
    preview: PreviewCapabilitiesResponse


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class InputPayload(StrictModel):
    format: Literal["records", "columns", "arrow_ipc"]
    data: JSONValue
    source_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]*$",
    )

    @field_validator("data", mode="before")
    @classmethod
    def copy_data(cls, value: object) -> JSONValue:
        return _copy_json_value(value)


_MAX_INPUT_BYTES = 10 * 1024 * 1024
_MAX_ROWS = 100_000


class RunOptions(StrictModel):
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    memory_limit_mb: int = Field(default=512, ge=64, le=4096)
    max_input_bytes: int = Field(default=_MAX_INPUT_BYTES, ge=1, le=_MAX_INPUT_BYTES)
    max_rows: int = Field(default=_MAX_ROWS, ge=1, le=_MAX_ROWS)
    output_rows: int = Field(default=1000, ge=1, le=10_000)


class RunRequest(StrictModel):
    inputs: dict[str, InputPayload] = Field(default_factory=dict)
    options: RunOptions | None = None


class JobCreateRequest(StrictModel):
    project_id: str = Field(min_length=1, max_length=120)


class ProjectCreateRequest(ProjectDocument):
    def to_project(self) -> ProjectDocument:
        return ProjectDocument.model_validate(self.root)


class ProjectSummary(StrictModel):
    id: str
    name: str
    description: str
    node_count: int


class ValidationIssue(StrictModel):
    path: str
    code: str
    message: str


class ValidValidationReport(StrictModel):
    kind: Literal["valid"] = "valid"
    valid: Literal[True] = True
    issues: tuple[ValidationIssue, ...] = Field(default=(), max_length=0)
    fingerprint: str = Field(min_length=1)


class InvalidValidationReport(StrictModel):
    kind: Literal["invalid"] = "invalid"
    valid: Literal[False] = False
    issues: tuple[ValidationIssue, ...] = Field(min_length=1)
    fingerprint: None = None


type ValidationReport = Annotated[
    ValidValidationReport | InvalidValidationReport,
    Field(discriminator="kind"),
]


class RunEvent(StrictModel):
    sequence: int
    timestamp: datetime
    type: str
    message: str
    state: str | None = None
    epoch: int | None = Field(default=None, ge=0)
    watermark: datetime | None = None
    throughput_rows: int | None = Field(default=None, ge=0)
    queue_envelopes: int | None = Field(default=None, ge=0)
    queue_rows: int | None = Field(default=None, ge=0)
    queue_bytes: int | None = Field(default=None, ge=0)
    backpressure_events: int | None = Field(default=None, ge=0)
    late_rows: int | None = Field(default=None, ge=0)


class OutputFieldPreview(StrictModel):
    name: str
    type: str
    nullable: StrictBool


class TableOutputPreview(StrictModel):
    kind: Literal["table"] = "table"
    total_rows: StrictInt = Field(ge=0)
    truncated: StrictBool
    schema_: tuple[OutputFieldPreview, ...] = Field(
        alias="schema",
        serialization_alias="schema",
    )
    rows: tuple[dict[str, JSONValue], ...]
    metadata: dict[str, JSONValue]

    @field_validator("rows", mode="before")
    @classmethod
    def copy_rows(cls, value: object) -> JSONValue:
        return _copy_json_value(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: object) -> JSONValue:
        return _copy_json_value(value)


class ArrayOutputPreview(StrictModel):
    kind: Literal["array"] = "array"
    backend: str
    total_rows: StrictInt = Field(ge=0)
    truncated: StrictBool
    data: JSONValue
    metadata: dict[str, JSONValue]

    @field_validator("data", "metadata", mode="before")
    @classmethod
    def copy_json(cls, value: object) -> JSONValue:
        return _copy_json_value(value)


type OutputPreview = Annotated[
    TableOutputPreview | ArrayOutputPreview,
    Field(discriminator="kind"),
]


class NodeTimingPreview(StrictModel):
    duration_ns: StrictInt = Field(ge=0)
    input_rows: dict[str, StrictInt]
    output_rows: dict[str, StrictInt]


class DataFusionMetricPreview(StrictModel):
    query_id: StrictInt = Field(ge=0)
    node_id: str | None
    planning_ns: StrictInt = Field(ge=0)
    execution_ns: StrictInt = Field(ge=0)
    output_rows: StrictInt = Field(ge=0)
    logical_plan: str
    physical_plan: str


class RunResultPreview(StrictModel):
    outputs: dict[str, OutputPreview]
    node_timings: dict[str, NodeTimingPreview]
    datafusion_metrics: tuple[DataFusionMetricPreview, ...]
    metadata: dict[str, JSONValue]

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: object) -> JSONValue:
        return _copy_json_value(value)


class RunResponseBase(StrictModel):
    id: str
    project_id: str
    created_at: datetime


class PendingRunResponse(RunResponseBase):
    status: Literal[RunStatus.PENDING]
    started_at: None = None
    finished_at: None = None
    error: None = None
    result: None = None


class RunningRunResponse(RunResponseBase):
    status: Literal[RunStatus.RUNNING]
    started_at: datetime
    finished_at: None = None
    error: None = None
    result: None = None


class CompletedRunResponse(RunResponseBase):
    status: Literal[RunStatus.COMPLETED]
    started_at: datetime
    finished_at: datetime
    error: None = None
    result: RunResultPreview


class FailedRunResponse(RunResponseBase):
    status: Literal[RunStatus.FAILED]
    started_at: datetime
    finished_at: datetime
    error: str = Field(min_length=1)
    result: None = None


class TimedOutRunResponse(RunResponseBase):
    status: Literal[RunStatus.TIMED_OUT]
    started_at: datetime
    finished_at: datetime
    error: str = Field(min_length=1)
    result: None = None


class CancelledRunResponse(RunResponseBase):
    status: Literal[RunStatus.CANCELLED]
    started_at: datetime | None = None
    finished_at: datetime
    error: None = None
    result: None = None


type RunResponseVariant = Annotated[
    PendingRunResponse
    | RunningRunResponse
    | CompletedRunResponse
    | FailedRunResponse
    | TimedOutRunResponse
    | CancelledRunResponse,
    Field(discriminator="status"),
]


class RunResponse(RootModel[RunResponseVariant]):
    model_config = ConfigDict(frozen=True)

    @property
    def id(self) -> str:
        return self.root.id

    @property
    def project_id(self) -> str:
        return self.root.project_id

    @property
    def status(self) -> RunStatus:
        return self.root.status

    @property
    def created_at(self) -> datetime:
        return self.root.created_at

    @property
    def started_at(self) -> datetime | None:
        return self.root.started_at

    @property
    def finished_at(self) -> datetime | None:
        return self.root.finished_at

    @property
    def error(self) -> str | None:
        return self.root.error

    @property
    def result(self) -> RunResultPreview | None:
        return self.root.result


class JobResponseBase(StrictModel):
    id: str
    project_id: str
    created_at: datetime


class PendingJobResponse(JobResponseBase):
    status: Literal[RunStatus.PENDING]
    started_at: None = None
    finished_at: None = None
    error_code: None = None
    error: None = None


class RunningJobResponse(JobResponseBase):
    status: Literal[RunStatus.RUNNING]
    started_at: datetime
    finished_at: None = None
    error_code: None = None
    error: None = None


class CompletedJobResponse(JobResponseBase):
    status: Literal[RunStatus.COMPLETED]
    started_at: datetime
    finished_at: datetime
    error_code: None = None
    error: None = None


class FailedJobResponse(JobResponseBase):
    status: Literal[RunStatus.FAILED]
    started_at: datetime
    finished_at: datetime
    error_code: Literal["job_limit_exceeded", "worker_failed"]
    error: str = Field(min_length=1)


class CancelledJobResponse(JobResponseBase):
    status: Literal[RunStatus.CANCELLED]
    started_at: datetime | None = None
    finished_at: datetime
    error_code: None = None
    error: None = None


type JobResponseVariant = Annotated[
    PendingJobResponse
    | RunningJobResponse
    | CompletedJobResponse
    | FailedJobResponse
    | CancelledJobResponse,
    Field(discriminator="status"),
]


class JobResponse(RootModel[JobResponseVariant]):
    """Typed lifecycle state for one persistent continuous job."""

    model_config = ConfigDict(frozen=True)

    @property
    def id(self) -> str:
        return self.root.id

    @property
    def project_id(self) -> str:
        return self.root.project_id

    @property
    def status(self) -> RunStatus:
        return self.root.status


class ResourceLimits(StrictModel):
    """Hard bounds for long-running continuous jobs."""

    max_concurrent_jobs: int = Field(default=4, ge=1, le=64)
    max_job_resident_memory_bytes: int = Field(
        default=1024 * 1024 * 1024, ge=64 * 1024 * 1024
    )
    max_global_resident_memory_bytes: int = Field(
        default=4 * 1024 * 1024 * 1024, ge=256 * 1024 * 1024
    )
    max_checkpoint_disk_bytes: int = Field(
        default=512 * 1024 * 1024, ge=16 * 1024 * 1024
    )
    job_lifecycle: Literal["user_explicit_stop"] = "user_explicit_stop"
