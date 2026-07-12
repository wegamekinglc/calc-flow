from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from calc_flow.batch import JSONValue
from calc_flow.config import (
    CONFIG_FORMAT_VERSION,
    DataSourceConfig,
    InputFormat,
    PipelineConfig,
    ProjectConfig,
    RunOptions,
    StrictModel,
)
from pydantic import Field, model_validator


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class InputPayload(StrictModel):
    format: InputFormat
    data: JSONValue
    source_id: str | None = None


class RunRequest(StrictModel):
    inputs: dict[str, InputPayload] = Field(default_factory=dict)
    options: RunOptions | None = None


class ProjectCreateRequest(StrictModel):
    format_version: Literal[CONFIG_FORMAT_VERSION] = CONFIG_FORMAT_VERSION
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    pipeline: PipelineConfig
    data_sources: tuple[DataSourceConfig, ...] = ()
    run_options: RunOptions = RunOptions()

    @model_validator(mode="after")
    def validate_project_content(self) -> Self:
        ProjectConfig(id="project_validation", **self.model_dump())
        return self

    def to_project(self, project_id: str) -> ProjectConfig:
        return ProjectConfig(id=project_id, **self.model_dump())


class ProjectSummary(StrictModel):
    id: str
    name: str
    description: str
    node_count: int


class CheckpointSummary(StrictModel):
    pipeline_name: str
    exists: bool
    compatible: bool | None = None
    pipeline_fingerprint: str | None = None
    sequence: int | None = None
    source_cursor: JSONValue = None
    created_at: datetime | None = None
    state_nodes: tuple[str, ...] = ()


class RunEvent(StrictModel):
    sequence: int
    timestamp: datetime
    type: str
    message: str


class RunResponse(StrictModel):
    id: str
    project_id: str
    status: RunStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: dict[str, JSONValue] | None = None


class CatalogResponse(StrictModel):
    config_format_version: str
    operators: tuple[dict[str, JSONValue], ...]
    udfs: tuple[dict[str, JSONValue], ...]
    arrow_types: tuple[str, ...]
    limits: dict[str, JSONValue]
