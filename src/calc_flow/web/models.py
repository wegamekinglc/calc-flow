from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from calc_flow.batch import JSONValue
from calc_flow.config import InputFormat, RunOptions, StrictModel


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
