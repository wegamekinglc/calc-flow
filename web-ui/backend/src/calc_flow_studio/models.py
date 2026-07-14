from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from calc_flow import ProjectDocument, Runtime
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class InputPayload(StrictModel):
    format: Literal["records", "columns", "arrow_ipc"]
    data: object
    source_id: str | None = None


class RunOptions(StrictModel):
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    memory_limit_mb: int = Field(default=512, ge=64, le=4096)
    max_input_bytes: int = Field(default=10 * 1024 * 1024, ge=1)
    max_rows: int = Field(default=100_000, ge=1)
    output_rows: int = Field(default=1000, ge=1, le=10_000)


class RunRequest(StrictModel):
    inputs: dict[str, InputPayload] = Field(default_factory=dict)
    options: RunOptions | None = None


class ProjectCreateRequest(ProjectDocument):
    @model_validator(mode="after")
    def validate_complete_project(self) -> Self:
        report = Runtime().validation_report(self.canonical_json())
        if not report["valid"]:
            issues = report.get("issues", [])
            details = "; ".join(
                str(issue.get("message", "invalid project")) for issue in issues
            )
            raise ValueError(details or "invalid project")
        return self

    def to_project(self) -> ProjectDocument:
        return ProjectDocument.model_validate(self.root)


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
    source_cursor: object = None
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
    result: dict[str, object] | None = None
