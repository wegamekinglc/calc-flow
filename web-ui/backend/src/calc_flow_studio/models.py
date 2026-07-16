from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Literal

from calc_flow import ProjectDocument
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class ProjectCreateRequest(ProjectDocument):
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
    source_cursor: JSONValue = None
    created_at: datetime | None = None
    state_nodes: tuple[str, ...] = ()

    @field_validator("source_cursor", mode="before")
    @classmethod
    def copy_source_cursor(cls, value: object) -> JSONValue:
        return _copy_json_value(value)


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

    @field_validator("result", mode="before")
    @classmethod
    def copy_result(cls, value: object) -> JSONValue:
        return _copy_json_value(value)
