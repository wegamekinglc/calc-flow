from __future__ import annotations

from datetime import UTC, datetime

import pytest
from calc_flow.config import DataSourceConfig, NodeConfig, PipelineConfig
from pydantic import ValidationError

from calc_flow_studio.models import (
    CheckpointSummary,
    InputPayload,
    ProjectCreateRequest,
    RunResponse,
    RunStatus,
)


def test_project_create_request_assigns_only_server_id() -> None:
    request = ProjectCreateRequest(
        name="Studio project",
        pipeline=PipelineConfig(
            id="main",
            name="Main",
            nodes=(
                NodeConfig(
                    id="calculate",
                    kind="expression",
                    expression="result = value + 1",
                ),
            ),
        ),
    )

    project = request.to_project("project_0123456789abcdef0123456789abcdef")

    assert project.id == "project_0123456789abcdef0123456789abcdef"
    assert project.name == "Studio project"
    with pytest.raises(ValidationError):
        ProjectCreateRequest.model_validate(
            {**request.model_dump(mode="json"), "id": "client_chosen"}
        )


def test_project_create_request_enforces_project_level_source_identity() -> None:
    source = DataSourceConfig(
        id="sample",
        input_name="input",
        format="inline_json",
        data=[{"value": 1}],
    )

    with pytest.raises(ValidationError, match="source IDs must be unique"):
        ProjectCreateRequest(
            name="Invalid sources",
            pipeline=PipelineConfig(
                id="main",
                name="Main",
                nodes=(
                    NodeConfig(
                        id="calculate",
                        kind="expression",
                        expression="result = value + 1",
                    ),
                ),
            ),
            data_sources=(source, source),
        )


def test_input_payload_rejects_executable_objects() -> None:
    with pytest.raises(ValidationError):
        InputPayload(format="inline_json", data=object())


def test_run_response_serializes_status_and_timestamps() -> None:
    response = RunResponse(
        id="run",
        project_id="project",
        status=RunStatus.COMPLETED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        result={"outputs": {}},
    )

    data = response.model_dump(mode="json")

    assert data["status"] == "completed"
    assert data["created_at"] == "2026-01-01T00:00:00Z"


def test_checkpoint_summary_serializes_recovery_metadata() -> None:
    summary = CheckpointSummary(
        pipeline_name="Main",
        exists=True,
        compatible=True,
        sequence=2,
        source_cursor={"offset": 10},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        state_nodes=("counter",),
    )

    data = summary.model_dump(mode="json")

    assert data["source_cursor"] == {"offset": 10}
    assert data["state_nodes"] == ["counter"]
