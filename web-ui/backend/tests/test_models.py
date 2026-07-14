from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from calc_flow_studio.models import (
    CheckpointSummary,
    InputPayload,
    ProjectCreateRequest,
    ProjectSummary,
    RunOptions,
    RunRequest,
    RunResponse,
    RunStatus,
)


def _project() -> dict[str, object]:
    return {
        "format_version": 2,
        "id": "project_alpha",
        "name": "Alpha",
        "description": "A v2 project",
        "pipeline": {
            "name": "Alpha pipeline",
            "nodes": [
                {
                    "id": "calculate",
                    "operator": {
                        "kind": "expression",
                        "expression": "result = value + 1",
                    },
                }
            ],
        },
        "data_sources": [
            {
                "id": "sample",
                "input": "input",
                "format": "inline_json",
                "data": [{"value": 1}],
            }
        ],
    }


def test_project_create_request_delegates_complete_validation_to_rust() -> None:
    request = ProjectCreateRequest.model_validate(_project())
    copied = request.to_project()
    copied.root["name"] = "Changed"

    assert request.root["format_version"] == 2
    assert request.root["id"] == "project_alpha"
    assert request.root["name"] == "Alpha"

    for invalid in (
        {**_project(), "format_version": 1},
        {**_project(), "unknown": True},
    ):
        with pytest.raises(ValidationError):
            ProjectCreateRequest.model_validate(invalid)

    with pytest.raises(ValidationError):
        ProjectCreateRequest.model_validate({**_project(), "description": object()})

    semantically_invalid = {
        **_project(),
        "pipeline": {"name": "empty", "nodes": []},
    }
    assert not ProjectCreateRequest.model_validate(semantically_invalid).root[
        "pipeline"
    ]["nodes"]


def test_transport_models_are_strict_frozen_and_v2_only() -> None:
    data = [{"value": [1]}]
    payload = InputPayload(format="records", data=data, source_id="sample")
    request = RunRequest(inputs={"input": payload})
    data[0]["value"].append(2)

    assert request.inputs["input"].format == "records"
    assert request.inputs["input"].data == [{"value": [1]}]
    with pytest.raises(ValidationError):
        InputPayload(format="inline_json", data=[])
    with pytest.raises(ValidationError):
        InputPayload(format="records", data=[], executable="no")
    with pytest.raises(ValidationError):
        RunRequest(inputs={}, unknown=True)
    with pytest.raises(ValidationError):
        request.options = RunOptions()  # type: ignore[misc]


def test_run_options_enforce_preview_limits() -> None:
    assert RunOptions().model_dump() == {
        "timeout_seconds": 30,
        "memory_limit_mb": 512,
        "max_input_bytes": 10 * 1024 * 1024,
        "max_rows": 100_000,
        "output_rows": 1000,
    }

    for field, value in (
        ("timeout_seconds", 301),
        ("memory_limit_mb", 32),
        ("max_input_bytes", 0),
        ("max_rows", 0),
        ("output_rows", 10_001),
    ):
        with pytest.raises(ValidationError):
            RunOptions.model_validate({field: value})


def test_project_and_checkpoint_summaries_are_json_transport_values() -> None:
    source_cursor = {"offsets": [10]}
    project = ProjectSummary(
        id="project_alpha", name="Alpha", description="A v2 project", node_count=1
    )
    checkpoint = CheckpointSummary(
        pipeline_name="Alpha pipeline",
        exists=True,
        compatible=True,
        pipeline_fingerprint="fingerprint",
        sequence=2,
        source_cursor=source_cursor,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        state_nodes=("counter",),
    )
    source_cursor["offsets"].append(11)

    assert project.node_count == 1
    assert checkpoint.model_dump(mode="json")["source_cursor"] == {"offsets": [10]}
    assert checkpoint.model_dump(mode="json")["created_at"] == "2026-01-01T00:00:00Z"


def test_run_response_serializes_status_and_timestamps() -> None:
    result = {"outputs": {"rows": [1]}}
    response = RunResponse(
        id="run",
        project_id="project_alpha",
        status=RunStatus.COMPLETED,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        result=result,
    )
    result["outputs"]["rows"].append(2)

    data = response.model_dump(mode="json")

    assert data["status"] == "completed"
    assert data["created_at"] == "2026-01-01T00:00:00Z"
    assert data["result"] == {"outputs": {"rows": [1]}}


@pytest.mark.parametrize("invalid", [object(), float("nan"), float("inf")])
def test_transport_json_values_reject_non_json_and_non_finite_values(
    invalid: object,
) -> None:
    with pytest.raises(ValidationError):
        InputPayload(format="records", data={"value": invalid})
    with pytest.raises(ValidationError):
        CheckpointSummary(
            pipeline_name="pipeline", exists=True, source_cursor={"value": invalid}
        )
    with pytest.raises(ValidationError):
        RunResponse(
            id="run",
            project_id="project",
            status=RunStatus.COMPLETED,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            result={"value": invalid},
        )
