from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import TypeAdapter, ValidationError

from calc_flow_studio.models import (
    CapabilitiesResponse,
    InputPayload,
    ProjectCreateRequest,
    ProjectSummary,
    RunOptions,
    RunRequest,
    RunResponse,
    RunStatus,
    ValidationReport,
)


def _project() -> dict[str, object]:
    return {
        "format_version": 3,
        "id": "project_alpha",
        "name": "Alpha",
        "description": "A v3 project",
        "runtime": {"mode": "batch", "options": {}},
        "graph": {
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


def test_project_create_request_uses_rust_structural_validation_only() -> None:
    request = ProjectCreateRequest.model_validate(_project())
    copied = request.to_project()
    copied.root["name"] = "Changed"

    assert request.root["format_version"] == 3
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
        "graph": {"name": "empty", "nodes": []},
    }
    assert not ProjectCreateRequest.model_validate(semantically_invalid).root["graph"][
        "nodes"
    ]


def test_transport_models_are_strict_and_frozen() -> None:
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

    assert (
        InputPayload(format="records", data=[], source_id="a" * 64).source_id
        == "a" * 64
    )
    for source_id in ("", "a" * 65, "not.portable", "é"):
        with pytest.raises(ValidationError):
            InputPayload(format="records", data=[], source_id=source_id)


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
        ("max_input_bytes", 10 * 1024 * 1024 + 1),
        ("max_rows", 0),
        ("max_rows", 100_001),
        ("output_rows", 10_001),
    ):
        with pytest.raises(ValidationError):
            RunOptions.model_validate({field: value})


def test_capabilities_response_is_a_closed_camel_case_v1_contract() -> None:
    document = {
        "schemaVersion": 1,
        "runtime": {
            "scope": {
                "kind": "runtimeSession",
                "sessionId": "session",
                "revision": 0,
            },
            "packageVersion": "4.0.0",
            "projectFormatVersions": [3],
            "batchKinds": ["array", "table"],
            "portableArrowTypes": ["int64"],
            "operators": [],
            "udfs": [],
            "providers": [],
            "connectors": [],
        },
        "preview": {
            "inputBatchKinds": ["table"],
            "requestInputFormats": ["arrow_ipc", "columns", "records"],
            "projectInputFormats": ["arrow_ipc", "csv", "inline_json", "json"],
            "workerRegistrations": [],
            "limits": {
                "maxInputBytes": {
                    "default": 10 * 1024 * 1024,
                    "minimum": 1,
                    "maximum": 10 * 1024 * 1024,
                },
                "maxRows": {
                    "default": 100_000,
                    "minimum": 1,
                    "maximum": 100_000,
                },
                "timeoutSeconds": {
                    "default": 30,
                    "minimum": 1,
                    "maximum": 300,
                },
                "memoryLimitMb": {
                    "default": 512,
                    "minimum": 64,
                    "maximum": 4096,
                },
                "outputRows": {
                    "default": 1000,
                    "minimum": 1,
                    "maximum": 10_000,
                },
            },
        },
    }

    response = CapabilitiesResponse.model_validate(document)

    assert response.model_dump(mode="json", by_alias=True) == document
    with pytest.raises(ValidationError):
        CapabilitiesResponse.model_validate({**document, "optionalFutureField": True})
    with pytest.raises(ValidationError):
        CapabilitiesResponse.model_validate({**document, "schemaVersion": 2})


def test_validation_report_discriminator_enforces_status_invariants() -> None:
    adapter = TypeAdapter(ValidationReport)

    valid = adapter.validate_python(
        {
            "kind": "valid",
            "valid": True,
            "issues": [],
            "fingerprint": "fingerprint",
        }
    )
    invalid = adapter.validate_python(
        {
            "kind": "invalid",
            "valid": False,
            "issues": [
                {
                    "path": "pipeline.nodes[0]",
                    "code": "invalid_expression",
                    "message": "bad expression",
                }
            ],
            "fingerprint": None,
        }
    )

    assert valid.kind == "valid"
    assert invalid.kind == "invalid"
    for malformed in (
        {
            "kind": "valid",
            "valid": True,
            "issues": [{"path": "x", "code": "bad", "message": "bad"}],
            "fingerprint": "fingerprint",
        },
        {
            "kind": "invalid",
            "valid": False,
            "issues": [],
            "fingerprint": None,
        },
        {
            "kind": "valid",
            "valid": False,
            "issues": [],
            "fingerprint": None,
        },
    ):
        with pytest.raises(ValidationError):
            adapter.validate_python(malformed)


def test_project_summary_is_a_json_transport_value() -> None:
    project = ProjectSummary(
        id="project_alpha", name="Alpha", description="A v3 project", node_count=1
    )

    assert project.node_count == 1


def test_run_response_serializes_status_and_timestamps() -> None:
    result = {
        "outputs": {},
        "node_timings": {},
        "datafusion_metrics": [],
        "metadata": {"values": [1]},
    }
    response = RunResponse.model_validate(
        {
            "id": "run",
            "project_id": "project_alpha",
            "status": RunStatus.COMPLETED,
            "created_at": datetime(2026, 1, 1, tzinfo=UTC),
            "started_at": datetime(2026, 1, 1, tzinfo=UTC),
            "finished_at": datetime(2026, 1, 1, tzinfo=UTC),
            "error": None,
            "result": result,
        }
    )
    result["metadata"]["values"].append(2)

    data = response.model_dump(mode="json")

    assert data["status"] == "completed"
    assert data["created_at"] == "2026-01-01T00:00:00Z"
    assert data["result"]["metadata"] == {"values": [1]}

    for malformed in (
        {
            **data,
            "status": "completed",
            "result": None,
        },
        {
            **data,
            "status": "failed",
            "result": None,
            "error": "",
        },
        {
            **data,
            "status": "pending",
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": None,
            "result": None,
        },
    ):
        with pytest.raises(ValidationError):
            RunResponse.model_validate(malformed)


@pytest.mark.parametrize("invalid", [object(), float("nan"), float("inf")])
def test_transport_json_values_reject_non_json_and_non_finite_values(
    invalid: object,
) -> None:
    with pytest.raises(ValidationError):
        InputPayload(format="records", data={"value": invalid})
    with pytest.raises(ValidationError):
        RunResponse(
            id="run",
            project_id="project",
            status=RunStatus.COMPLETED,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            result={"value": invalid},
        )
