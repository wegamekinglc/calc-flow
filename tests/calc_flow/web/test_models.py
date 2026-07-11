from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from calc_flow.web.models import (
    CheckpointSummary,
    InputPayload,
    RunResponse,
    RunStatus,
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
