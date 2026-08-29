from __future__ import annotations

import json
import queue
import time
from pathlib import Path

import pytest
from calc_flow import ProjectDocument
from fastapi.testclient import TestClient

import calc_flow_studio.run_manager as run_manager_module
from calc_flow_studio.app import create_app
from calc_flow_studio.models import ResourceLimits, RunStatus
from calc_flow_studio.run_manager import RunManager, RunManagerError


def _stream_project(
    tmp_path: Path, project_id: str = "stream_project"
) -> dict[str, object]:
    return {
        "format_version": 3,
        "id": project_id,
        "name": "Stream project",
        "description": "A project-v3 continuous job",
        "runtime": {
            "mode": "stream",
            "options": {"checkpoint_interval_ms": 60_000},
        },
        "graph": {
            "name": "stream-project",
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
        "sources": [
            {
                "binding": "input",
                "connector": {
                    "provider": "calc-flow-connectors",
                    "name": "file",
                    "version": "2.0.0",
                },
                "options": {
                    "path": str(tmp_path / "input.json"),
                    "format": "json",
                },
                "watermark": {"policy": "disabled"},
            }
        ],
        "sinks": [
            {
                "binding": "output",
                "connector": {
                    "provider": "calc-flow-connectors",
                    "name": "file",
                    "version": "2.0.0",
                },
                "options": {"path": str(tmp_path), "output": "results"},
                "delivery": "at_least_once",
            }
        ],
        "state": {"root": str(tmp_path / "state"), "retention": 3},
    }


def _batch_project() -> dict[str, object]:
    return {
        "format_version": 3,
        "id": "batch_project",
        "name": "Batch project",
        "runtime": {"mode": "batch", "options": {}},
        "graph": {
            "name": "batch",
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
                "id": "fixture",
                "input": "input",
                "format": "inline_json",
                "data": [{"value": 1}],
            }
        ],
    }


def _controlled_worker(project_json: str, commands: object, output: object) -> None:
    json.loads(project_json)
    output.put({"kind": "state", "state": "running"})
    output.put(
        {
            "kind": "progress",
            "state": "running",
            "epoch": 0,
            "watermark": "2026-08-19T00:00:00Z",
            "throughput_rows": 7,
            "backpressure_events": 2,
            "late_rows": 1,
        }
    )
    while True:
        try:
            command = commands.get(timeout=1)
        except queue.Empty:
            continue
        if command == "checkpoint":
            output.put({"kind": "checkpoint", "epoch": 1})
        elif command == "shutdown":
            output.put(
                {
                    "kind": "terminal",
                    "state": "completed",
                    "cause": "graceful_shutdown",
                }
            )
            return
        elif command == "cancel":
            output.put(
                {
                    "kind": "terminal",
                    "state": "cancelled",
                    "cause": "explicit_cancel",
                }
            )
            return


def _wait_for_status(
    manager: RunManager,
    job_id: str,
    expected: RunStatus,
    *,
    timeout: float = 3,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager.get_job(job_id).status is expected:
            return
        time.sleep(0.01)
    raise AssertionError(
        f"job {job_id} did not reach {expected.value}: "
        f"{manager.get_job(job_id).model_dump(mode='json')}"
    )


def test_continuous_progress_formats_the_native_aggregate_watermark() -> None:
    progress = run_manager_module._continuous_progress(
        {
            "state": "running",
            "watermark_micros": 1_787_097_600_123_456,
            "sources": {},
            "operators": {},
            "edges": {},
            "checkpoint": {},
        }
    )

    assert progress["watermark"] == "2026-08-19T00:00:00.123456Z"


@pytest.fixture()
def controlled_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunManager:
    monkeypatch.setattr(
        run_manager_module, "_execute_continuous_worker", _controlled_worker
    )
    return RunManager(
        use_processes=False,
        checkpoint_directory=tmp_path / "checkpoints",
    )


def test_openapi_exposes_only_project_v3_and_continuous_job_routes(
    tmp_path: Path,
) -> None:
    with TestClient(create_app(project_directory=tmp_path / "projects")) as client:
        paths = set(client.get("/openapi.json").json()["paths"])

    expected = {
        "/api/v3/jobs",
        "/api/v3/jobs/{job_id}",
        "/api/v3/jobs/{job_id}/events",
        "/api/v3/jobs/{job_id}/checkpoint",
        "/api/v3/jobs/{job_id}/shutdown",
        "/api/v3/jobs/{job_id}/cancel",
    }
    assert expected <= paths
    assert not any(path.startswith("/api/v2/") for path in paths)
    assert not any("/runs" in path for path in paths)
    assert not any(
        path.endswith("/checkpoint") and "/projects/" in path for path in paths
    )


def test_resource_limits_use_exact_byte_contract(tmp_path: Path) -> None:
    with TestClient(create_app(project_directory=tmp_path / "projects")) as client:
        response = client.get("/api/v3/resource-limits")

    assert response.status_code == 200
    assert response.json() == {
        "max_concurrent_jobs": 4,
        "max_job_resident_memory_bytes": 1024 * 1024 * 1024,
        "max_global_resident_memory_bytes": 4 * 1024 * 1024 * 1024,
        "max_checkpoint_disk_bytes": 512 * 1024 * 1024,
        "job_lifecycle": "user_explicit_stop",
    }


def test_job_lifecycle_checkpoint_sse_and_reconnect_are_payload_free(
    tmp_path: Path,
    controlled_manager: RunManager,
) -> None:
    app = create_app(
        project_directory=tmp_path / "projects",
        run_manager=controlled_manager,
    )
    with TestClient(app) as client:
        created = client.post("/api/v3/projects", json=_stream_project(tmp_path))
        started = client.post("/api/v3/jobs", json={"project_id": "stream_project"})
        job_id = started.json()["id"]
        checkpointed = client.post(f"/api/v3/jobs/{job_id}/checkpoint")
        stopped = client.post(f"/api/v3/jobs/{job_id}/shutdown")
        _wait_for_status(controlled_manager, job_id, RunStatus.COMPLETED)
        fetched = client.get(f"/api/v3/jobs/{job_id}")
        listed = client.get("/api/v3/jobs")
        events = client.get(
            f"/api/v3/jobs/{job_id}/events", headers={"Last-Event-ID": "0"}
        )

    assert created.status_code == 201
    assert started.status_code == 202
    assert checkpointed.status_code == 200
    assert stopped.status_code == 200
    assert fetched.json()["status"] == "completed"
    assert listed.json()[0]["id"] == job_id
    assert events.status_code == 200
    assert events.text.startswith("retry: 500\n\n")
    assert "event: progress" in events.text
    assert "event: checkpoint" in events.text
    assert "event: terminal" in events.text
    assert '"throughput_rows":7' in events.text
    assert '"backpressure_events":2' in events.text
    assert '"late_rows":1' in events.text
    for forbidden in ("password", "secret", "raw_payload", "value + 1"):
        assert forbidden not in events.text


def test_cancel_produces_a_persistent_terminal_status(
    tmp_path: Path,
    controlled_manager: RunManager,
) -> None:
    project = ProjectDocument.model_validate(_stream_project(tmp_path))
    started = controlled_manager.submit_job(project)

    controlled_manager.cancel_job(started.id)
    _wait_for_status(controlled_manager, started.id, RunStatus.CANCELLED)

    assert controlled_manager.get_job(started.id).status is RunStatus.CANCELLED
    assert controlled_manager.list_jobs()[0].id == started.id


def test_batch_projects_cannot_start_continuous_jobs(tmp_path: Path) -> None:
    app = create_app(project_directory=tmp_path / "projects")
    with TestClient(app) as client:
        assert client.post("/api/v3/projects", json=_batch_project()).status_code == 201
        response = client.post("/api/v3/jobs", json={"project_id": "batch_project"})

    assert response.status_code == 422
    assert "runtime.mode 'stream'" in response.json()["detail"]


def test_static_input_projects_fail_closed_before_worker_spawn(tmp_path: Path) -> None:
    app = create_app(project_directory=tmp_path / "projects")
    project = _stream_project(tmp_path, project_id="static_project")
    project["graph"]["nodes"] = [
        {
            "id": "merge",
            "operator": {"kind": "union"},
            "input_ports": [
                {"name": "input", "kind": "table"},
                {"name": "weights", "kind": "table"},
            ],
        }
    ]
    project["static_inputs"] = [
        {
            "kind": "table",
            "name": "weights",
            "mutability": "static",
            "schema": [{"name": "factor", "data_type": "float64", "nullable": False}],
        }
    ]
    with TestClient(app) as client:
        assert client.post("/api/v3/projects", json=project).status_code == 201
        response = client.post("/api/v3/jobs", json={"project_id": "static_project"})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "static_inputs.weights" in detail
    assert "cannot resolve" in detail


def test_concurrency_limit_is_enforced_before_starting_another_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_manager_module, "_execute_continuous_worker", _controlled_worker
    )
    manager = RunManager(
        use_processes=False,
        checkpoint_directory=tmp_path / "checkpoints",
        resource_limits=ResourceLimits(max_concurrent_jobs=1),
    )
    project = ProjectDocument.model_validate(_stream_project(tmp_path))
    first = manager.submit_job(project)

    with pytest.raises(RunManagerError, match="max_concurrent_jobs"):
        manager.submit_job(project)

    manager.cancel_job(first.id)
    _wait_for_status(manager, first.id, RunStatus.CANCELLED)


def test_checkpoint_disk_limit_becomes_a_typed_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 16 * 1024 * 1024

    def oversized_worker(project_json: str, _commands: object, output: object) -> None:
        document = json.loads(project_json)
        state = Path(document["state"]["root"])
        state.mkdir(parents=True, exist_ok=True)
        with (state / "oversized").open("wb") as file:
            file.truncate(limit + 1)
        output.put({"kind": "state", "state": "running"})
        time.sleep(1)

    monkeypatch.setattr(
        run_manager_module, "_execute_continuous_worker", oversized_worker
    )
    manager = RunManager(
        use_processes=False,
        checkpoint_directory=tmp_path / "checkpoints",
        resource_limits=ResourceLimits(max_checkpoint_disk_bytes=limit),
    )
    started = manager.submit_job(
        ProjectDocument.model_validate(_stream_project(tmp_path))
    )

    _wait_for_status(manager, started.id, RunStatus.FAILED)
    failed = manager.get_job(started.id).model_dump(mode="json")

    assert failed["error_code"] == "job_limit_exceeded"
    assert failed["error"] == "job_limit_exceeded: max_checkpoint_disk_bytes"


def test_worker_death_without_terminal_event_is_a_typed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dying_worker(_project_json: str, _commands: object, output: object) -> None:
        output.put({"kind": "state", "state": "running"})

    monkeypatch.setattr(run_manager_module, "_execute_continuous_worker", dying_worker)
    manager = RunManager(
        use_processes=False,
        checkpoint_directory=tmp_path / "checkpoints",
    )
    started = manager.submit_job(
        ProjectDocument.model_validate(_stream_project(tmp_path))
    )

    _wait_for_status(manager, started.id, RunStatus.FAILED)
    failed = manager.get_job(started.id).model_dump(mode="json")

    assert failed["error_code"] == "worker_failed"
    assert failed["error"] == "worker exited without a terminal event"


def test_worker_join_failure_reason_survives_terminal_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed_worker(_project_json: str, _commands: object, output: object) -> None:
        output.put(
            {
                "kind": "terminal",
                "state": "failed",
                "cause": "failure",
                "error": "operator failed",
                "reason_code": "join_state_limit_exceeded",
            }
        )

    monkeypatch.setattr(run_manager_module, "_execute_continuous_worker", failed_worker)
    manager = RunManager(
        use_processes=False,
        checkpoint_directory=tmp_path / "checkpoints",
    )
    started = manager.submit_job(
        ProjectDocument.model_validate(_stream_project(tmp_path))
    )

    _wait_for_status(manager, started.id, RunStatus.FAILED)
    failed = manager.get_job(started.id).model_dump(mode="json")

    assert failed["error_code"] == "worker_failed"
    assert failed["reason_code"] == "join_state_limit_exceeded"
    assert failed["error"] == "operator failed"


def test_missing_job_operations_return_not_found(tmp_path: Path) -> None:
    with TestClient(create_app(project_directory=tmp_path / "projects")) as client:
        responses = (
            client.get("/api/v3/jobs/missing"),
            client.get("/api/v3/jobs/missing/events"),
            client.post("/api/v3/jobs/missing/checkpoint"),
            client.post("/api/v3/jobs/missing/shutdown"),
            client.post("/api/v3/jobs/missing/cancel"),
        )

    assert all(response.status_code == 404 for response in responses)


def test_join_metrics_reach_progress_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def metrics_worker(_project_json: str, _commands: object, output: object) -> None:
        output.put({"kind": "state", "state": "running"})
        output.put(
            {
                "kind": "progress",
                "state": "running",
                "epoch": 1,
                "watermark": None,
                "throughput_rows": 2,
                "queue_envelopes": 0,
                "queue_rows": 0,
                "queue_bytes": 0,
                "backpressure_events": 0,
                "late_rows": 0,
                "stream_joins": (
                    {
                        "node_id": "match",
                        "left": {
                            "retained_rows": 1,
                            "retained_bytes": 124,
                            "evicted_rows": 0,
                            "late_rows": 0,
                            "late_affected_batches": 0,
                            "max_lateness_micros": None,
                            "null_event_time_rows": 0,
                            "null_key_rows": 0,
                        },
                        "right": {
                            "retained_rows": 0,
                            "retained_bytes": 0,
                            "evicted_rows": 0,
                            "late_rows": 0,
                            "late_affected_batches": 0,
                            "max_lateness_micros": None,
                            "null_event_time_rows": 0,
                            "null_key_rows": 0,
                        },
                        "emitted_match_rows": 1,
                        "state_limit_failures": 0,
                        "match_limit_failures": 0,
                    },
                ),
            }
        )
        output.put(
            {
                "kind": "terminal",
                "state": "completed",
                "cause": "graceful_shutdown",
            }
        )

    monkeypatch.setattr(
        run_manager_module, "_execute_continuous_worker", metrics_worker
    )
    manager = RunManager(
        use_processes=False,
        checkpoint_directory=tmp_path / "checkpoints",
    )
    started = manager.submit_job(
        ProjectDocument.model_validate(_stream_project(tmp_path))
    )

    _wait_for_status(manager, started.id, RunStatus.COMPLETED)
    events = [event.model_dump(mode="json") for event in manager.events(started.id)]
    progress = [event for event in events if event["type"] == "progress"]
    assert progress, events
    joins = progress[0]["stream_joins"]
    assert joins is not None
    assert joins[0]["node_id"] == "match"
    assert joins[0]["left"]["retained_rows"] == 1
    assert joins[0]["emitted_match_rows"] == 1
