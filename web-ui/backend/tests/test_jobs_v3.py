from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from calc_flow_studio.app import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        project_directory=tmp_path / "projects",
        checkpoint_directory=tmp_path / "checkpoints",
    )
    return TestClient(app)


class TestApiPrefixV3:
    def test_openapi_uses_v3_prefix(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        paths = spec["paths"]
        assert any(p.startswith("/api/v3/") for p in paths)
        assert not any(p.startswith("/api/v2/") for p in paths)

    def test_catalog_under_v3(self, client):
        response = client.get("/api/v3/catalog")
        assert response.status_code == 200


class TestResourceLimits:
    def test_returns_defaults(self, client):
        response = client.get("/api/v3/resource-limits")
        assert response.status_code == 200
        limits = response.json()
        assert limits["max_concurrent_jobs"] == 4
        assert limits["max_job_memory_mb"] == 1024
        assert limits["max_global_memory_mb"] == 4096
        assert limits["max_checkpoint_disk_mb"] == 512
        assert limits["job_lifecycle"] == "user_explicit_stop"

    def test_appears_in_openapi(self, client):
        response = client.get("/openapi.json")
        spec = response.json()
        assert "/api/v3/resource-limits" in spec["paths"]


class TestJobRoutes:
    def test_list_jobs_empty(self, client):
        response = client.get("/api/v3/jobs")
        assert response.status_code == 200
        assert response.json() == []

    def test_get_missing_job_404(self, client):
        response = client.get("/api/v3/jobs/missing")
        assert response.status_code == 404

    def test_checkpoint_missing_job_404(self, client):
        response = client.post("/api/v3/jobs/missing/checkpoint")
        assert response.status_code == 404

    def test_shutdown_missing_job_404(self, client):
        response = client.post("/api/v3/jobs/missing/shutdown")
        assert response.status_code == 404

    def test_cancel_missing_job_404(self, client):
        response = client.post("/api/v3/jobs/missing/cancel")
        assert response.status_code == 404

    def test_all_job_routes_in_openapi(self, client):
        response = client.get("/openapi.json")
        spec = response.json()
        paths = spec["paths"]
        assert "/api/v3/jobs" in paths
        assert "/api/v3/jobs/{run_id}" in paths
        assert "/api/v3/jobs/{run_id}/checkpoint" in paths
        assert "/api/v3/jobs/{run_id}/shutdown" in paths
        assert "/api/v3/jobs/{run_id}/cancel" in paths
        assert "/api/v3/jobs/{run_id}/events" in paths


class TestProjectRoutesStillUnderV3:
    def test_projects_list(self, client):
        response = client.get("/api/v3/projects")
        assert response.status_code == 200


class TestJobLifecycleWithRun:
    """Exercises the job surface against a real RunManager."""

    def _create_project(self, client):
        project = {
            "format_version": 2,
            "id": "p1",
            "name": "test",
            "description": "",
            "pipeline": {
                "name": "pipe",
                "nodes": [
                    {
                        "id": "n1",
                        "operator": {
                            "kind": "expression",
                            "expression": "x = value + 1",
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
        create = client.post("/api/v3/projects", json=project)
        assert create.status_code == 201, create.text
        return create.json()

    def test_list_jobs_with_created_project(self, client):
        self._create_project(client)
        jobs = client.get("/api/v3/jobs")
        assert jobs.status_code == 200
        assert isinstance(jobs.json(), list)

    def test_get_job_with_created_project(self, client):
        self._create_project(client)
        job = client.get("/api/v3/jobs/missing")
        assert job.status_code == 404

    def test_checkpoint_missing_422(self, client):
        self._create_project(client)
        response = client.post("/api/v3/jobs/missing/checkpoint")
        assert response.status_code == 404

    def test_shutdown_missing_404(self, client):
        self._create_project(client)
        response = client.post("/api/v3/jobs/missing/shutdown")
        assert response.status_code == 404

    def test_cancel_missing_404(self, client):
        self._create_project(client)
        response = client.post("/api/v3/jobs/missing/cancel")
        assert response.status_code == 404


class FakeJobManager:
    """A minimal RunManager that owns one running job."""

    def __init__(self) -> None:
        from datetime import UTC, datetime

        from calc_flow_studio.models import RunResponse, RunStatus

        self.run = RunResponse.model_validate(
            {
                "id": "job_1",
                "project_id": "project_alpha",
                "status": RunStatus.RUNNING,
                "created_at": datetime(2026, 1, 1, tzinfo=UTC),
                "started_at": datetime(2026, 1, 1, tzinfo=UTC),
            }
        )
        self.checkpoint_calls: list[str] = []
        self.shutdown_calls: list[str] = []
        self.cancel_calls: list[str] = []

    def capabilities(self):
        raise NotImplementedError

    def submit(self, project, request):
        raise NotImplementedError

    def get(self, run_id):
        if run_id != "job_1":
            raise KeyError(run_id)
        return self.run

    def wait_for_events(self, run_id, *, after_sequence, timeout):
        raise NotImplementedError

    def cancel(self, run_id):
        if run_id != "job_1":
            raise KeyError(run_id)
        self.cancel_calls.append(run_id)
        return self.run

    def shutdown(self):
        pass

    def list_jobs(self):
        return (self.run,)

    def trigger_checkpoint(self, run_id):
        if run_id != "job_1":
            raise KeyError(run_id)
        self.checkpoint_calls.append(run_id)
        return self.run

    def shutdown_job(self, run_id):
        if run_id != "job_1":
            raise KeyError(run_id)
        self.shutdown_calls.append(run_id)
        return self.run

    def resource_limits(self):
        from calc_flow_studio.models import ResourceLimits

        return ResourceLimits()


class TestJobRoutesWithFakeManager:
    def _client(self):
        from fastapi.testclient import TestClient

        from calc_flow_studio.app import create_app

        app = create_app(run_manager=FakeJobManager())
        return TestClient(app)

    def test_list_returns_running_job(self):
        client = self._client()
        jobs = client.get("/api/v3/jobs")
        assert jobs.status_code == 200
        data = jobs.json()
        assert len(data) == 1
        assert data[0]["id"] == "job_1"

    def test_get_returns_running_job(self):
        client = self._client()
        job = client.get("/api/v3/jobs/job_1")
        assert job.status_code == 200
        assert job.json()["id"] == "job_1"

    def test_checkpoint_triggers(self):
        client = self._client()
        response = client.post("/api/v3/jobs/job_1/checkpoint")
        assert response.status_code == 200

    def test_shutdown_stops(self):
        client = self._client()
        response = client.post("/api/v3/jobs/job_1/shutdown")
        assert response.status_code == 200

    def test_cancel_stops(self):
        client = self._client()
        response = client.post("/api/v3/jobs/job_1/cancel")
        assert response.status_code == 200

    def test_resource_limits_returns_model(self):
        client = self._client()
        response = client.get("/api/v3/resource-limits")
        assert response.status_code == 200
        assert response.json()["job_lifecycle"] == "user_explicit_stop"
