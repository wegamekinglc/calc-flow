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
