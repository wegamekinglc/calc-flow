from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from calc_flow import (
    FileCheckpointStore,
    FileProjectStore,
    ProjectDocument,
    Runtime,
    project_json_schema,
)
from fastapi.testclient import TestClient

import calc_flow_studio.app as app_module
from calc_flow_studio.app import API_PREFIX, create_app, validate_bind_host
from calc_flow_studio.models import RunEvent, RunResponse, RunStatus


def _project(
    project_id: str = "project_alpha", *, name: str = "Alpha"
) -> dict[str, object]:
    return {
        "format_version": 2,
        "id": project_id,
        "name": name,
        "description": "A v2 project",
        "pipeline": {
            "name": f"{name} pipeline",
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


class FakeManager:
    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.wait_calls: list[tuple[str, int, float]] = []
        self.run = RunResponse(
            id="run_1",
            project_id="project_alpha",
            status=RunStatus.RUNNING,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    def submit(self, project: ProjectDocument, request: object) -> RunResponse:
        assert project.root["id"] == "project_alpha"
        assert request is not None
        return self.run

    def get(self, run_id: str) -> RunResponse:
        if run_id != self.run.id:
            raise KeyError(run_id)
        return self.run

    def wait_for_events(
        self, run_id: str, *, after_sequence: int, timeout: float
    ) -> tuple[tuple[RunEvent, ...], RunStatus]:
        self.wait_calls.append((run_id, after_sequence, timeout))
        if len(self.wait_calls) == 1:
            return (), RunStatus.RUNNING
        event = RunEvent(
            sequence=after_sequence + 1,
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            type="completed",
            message="Run completed",
        )
        return (event,), RunStatus.COMPLETED

    def cancel(self, run_id: str) -> RunResponse:
        current = self.get(run_id)
        return current.model_copy(update={"status": RunStatus.CANCELLED})

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _client(tmp_path: Path, **kwargs: Any) -> TestClient:
    return TestClient(
        create_app(
            project_directory=tmp_path / "projects",
            checkpoint_directory=tmp_path / "checkpoints",
            **kwargs,
        )
    )


def _create(client: TestClient, project: dict[str, object] | None = None):
    return client.post(f"{API_PREFIX}/projects", json=project or _project())


def test_openapi_contains_only_v2_routes_and_exact_rust_schema(tmp_path) -> None:
    with _client(tmp_path) as client:
        openapi = client.get("/openapi.json").json()
        schema = client.get(f"{API_PREFIX}/schema/project").json()

    assert API_PREFIX == "/api/v2"
    assert f"{API_PREFIX}/catalog" in openapi["paths"]
    assert f"{API_PREFIX}/schema/project" in openapi["paths"]
    assert f"{API_PREFIX}/projects/{{project_id}}" in openapi["paths"]
    assert f"{API_PREFIX}/runs/{{run_id}}/events" in openapi["paths"]
    assert not any(path.startswith("/api/v1/") for path in openapi["paths"])
    assert schema == json.loads(project_json_schema())
    assert schema["properties"]["format_version"]["const"] == 2
    assert "backend" not in json.dumps(schema).lower()


def test_catalog_is_exact_runtime_metadata_and_validation_uses_canonical_json(
    tmp_path,
) -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.documents: list[str] = []

        def catalog(self) -> list[dict[str, object]]:
            return [{"kind": "test", "name": "only-runtime-metadata"}]

        def validation_report(self, project_json: str) -> dict[str, object]:
            self.documents.append(project_json)
            return {"valid": True, "issues": [], "fingerprint": "fake"}

    runtime = FakeRuntime()
    with _client(tmp_path, runtime=runtime) as client:
        assert _create(client).status_code == 201
        catalog = client.get(f"{API_PREFIX}/catalog")
        validation = client.post(f"{API_PREFIX}/projects/project_alpha/validate")

    assert catalog.json() == [{"kind": "test", "name": "only-runtime-metadata"}]
    assert validation.json() == {
        "valid": True,
        "issues": [],
        "fingerprint": "fake",
    }
    assert len(runtime.documents) == 1
    assert json.loads(runtime.documents[0])["id"] == "project_alpha"


def test_project_crud_preserves_client_ids_sorting_and_request_values(tmp_path) -> None:
    original = _project()
    with _client(tmp_path) as client:
        created = _create(client, original)
        duplicate = _create(client, original)
        assert _create(client, _project("project_beta", name="Beta")).status_code == 201
        listed = client.get(f"{API_PREFIX}/projects")
        fetched = client.get(f"{API_PREFIX}/projects/project_alpha")
        updated_document = {**original, "name": "Updated"}
        updated = client.put(
            f"{API_PREFIX}/projects/project_alpha", json=updated_document
        )
        mismatch = client.put(f"{API_PREFIX}/projects/other", json=updated_document)
        deleted = client.delete(f"{API_PREFIX}/projects/project_alpha")
        missing = client.get(f"{API_PREFIX}/projects/project_alpha")
        missing_delete = client.delete(f"{API_PREFIX}/projects/project_alpha")

    assert created.status_code == 201
    assert created.json()["id"] == "project_alpha"
    assert duplicate.status_code == 409
    assert [item["id"] for item in listed.json()] == [
        "project_alpha",
        "project_beta",
    ]
    assert listed.json()[0]["node_count"] == 1
    assert fetched.json()["pipeline"]["nodes"][0]["id"] == "calculate"
    assert updated.json()["name"] == "Updated"
    assert mismatch.status_code == 409
    assert original["name"] == "Alpha"
    assert deleted.status_code == 204
    assert missing.status_code == 404
    assert missing_delete.status_code == 404


def test_project_routes_await_async_store_and_never_use_blocking_facades(
    tmp_path,
) -> None:
    class AsyncOnlyStore:
        def __init__(self) -> None:
            self.projects: dict[str, ProjectDocument] = {}

        async def create(self, project: ProjectDocument) -> None:
            self.projects[str(project.root["id"])] = ProjectDocument.model_validate(
                project.root
            )

        async def put(self, project: ProjectDocument) -> None:
            self.projects[str(project.root["id"])] = ProjectDocument.model_validate(
                project.root
            )

        async def get(self, project_id: str) -> ProjectDocument:
            return ProjectDocument.model_validate(self.projects[project_id].root)

        async def list(self) -> list[ProjectDocument]:
            return list(self.projects.values())

        async def delete(self, project_id: str) -> None:
            del self.projects[project_id]

        def create_blocking(self, *_: object) -> None:
            raise AssertionError("blocking facade used")

    store = AsyncOnlyStore()
    with _client(tmp_path, project_store=store) as client:
        assert _create(client).status_code == 201
        assert client.get(f"{API_PREFIX}/projects/project_alpha").status_code == 200
        assert (
            client.put(
                f"{API_PREFIX}/projects/project_alpha", json=_project()
            ).status_code
            == 200
        )
        assert client.get(f"{API_PREFIX}/projects").status_code == 200
        assert client.delete(f"{API_PREFIX}/projects/project_alpha").status_code == 204
        assert client.get(f"{API_PREFIX}/projects/project_alpha").status_code == 404


def test_import_export_use_bounded_strict_rust_transforms_and_conflicts(
    tmp_path,
) -> None:
    document = _project()
    with _client(tmp_path) as client:
        imported = client.post(
            f"{API_PREFIX}/projects/import?format=json",
            content=json.dumps(document),
        )
        conflict = client.post(
            f"{API_PREFIX}/projects/import?format=json",
            content=json.dumps(document),
        )
        replaced = client.post(
            f"{API_PREFIX}/projects/import?format=json&replace=true",
            content=json.dumps({**document, "name": "Replaced"}),
        )
        json_export = client.get(
            f"{API_PREFIX}/projects/project_alpha/export?format=json"
        )
        yaml_export = client.get(
            f"{API_PREFIX}/projects/project_alpha/export?format=yaml"
        )
        unsafe = client.post(
            f"{API_PREFIX}/projects/import?format=yaml",
            content="!!python/object/apply:os.system ['id']",
        )
        alias = client.post(
            f"{API_PREFIX}/projects/import?format=yaml",
            content=(
                "format_version: 2\nid: alias\nname: &name alias\n"
                "description: *name\npipeline: {name: p, nodes: []}\n"
            ),
        )
        oversized = client.post(
            f"{API_PREFIX}/projects/import?format=json",
            content=b"x" * (10 * 1024 * 1024 + 1),
        )
        missing = client.get(f"{API_PREFIX}/projects/missing/export")

    assert imported.status_code == 201
    assert conflict.status_code == 409
    assert replaced.status_code == 201
    assert replaced.json()["name"] == "Replaced"
    assert json_export.status_code == 200
    assert json_export.text.endswith("\n")
    assert (
        json_export.text
        == json.dumps(json_export.json(), indent=2, sort_keys=True) + "\n"
    )
    assert yaml_export.status_code == 200
    assert "format_version: 2" in yaml_export.text
    assert unsafe.status_code == 422
    assert alias.status_code == 422
    assert oversized.status_code == 422
    assert missing.status_code == 404


def test_import_and_export_threadpool_only_pure_rust_transformations(
    tmp_path, monkeypatch
) -> None:
    calls: list[object] = []
    original = app_module.run_in_threadpool

    async def tracking(function, *args, **kwargs):
        calls.append(function)
        return await original(function, *args, **kwargs)

    monkeypatch.setattr(app_module, "run_in_threadpool", tracking)
    with _client(tmp_path) as client:
        assert (
            client.post(
                f"{API_PREFIX}/projects/import?format=json",
                content=json.dumps(_project()),
            ).status_code
            == 201
        )
        assert (
            client.get(
                f"{API_PREFIX}/projects/project_alpha/export?format=yaml"
            ).status_code
            == 200
        )

    assert app_module.import_project_json in calls
    assert app_module.export_project_yaml in calls
    assert FileProjectStore.create not in calls
    assert FileProjectStore.get not in calls


def test_checkpoint_routes_use_compiled_plan_identity_and_async_store(tmp_path) -> None:
    projects = FileProjectStore(tmp_path / "projects")
    checkpoints = FileCheckpointStore(tmp_path / "checkpoints")
    runtime = Runtime()
    project = ProjectDocument.model_validate(_project())
    asyncio.run(projects.create(project))
    plan = runtime.compile_project(project.canonical_json())
    checkpoint = {
        "format_version": 2,
        "pipeline_name": plan.name,
        "pipeline_fingerprint": plan.fingerprint,
        "source_cursor": {"offset": 12},
        "sequence": 4,
        "state": {"calculate": {"rows": 12}},
        "created_at": "2026-01-01T00:00:00Z",
    }
    asyncio.run(checkpoints.save(checkpoint))

    with _client(
        tmp_path,
        project_store=projects,
        checkpoint_store=checkpoints,
        runtime=runtime,
    ) as client:
        inspected = client.get(f"{API_PREFIX}/projects/project_alpha/checkpoint")
        asyncio.run(checkpoints.save({**checkpoint, "pipeline_fingerprint": "stale"}))
        stale = client.get(f"{API_PREFIX}/projects/project_alpha/checkpoint")
        reset = client.delete(f"{API_PREFIX}/projects/project_alpha/checkpoint")
        absent = client.get(f"{API_PREFIX}/projects/project_alpha/checkpoint")
        missing = client.get(f"{API_PREFIX}/projects/missing/checkpoint")

    assert inspected.json() == {
        "pipeline_name": "Alpha pipeline",
        "exists": True,
        "compatible": True,
        "pipeline_fingerprint": plan.fingerprint,
        "sequence": 4,
        "source_cursor": {"offset": 12},
        "created_at": "2026-01-01T00:00:00Z",
        "state_nodes": ["calculate"],
    }
    assert stale.json()["compatible"] is False
    assert reset.json()["exists"] is False
    assert absent.json()["exists"] is False
    assert missing.status_code == 404


def test_run_routes_use_injected_manager_and_preserve_sse_contract(tmp_path) -> None:
    manager = FakeManager()
    with _client(tmp_path, run_manager=manager) as client:
        assert _create(client).status_code == 201
        submitted = client.post(
            f"{API_PREFIX}/projects/project_alpha/runs",
            json={"inputs": {"input": {"format": "records", "data": [{"value": 2}]}}},
        )
        fetched = client.get(f"{API_PREFIX}/runs/run_1")
        events = client.get(
            f"{API_PREFIX}/runs/run_1/events", headers={"Last-Event-ID": "0"}
        )
        cancelled = client.delete(f"{API_PREFIX}/runs/run_1")
        missing_run = client.get(f"{API_PREFIX}/runs/missing")
        missing_events = client.get(f"{API_PREFIX}/runs/missing/events")
        missing_cancel = client.delete(f"{API_PREFIX}/runs/missing")

    assert submitted.status_code == 202
    assert fetched.json()["status"] == "running"
    assert events.status_code == 200
    assert events.text.startswith("retry: 500\n\n")
    assert ": keep-alive\n\n" in events.text
    assert "id: 1\nevent: completed\n" in events.text
    assert events.headers["cache-control"] == "no-cache"
    assert events.headers["x-accel-buffering"] == "no"
    assert manager.wait_calls[0][1] == 0
    assert cancelled.json()["status"] == "cancelled"
    assert missing_run.status_code == 404
    assert missing_events.status_code == 404
    assert missing_cancel.status_code == 404
    assert manager.shutdown_calls == 1


def test_run_routes_return_typed_503_without_a_task22_manager(tmp_path) -> None:
    with _client(tmp_path) as client:
        assert _create(client).status_code == 201
        responses = (
            client.post(f"{API_PREFIX}/projects/project_alpha/runs", json={}),
            client.get(f"{API_PREFIX}/runs/run"),
            client.get(f"{API_PREFIX}/runs/run/events"),
            client.delete(f"{API_PREFIX}/runs/run"),
        )

    assert all(response.status_code == 503 for response in responses)
    assert all("run manager" in response.json()["detail"] for response in responses)


def test_app_serves_static_frontend_and_accepts_only_loopback(tmp_path) -> None:
    frontend = tmp_path / "frontend"
    assets = frontend / "assets"
    assets.mkdir(parents=True)
    (frontend / "index.html").write_text("<h1>Local studio</h1>", encoding="utf-8")
    (assets / "app.js").write_text("export {};", encoding="utf-8")
    app = create_app(
        project_directory=tmp_path / "projects", frontend_directory=frontend
    )

    with TestClient(app) as client:
        index = client.get("/")
        asset = client.get("/assets/app.js")

    assert index.status_code == 200
    assert "Local studio" in index.text
    assert asset.status_code == 200
    assert validate_bind_host("127.0.0.1") == "127.0.0.1"
    assert validate_bind_host("::1") == "::1"
    assert validate_bind_host("localhost") == "localhost"
    with pytest.raises(ValueError, match="loopback"):
        validate_bind_host("0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        validate_bind_host("example.com")
