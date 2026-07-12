from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from calc_flow.checkpoint import Checkpoint, FileCheckpointStore
from calc_flow.config import (
    DataSourceConfig,
    EdgeConfig,
    NodeConfig,
    PipelineConfig,
    ProjectConfig,
    UdfReferenceConfig,
    compile_project,
)
from calc_flow.udf import UdfRegistry
from fastapi.testclient import TestClient

from calc_flow_studio.app import create_app, validate_bind_host
from calc_flow_studio.models import RunEvent, RunResponse, RunStatus
from calc_flow_studio.run_manager import RunManager


def _registry() -> UdfRegistry:
    registry = UdfRegistry()

    @registry.datafusion_scalar(
        name="double_value",
        version="1",
        input_fields=[pa.int64()],
        return_field=pa.int64(),
        description="Double a value",
    )
    def double_value(values):
        return pc.multiply(values, 2)

    return registry


def _project() -> ProjectConfig:
    return ProjectConfig(
        id="branching",
        name="Branching UDF",
        description="A browser-configurable branching pipeline",
        pipeline=PipelineConfig(
            id="main",
            name="Main",
            nodes=(
                NodeConfig(
                    id="calculate",
                    kind="expression",
                    expression="doubled = double_value(value)",
                    udfs=(UdfReferenceConfig(name="double_value", version="1"),),
                ),
                NodeConfig(
                    id="left",
                    kind="expression",
                    expression="left_value = doubled + 1",
                ),
                NodeConfig(
                    id="right",
                    kind="expression",
                    expression="right_value = doubled * 10",
                ),
            ),
            edges=(
                EdgeConfig(source_node="calculate", target_node="left"),
                EdgeConfig(source_node="calculate", target_node="right"),
            ),
        ),
        data_sources=(
            DataSourceConfig(
                id="sample",
                input_name="input",
                format="inline_json",
                data=[{"value": 2}, {"value": 4}],
            ),
        ),
    )


def _join_project() -> ProjectConfig:
    return ProjectConfig(
        id="join_project",
        name="Join project",
        pipeline=PipelineConfig(
            id="main",
            name="Join and branch",
            nodes=(
                NodeConfig(
                    id="orders",
                    kind="expression",
                    expression="doubled_amount = double_value(amount)",
                    udfs=(UdfReferenceConfig(name="double_value", version="1"),),
                ),
                NodeConfig(
                    id="fees",
                    kind="expression",
                    expression="normalized_fee = fee + 0",
                ),
                NodeConfig(
                    id="join",
                    kind="sql",
                    query=(
                        "select o.id, o.doubled_amount + f.normalized_fee as total "
                        "from orders_table o join fees_table f on o.id = f.id"
                    ),
                    inputs=("orders_table", "fees_table"),
                ),
                NodeConfig(
                    id="audit",
                    kind="expression",
                    expression="audit_total = total",
                ),
                NodeConfig(
                    id="report",
                    kind="expression",
                    expression="report_total = total * 10",
                ),
            ),
            edges=(
                EdgeConfig(
                    source_node="orders",
                    target_node="join",
                    target_port="orders_table",
                ),
                EdgeConfig(
                    source_node="fees",
                    target_node="join",
                    target_port="fees_table",
                ),
                EdgeConfig(source_node="join", target_node="audit"),
                EdgeConfig(source_node="join", target_node="report"),
            ),
        ),
        data_sources=(
            DataSourceConfig(
                id="orders_sample",
                input_name="orders.input",
                format="inline_json",
                data=[{"id": 1, "amount": 10}],
            ),
            DataSourceConfig(
                id="fees_sample",
                input_name="fees.input",
                format="inline_json",
                data=[{"id": 1, "fee": 3}],
            ),
        ),
    )


def _client(tmp_path) -> TestClient:
    registry = _registry()
    manager = RunManager(udf_registry=registry, use_processes=False)
    return TestClient(
        create_app(
            project_directory=tmp_path / "projects",
            checkpoint_directory=tmp_path / "checkpoints",
            udf_registry=registry,
            run_manager=manager,
        )
    )


def _wait_for_run(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] not in {"pending", "running"}:
            return run
        time.sleep(0.01)
    raise AssertionError("run did not finish")


def _import_project(client: TestClient, project: dict) -> None:
    response = client.post(
        "/api/v1/projects/import?format=json",
        content=json.dumps(project),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 201


def test_catalog_exposes_datafusion_nodes_udfs_and_limits(tmp_path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/api/v1/catalog")

    assert response.status_code == 200
    catalog = response.json()
    assert [operator["kind"] for operator in catalog["operators"]] == [
        "expression",
        "sql",
        "array_expression",
    ]
    assert catalog["operators"][0]["backend_selector"] is False
    assert catalog["udfs"][0]["name"] == "double_value"
    assert "pandas" not in response.text.lower()
    assert "polars" not in response.text.lower()
    assert catalog["limits"]["max_rows"] == 100_000


def test_project_crud_uses_server_generated_ids(tmp_path) -> None:
    project = _project().model_dump(mode="json", by_alias=True)
    draft = {key: value for key, value in project.items() if key != "id"}
    with _client(tmp_path) as client:
        created = client.post("/api/v1/projects", json=draft)
        project_id = created.json()["id"]
        duplicate = client.post("/api/v1/projects", json=draft)
        listed = client.get("/api/v1/projects")
        fetched = client.get(f"/api/v1/projects/{project_id}")
        updated_document = {**created.json(), "name": "Updated"}
        updated = client.put(
            f"/api/v1/projects/{project_id}",
            json=updated_document,
        )
        validation = client.post(f"/api/v1/projects/{project_id}/validate")
        checkpoint = client.get(f"/api/v1/projects/{project_id}/checkpoint")
        mismatch = client.put(
            "/api/v1/projects/other",
            json=created.json(),
        )
        deleted = client.delete(f"/api/v1/projects/{project_id}")
        missing = client.get(f"/api/v1/projects/{project_id}")

    assert created.status_code == 201
    assert project_id.startswith("project_")
    assert len(project_id) == len("project_") + 32
    assert duplicate.status_code == 201
    assert duplicate.json()["id"] != project_id
    assert len(listed.json()) == 2
    assert listed.json()[0]["node_count"] == 3
    assert fetched.json()["pipeline"]["nodes"][0]["kind"] == "expression"
    assert updated.json()["name"] == "Updated"
    assert validation.json()["valid"] is True
    assert checkpoint.json()["exists"] is False
    assert validation.json()["graph_outputs"] == ["left.output", "right.output"]
    assert mismatch.status_code == 409
    assert deleted.status_code == 204
    assert missing.status_code == 404


def test_project_import_and_export_json_and_yaml(tmp_path) -> None:
    project = _project().model_dump(mode="json", by_alias=True)
    with _client(tmp_path) as client:
        imported = client.post(
            "/api/v1/projects/import?format=json",
            content=json.dumps(project),
            headers={"Content-Type": "application/json"},
        )
        conflict = client.post(
            "/api/v1/projects/import?format=json",
            content=json.dumps(project),
            headers={"Content-Type": "application/json"},
        )
        replaced = client.post(
            "/api/v1/projects/import?format=json&replace=true",
            content=json.dumps({**project, "name": "Replaced"}),
            headers={"Content-Type": "application/json"},
        )
        json_export = client.get("/api/v1/projects/branching/export?format=json")
        yaml_export = client.get("/api/v1/projects/branching/export?format=yaml")
        unsafe = client.post(
            "/api/v1/projects/import?format=yaml",
            content="!!python/object/apply:os.system ['id']",
        )

    assert imported.status_code == 201
    assert conflict.status_code == 409
    assert replaced.status_code == 201
    assert replaced.json()["name"] == "Replaced"
    assert json_export.status_code == 200
    assert json_export.headers["content-disposition"] == (
        'attachment; filename="branching.json"'
    )
    assert json_export.json()["id"] == "branching"
    assert yaml_export.status_code == 200
    assert "format_version: '1'" in yaml_export.text
    assert unsafe.status_code == 422


def test_run_api_executes_branching_registered_udf_project(tmp_path) -> None:
    project = _project().model_dump(mode="json", by_alias=True)
    with _client(tmp_path) as client:
        _import_project(client, project)
        submitted = client.post("/api/v1/projects/branching/runs", json={})
        assert submitted.status_code == 202
        run = _wait_for_run(client, submitted.json()["id"])
        events = client.get(
            f"/api/v1/runs/{run['id']}/events",
            headers={"Last-Event-ID": "1"},
        )

    assert run["status"] == "completed"
    assert run["result"]["outputs"]["left.output"]["rows"] == [
        {"value": 2, "doubled": 4, "left_value": 5},
        {"value": 4, "doubled": 8, "left_value": 9},
    ]
    assert run["result"]["outputs"]["right.output"]["rows"][0]["right_value"] == 40
    assert run["result"]["datafusion_metrics"]
    assert events.status_code == 200
    assert "retry: 500" in events.text
    assert "event: completed" in events.text
    assert "event: running" not in events.text
    assert "data:" in events.text
    assert events.headers["cache-control"] == "no-cache"


def test_run_event_stream_emits_heartbeat_before_terminal_event(tmp_path) -> None:
    class EventManager:
        udf_registry = _registry().snapshot()

        def __init__(self) -> None:
            self.waits = 0

        def get(self, run_id: str) -> RunResponse:
            return RunResponse(
                id=run_id,
                project_id="project",
                status=RunStatus.RUNNING,
                created_at=datetime.now(UTC),
            )

        def wait_for_events(self, run_id, *, after_sequence, timeout):
            self.waits += 1
            if self.waits == 1:
                return (), RunStatus.RUNNING
            return (
                (
                    RunEvent(
                        sequence=0,
                        timestamp=datetime.now(UTC),
                        type="completed",
                        message="Run completed",
                    ),
                ),
                RunStatus.COMPLETED,
            )

        def shutdown(self) -> None:
            return None

    manager = EventManager()
    app = create_app(
        project_directory=tmp_path / "projects",
        run_manager=manager,
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/runs/run/events")

    assert response.status_code == 200
    assert ": keep-alive\n\n" in response.text
    assert "event: completed" in response.text


def test_run_api_executes_udf_multi_input_sql_and_fan_out(tmp_path) -> None:
    project = _join_project().model_dump(mode="json", by_alias=True)
    with _client(tmp_path) as client:
        _import_project(client, project)
        validation = client.post("/api/v1/projects/join_project/validate")
        submitted = client.post("/api/v1/projects/join_project/runs", json={})
        run = _wait_for_run(client, submitted.json()["id"])

    assert validation.json()["valid"] is True
    assert validation.json()["graph_inputs"] == ["orders.input", "fees.input"]
    assert run["status"] == "completed"
    assert run["result"]["outputs"]["audit.output"]["rows"][0]["audit_total"] == 23
    assert run["result"]["outputs"]["report.output"]["rows"][0]["report_total"] == 230
    assert len(run["result"]["datafusion_metrics"]) == 5


def test_run_api_rejects_wrong_inputs_and_unknown_resources(tmp_path) -> None:
    project = _project().model_copy(update={"data_sources": ()})
    with _client(tmp_path) as client:
        _import_project(client, project.model_dump(mode="json", by_alias=True))
        bad_run = client.post(
            "/api/v1/projects/branching/runs",
            json={
                "inputs": {
                    "wrong": {
                        "format": "inline_json",
                        "data": [{"value": 1}],
                    }
                }
            },
        )
        missing_project = client.get("/api/v1/projects/missing")
        missing_run = client.get("/api/v1/runs/missing")
        missing_responses = [
            client.delete("/api/v1/projects/missing"),
            client.get("/api/v1/projects/missing/export"),
            client.post("/api/v1/projects/missing/validate"),
            client.get("/api/v1/projects/missing/checkpoint"),
            client.delete("/api/v1/projects/missing/checkpoint"),
            client.post("/api/v1/projects/missing/runs", json={}),
            client.get("/api/v1/runs/missing/events"),
            client.delete("/api/v1/runs/missing"),
        ]

    assert bad_run.status_code == 422
    assert missing_project.status_code == 404
    assert missing_run.status_code == 404
    assert all(response.status_code == 404 for response in missing_responses)


def test_openapi_contains_versioned_project_and_run_endpoints(tmp_path) -> None:
    with _client(tmp_path) as client:
        openapi = client.get("/openapi.json").json()
        schema = client.get("/api/v1/schema/project").json()

    assert "/api/v1/projects/{project_id}/runs" in openapi["paths"]
    assert "/api/v1/projects/{project_id}/checkpoint" in openapi["paths"]
    assert "/api/v1/runs/{run_id}/events" in openapi["paths"]
    assert schema["properties"]["format_version"]["const"] == "1"


def test_app_serves_built_frontend_when_directory_is_supplied(tmp_path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<h1>Local studio</h1>", encoding="utf-8")
    app = create_app(
        project_directory=tmp_path / "projects",
        run_manager=RunManager(use_processes=False),
        frontend_directory=frontend,
    )

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Local studio" in response.text


def test_server_binding_accepts_only_loopback() -> None:
    assert validate_bind_host("127.0.0.1") == "127.0.0.1"
    assert validate_bind_host("::1") == "::1"
    assert validate_bind_host("localhost") == "localhost"

    with pytest.raises(ValueError, match="loopback"):
        validate_bind_host("0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        validate_bind_host("example.com")


def test_checkpoint_api_inspects_compatibility_and_resets_state(tmp_path) -> None:
    project = _project()
    registry = _registry()
    plan = compile_project(project, udf_registry=registry)
    checkpoint_store = FileCheckpointStore(tmp_path / "checkpoints")
    checkpoint_store.save(
        Checkpoint(
            pipeline_name=plan.name,
            pipeline_fingerprint=plan.fingerprint,
            source_cursor={"offset": 12},
            sequence=4,
            state={"calculate": {"rows": 12}},
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )

    with _client(tmp_path) as client:
        document = project.model_dump(mode="json", by_alias=True)
        _import_project(client, document)
        inspected = client.get("/api/v1/projects/branching/checkpoint")
        reset = client.delete("/api/v1/projects/branching/checkpoint")
        missing = client.get("/api/v1/projects/missing/checkpoint")

    assert inspected.status_code == 200
    assert inspected.json() == {
        "pipeline_name": "Main",
        "exists": True,
        "compatible": True,
        "pipeline_fingerprint": plan.fingerprint,
        "sequence": 4,
        "source_cursor": {"offset": 12},
        "created_at": "2026-01-01T00:00:00Z",
        "state_nodes": ["calculate"],
    }
    assert reset.json()["exists"] is False
    assert checkpoint_store.load(plan.name) is None
    assert missing.status_code == 404


def test_checkpoint_api_reports_stale_fingerprint(tmp_path) -> None:
    project = _project()
    checkpoint_store = FileCheckpointStore(tmp_path / "checkpoints")
    checkpoint_store.save(
        Checkpoint(
            pipeline_name=project.pipeline.name,
            pipeline_fingerprint="stale",
            source_cursor=1,
            sequence=0,
            state={},
        )
    )

    with _client(tmp_path) as client:
        _import_project(client, project.model_dump(mode="json", by_alias=True))
        inspected = client.get("/api/v1/projects/branching/checkpoint")

    assert inspected.status_code == 200
    assert inspected.json()["compatible"] is False
