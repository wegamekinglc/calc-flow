from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from calc_flow import (
    CalcFlowError,
    FileProjectStore,
    PipelineBuilder,
    ProjectDocument,
    Runtime,
    project_json_schema,
)
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest

import calc_flow_studio.app as app_module
from calc_flow_studio.app import API_PREFIX, create_app, validate_bind_host
from calc_flow_studio.models import RunEvent, RunResponse, RunStatus
from calc_flow_studio.run_manager import CapabilitySnapshotError, RunManagerError


def _project(
    project_id: str = "project_alpha", *, name: str = "Alpha"
) -> dict[str, object]:
    return {
        "format_version": 3,
        "id": project_id,
        "name": name,
        "description": "A v3 project",
        "runtime": {"mode": "batch", "options": {}},
        "graph": {
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
        self.run = RunResponse.model_validate(
            {
                "id": "run_1",
                "project_id": "project_alpha",
                "status": RunStatus.RUNNING,
                "created_at": datetime(2026, 1, 1, tzinfo=UTC),
                "started_at": datetime(2026, 1, 1, tzinfo=UTC),
            }
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
        return RunResponse.model_validate(
            {
                **current.model_dump(),
                "status": RunStatus.CANCELLED,
                "finished_at": datetime(2026, 1, 1, tzinfo=UTC),
            }
        )

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


def test_openapi_contains_only_v3_routes_and_exact_rust_schema(tmp_path) -> None:
    with _client(tmp_path) as client:
        openapi = client.get("/openapi.json").json()
        schema = client.get(f"{API_PREFIX}/schema/project").json()

    assert API_PREFIX == "/api/v3"
    assert f"{API_PREFIX}/catalog" in openapi["paths"]
    assert f"{API_PREFIX}/schema/project" in openapi["paths"]
    assert f"{API_PREFIX}/projects/{{project_id}}" in openapi["paths"]
    assert f"{API_PREFIX}/jobs/{{job_id}}/events" in openapi["paths"]
    assert not any("/runs" in path for path in openapi["paths"])
    assert not any(path.startswith("/api/v2/") for path in openapi["paths"])
    assert schema == json.loads(project_json_schema())
    assert schema["properties"]["format_version"]["const"] == 3

    # The v2 table backend selector must stay dead: no part of the project
    # schema may mention "backend" except the static array declaration
    # descriptor, whose `backend` field names the array provider (SCE-11).
    without_static_defs = dict(schema)
    without_static_defs["$defs"] = {
        name: definition
        for name, definition in schema["$defs"].items()
        if name != "StaticInputSpec"
    }
    assert "backend" not in json.dumps(without_static_defs).lower()

    def without_none_defaults(value: object) -> object:
        if isinstance(value, dict):
            return {
                key: without_none_defaults(item)
                for key, item in value.items()
                if not (key == "default" and item is None)
            }
        if isinstance(value, list):
            return [without_none_defaults(item) for item in value]
        return value

    def resolve_json_pointer(document: object, pointer: str) -> object:
        assert pointer.startswith("#/")
        current = document
        for encoded_part in pointer[2:].split("/"):
            part = encoded_part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict):
                current = current[part]
            else:
                assert isinstance(current, list)
                current = current[int(part)]
        return current

    def schema_references(value: object):
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str):
                yield reference
            for item in value.values():
                yield from schema_references(item)
        elif isinstance(value, list):
            for item in value:
                yield from schema_references(item)

    def normalize_component_references(value: object, component_name: str) -> object:
        if isinstance(value, dict):
            normalized = {
                key: normalize_component_references(item, component_name)
                for key, item in value.items()
            }
            reference = normalized.get("$ref")
            component_prefix = f"#/components/schemas/{component_name}/$defs/"
            if isinstance(reference, str) and reference.startswith(component_prefix):
                normalized["$ref"] = "#/$defs/" + reference.removeprefix(
                    component_prefix
                )
            return normalized
        if isinstance(value, list):
            return [
                normalize_component_references(item, component_name) for item in value
            ]
        return value

    for component_name in ("ProjectCreateRequest", "ProjectDocument"):
        component = openapi["components"]["schemas"][component_name]
        assert (
            normalize_component_references(component["properties"], component_name)
            == schema["properties"]
        )
        assert component["required"] == schema["required"]
        assert component["additionalProperties"] is False
        assert component["$defs"].keys() == schema["$defs"].keys()
        assert without_none_defaults(
            normalize_component_references(component["$defs"], component_name)
        ) == without_none_defaults(schema["$defs"])
        references = tuple(schema_references(component))
        assert references
        for reference in references:
            assert resolve_json_pointer(openapi, reference)
        assert component["$defs"]["NodeSpec"]["properties"]["operator"]["$ref"] == (
            f"#/components/schemas/{component_name}/$defs/OperatorSpec"
        )
        assert "JSONValue" not in json.dumps(component)


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
        "kind": "valid",
        "valid": True,
        "issues": [],
        "fingerprint": "fake",
    }
    assert len(runtime.documents) == 2
    assert [json.loads(document)["id"] for document in runtime.documents] == [
        "project_alpha",
        "project_alpha",
    ]


def test_capabilities_route_exposes_the_typed_runtime_session_snapshot(
    tmp_path,
) -> None:
    runtime = Runtime()
    runtime.register_provider("test", "identity", "1", lambda batch, _options: batch)

    with _client(tmp_path, runtime=runtime) as client:
        response = client.get(f"{API_PREFIX}/capabilities")
        catalog = client.get(f"{API_PREFIX}/catalog")
        openapi = client.get("/openapi.json").json()

    assert response.status_code == 200
    document = response.json()
    assert document["schemaVersion"] == 3
    assert document["runtime"]["scope"]["kind"] == "runtimeSession"
    assert document["runtime"]["scope"]["revision"] == 1
    assert document["runtime"]["providers"][0]["name"] == "identity"
    assert document["runtime"]["providers"][0]["modes"] == ["batch"]
    assert document["runtime"]["providers"][0]["finality"] == "unproven"
    assert document["runtime"]["providers"][0]["supportsStaticInputs"] is False
    assert [operator["kind"] for operator in document["runtime"]["operators"]] == [
        "cross_section",
        "expression",
        "rolling",
        "sql",
        "stream_join",
    ]
    assert document["runtime"]["operators"][0]["modes"] == ["batch", "stream"]
    assert document["runtime"]["operators"][0]["finality"] == "group_final_append_only"
    assert document["preview"]["inputBatchKinds"] == ["table"]
    assert catalog.json() == []
    capability_operation = openapi["paths"][f"{API_PREFIX}/capabilities"]["get"]
    assert capability_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/CapabilitiesResponse"}


def test_capabilities_route_reports_a_malformed_runtime_snapshot_as_internal(
    tmp_path,
) -> None:
    class MalformedCapabilityManager(FakeManager):
        def capabilities(self):
            raise CapabilitySnapshotError(
                "runtime capability snapshot violates schema version 1 at "
                "runtime.batchKinds.0: Input should be 'table' or 'array'"
            )

    with _client(
        tmp_path,
        runtime=Runtime(),
        run_manager=MalformedCapabilityManager(),
    ) as client:
        response = client.get(f"{API_PREFIX}/capabilities")

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "runtime capability snapshot violates schema version 1 at "
            "runtime.batchKinds.0: Input should be 'table' or 'array'"
        )
    }


def test_capabilities_route_reports_an_unavailable_runtime_session(
    tmp_path,
) -> None:
    class UnavailableCapabilityManager(FakeManager):
        def capabilities(self):
            raise RunManagerError("no parent runtime")

    with _client(
        tmp_path,
        runtime=Runtime(),
        run_manager=UnavailableCapabilityManager(),
    ) as client:
        response = client.get(f"{API_PREFIX}/capabilities")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "runtime capability snapshot is unavailable for this session"
    }


def test_mutating_routes_use_the_injected_runtime_before_store_mutation(
    tmp_path,
) -> None:
    runtime = Runtime()
    runtime.register_provider("test", "identity", "1", lambda batch, _options: batch)
    runtime.register_scalar_udf(
        provider="python",
        name="identity",
        version="1",
        input_types=("int64",),
        return_type="int64",
        volatility="immutable",
        function=lambda value: value,
    )
    provider_project = (
        PipelineBuilder("provider_project")
        .external("calc", "test", "identity", "1", {})
        .project
    )
    udf_project = (
        PipelineBuilder("udf_project")
        .expression(
            "calc",
            "result = identity(value)",
            udfs=(("python", "identity", "1"),),
        )
        .project
    )
    invalid_project = {
        **_project("invalid_project"),
        "pipeline": {"name": "empty", "nodes": []},
    }

    with _client(tmp_path, runtime=runtime) as client:
        created = _create(client, provider_project)
        imported = client.post(
            f"{API_PREFIX}/projects/import?format=json",
            content=json.dumps(udf_project),
        )
        rejected_create = _create(client, invalid_project)
        rejected_put = client.put(
            f"{API_PREFIX}/projects/invalid_project", json=invalid_project
        )
        rejected_import = client.post(
            f"{API_PREFIX}/projects/import?format=json",
            content=json.dumps({**invalid_project, "id": "unsafe\r\nid"}),
        )
        listed = client.get(f"{API_PREFIX}/projects")

    assert created.status_code == 201
    assert imported.status_code == 201
    assert rejected_create.status_code == 422
    assert rejected_put.status_code == 422
    assert rejected_import.status_code == 422
    assert [project["id"] for project in listed.json()] == [
        "provider_project",
        "udf_project",
    ]


@pytest.mark.parametrize(
    "report",
    [
        [],
        {"valid": "yes", "issues": []},
        {"valid": False, "issues": object()},
        {"valid": False, "issues": [object()]},
        {"valid": True, "issues": [], "fingerprint": float("nan")},
        {"valid": True, "issues": [{"message": "contradictory"}]},
    ],
)
def test_mutating_routes_reject_malformed_runtime_validation_reports(
    tmp_path, report
) -> None:
    class MalformedRuntime:
        def validation_report(self, project_json: str):
            assert json.loads(project_json)["id"] == "project_alpha"
            return report

    app = create_app(
        project_directory=tmp_path / "projects",
        checkpoint_directory=tmp_path / "checkpoints",
        runtime=MalformedRuntime(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = _create(client)
        listed = client.get(f"{API_PREFIX}/projects")

    assert response.status_code == 500
    assert listed.json() == []


def test_every_validation_caller_treats_malformed_runtime_data_as_internal(
    tmp_path,
) -> None:
    class SwitchingRuntime:
        malformed = False

        def validation_report(self, project_json: str) -> dict[str, object]:
            json.loads(project_json)
            if self.malformed:
                return {"valid": False, "issues": [], "fingerprint": None}
            return {"valid": True, "issues": [], "fingerprint": "valid"}

    runtime = SwitchingRuntime()
    with _client(tmp_path, runtime=runtime) as client:
        assert _create(client).status_code == 201
        runtime.malformed = True
        responses = (
            client.post(f"{API_PREFIX}/projects/project_alpha/validate"),
            client.post(
                f"{API_PREFIX}/projects",
                json=_project("project_new", name="New"),
            ),
            client.put(
                f"{API_PREFIX}/projects/project_alpha",
                json={**_project(), "name": "Overwritten"},
            ),
            client.post(
                f"{API_PREFIX}/projects/import?format=json&replace=true",
                content=json.dumps({**_project(), "name": "Imported"}),
            ),
        )
        stored = client.get(f"{API_PREFIX}/projects/project_alpha")
        listed = client.get(f"{API_PREFIX}/projects")

    assert [response.status_code for response in responses] == [500, 500, 500, 500]
    assert all(
        response.json()["detail"].startswith(
            "runtime validation report violates the v1 contract at "
        )
        for response in responses
    )
    assert stored.json()["name"] == "Alpha"
    assert [project["id"] for project in listed.json()] == ["project_alpha"]


def test_validation_route_preserves_native_calc_flow_error_details(
    tmp_path,
) -> None:
    class FailingRuntime:
        should_fail = False

        def validation_report(self, project_json: str) -> dict[str, object]:
            json.loads(project_json)
            if self.should_fail:
                raise CalcFlowError(
                    "pipeline.nodes[0]: provider test:missing@1 is not registered"
                )
            return {"valid": True, "issues": [], "fingerprint": "valid"}

    runtime = FailingRuntime()
    with _client(tmp_path, runtime=runtime) as client:
        assert _create(client).status_code == 201
        runtime.should_fail = True
        response = client.post(f"{API_PREFIX}/projects/project_alpha/validate")

    assert response.status_code == 422
    assert response.json() == {
        "detail": "pipeline.nodes[0]: provider test:missing@1 is not registered"
    }


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
    assert fetched.json()["graph"]["nodes"][0]["id"] == "calculate"
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
        missing_validation = client.post(f"{API_PREFIX}/projects/missing/validate")
        missing_checkpoint = client.get(f"{API_PREFIX}/projects/missing/checkpoint")

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
    assert "format_version: 3" in yaml_export.text
    assert unsafe.status_code == 422
    assert alias.status_code == 422
    assert oversized.status_code == 422
    assert missing.status_code == 404
    assert missing_validation.status_code == 404
    assert missing_checkpoint.status_code == 404


def test_import_streams_with_early_and_running_size_limits(
    tmp_path, monkeypatch
) -> None:
    maximum = 10 * 1024 * 1024
    parser_calls: list[int] = []

    def tracking_parser(document: str | bytes) -> ProjectDocument:
        parser_calls.append(len(document))
        return ProjectDocument.model_validate(_project())

    monkeypatch.setattr(app_module, "import_project_json", tracking_parser)
    with _client(tmp_path) as client:
        declared_oversize = client.post(
            f"{API_PREFIX}/projects/import?format=json",
            content=json.dumps(_project()),
            headers={"Content-Length": str(maximum + 1)},
        )

        def chunks():
            yield b" " * maximum
            yield b"x"

        streamed_oversize = client.post(
            f"{API_PREFIX}/projects/import?format=json", content=chunks()
        )
        malformed_length = client.post(
            f"{API_PREFIX}/projects/import?format=json",
            content=b"",
            headers={"Content-Length": "not-an-integer"},
        )
        negative_length = client.post(
            f"{API_PREFIX}/projects/import?format=json",
            content=b"",
            headers={"Content-Length": "-1"},
        )
        duplicate_length = client.post(
            f"{API_PREFIX}/projects/import?format=json",
            content=b"",
            headers=[("Content-Length", "0"), ("Content-Length", "1")],
        )

        def exact_chunks():
            yield b"x" * maximum

        exact_limit = client.post(
            f"{API_PREFIX}/projects/import?format=json", content=exact_chunks()
        )

    assert declared_oversize.status_code == 422
    assert streamed_oversize.status_code == 422
    assert malformed_length.status_code == 422
    assert negative_length.status_code == 422
    assert duplicate_length.status_code == 422
    assert exact_limit.status_code == 201
    assert parser_calls == [maximum]


def test_import_never_uses_request_body_buffering(tmp_path, monkeypatch) -> None:
    async def fail_body(_: StarletteRequest) -> bytes:
        raise AssertionError("Request.body() buffered the project import")

    monkeypatch.setattr(StarletteRequest, "body", fail_body)
    with _client(tmp_path) as client:
        imported = client.post(
            f"{API_PREFIX}/projects/import?format=json",
            content=json.dumps(_project()),
        )

    assert imported.status_code == 201


def test_export_content_disposition_encodes_untrusted_project_ids(tmp_path) -> None:
    class UntrustedStore:
        async def get(self, project_id: str) -> ProjectDocument:
            return ProjectDocument.model_validate(_project(project_id))

    app = create_app(
        project_store=UntrustedStore(),
        checkpoint_directory=tmp_path / "checkpoints",
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        exported = client.get(
            f"{API_PREFIX}/projects/unsafe%0D%0AX-Injected%3Ayes/export"
        )

    assert exported.status_code == 200
    disposition = exported.headers["content-disposition"]
    assert disposition.startswith("attachment; filename*=UTF-8''")
    assert "%0D%0A" in disposition
    assert "\r" not in disposition
    assert "\n" not in disposition


def test_injected_key_error_maps_missing_project_dependencies_to_404(tmp_path) -> None:
    class MissingStore:
        async def get(self, project_id: str) -> ProjectDocument:
            raise KeyError(project_id)

    with _client(tmp_path, project_store=MissingStore()) as client:
        responses = (
            client.get(f"{API_PREFIX}/projects/missing/export"),
            client.post(f"{API_PREFIX}/projects/missing/validate"),
            client.get(f"{API_PREFIX}/projects/missing/checkpoint"),
            client.delete(f"{API_PREFIX}/projects/missing/checkpoint"),
        )

    assert [response.status_code for response in responses] == [404, 404, 404, 404]


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


def test_v2_checkpoint_routes_are_removed(tmp_path) -> None:
    with _client(tmp_path) as client:
        assert _create(client).status_code == 201
        inspected = client.get(f"{API_PREFIX}/projects/project_alpha/checkpoint")
        reset = client.delete(f"{API_PREFIX}/projects/project_alpha/checkpoint")

    assert inspected.status_code == 404
    assert reset.status_code == 404


def test_v2_checkpoint_store_is_not_attached_to_studio(tmp_path) -> None:
    with _client(tmp_path) as client:
        state = client.app.state

    assert not hasattr(state, "checkpoint_store")


def test_v2_run_routes_are_removed(tmp_path) -> None:
    with _client(tmp_path) as client:
        assert _create(client).status_code == 201
        submitted = client.post(f"{API_PREFIX}/projects/project_alpha/runs", json={})
        fetched = client.get(f"{API_PREFIX}/runs/run_1")
        events = client.get(f"{API_PREFIX}/runs/run_1/events")
        cancelled = client.delete(f"{API_PREFIX}/runs/run_1")

    statuses = {
        submitted.status_code,
        fetched.status_code,
        events.status_code,
        cancelled.status_code,
    }
    assert statuses == {404}


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
