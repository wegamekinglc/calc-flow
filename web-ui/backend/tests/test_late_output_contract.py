from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from calc_flow import ProjectDocument, Runtime
from fastapi.testclient import TestClient

from calc_flow_studio.app import create_app
from calc_flow_studio.run_manager import RunManager


def _project(tmp_path: Path, kind: str = "rolling") -> dict:
    schema = [
        {"name": "ts", "data_type": "timestamp[us, UTC]", "nullable": False},
        {"name": "symbol", "data_type": "string", "nullable": False},
        {"name": "seq", "data_type": "uint64", "nullable": False},
        {"name": "x", "data_type": "float64", "nullable": False},
    ]
    spec = {
        "configuration_version": 1,
        "state_layout_version": 1,
        "sequence_by": ["seq"],
        "value_policy": "stateful_numeric_v1"
        if kind == "rolling"
        else "nan_exclude_preserve_v1",
        "event_time": "ts",
        "allowed_lateness_micros": 0,
        "late_policy": {
            "kind": "side_output",
            "metrics_version": 1,
            "schema_version": 1,
        },
    }
    if kind == "rolling":
        spec.update(
            partition_by=["symbol"],
            outputs=[
                {
                    "kind": "mean",
                    "primitive_version": 1,
                    "input": "x",
                    "output": "mean",
                    "frame": {"kind": "rows", "size": 2},
                    "min_periods": 1,
                }
            ],
        )
    else:
        spec.update(
            entity_by=["symbol"],
            grouping={"kind": "exact_time"},
            outputs=[
                {
                    "kind": "rank",
                    "primitive_version": 1,
                    "input": "x",
                    "output": "rank",
                    "tie_method": "average",
                    "direction": "ascending",
                    "null_placement": "last",
                    "min_samples": 1,
                }
            ],
        )
    connector = {"provider": "calc-flow-connectors", "name": "file", "version": "2.0.0"}
    return {
        "format_version": 3,
        "id": "late_contract",
        "name": "Late contract",
        "runtime": {"mode": "stream", "options": {}},
        "graph": {
            "name": "late-contract",
            "nodes": [
                {
                    "id": "roll",
                    "operator": {"kind": kind, "spec": spec},
                    "input_ports": [
                        {
                            "name": "input",
                            "kind": "table",
                            "required": True,
                            "schema": schema,
                        }
                    ],
                    "output_ports": [],
                }
            ],
            "edges": [],
        },
        "sources": [
            {
                "binding": "input",
                "connector": connector,
                "schema": schema,
                "options": {
                    "path": str(tmp_path / "input.parquet"),
                    "format": "parquet",
                },
                "watermark": {
                    "policy": "bounded_out_of_orderness",
                    "column": "ts",
                    "delay_ms": 1,
                    "emit_interval_ms": 1,
                },
            }
        ],
        "sinks": [
            {
                "binding": name,
                "connector": connector,
                "options": {"path": str(tmp_path / name), "output": "rows"},
                "delivery": "at_least_once",
            }
            for name in ("output", "late")
        ],
        "state": {"root": str(tmp_path / "state"), "retention": 3},
    }


@pytest.mark.parametrize("kind", ["rolling", "cross_section"])
def test_normal_schema_edit_matches_frontend_fixture_and_native_validation(
    tmp_path, kind
):
    fields = json.loads(
        (
            Path(__file__).parents[2] / "src/components/lateSchema.fixture.json"
        ).read_text()
    )
    document = _project(tmp_path, kind)
    node = document["graph"]["nodes"][0]
    node["operator"]["spec"]["outputs"][0]["output"] = "result"
    assert node["input_ports"][0]["schema"] == fields["input"]
    app = create_app(
        project_directory=tmp_path / "projects", checkpoint_directory=tmp_path / "jobs"
    )
    with TestClient(app) as client:
        assert client.post("/api/v3/projects", json=document).status_code == 201
        path = "/api/v3/projects/late_contract"
        assert client.post(path + "/validate").json()["valid"] is True
        node["output_ports"] = [
            {"name": name, "kind": "table", "required": True, "schema": fields[name]}
            for name in ("output", "late")
        ]
        updated = client.put(path, json=document)
        assert updated.status_code == 200, updated.text
        assert (
            updated.json()["graph"]["nodes"][0]["output_ports"] == node["output_ports"]
        )
        assert client.post(path + "/validate").json()["valid"] is True
        node["output_ports"][1]["schema"] = []
        invalid = client.put(path, json=document)
        assert invalid.status_code == 422, invalid.text
        issue = invalid.json()["detail"]["issues"][0]
        assert (issue["code"], issue["path"]) == (
            "schema_mismatch",
            "graph.nodes[0].output_ports",
        )


class _StoredProject:
    def __init__(self, document: dict) -> None:
        self.project = ProjectDocument.model_validate(document)

    async def get(self, project_id: str) -> ProjectDocument:
        return self.project


def _diagnostic_fields() -> list[dict]:
    return [
        {"name": "_cf_late_" + name, "data_type": dtype, "nullable": False}
        for name, dtype in (
            ("node", "string"),
            ("input_port", "string"),
            ("event_time_micros", "int64"),
            ("closing_time_micros", "int64"),
            ("watermark_micros", "int64"),
            ("reason", "string"),
            ("source", "string"),
            ("sequence", "uint64"),
            ("row_index", "uint64"),
        )
    ]


def test_late_job_revalidates_bindings_before_worker_submission(tmp_path):
    document = _project(tmp_path)
    document["sinks"] = document["sinks"][:1]
    manager = Mock()
    manager.submit_job.side_effect = AssertionError("worker submission reached")
    app = create_app(project_store=_StoredProject(document), run_manager=manager)
    with TestClient(app) as client:
        response = client.post("/api/v3/jobs", json={"project_id": document["id"]})
    assert response.status_code == 422, response.text
    issue = response.json()["detail"]["issues"][0]
    assert issue["code"] == "missing_binding"
    assert issue["path"] == "sinks"
    manager.submit_job.assert_not_called()


@pytest.mark.parametrize("kind", ["rolling", "cross_section"])
def test_late_project_roundtrip_preserves_policy_bindings_and_delivery(tmp_path, kind):
    document = _project(tmp_path, kind)
    original = copy.deepcopy(document)
    app = create_app(
        project_directory=tmp_path / "projects", checkpoint_directory=tmp_path / "jobs"
    )
    with TestClient(app) as client:
        created = client.post("/api/v3/projects", json=document)
        assert created.status_code == 201, created.text
        canonical = created.json()
        path = "/api/v3/projects/late_contract"
        assert client.get(path).json() == canonical
        assert client.put(path, json=canonical).json() == canonical
        assert client.post(path + "/validate").json()["valid"] is True
        exported = client.get(path + "/export?format=json")
        assert json.loads(exported.text) == canonical
        assert client.delete(path).status_code == 204
        imported = client.post(
            "/api/v3/projects/import?format=json", content=exported.text
        )
        assert imported.status_code == 201, imported.text
        assert imported.json() == canonical
    assert document == original


def test_late_explicit_ports_and_row_local_edges_roundtrip(tmp_path):
    document = _project(tmp_path)
    node = document["graph"]["nodes"][0]
    fields = node["input_ports"][0]["schema"]
    node["output_ports"] = [
        {
            "name": "output",
            "kind": "table",
            "required": True,
            "schema": fields
            + [{"name": "mean", "data_type": "float64", "nullable": True}],
        },
        {
            "name": "late",
            "kind": "table",
            "required": True,
            "schema": fields + _diagnostic_fields(),
        },
    ]
    document["graph"]["nodes"].append(
        {
            "id": "diagnostics",
            "operator": {
                "kind": "sql",
                "query": "SELECT * FROM input",
                "aliases": ["input"],
            },
            "input_ports": [
                {
                    "name": "input",
                    "kind": "table",
                    "required": True,
                    "schema": fields + _diagnostic_fields(),
                }
            ],
        }
    )
    document["graph"]["edges"] = [
        {
            "source_node": "roll",
            "source_port": "late",
            "target_node": "diagnostics",
            "target_port": "input",
        }
    ]
    document["sinks"][0]["binding"] = "roll.output"
    document["sinks"][1]["binding"] = "diagnostics.output"
    document["sinks"][1]["delivery"] = "best_effort"
    with TestClient(create_app(project_directory=tmp_path / "projects")) as client:
        response = client.post("/api/v3/projects", json=document)
        assert response.status_code == 201, response.text
        exported = client.get(
            "/api/v3/projects/late_contract/export?format=json"
        ).json()
    assert exported["graph"]["nodes"][0]["output_ports"] == node["output_ports"]
    assert exported["graph"]["edges"] == document["graph"]["edges"]
    assert [sink["binding"] for sink in exported["sinks"]] == [
        "roll.output",
        "diagnostics.output",
    ]
    assert exported["sinks"][1]["delivery"] == "best_effort"


def test_late_job_entry_runs_native_two_sink_project_without_json_rows(tmp_path):
    document = _project(tmp_path)
    pq.write_table(
        pa.table(
            {
                "ts": pa.array(
                    [9_007_199_254_740_993], type=pa.timestamp("us", tz="UTC")
                ),
                "symbol": ["A"],
                "seq": pa.array([2**64 - 1], type=pa.uint64()),
                "x": [2.0],
            },
            schema=pa.schema(
                [
                    pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                    pa.field("symbol", pa.string(), nullable=False),
                    pa.field("seq", pa.uint64(), nullable=False),
                    pa.field("x", pa.float64(), nullable=False),
                ]
            ),
        ),
        tmp_path / "input.parquet",
    )
    manager = RunManager(
        runtime=Runtime(), use_processes=False, checkpoint_directory=tmp_path / "jobs"
    )
    with TestClient(
        create_app(project_directory=tmp_path / "projects", run_manager=manager)
    ) as client:
        created = client.post("/api/v3/projects", json=document)
        assert created.status_code == 201, created.text
        started = client.post("/api/v3/jobs", json={"project_id": document["id"]})
        assert started.status_code == 202, started.text
        job_id = started.json()["id"]
        events = client.get(f"/api/v3/jobs/{job_id}/events")
        assert events.status_code == 200
        finished = client.get(f"/api/v3/jobs/{job_id}").json()
        assert finished["status"] == "completed", json.dumps(finished)
        assert "outputs" not in finished
        assert "9007199254740993" not in events.text
        assert "18446744073709551615" not in events.text
    tables = [pq.read_table(path) for path in (tmp_path / "output").rglob("*.parquet")]
    assert pa.concat_tables(tables)["seq"].to_pylist() == [2**64 - 1]
    assert pa.concat_tables(tables)["ts"].cast(pa.int64()).to_pylist() == [
        9_007_199_254_740_993
    ]


@pytest.mark.parametrize("component", ["ProjectCreateRequest", "ProjectDocument"])
def test_late_openapi_embedded_schema_matches_native_and_checked_schema(component):
    from calc_flow import project_json_schema

    root = Path(__file__).resolve().parents[3]
    canonical = json.loads(project_json_schema())
    checked = json.loads((root / "schemas/project-v3.schema.json").read_text())
    openapi = create_app(run_manager=RunManager(use_processes=False)).openapi()
    serialized = json.dumps(openapi["components"]["schemas"][component])
    normalized = json.loads(
        serialized.replace(f"#/components/schemas/{component}/$defs/", "#/$defs/")
    )
    assert normalized == canonical == checked
    diagnostics = _diagnostic_fields()
    assert [field["data_type"] for field in diagnostics[2:5]] == ["int64"] * 3
    assert [field["data_type"] for field in diagnostics[-2:]] == ["uint64"] * 2


@pytest.mark.parametrize("case", ["batch", "reserved", "schema", "extra", "temporal"])
def test_raw_late_requests_keep_validation_report_envelope(tmp_path, case):
    document = _project(tmp_path)
    _invalidate(document, document["graph"]["nodes"][0], case)
    with TestClient(create_app(project_directory=tmp_path / "projects")) as client:
        response = client.post("/api/v3/projects", json=document)
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, dict), detail
    assert detail["kind"] == "invalid"
    assert detail["issues"][0]["path"].startswith("graph.")


@pytest.mark.parametrize(
    "code,field,value",
    [
        ("unsupported_version", "schema_version", 2),
        ("unsupported_version", "metrics_version", 2),
    ],
)
def test_late_policy_errors_keep_structured_locations(tmp_path, code, field, value):
    document = _project(tmp_path)
    document["graph"]["nodes"][0]["operator"]["spec"]["late_policy"][field] = value
    with TestClient(create_app(project_directory=tmp_path / "projects")) as client:
        response = client.post("/api/v3/projects", json=document)
    assert response.status_code == 422, response.text
    issue = response.json()["detail"]["issues"][0]
    assert issue["code"] == code
    assert issue["path"].startswith("graph.nodes[0].operator.spec.late_policy")


def test_late_capability_comes_from_runtime_validation(tmp_path):
    runtime = Runtime()
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.get("/api/v3/capabilities")
    assert response.status_code == 200, response.text
    assert response.json()["runtime"]["lateOutput"] == {
        "operators": ["cross_section", "rolling"],
        "schemaVersion": 1,
        "metricsVersion": 1,
    }


def test_old_runtime_cannot_claim_late_support_from_package_version(
    tmp_path, monkeypatch
):
    runtime = Runtime()
    validation = Mock(return_value={"valid": False, "issues": [], "fingerprint": None})
    monkeypatch.setattr(Runtime, "validation_report", validation)
    with TestClient(create_app(runtime=runtime)) as client:
        response = client.get("/api/v3/capabilities")
    assert response.json()["runtime"]["packageVersion"] == "2026.9.24"
    assert response.json()["runtime"]["lateOutput"]["operators"] == []
    assert validation.call_count == 2


def test_old_runtime_rejects_stored_late_project_before_submission(
    tmp_path, monkeypatch
):
    runtime = Runtime()
    issue = {
        "path": "graph.nodes[0].operator.spec.late_policy",
        "code": "unsupported_mode",
        "message": "late side output is unavailable",
    }
    monkeypatch.setattr(
        Runtime,
        "validation_report",
        Mock(
            return_value={
                "valid": False,
                "issues": [issue],
                "fingerprint": None,
            }
        ),
    )
    manager = Mock()
    with TestClient(
        create_app(
            runtime=runtime,
            project_store=_StoredProject(_project(tmp_path)),
            run_manager=manager,
        )
    ) as client:
        response = client.post("/api/v3/jobs", json={"project_id": "late_contract"})
    assert response.status_code == 422
    assert response.json()["detail"]["issues"] == [issue]
    manager.submit_job.assert_not_called()


def test_late_malformed_runtime_report_is_internal_before_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Runtime, "validation_report", Mock(return_value={"valid": False})
    )
    manager = Mock()
    with TestClient(
        create_app(
            project_store=_StoredProject(_project(tmp_path)), run_manager=manager
        )
    ) as client:
        response = client.post("/api/v3/jobs", json={"project_id": "late_contract"})
    assert response.status_code == 500
    assert "runtime validation report violates" in response.json()["detail"]
    manager.submit_job.assert_not_called()


@pytest.mark.parametrize("kind", ["union", "external"])
def test_late_illegal_merge_and_provider_are_rejected_before_worker(tmp_path, kind):
    from calc_flow import register_numpy

    document = _project(tmp_path)
    fields = (
        document["graph"]["nodes"][0]["input_ports"][0]["schema"] + _diagnostic_fields()
    )
    names = ["left", "right"] if kind == "union" else ["input"]
    operator = (
        {"kind": "union"}
        if kind == "union"
        else {
            "kind": "external",
            "provider": "numpy",
            "name": "expression",
            "version": "1",
            "options": {"expression": "x + 1"},
        }
    )
    document["graph"]["nodes"].append(
        {
            "id": "next",
            "operator": operator,
            "input_ports": [
                {"name": name, "kind": "table", "required": True, "schema": fields}
                for name in names
            ],
            "output_ports": [
                {"name": "output", "kind": "table", "required": True, "schema": fields}
            ],
        }
    )
    document["graph"]["edges"] = [
        {
            "source_node": "roll",
            "source_port": "late",
            "target_node": "next",
            "target_port": name,
        }
        for name in names
    ]
    manager = Mock()
    runtime = Runtime()
    register_numpy(runtime)
    store = _StoredProject(document)
    with TestClient(
        create_app(runtime=runtime, project_store=store, run_manager=manager)
    ) as client:
        response = client.post("/api/v3/jobs", json={"project_id": "late_contract"})
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["valid"] is False
    manager.submit_job.assert_not_called()


@pytest.mark.parametrize(
    "case,code,path",
    [
        ("reserved", "reserved_field", "graph.nodes[0].input_ports[0].schema[0].name"),
        ("schema", "schema_mismatch", "graph.nodes[0].output_ports"),
        ("extra", "schema_mismatch", "graph.nodes[0].output_ports"),
        ("batch", "unsupported_mode", "graph.nodes[0].operator.spec.late_policy"),
        ("temporal", "temporal_output_unavailable", "graph.edges[0]"),
    ],
)
def test_invalid_late_project_never_submits_worker(tmp_path, case, code, path):
    document = _project(tmp_path)
    node = document["graph"]["nodes"][0]
    _invalidate(document, node, case)
    manager = Mock()
    manager.submit_job.side_effect = AssertionError("worker submission reached")
    store = _StoredProject.__new__(_StoredProject)
    store.project = ProjectDocument.model_construct(root=document)
    with TestClient(create_app(project_store=store, run_manager=manager)) as client:
        response = client.post("/api/v3/jobs", json={"project_id": document["id"]})
    assert response.status_code == 422, response.text
    issue = response.json()["detail"]["issues"][0]
    assert issue["code"] == code, issue
    assert issue["path"] == path, issue
    manager.submit_job.assert_not_called()


def _invalidate(document, node, case):
    if case == "reserved":
        node["input_ports"][0]["schema"][0]["name"] = "_cf_late_ts"
    elif case in {"schema", "extra"}:
        node["output_ports"] = [
            {
                "name": name,
                "kind": "table",
                "required": True,
                "schema": [{"name": "bad", "data_type": "int64", "nullable": False}],
            }
            for name in ("output", "late")
        ]
        if case == "extra":
            node["output_ports"].append(
                {"name": "extra", "kind": "table", "required": True, "schema": []}
            )
    elif case == "batch":
        document["runtime"] = {"mode": "batch", "options": {}}
        document["sources"] = []
        document["sinks"] = []
    else:
        successor = copy.deepcopy(node)
        successor["id"] = "next"
        successor["operator"]["spec"]["late_policy"] = {
            "kind": "drop",
            "metrics_version": 1,
        }
        successor["input_ports"][0]["schema"] += _diagnostic_fields()
        document["graph"]["nodes"].append(successor)
        document["graph"]["edges"] = [
            {
                "source_node": "roll",
                "source_port": "late",
                "target_node": "next",
                "target_port": "input",
            }
        ]


def test_stream_compile_message_with_digit_code_keeps_parsed_path(
    tmp_path, monkeypatch
):
    issue = {
        "path": "project",
        "code": "stream_compile",
        "message": "stored document is invalid:"
        " graph.nodes[0].operator.spec.late_policy.metrics_version"
        " [unsupported_version_2]: drop metrics_version must equal 1; found 2",
    }
    monkeypatch.setattr(
        Runtime,
        "validation_report",
        Mock(return_value={"valid": False, "issues": [issue], "fingerprint": None}),
    )
    store = _StoredProject(_project(tmp_path))
    with TestClient(create_app(project_store=store, run_manager=Mock())) as client:
        response = client.post("/api/v3/jobs", json={"project_id": "late_contract"})
    assert response.status_code == 422, response.text
    parsed = response.json()["detail"]["issues"][0]
    assert parsed["code"] == "unsupported_version_2"
    assert parsed["path"] == "graph.nodes[0].operator.spec.late_policy.metrics_version"


def test_unmatched_stream_compile_message_logs_and_keeps_generic_issue(
    tmp_path, monkeypatch, caplog
):
    issue = {
        "path": "project",
        "code": "stream_compile",
        "message": "graph compilation failed: rewired native wording sentinel",
    }
    monkeypatch.setattr(
        Runtime,
        "validation_report",
        Mock(return_value={"valid": False, "issues": [issue], "fingerprint": None}),
    )
    store = _StoredProject(_project(tmp_path))
    with (
        caplog.at_level(logging.WARNING, logger="calc_flow_studio.late_output"),
        TestClient(create_app(project_store=store, run_manager=Mock())) as client,
    ):
        response = client.post("/api/v3/jobs", json={"project_id": "late_contract"})
    assert response.status_code == 422, response.text
    assert response.json()["detail"]["issues"] == [issue]
    assert any(
        record.levelno == logging.WARNING
        and "matched no known late-output pattern" in record.message
        and "rewired native wording sentinel" in record.message
        for record in caplog.records
    )
