from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from calc_flow_studio.app import create_app


def _project() -> dict[str, object]:
    schema = [
        {"name": "symbol", "data_type": "string", "nullable": False},
        {"name": "time", "data_type": "timestamp[us, UTC]", "nullable": False},
        {"name": "sequence", "data_type": "int64", "nullable": False},
    ]
    return {
        "format_version": 3,
        "id": "asof_contract",
        "name": "ASOF contract",
        "runtime": {"mode": "stream", "options": {}},
        "sources": [
            {
                "binding": side,
                "connector": {
                    "provider": "calc-flow-connectors",
                    "name": "file",
                    "version": "2.0.0",
                },
                "format": {"name": "parquet", "version": "1"},
                "options": {"path": f"fixtures/{side}.parquet", "format": "parquet"},
                "watermark": {
                    "policy": "bounded_out_of_orderness",
                    "column": "time",
                    "delay_ms": 0,
                    "emit_interval_ms": 100,
                },
            }
            for side in ("left", "right")
        ],
        "sinks": [
            {
                "binding": "output",
                "connector": {
                    "provider": "calc-flow-connectors",
                    "name": "file",
                    "version": "2.0.0",
                },
                "options": {"path": "fixtures/output", "output": "matches"},
                "delivery": "at_least_once",
            }
        ],
        "graph": {
            "name": "ASOF",
            "nodes": [
                {
                    "id": "match",
                    "input_ports": [
                        {
                            "name": side,
                            "kind": "table",
                            "required": True,
                            "schema": schema,
                        }
                        for side in ("left", "right")
                    ],
                    "output_ports": [],
                    "operator": {
                        "kind": "stream_asof_join",
                        "spec": {
                            "left": {
                                "keys": ["symbol"],
                                "event_time": "time",
                                "sequence_by": ["sequence"],
                                "prefix": "trade",
                            },
                            "right": {
                                "keys": ["symbol"],
                                "event_time": "time",
                                "sequence_by": ["sequence"],
                                "prefix": "quote",
                            },
                            "tolerance_micros": 9_007_199_254_740_991,
                            "late_policy": "drop",
                            "limits": {
                                "max_state_rows": 100_000,
                                "max_state_bytes": 67_108_864,
                            },
                        },
                    },
                }
            ],
        },
    }


def _client(tmp_path: Path) -> TestClient:
    return TestClient(
        create_app(
            project_directory=tmp_path / "projects",
            checkpoint_directory=tmp_path / "checkpoints",
        )
    )


def test_asof_import_view_save_preserves_complete_safe_integer_declaration(tmp_path):
    document = _project()
    original = copy.deepcopy(document)
    with _client(tmp_path) as client:
        imported = client.post(
            "/api/v3/projects/import?format=json", content=json.dumps(document)
        )
        assert imported.status_code == 201, imported.text
        fetched = client.get("/api/v3/projects/asof_contract")
        assert fetched.status_code == 200, fetched.text
        stored = fetched.json()
        assert (
            stored["graph"]["nodes"][0]["operator"]
            == (original["graph"]["nodes"][0]["operator"])
        )
        stored["description"] = "updated outside the operator"
        saved = client.put("/api/v3/projects/asof_contract", json=stored)
        assert saved.status_code == 200, saved.text
        assert (
            saved.json()["graph"]["nodes"][0]["operator"]
            == (original["graph"]["nodes"][0]["operator"])
        )
    assert document == original


@pytest.mark.parametrize("invalid", [True, 1.5, -1, 9_007_199_254_740_992])
def test_asof_invalid_tolerance_preserves_structured_field_path(tmp_path, invalid):
    document = _project()
    document["graph"]["nodes"][0]["operator"]["spec"]["tolerance_micros"] = invalid
    with _client(tmp_path) as client:
        response = client.post("/api/v3/projects", json=document)
    assert response.status_code == 422, response.text
    issue = response.json()["detail"]["issues"][0]
    assert issue["path"] == "graph.nodes[0].operator.spec.tolerance_micros"
    assert issue["code"] == (
        "invalid_type" if type(invalid) in {bool, float} else "out_of_range"
    )
