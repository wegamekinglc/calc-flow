from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from calc_flow_studio.app import create_app

API_PREFIX = "/api/v3"


def _join_project(project_id: str = "join_project") -> dict[str, Any]:
    return {
        "format_version": 3,
        "id": project_id,
        "name": "Join project",
        "description": "",
        "runtime": {"mode": "stream", "options": {}},
        "graph": {
            "name": "join-project",
            "nodes": [
                {
                    "id": "match",
                    "input_ports": [
                        {
                            "name": "left",
                            "kind": "table",
                            "required": True,
                            "schema": [
                                {"name": "account_id", "data_type": "int64"},
                                {"name": "authorized_at", "data_type": "timestamp[us]"},
                            ],
                        },
                        {
                            "name": "right",
                            "kind": "table",
                            "required": True,
                            "schema": [
                                {"name": "account_id", "data_type": "int64"},
                                {"name": "paid_at", "data_type": "timestamp[us]"},
                            ],
                        },
                    ],
                    "output_ports": [],
                    "operator": {
                        "kind": "stream_join",
                        "spec": {
                            "join_type": "inner",
                            "left_keys": ["account_id"],
                            "right_keys": ["account_id"],
                            "left_event_time": "authorized_at",
                            "right_event_time": "paid_at",
                            "bounds": {
                                "before_micros": 300_000_000,
                                "after_micros": 30_000_000,
                            },
                            "limits": {
                                "max_state_rows_per_side": 100_000,
                                "max_state_bytes_per_side": 134_217_728,
                                "max_matches_per_input_batch": 1_000_000,
                            },
                            "left_prefix": "authorization",
                            "right_prefix": "payment",
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


def test_create_with_malformed_raw_join_returns_the_validation_report_envelope(
    tmp_path,
) -> None:
    document = _join_project()
    document["graph"]["nodes"][0]["operator"]["spec"]["bounds"]["before_micros"] = -1
    with _client(tmp_path) as client:
        response = client.post(f"{API_PREFIX}/projects", json=document)

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "kind": "invalid",
            "valid": False,
            "issues": [
                {
                    "path": ("graph.nodes[0].operator.spec.bounds.before_micros"),
                    "code": "invalid_time_bound",
                    "message": (
                        "before_micros must be an integer microsecond count "
                        "in 0..=9007199254740991"
                    ),
                }
            ],
            "fingerprint": None,
        }
    }


def test_import_with_malformed_raw_join_returns_the_validation_report_envelope(
    tmp_path,
) -> None:
    document = _join_project()
    document["graph"]["nodes"][0]["operator"]["spec"]["join_type"] = "outer"
    with _client(tmp_path) as client:
        response = client.post(
            f"{API_PREFIX}/projects/import?format=json",
            content=json.dumps(document),
        )

    assert response.status_code == 422
    body = response.json()["detail"]
    assert body["kind"] == "invalid"
    assert body["valid"] is False
    assert body["fingerprint"] is None
    assert body["issues"] == [
        {
            "path": "graph.nodes[0].operator.spec.join_type",
            "code": "unsupported_join_type",
            "message": "join_type must be the string inner",
        }
    ]


def test_semantic_join_issues_return_the_validation_report_envelope(tmp_path) -> None:
    class InvalidJoinRuntime:
        def catalog(self) -> list[dict[str, object]]:
            return []

        def validation_report(self, project_json: str) -> dict[str, object]:
            return {
                "valid": False,
                "issues": [
                    {
                        "path": "graph.nodes[0].operator.spec.left_event_time",
                        "code": "invalid_event_time",
                        "message": (
                            "left_event_time must be a timezone-naive or UTC "
                            "Arrow timestamp; found Int64"
                        ),
                    }
                ],
                "fingerprint": None,
            }

    with TestClient(
        create_app(
            project_directory=tmp_path / "projects",
            checkpoint_directory=tmp_path / "checkpoints",
            runtime=InvalidJoinRuntime(),
        )
    ) as client:
        response = client.post(f"{API_PREFIX}/projects", json=_join_project())

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "kind": "invalid",
            "valid": False,
            "issues": [
                {
                    "path": "graph.nodes[0].operator.spec.left_event_time",
                    "code": "invalid_event_time",
                    "message": (
                        "left_event_time must be a timezone-naive or UTC "
                        "Arrow timestamp; found Int64"
                    ),
                }
            ],
            "fingerprint": None,
        }
    }
