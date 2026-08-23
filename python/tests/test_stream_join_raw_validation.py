from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from calc_flow import _native
from calc_flow.config import ProjectDocument


def _join_document() -> dict[str, Any]:
    return {
        "format_version": 3,
        "id": "payments",
        "name": "Payments",
        "runtime": {"mode": "stream", "options": {}},
        "graph": {
            "name": "payments",
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
                                {
                                    "name": "authorized_at",
                                    "data_type": "timestamp[us]",
                                },
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


def test_config_error_exposes_structured_raw_join_issues() -> None:
    document = _join_document()
    document["graph"]["nodes"][0]["operator"]["spec"]["bounds"]["before_micros"] = -1
    with pytest.raises(_native.ConfigError) as info:
        _native.validate_project_json(json.dumps(document))
    issues = getattr(info.value, "issues", ())
    assert len(issues) == 1
    assert issues[0]["path"] == "graph.nodes[0].operator.spec.bounds.before_micros"
    assert issues[0]["code"] == "invalid_time_bound"
    assert issues[0]["message"] == (
        "before_micros must be an integer microsecond count in 0..=9007199254740991"
    )


def test_config_error_issues_stay_empty_for_ordinary_format_errors() -> None:
    with pytest.raises(_native.ConfigError) as info:
        _native.validate_project_json('{"format_version": 3}')
    assert getattr(info.value, "issues", ()) == ()


def test_project_document_maps_raw_join_issues_to_typed_errors() -> None:
    document = _join_document()
    document["graph"]["nodes"][0]["operator"]["spec"]["join_type"] = "outer"
    with pytest.raises(ValidationError) as info:
        ProjectDocument.model_validate(document)
    entries = info.value.errors(include_input=False, include_url=False)
    assert entries[0]["type"] == "unsupported_join_type"
    assert entries[0]["loc"] == ("graph", "nodes", 0, "operator", "spec", "join_type")
    assert entries[0]["msg"] == "join_type must be the string inner"


def test_project_document_accepts_the_clean_raw_ceiling() -> None:
    document = _join_document()
    spec = document["graph"]["nodes"][0]["operator"]["spec"]
    spec["bounds"]["before_micros"] = 9_007_199_254_740_991
    spec["limits"]["max_matches_per_input_batch"] = 9_007_199_254_740_991
    parsed = ProjectDocument.model_validate(document)
    assert (
        parsed.root["graph"]["nodes"][0]["operator"]["spec"]["limits"][
            "max_matches_per_input_batch"
        ]
        == 9_007_199_254_740_991
    )
