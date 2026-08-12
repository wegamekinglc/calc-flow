from __future__ import annotations

import json
from pathlib import Path

_EXPECTED_CATEGORIES = (
    "validation",
    "compile",
    "conflict",
    "cancelled",
    "checkpoint_timeout",
    "checkpoint_mismatch",
    "checkpoint_publication_unknown",
    "io",
    "operator",
    "connector",
    "task_panicked",
    "internal",
)


def test_streaming_error_vectors_freeze_python_safe_projection() -> None:
    fixture = (
        Path(__file__).parents[2]
        / "tests"
        / "fixtures"
        / "a6"
        / "streaming_error_projection.json"
    )

    document = json.loads(fixture.read_text(encoding="utf-8"))

    assert document["schema_version"] == 1
    assert tuple(document["categories"]) == _EXPECTED_CATEGORIES
    assert [vector["case"] for vector in document["vectors"]] == sorted(
        vector["case"] for vector in document["vectors"]
    )
    for vector in document["vectors"]:
        expected = vector["expected"]
        assert set(expected) == {
            "category",
            "message",
            "job_id",
            "epoch",
            "checkpoint_phase",
            "component_kind",
            "component_id",
            "diagnostic_id",
            "position",
        }
        assert expected["category"] in _EXPECTED_CATEGORIES
        rendered = json.dumps(expected, sort_keys=True)
        for sentinel in vector["private_sentinels"]:
            assert sentinel not in rendered
