from __future__ import annotations

import json
from collections import UserDict
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from calc_flow import (
    ConfigError,
    PipelineBuilder,
    ProjectDocument,
    Runtime,
    project_json_schema,
)


def _minimal_project() -> dict[str, object]:
    return PipelineBuilder("config").expression("calc", "b = a + 1").project


def test_project_document_delegates_schema_and_rust_defaults() -> None:
    schema = ProjectDocument.model_json_schema()
    assert schema == json.loads(project_json_schema())
    assert schema["title"] == "Calc Flow Project V2"
    assert schema["properties"]["format_version"]["const"] == 2

    project = _minimal_project()
    project.pop("data_sources")
    document = ProjectDocument.model_validate(project)

    canonical = document.canonical_json()
    assert canonical == json.dumps(
        json.loads(canonical), sort_keys=True, separators=(",", ":")
    )
    assert document.root["description"] == ""
    assert document.root["run_options"]["timeout_seconds"] == 30


def test_project_document_converts_native_errors_to_validation_errors() -> None:
    project = _minimal_project()
    for invalid in (
        {**project, "format_version": 1},
        {**project, "callable": "os.system"},
    ):
        with pytest.raises(ValidationError):
            ProjectDocument.model_validate(invalid)


@pytest.mark.parametrize(
    "invalid",
    [
        {1: "not a string key"},
        {"value": object()},
        {"value": (1, 2)},
        {"value": float("nan")},
        {"value": float("inf")},
    ],
)
def test_project_document_rejects_non_json_values(invalid: object) -> None:
    project = _minimal_project()
    project["data_sources"] = [
        {"id": "input", "input": "calc.input", "format": "json", "data": invalid}
    ]
    with pytest.raises(ValidationError):
        ProjectDocument.model_validate(project)


def test_project_document_rejects_non_builtin_json_containers_without_hooks() -> None:
    class HostileDict(dict[str, object]):
        hooks_called = False

        def items(self):
            type(self).hooks_called = True
            raise AssertionError("dict subclass hook must not run")

    class HostileList(list[object]):
        hooks_called = False

        def __iter__(self):
            type(self).hooks_called = True
            raise AssertionError("list subclass hook must not run")

    containers = (
        HostileDict(value=1),
        HostileList([1]),
        UserDict({"value": 1}),
        MappingProxyType({"value": 1}),
    )
    for container in containers:
        project = _minimal_project()
        project["data_sources"] = [
            {
                "id": "input",
                "input": "calc.input",
                "format": "inline_json",
                "data": container,
            }
        ]
        with pytest.raises(ValidationError, match="non-JSON"):
            ProjectDocument.model_validate(project)

    assert HostileDict.hooks_called is False
    assert HostileList.hooks_called is False


def test_project_document_rejects_cycles_and_excessive_depth() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValidationError, match="cycle"):
        ProjectDocument.model_validate(cyclic)

    nested: object = None
    for _ in range(34):
        nested = {"value": nested}
    with pytest.raises(ValidationError, match="depth"):
        ProjectDocument.model_validate(nested)


def test_project_document_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        ProjectDocument.model_validate_json('{"format_version":2,"format_version":2}')


def test_project_document_json_translates_decoder_recursion_errors() -> None:
    project = _minimal_project()
    project["data_sources"][0]["data"] = "deep-value"
    encoded = json.dumps(project)
    nested = "[" * 10_000 + "null" + "]" * 10_000
    encoded = encoded.replace('"deep-value"', nested)

    with pytest.raises(ValidationError, match="depth|nesting"):
        ProjectDocument.model_validate_json(encoded)


def test_project_document_does_not_mutate_inputs_or_expose_owned_values() -> None:
    project = _minimal_project()
    original = json.loads(json.dumps(project))
    document = ProjectDocument.model_validate(project)
    assert project == original

    project["pipeline"]["nodes"].clear()
    returned = document.model_dump()
    returned["pipeline"]["nodes"].clear()
    schema = ProjectDocument.model_json_schema()
    schema.clear()

    assert document.root["pipeline"]["nodes"]
    assert ProjectDocument.model_json_schema()["title"] == "Calc Flow Project V2"


def test_runtime_validation_report_uses_runtime_snapshots_defensively() -> None:
    runtime = Runtime()
    missing = PipelineBuilder("report").external(
        "calc", "missing", "expression", "1", {"expression": "x + 1"}
    )
    report = runtime.validation_report(json.dumps(missing.project))
    assert report["valid"] is False
    assert report["fingerprint"] is None
    assert any(issue["code"] == "missing_provider" for issue in report["issues"])

    report["issues"].clear()
    repeated = runtime.validation_report(json.dumps(missing.project))
    assert repeated["issues"]

    valid = runtime.validation_report(
        json.dumps(PipelineBuilder("valid").expression("calc", "b = a + 1").project)
    )
    assert valid["valid"] is True
    assert valid["issues"] == []
    assert valid["fingerprint"]


def test_runtime_compile_stream_project_requires_source_coverage() -> None:
    project = _minimal_project()
    project["data_sources"] = []

    with pytest.raises(ConfigError, match=r"data_sources \[source_input_mismatch\]"):
        Runtime().compile_stream_project(json.dumps(project))


def test_runtime_validation_report_uses_current_udf_snapshot() -> None:
    runtime = Runtime()
    project = PipelineBuilder("report_udf").expression(
        "calc",
        "result = identity(value)",
        udfs=(("python", "identity", "1"),),
    )
    missing = runtime.validation_report(json.dumps(project.project))
    assert missing["valid"] is False
    assert any(issue["code"] == "missing_udf" for issue in missing["issues"])

    runtime.register_scalar_udf(
        provider="python",
        name="identity",
        version="1",
        input_types=("int64",),
        return_type="int64",
        volatility="immutable",
        function=lambda value: value,
    )
    available = runtime.validation_report(json.dumps(project.project))
    assert available["valid"] is True
    assert available["fingerprint"]
