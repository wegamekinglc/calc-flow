from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import UserDict
from copy import deepcopy
from types import MappingProxyType

import pytest

from calc_flow import (
    ConfigError,
    PipelineBuilder,
    ProjectDocument,
    Runtime,
    project_json_schema,
)
from calc_flow import ProjectValidationError as ValidationError
from calc_flow.config import _openapi_project_schema


def _minimal_project() -> dict[str, object]:
    return PipelineBuilder("config").expression("calc", "b = a + 1").project


def _project_with_data(value: object) -> dict[str, object]:
    project = _minimal_project()
    project["data_sources"][0]["data"] = value
    return project


def _assert_json_error(project: object, message: str) -> None:
    with pytest.raises(ValidationError) as caught:
        ProjectDocument.model_validate(project)
    assert [
        (error["type"], error["loc"], error["msg"])
        for error in caught.value.errors(include_url=False)
    ] == [("value_error", (), f"Value error, {message}")]


def test_project_document_works_without_pydantic_installed() -> None:
    script = """
import os
import sys
expected = os.environ["CALC_FLOW_TEST_PYTHON"]
assert os.path.samefile(sys.executable, expected)
sys.modules["pydantic"] = None
sys.modules["pydantic_core"] = None
from calc_flow import PipelineBuilder, ProjectDocument
project = PipelineBuilder("config").expression("calc", "b = a + 1").project
document = ProjectDocument.model_validate(project)
assert document.model_dump(mode="json")["format_version"] == 3
round_tripped = ProjectDocument.model_validate_json(document.model_dump_json())
assert round_tripped.root == document.root
"""
    environment = os.environ.copy()
    environment["PATH"] = (
        os.path.dirname(sys.executable) + os.pathsep + environment.get("PATH", "")
    )
    environment["CALC_FLOW_TEST_PYTHON"] = sys.executable
    completed = subprocess.run(
        ["python", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_project_document_delegates_schema_and_rust_defaults() -> None:
    schema = ProjectDocument.model_json_schema()
    assert schema == json.loads(project_json_schema())
    assert schema["title"] == "Calc Flow Project V3"
    assert schema["properties"]["format_version"]["const"] == 3

    project = _minimal_project()
    project.pop("data_sources")
    document = ProjectDocument.model_validate(project)

    canonical = document.canonical_json()
    assert canonical == json.dumps(
        json.loads(canonical), sort_keys=True, separators=(",", ":")
    )
    assert document.root["description"] == ""
    assert document.root["runtime"]["options"]["timeout_seconds"] == 30


def test_project_document_exclude_none_preserves_nested_json_nulls() -> None:
    value = {"keep_null": None, "nested": [{"also_keep": None}]}
    document = ProjectDocument.model_validate(_project_with_data(value))

    dumped = document.model_dump(exclude_none=True)
    dumped_json = json.loads(document.model_dump_json(exclude_none=True))
    assert dumped["data_sources"][0]["data"] == value
    assert dumped_json["data_sources"][0]["data"] == value


def test_project_document_equality_requires_the_same_concrete_type() -> None:
    class ChildDocument(ProjectDocument):
        pass

    class OtherChildDocument(ProjectDocument):
        pass

    project = _minimal_project()
    document = ProjectDocument.model_validate(project)
    child = ChildDocument.model_validate(project)
    other_child = OtherChildDocument.model_validate(project)

    assert child == ChildDocument.model_validate(project)
    assert document != child
    assert child != document
    assert child != other_child
    assert other_child != child


def test_project_document_openapi_schema_rewrites_component_references() -> None:
    schema = _openapi_project_schema(ProjectDocument.__name__)
    references: list[str] = []
    pending: list[object] = [schema]
    while pending:
        value = pending.pop()
        if type(value) is dict:
            reference = value.get("$ref")
            if isinstance(reference, str):
                references.append(reference)
            pending.extend(value.values())
        elif type(value) is list:
            pending.extend(value)

    assert references
    assert all(
        reference.startswith("#/components/schemas/ProjectDocument/$defs/")
        for reference in references
    )


def test_project_document_converts_native_errors_to_validation_errors() -> None:
    project = _minimal_project()
    for invalid in (
        {**project, "format_version": 1},
        {**project, "callable": "os.system"},
    ):
        with pytest.raises(ValidationError):
            ProjectDocument.model_validate(invalid)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ({1: "not a string key"}, "project JSON object keys must be strings"),
        (object(), "project contains a non-JSON value of type object"),
        ((1, 2), "project contains a non-JSON value of type tuple"),
        (b"x", "project contains a non-JSON value of type bytes"),
        ({1}, "project contains a non-JSON value of type set"),
        (float("nan"), "project JSON numbers must be finite"),
        (float("inf"), "project JSON numbers must be finite"),
        (float("-inf"), "project JSON numbers must be finite"),
        (-(2**63) - 1, "project integer is outside the portable JSON range"),
        (2**64, "project integer is outside the portable JSON range"),
    ],
)
def test_project_document_rejects_non_json_values(
    invalid: object, message: str
) -> None:
    _assert_json_error(_project_with_data(invalid), message)


@pytest.mark.parametrize("key", [1, None, False, (1,)])
def test_project_document_rejects_non_string_json_keys(key: object) -> None:
    _assert_json_error(
        _project_with_data({key: "value"}), "project JSON object keys must be strings"
    )


@pytest.mark.parametrize(
    "value", [None, False, True, "", "中文", -(2**63), 2**64 - 1, 0, 1.25, [], {}]
)
def test_project_document_accepts_portable_json_values(value: object) -> None:
    document = ProjectDocument.model_validate(_project_with_data(value))
    assert document.root["data_sources"][0]["data"] == value


def test_project_document_accepts_scalar_subclasses_and_string_subclass_keys() -> None:
    class IntSubclass(int):
        pass

    class FloatSubclass(float):
        pass

    class StrSubclass(str):
        pass

    value = {StrSubclass("key"): [IntSubclass(1), FloatSubclass(1.5), StrSubclass("x")]}
    document = ProjectDocument.model_validate(_project_with_data(value))
    assert document.root["data_sources"][0]["data"] == {"key": [1, 1.5, "x"]}


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
        _assert_json_error(
            _project_with_data(container),
            f"project contains a non-JSON value of type {type(container).__name__}",
        )

    assert HostileDict.hooks_called is False
    assert HostileList.hooks_called is False


@pytest.mark.parametrize("kind", ["dict", "list", "mixed"])
def test_project_document_rejects_cycles_without_mutating_them(kind: str) -> None:
    mapping: dict[str, object] = {}
    items: list[object] = []
    if kind == "dict":
        mapping["self"] = mapping
        cyclic = mapping
    elif kind == "list":
        items.append(items)
        cyclic = items
    else:
        mapping["items"] = items
        items.append(mapping)
        cyclic = mapping

    _assert_json_error(_project_with_data(cyclic), "project contains a cycle")
    if kind == "dict":
        assert mapping["self"] is mapping
    elif kind == "list":
        assert items[0] is items
    else:
        assert mapping["items"] is items
        assert items[0] is mapping


@pytest.mark.parametrize("kind", ["dict", "list"])
@pytest.mark.parametrize("depth", [32, 33])
def test_project_document_preserves_json_depth_boundary(kind: str, depth: int) -> None:
    nested: object = None
    # Source data starts at depth 3 in the project document.
    for _ in range(depth - 3):
        nested = {"value": nested} if kind == "dict" else [nested]
    project = _project_with_data(nested)
    if depth == 32:
        document = ProjectDocument.model_validate(project)
        assert document.root["data_sources"][0]["data"] == nested
    else:
        _assert_json_error(project, "project exceeds the maximum JSON depth of 32")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            {"nested": [object()], 1: "bad key"},
            "project JSON object keys must be strings",
        ),
        ([float("nan"), (1,)], "project contains a non-JSON value of type tuple"),
        (
            {"first": float("nan"), "last": (1,)},
            "project contains a non-JSON value of type tuple",
        ),
        (
            {"nested": {1: "bad key"}, "last": (1,)},
            "project contains a non-JSON value of type tuple",
        ),
    ],
)
def test_project_document_preserves_first_json_error(
    value: object, message: str
) -> None:
    _assert_json_error(_project_with_data(value), message)


def test_project_document_checks_depth_before_scalar_type_or_cycle() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    for leaf in (object(), cyclic):
        nested = leaf
        for _ in range(30):
            nested = [nested]
        _assert_json_error(
            _project_with_data(nested), "project exceeds the maximum JSON depth of 32"
        )


def test_project_document_accepts_shared_children_without_mutating_inputs() -> None:
    shared = [1, {"x": False}]
    value = {"left": shared, "right": [shared, shared]}
    project = _project_with_data(value)
    original = deepcopy(project)

    document = ProjectDocument.model_validate(project)

    assert project == original
    assert project["data_sources"][0]["data"] is value
    assert value["left"] is shared
    assert all(child is shared for child in value["right"])
    assert document.root["data_sources"][0]["data"] == value


def test_project_document_json_rejects_duplicate_keys() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        ProjectDocument.model_validate_json('{"format_version":3,"format_version":3}')


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

    project["graph"]["nodes"].clear()
    returned = document.model_dump()
    returned["graph"]["nodes"].clear()
    schema = ProjectDocument.model_json_schema()
    schema.clear()

    assert document.root["graph"]["nodes"]
    assert ProjectDocument.model_json_schema()["title"] == "Calc Flow Project V3"


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
    project["runtime"] = {"mode": "stream", "options": {}}

    with pytest.raises(ConfigError, match=r"sources \[source_input_mismatch\]"):
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
