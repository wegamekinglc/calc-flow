from __future__ import annotations

import json

import pytest

from calc_flow.config import NodeConfig, PipelineConfig, ProjectConfig
from calc_flow.project_store import (
    FileProjectStore,
    ProjectConflictError,
    ProjectFormatError,
    ProjectNotFoundError,
    canonical_project_json,
    export_project_document,
    load_project_document,
)


def _project(project_id: str = "demo", name: str = "Demo") -> ProjectConfig:
    return ProjectConfig(
        id=project_id,
        name=name,
        pipeline=PipelineConfig(
            id="main",
            name="Main",
            nodes=(
                NodeConfig(
                    id="calculate",
                    kind="expression",
                    expression="result = value + 1",
                ),
            ),
        ),
    )


def test_canonical_project_json_is_stable_and_formatted() -> None:
    first = canonical_project_json(_project())
    second = canonical_project_json(_project())

    assert first == second
    assert first.endswith("\n")
    assert json.loads(first)["id"] == "demo"
    assert first.index('"format_version"') < first.index('"id"')


def test_file_project_store_crud_and_sorted_listing(tmp_path) -> None:
    store = FileProjectStore(tmp_path)
    store.create(_project("zulu", "Zulu"))
    store.create(_project("alpha", "Alpha"))

    assert [project.id for project in store.list()] == ["alpha", "zulu"]
    assert store.get("alpha").name == "Alpha"

    store.put(_project("alpha", "Updated"))
    assert store.get("alpha").name == "Updated"

    store.delete("alpha")
    with pytest.raises(ProjectNotFoundError):
        store.get("alpha")


def test_file_project_store_rejects_conflict_and_missing_delete(tmp_path) -> None:
    store = FileProjectStore(tmp_path)
    store.create(_project())

    with pytest.raises(ProjectConflictError, match="already exists"):
        store.create(_project())
    with pytest.raises(ProjectNotFoundError):
        store.delete("missing")


def test_file_project_store_writes_atomically_and_contains_paths(tmp_path) -> None:
    store = FileProjectStore(tmp_path)
    store.create(_project("safe_id"))

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].name != "safe_id.json"
    assert list(tmp_path.glob("*.tmp")) == []


def test_project_json_and_yaml_import_export_round_trip() -> None:
    project = _project()

    json_document = export_project_document(project, format="json")
    yaml_document = export_project_document(project, format="yaml")

    assert load_project_document(json_document, format="json") == project
    assert load_project_document(yaml_document, format="yaml") == project


def test_project_import_rejects_invalid_unsafe_and_oversized_documents() -> None:
    with pytest.raises(ProjectFormatError, match="invalid"):
        load_project_document("not-json", format="json")
    with pytest.raises(ProjectFormatError, match="invalid or unsafe"):
        load_project_document("!!python/object/apply:os.system ['id']", format="yaml")
    with pytest.raises(ProjectFormatError, match="exceeds"):
        load_project_document("{}" * 10, format="json", max_bytes=5)
    with pytest.raises(ProjectFormatError, match="format"):
        load_project_document("{}", format="toml")


def test_file_project_store_rejects_corrupt_persisted_data(tmp_path) -> None:
    store = FileProjectStore(tmp_path)
    store.create(_project())
    path = next(tmp_path.glob("*.json"))
    path.write_text("corrupt", encoding="utf-8")

    with pytest.raises(ProjectFormatError):
        store.get("demo")
