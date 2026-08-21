from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from calc_flow import ConfigError, FileProjectStore, PipelineBuilder, ProjectDocument
from calc_flow.store import (
    export_project_json,
    export_project_yaml,
    import_project_json,
    import_project_yaml,
)


def _project(name: str) -> ProjectDocument:
    return ProjectDocument.model_validate(
        PipelineBuilder(name).expression("calc", "b = a + 1").project
    )


def test_project_store_round_trips_sorted_defensive_v3_documents(tmp_path) -> None:
    async def exercise() -> None:
        store = FileProjectStore(tmp_path)
        zeta = _project("zeta")
        alpha = _project("alpha")
        await store.create(zeta)
        await store.create(alpha.root)

        loaded = await store.get("alpha")
        listed = await store.list()
        loaded.root["name"] = "changed"
        listed[0].root["name"] = "changed"

        assert (await store.get("alpha")).root == alpha.root
        assert [project.root["id"] for project in await store.list()] == [
            "alpha",
            "zeta",
        ]
        with pytest.raises(ConfigError, match="already exists"):
            await store.create(alpha)
        with pytest.raises(ConfigError, match="not found"):
            await store.get("missing")

        replacement = ProjectDocument.model_validate({**alpha.root, "name": "Alpha 2"})
        await store.put(replacement)
        assert (await store.get("alpha")).root["name"] == "Alpha 2"
        await store.delete("alpha")
        with pytest.raises(ConfigError, match="not found"):
            await store.delete("alpha")

    asyncio.run(exercise())


def test_project_store_persists_canonical_pretty_json(tmp_path) -> None:
    project = _project("canonical")
    store = FileProjectStore(tmp_path)
    asyncio.run(store.create(project))

    file_name = hashlib.sha256(b"canonical").hexdigest() + ".json"
    document = (tmp_path / file_name).read_text()
    assert document.endswith("\n")
    assert json.loads(document) == project.root
    assert document == json.dumps(project.root, indent=2, sort_keys=True) + "\n"


def test_project_store_validates_before_filesystem_io(tmp_path) -> None:
    store = FileProjectStore(tmp_path / "not-created")

    with pytest.raises(ValueError):
        store.create({"format_version": 1})
    assert not (tmp_path / "not-created").exists()


def test_store_awaitables_copy_inputs_before_the_caller_can_mutate_them(
    tmp_path,
) -> None:
    async def exercise() -> None:
        projects = FileProjectStore(tmp_path / "projects")
        project = _project("copied").root
        create = projects.create(project)
        project["name"] = "mutated"
        await create
        assert (await projects.get("copied")).root["name"] == "copied"

    asyncio.run(exercise())


def test_store_blocking_conveniences_work_and_reject_running_loop(tmp_path) -> None:
    store = FileProjectStore(tmp_path / "projects")
    project = _project("blocking")

    store.create_blocking(project)
    assert store.get_blocking("blocking").root == project.root
    assert [item.root["id"] for item in store.list_blocking()] == ["blocking"]
    store.put_blocking(project)
    store.delete_blocking("blocking")

    async def reject() -> None:
        with pytest.raises(RuntimeError, match="await create"):
            store.create_blocking(project)

    asyncio.run(reject())


def test_project_document_transforms_are_bounded_strict_and_rust_backed() -> None:
    project = _project("portable")

    imported_json = import_project_json(project.canonical_json().encode())
    exported_json = export_project_json(imported_json)
    exported_yaml = export_project_yaml(project)
    imported_yaml = import_project_yaml(exported_yaml)

    assert imported_json.root == project.root
    assert imported_yaml.root == project.root
    assert exported_json == json.dumps(project.root, indent=2, sort_keys=True) + "\n"
    assert "format_version: 3" in exported_yaml

    with pytest.raises(ConfigError):
        import_project_json('{"format_version":1}')
    with pytest.raises(ConfigError, match="alias"):
        import_project_yaml(
            "format_version: 3\nid: aliases\nname: &name aliases\n"
            "description: *name\nruntime: {mode: batch, options: {}}\n"
            "graph: {name: p, nodes: []}\n"
        )
    with pytest.raises(ConfigError, match="exceeds"):
        import_project_json(b"x" * (10 * 1024 * 1024 + 1))


def test_project_document_transforms_reject_subclass_hooks_without_calling_them() -> (
    None
):
    class HostileDict(dict):
        def items(self):
            raise AssertionError("caller-controlled hook ran")

    with pytest.raises(TypeError, match="strict dict"):
        export_project_json(HostileDict(_project("hostile").root))
    with pytest.raises(TypeError, match="bytes or str"):
        import_project_json(bytearray(b"{}"))  # type: ignore[arg-type]
