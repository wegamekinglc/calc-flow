from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime

import pytest

from calc_flow import (
    ConfigError,
    FileCheckpointStore,
    FileProjectStore,
    PipelineBuilder,
    ProjectDocument,
)


def _project(name: str) -> ProjectDocument:
    return ProjectDocument.model_validate(
        PipelineBuilder(name).expression("calc", "b = a + 1").project
    )


def _checkpoint(name: str = "stored") -> dict[str, object]:
    return {
        "created_at": datetime(2026, 7, 14, tzinfo=UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "format_version": 2,
        "pipeline_fingerprint": "fingerprint",
        "pipeline_name": name,
        "sequence": 4,
        "source_cursor": {"offset": 3},
        "state": {"calc": {}},
    }


def test_project_store_round_trips_sorted_defensive_v2_documents(tmp_path) -> None:
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

        checkpoints = FileCheckpointStore(tmp_path / "checkpoints")
        checkpoint = _checkpoint("copied")
        save = checkpoints.save(checkpoint)
        checkpoint["sequence"] = 99
        await save
        loaded = await checkpoints.load("copied")
        assert loaded is not None
        assert loaded["sequence"] == 4

    asyncio.run(exercise())


def test_checkpoint_store_round_trips_strict_defensive_documents(tmp_path) -> None:
    async def exercise() -> None:
        store = FileCheckpointStore(tmp_path)
        checkpoint = _checkpoint()
        assert await store.load("stored") is None
        await store.save(checkpoint)
        checkpoint["state"] = {}

        loaded = await store.load("stored")
        assert loaded == _checkpoint()
        assert loaded is not None
        loaded["state"] = {}
        assert await store.load("stored") == _checkpoint()

        await store.delete("stored")
        await store.delete("stored")
        assert await store.load("stored") is None

        with pytest.raises((ConfigError, ValueError)):
            await store.save({**_checkpoint(), "unknown": True})
        with pytest.raises((ConfigError, ValueError)):
            await store.save({**_checkpoint(), "format_version": 1})

    asyncio.run(exercise())


def test_checkpoint_store_rejects_malformed_persisted_documents(tmp_path) -> None:
    async def exercise() -> None:
        store = FileCheckpointStore(tmp_path)
        assert await store.load("broken") is None
        file_name = hashlib.sha256(b"broken").hexdigest() + ".json"
        (tmp_path / file_name).write_text(
            json.dumps({**_checkpoint("broken"), "unknown": True})
        )
        with pytest.raises(ConfigError, match="unknown field"):
            await store.load("broken")

    asyncio.run(exercise())


def test_store_blocking_conveniences_work_and_reject_running_loop(tmp_path) -> None:
    project_store = FileProjectStore(tmp_path / "projects")
    checkpoint_store = FileCheckpointStore(tmp_path / "checkpoints")
    project = _project("blocking")

    project_store.create_blocking(project)
    assert project_store.get_blocking("blocking").root == project.root
    assert [item.root["id"] for item in project_store.list_blocking()] == ["blocking"]
    project_store.put_blocking(project)
    project_store.delete_blocking("blocking")

    checkpoint = _checkpoint("blocking")
    checkpoint_store.save_blocking(checkpoint)
    assert checkpoint_store.load_blocking("blocking") == checkpoint
    checkpoint_store.delete_blocking("blocking")

    async def reject() -> None:
        with pytest.raises(RuntimeError, match="await create"):
            project_store.create_blocking(project)
        with pytest.raises(RuntimeError, match="await load"):
            checkpoint_store.load_blocking("blocking")

    asyncio.run(reject())
