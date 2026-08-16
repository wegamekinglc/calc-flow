from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import calc_flow_studio.checkpoint_store as checkpoint_store_module
from calc_flow_studio.checkpoint_store import (
    CheckpointDocumentError,
    FileCheckpointDocumentStore,
)


def _checkpoint() -> dict[str, object]:
    return {
        "format_version": 2,
        "pipeline_name": "orders",
        "pipeline_fingerprint": "fingerprint",
        "source_cursor": {"offset": 12},
        "sequence": 4,
        "state": {"calculate": {"rows": 12}},
        "created_at": "2026-01-01T00:00:00Z",
    }


def _create_directory_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip("Windows host does not permit unprivileged symlinks")
        raise


def test_load_rejects_linked_checkpoint_store_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    leaked = {
        **_checkpoint(),
        "pipeline_fingerprint": "outside-fingerprint",
        "state": {"secret": "outside-store"},
    }
    checkpoint_store_module._path_for(outside, "orders").write_text(
        json.dumps(leaked), encoding="utf-8"
    )
    store_root = tmp_path / "store"
    _create_directory_symlink(store_root, outside)

    with pytest.raises(CheckpointDocumentError, match="checkpoint directory"):
        asyncio.run(FileCheckpointDocumentStore(store_root).load("orders"))


def test_save_rejects_linked_checkpoint_store_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    store_root = tmp_path / "store"
    _create_directory_symlink(store_root, outside)

    with pytest.raises(CheckpointDocumentError, match="checkpoint directory"):
        asyncio.run(FileCheckpointDocumentStore(store_root).save(_checkpoint()))

    assert not checkpoint_store_module._path_for(outside, "orders").exists()


def test_delete_rejects_linked_checkpoint_store_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    checkpoint_path = checkpoint_store_module._path_for(outside, "orders")
    checkpoint_path.write_text(json.dumps(_checkpoint()), encoding="utf-8")
    store_root = tmp_path / "store"
    _create_directory_symlink(store_root, outside)

    with pytest.raises(CheckpointDocumentError, match="checkpoint directory"):
        asyncio.run(FileCheckpointDocumentStore(store_root).delete("orders"))

    assert checkpoint_path.exists()


def test_load_rejects_non_regular_checkpoint_entry(tmp_path: Path) -> None:
    checkpoint_path = checkpoint_store_module._path_for(tmp_path, "orders")
    checkpoint_path.mkdir()

    with pytest.raises(CheckpointDocumentError, match="regular file"):
        asyncio.run(FileCheckpointDocumentStore(tmp_path).load("orders"))


def test_checkpoint_directory_validation_rejects_windows_reparse_point() -> None:
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    metadata = SimpleNamespace(
        st_mode=stat.S_IFDIR,
        st_file_attributes=reparse_point,
    )

    with pytest.raises(CheckpointDocumentError, match="checkpoint directory"):
        checkpoint_store_module._validate_checkpoint_directory(metadata)


def test_save_and_delete_sync_parent_directory_from_worker_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def record_sync(_descriptor: int) -> None:
        calls.append(threading.get_ident())

    monkeypatch.setattr(checkpoint_store_module, "_sync_directory", record_sync)
    store = FileCheckpointDocumentStore(tmp_path)
    owner_thread = threading.get_ident()

    asyncio.run(store.save(_checkpoint()))
    asyncio.run(store.delete("orders"))

    assert len(calls) == 2
    assert all(thread_id != owner_thread for thread_id in calls)


@pytest.mark.parametrize("operation", ("save", "delete"))
def test_parent_directory_sync_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = FileCheckpointDocumentStore(tmp_path)
    if operation == "delete":
        asyncio.run(store.save(_checkpoint()))

    def fail_sync(_descriptor: int) -> None:
        raise OSError("cannot sync checkpoint directory")

    monkeypatch.setattr(checkpoint_store_module, "_sync_directory", fail_sync)

    with pytest.raises(OSError, match="cannot sync"):
        if operation == "save":
            asyncio.run(store.save(_checkpoint()))
        else:
            asyncio.run(store.delete("orders"))


def test_delete_of_missing_checkpoint_does_not_sync_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        checkpoint_store_module,
        "_sync_directory",
        lambda _directory: pytest.fail("missing delete must not sync"),
    )

    asyncio.run(FileCheckpointDocumentStore(tmp_path).delete("orders"))


def test_directory_sync_is_a_noop_off_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        checkpoint_store_module,
        "os",
        SimpleNamespace(
            name="nt",
            fsync=lambda *_args, **_kwargs: pytest.fail(
                "non-POSIX directory sync must be a no-op"
            ),
        ),
    )

    checkpoint_store_module._sync_directory(123)


def test_checkpoint_copy_rejects_non_object_decoder_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(checkpoint_store_module.json, "loads", lambda _value: [])

    with pytest.raises(CheckpointDocumentError, match="strict JSON object"):
        checkpoint_store_module._validate_checkpoint(_checkpoint())


def test_load_rejects_checkpoint_symlink(tmp_path: Path) -> None:
    store_directory = tmp_path / "store"
    store_directory.mkdir()
    outside_checkpoint = tmp_path / "outside-checkpoint.json"
    leaked = {**_checkpoint(), "state": {"secret": "outside-store"}}
    outside_checkpoint.write_text(json.dumps(leaked), encoding="utf-8")
    checkpoint_path = checkpoint_store_module._path_for(store_directory, "orders")
    try:
        checkpoint_path.symlink_to(outside_checkpoint)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip("Windows host does not permit unprivileged symlinks")
        raise

    store = FileCheckpointDocumentStore(store_directory)

    with pytest.raises(CheckpointDocumentError, match="symbolic link"):
        asyncio.run(store.load("orders"))
