from __future__ import annotations

import asyncio
import json
import os
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


def test_save_and_delete_sync_parent_directory_from_worker_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, int]] = []

    def record_sync(directory: Path) -> None:
        calls.append((directory, threading.get_ident()))

    monkeypatch.setattr(checkpoint_store_module, "_sync_directory", record_sync)
    store = FileCheckpointDocumentStore(tmp_path)
    owner_thread = threading.get_ident()

    asyncio.run(store.save(_checkpoint()))
    asyncio.run(store.delete("orders"))

    assert calls == [(tmp_path, calls[0][1]), (tmp_path, calls[1][1])]
    assert all(thread_id != owner_thread for _, thread_id in calls)


@pytest.mark.parametrize("operation", ("save", "delete"))
def test_parent_directory_sync_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    store = FileCheckpointDocumentStore(tmp_path)
    if operation == "delete":
        asyncio.run(store.save(_checkpoint()))

    def fail_sync(directory: Path) -> None:
        raise OSError(f"cannot sync {directory.name}")

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
            open=lambda *_args, **_kwargs: pytest.fail(
                "non-POSIX directory sync must be a no-op"
            ),
        ),
    )

    checkpoint_store_module._sync_directory(tmp_path)


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
