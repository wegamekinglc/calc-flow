from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from calc_flow.checkpoint import (
    Checkpoint,
    CheckpointFormatError,
    CheckpointMismatchError,
    FileCheckpointStore,
    validate_checkpoint,
)


def _checkpoint(**changes) -> Checkpoint:
    values = {
        "pipeline_name": "orders",
        "pipeline_fingerprint": "abc123",
        "source_cursor": {"offset": 4},
        "sequence": 2,
        "state": {"counter": {"rows": 10}},
        "created_at": datetime(2026, 1, 2, tzinfo=UTC),
    }
    values.update(changes)
    return Checkpoint(**values)


def test_checkpoint_round_trip_is_versioned() -> None:
    checkpoint = _checkpoint()

    restored = Checkpoint.from_dict(checkpoint.to_dict())

    assert restored == checkpoint
    assert restored.format_version == 1
    assert restored.source_cursor == {"offset": 4}


def test_checkpoint_rejects_unknown_format_version() -> None:
    data = _checkpoint().to_dict()
    data["format_version"] = 99

    with pytest.raises(CheckpointFormatError, match="unsupported"):
        Checkpoint.from_dict(data)


def test_checkpoint_rejects_non_json_state() -> None:
    with pytest.raises(TypeError, match="JSON-compatible"):
        _checkpoint(state={"node": {"bad": object()}})


def test_file_checkpoint_store_saves_and_loads_atomically(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    checkpoint = _checkpoint()

    store.save(checkpoint)
    restored = store.load("orders")

    assert restored == checkpoint
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert list(tmp_path.glob("*.tmp")) == []


def test_file_checkpoint_store_loads_missing_and_deletes(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    assert store.load("orders") is None

    store.save(_checkpoint())
    store.delete("orders")

    assert store.load("orders") is None


def test_file_checkpoint_store_rejects_corrupt_json(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    store.save(_checkpoint())
    path = next(tmp_path.glob("*.json"))
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(CheckpointFormatError, match="could not read"):
        store.load("orders")


def test_file_checkpoint_store_contains_path_traversal_names(tmp_path) -> None:
    store = FileCheckpointStore(tmp_path)
    store.save(_checkpoint(pipeline_name="../../outside"))

    stored = list(tmp_path.glob("*.json"))
    assert len(stored) == 1
    assert stored[0].parent == tmp_path


def test_validate_checkpoint_rejects_fingerprint_mismatch() -> None:
    with pytest.raises(CheckpointMismatchError, match="reset or migrate"):
        validate_checkpoint(
            _checkpoint(), pipeline_name="orders", fingerprint="different"
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"pipeline_name": ""},
        {"pipeline_fingerprint": ""},
        {"sequence": -1},
        {"created_at": datetime(2026, 1, 2)},
    ),
)
def test_checkpoint_rejects_invalid_identity_values(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _checkpoint(**changes)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda document: document.pop("state"),
        lambda document: document.update({"unknown": True}),
    ),
)
def test_checkpoint_rejects_missing_and_unknown_fields(mutation) -> None:
    document = _checkpoint().to_dict()
    mutation(document)

    with pytest.raises(CheckpointFormatError, match="invalid checkpoint fields"):
        Checkpoint.from_dict(document)


def test_checkpoint_rejects_invalid_serialized_values() -> None:
    document = _checkpoint().to_dict()
    document["created_at"] = "not-a-date"

    with pytest.raises(CheckpointFormatError, match="invalid values"):
        Checkpoint.from_dict(document)


def test_file_checkpoint_store_rejects_non_object_and_wrong_storage_key(
    tmp_path,
) -> None:
    store = FileCheckpointStore(tmp_path)
    store.save(_checkpoint())
    path = next(tmp_path.glob("*.json"))
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(CheckpointFormatError, match="could not read"):
        store.load("orders")

    other_path = tmp_path / f"{hashlib.sha256(b'other').hexdigest()}.json"
    other_path.write_text(json.dumps(_checkpoint().to_dict()), encoding="utf-8")
    with pytest.raises(CheckpointFormatError, match="storage key"):
        store.load("other")
