from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Awaitable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

CHECKPOINT_FORMAT_VERSION = 2
MAX_CHECKPOINT_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_JSON_DEPTH = 32
_FIELDS = {
    "created_at",
    "format_version",
    "pipeline_fingerprint",
    "pipeline_name",
    "sequence",
    "source_cursor",
    "state",
}


class CheckpointDocumentError(ValueError):
    """A Studio-private v2 checkpoint document is invalid."""


class FileCheckpointDocumentStore:
    """Async file store for the Studio's legacy v2 checkpoint inspection routes."""

    __slots__ = ("_directory",)

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def save(self, checkpoint: Mapping[str, object]) -> Awaitable[None]:
        copied = _validate_checkpoint(checkpoint)
        document = (json.dumps(copied, indent=2, sort_keys=True) + "\n").encode()
        if len(document) > MAX_CHECKPOINT_DOCUMENT_BYTES:
            raise CheckpointDocumentError(
                f"checkpoint exceeds the {MAX_CHECKPOINT_DOCUMENT_BYTES}-byte limit"
            )

        async def persist() -> None:
            await asyncio.to_thread(
                _atomic_write,
                self._directory,
                _path_for(self._directory, copied["pipeline_name"]),
                document,
            )

        return persist()

    async def load(self, pipeline_name: str) -> dict[str, object] | None:
        _require_pipeline_name(pipeline_name)
        document = await asyncio.to_thread(
            _bounded_read,
            _path_for(self._directory, pipeline_name),
        )
        if document is None:
            return None
        checkpoint = _parse_checkpoint(document)
        if checkpoint["pipeline_name"] != pipeline_name:
            raise CheckpointDocumentError(
                "stored checkpoint pipeline name "
                f"{checkpoint['pipeline_name']!r} does not match key {pipeline_name!r}"
            )
        return checkpoint

    async def delete(self, pipeline_name: str) -> None:
        _require_pipeline_name(pipeline_name)
        await asyncio.to_thread(
            _delete_file,
            _path_for(self._directory, pipeline_name),
        )


def _require_pipeline_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointDocumentError("pipeline_name must be a non-empty string")
    return value


def _validate_checkpoint(checkpoint: Mapping[str, object]) -> dict[str, object]:
    _validate_checkpoint_fields(checkpoint)
    pipeline_name = _require_pipeline_name(checkpoint["pipeline_name"])
    _validate_format_version(checkpoint["format_version"])
    _validate_fingerprint(checkpoint["pipeline_fingerprint"])
    _validate_sequence(checkpoint["sequence"])
    _validate_state(checkpoint["state"])
    _validate_created_at(checkpoint["created_at"])
    _validate_json_depth(checkpoint)
    copied = _copy_checkpoint(checkpoint)
    copied["pipeline_name"] = pipeline_name
    return copied


def _validate_checkpoint_fields(checkpoint: Mapping[str, object]) -> None:
    if type(checkpoint) is not dict:
        raise CheckpointDocumentError("checkpoint must be a strict JSON object")
    if any(not isinstance(key, str) for key in checkpoint):
        raise CheckpointDocumentError("checkpoint field names must be strings")
    unknown = set(checkpoint) - _FIELDS
    missing = _FIELDS - set(checkpoint)
    if unknown:
        raise CheckpointDocumentError(
            f"checkpoint contains unknown field {min(unknown)!r}"
        )
    if missing:
        raise CheckpointDocumentError(
            f"checkpoint is missing required field {min(missing)!r}"
        )


def _validate_format_version(value: object) -> None:
    if type(value) is not int or value != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointDocumentError(
            "checkpoint format version is unsupported; expected 2"
        )


def _validate_fingerprint(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise CheckpointDocumentError("pipeline_fingerprint must be a non-empty string")


def _validate_sequence(value: object) -> None:
    if type(value) is not int or not 0 <= value <= 2**64 - 1:
        raise CheckpointDocumentError("sequence must be a non-negative u64 integer")


def _validate_state(value: object) -> None:
    if type(value) is not dict or any(
        not isinstance(node_id, str) or not node_id for node_id in value
    ):
        raise CheckpointDocumentError(
            "state must be an object with non-empty string node IDs"
        )


def _validate_created_at(value: object) -> None:
    if not isinstance(value, str) or not _is_offset_datetime(value):
        raise CheckpointDocumentError("created_at must be an offset-aware timestamp")


def _copy_checkpoint(checkpoint: Mapping[str, object]) -> dict[str, object]:
    try:
        encoded = json.dumps(
            checkpoint,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise CheckpointDocumentError(
            "checkpoint must contain strict JSON-compatible data"
        ) from error
    copied = json.loads(encoded)
    if type(copied) is not dict:
        raise CheckpointDocumentError("checkpoint must be a strict JSON object")
    return copied


def _is_offset_datetime(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate_json_depth(value: object) -> None:
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise CheckpointDocumentError(
                f"checkpoint exceeds the maximum JSON depth of {MAX_JSON_DEPTH}"
            )
        if type(current) is dict:
            pending.extend((item, depth + 1) for item in current.values())
        elif type(current) is list:
            pending.extend((item, depth + 1) for item in current)


def _parse_checkpoint(document: bytes) -> dict[str, object]:
    try:
        parsed = json.loads(
            document,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: _reject_constant(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointDocumentError(
            f"checkpoint document is invalid: {error}"
        ) from error
    return _validate_checkpoint(parsed)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise CheckpointDocumentError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise CheckpointDocumentError(f"JSON number {value} is not finite")


def _path_for(directory: Path, pipeline_name: str) -> Path:
    digest = hashlib.sha256(pipeline_name.encode()).hexdigest()
    return directory / f"{digest}.json"


def _bounded_read(path: Path) -> bytes | None:
    opened = _open_checkpoint(path)
    if opened is None:
        return None
    descriptor, metadata = opened
    try:
        if metadata.st_size > MAX_CHECKPOINT_DOCUMENT_BYTES:
            raise CheckpointDocumentError(
                f"checkpoint exceeds the {MAX_CHECKPOINT_DOCUMENT_BYTES}-byte limit"
            )
        document = _read_descriptor(descriptor)
    finally:
        os.close(descriptor)
    if len(document) > MAX_CHECKPOINT_DOCUMENT_BYTES:
        raise CheckpointDocumentError(
            f"checkpoint exceeds the {MAX_CHECKPOINT_DOCUMENT_BYTES}-byte limit"
        )
    return document


def _open_checkpoint(path: Path) -> tuple[int, os.stat_result] | None:
    try:
        expected = path.lstat()
    except FileNotFoundError:
        return None
    if _is_symbolic_link(expected):
        raise CheckpointDocumentError("checkpoint path must not be a symbolic link")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise CheckpointDocumentError(
                "checkpoint path must not be a symbolic link"
            ) from error
        raise
    try:
        metadata = os.fstat(descriptor)
        if not os.path.samestat(expected, metadata):
            raise CheckpointDocumentError(
                "checkpoint changed while it was being opened"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, metadata


def _is_symbolic_link(metadata: os.stat_result) -> bool:
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & reparse_point)


def _read_descriptor(descriptor: int) -> bytes:
    remaining = MAX_CHECKPOINT_DOCUMENT_BYTES + 1
    chunks: list[bytes] = []
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _atomic_write(directory: Path, path: Path, document: bytes) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _delete_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _sync_directory(path.parent)


def _sync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
