from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import secrets
import stat
import tempfile
from collections.abc import Awaitable, Iterator, Mapping
from contextlib import contextmanager, suppress
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
                _checkpoint_name(copied["pipeline_name"]),
                document,
            )

        return persist()

    async def load(self, pipeline_name: str) -> dict[str, object] | None:
        _require_pipeline_name(pipeline_name)
        document = await asyncio.to_thread(
            _bounded_read,
            self._directory,
            _checkpoint_name(pipeline_name),
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
            self._directory,
            _checkpoint_name(pipeline_name),
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
    return directory / _checkpoint_name(pipeline_name)


def _checkpoint_name(pipeline_name: str) -> str:
    digest = hashlib.sha256(pipeline_name.encode()).hexdigest()
    return f"{digest}.json"


def _bounded_read(directory: Path, checkpoint_name: str) -> bytes | None:
    with _checkpoint_directory(directory, create=False) as anchor:
        if anchor is None:
            return None
        if os.name == "posix":
            return _bounded_read_at(anchor, checkpoint_name)
        return _bounded_read_path(_path_from_name(directory, checkpoint_name))


def _bounded_read_at(directory_descriptor: int, checkpoint_name: str) -> bytes | None:
    opened = _open_checkpoint_at(directory_descriptor, checkpoint_name)
    if opened is None:
        return None
    return _bounded_read_descriptor(*opened)


def _bounded_read_path(path: Path) -> bytes | None:
    opened = _open_checkpoint_path(path)
    if opened is None:
        return None
    return _bounded_read_descriptor(*opened)


def _bounded_read_descriptor(descriptor: int, metadata: os.stat_result) -> bytes:
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


def _open_checkpoint_at(
    directory_descriptor: int, checkpoint_name: str
) -> tuple[int, os.stat_result] | None:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | _required_posix_flag("O_NOFOLLOW")
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(checkpoint_name, flags, dir_fd=directory_descriptor)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise CheckpointDocumentError(
                "checkpoint entry must be a regular file, not a symbolic link"
            ) from error
        raise
    try:
        metadata = os.fstat(descriptor)
        _validate_checkpoint_entry(metadata)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, metadata


def _open_checkpoint_path(path: Path) -> tuple[int, os.stat_result] | None:
    try:
        expected = path.lstat()
    except FileNotFoundError:
        return None
    _validate_checkpoint_entry(expected)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise CheckpointDocumentError(
                "checkpoint entry must be a regular file, not a symbolic link"
            ) from error
        raise
    try:
        metadata = os.fstat(descriptor)
        _validate_checkpoint_entry(metadata)
        if not os.path.samestat(expected, metadata):
            raise CheckpointDocumentError(
                "checkpoint changed while it was being opened"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, metadata


def _validate_checkpoint_entry(metadata: os.stat_result) -> None:
    if _is_symbolic_link(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise CheckpointDocumentError(
            "checkpoint entry must be a regular file, "
            "not a symbolic link or reparse point"
        )


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


def _atomic_write(directory: Path, checkpoint_name: str, document: bytes) -> None:
    with _checkpoint_directory(directory, create=True) as anchor:
        if anchor is None:
            raise CheckpointDocumentError("checkpoint directory could not be created")
        if os.name == "posix":
            _atomic_write_at(anchor, checkpoint_name, document)
        else:
            _atomic_write_path(
                directory,
                _path_from_name(directory, checkpoint_name),
                document,
            )
            _sync_directory(anchor)


def _atomic_write_at(
    directory_descriptor: int, checkpoint_name: str, document: bytes
) -> None:
    temporary_name = f".checkpoint-{secrets.token_hex(16)}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | _required_posix_flag("O_NOFOLLOW")
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(
        temporary_name,
        flags,
        0o600,
        dir_fd=directory_descriptor,
    )
    try:
        try:
            _write_descriptor(descriptor, document)
        finally:
            os.close(descriptor)
        _validate_checkpoint_name_at(directory_descriptor, checkpoint_name)
        os.replace(
            temporary_name,
            checkpoint_name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        _sync_directory(directory_descriptor)
    finally:
        _unlink_at(directory_descriptor, temporary_name)


def _atomic_write_path(directory: Path, path: Path, document: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=".checkpoint-", dir=directory)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        _validate_checkpoint_path(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_descriptor(descriptor: int, document: bytes) -> None:
    remaining = memoryview(document)
    while remaining:
        written = os.write(descriptor, remaining)
        if written == 0:
            raise OSError("checkpoint write made no progress")
        remaining = remaining[written:]
    os.fsync(descriptor)


def _delete_file(directory: Path, checkpoint_name: str) -> None:
    with _checkpoint_directory(directory, create=False) as anchor:
        if anchor is None:
            return
        if os.name == "posix":
            if not _validate_checkpoint_name_at(anchor, checkpoint_name):
                return
            os.unlink(checkpoint_name, dir_fd=anchor)
            _sync_directory(anchor)
            return
        path = _path_from_name(directory, checkpoint_name)
        if not _validate_checkpoint_path(path):
            return
        path.unlink()
        _sync_directory(anchor)


def _validate_checkpoint_name_at(
    directory_descriptor: int, checkpoint_name: str
) -> bool:
    try:
        metadata = os.stat(
            checkpoint_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return False
    _validate_checkpoint_entry(metadata)
    return True


def _validate_checkpoint_path(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    _validate_checkpoint_entry(metadata)
    return True


def _unlink_at(directory_descriptor: int, name: str) -> None:
    with suppress(FileNotFoundError):
        os.unlink(name, dir_fd=directory_descriptor)


def _path_from_name(directory: Path, checkpoint_name: str) -> Path:
    return directory / checkpoint_name


@contextmanager
def _checkpoint_directory(directory: Path, *, create: bool) -> Iterator[int | None]:
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    try:
        expected = directory.lstat()
    except FileNotFoundError:
        yield None
        return
    _validate_checkpoint_directory(expected)
    if os.name == "posix":
        anchor = _open_posix_directory(directory, expected)
        close = os.close
    elif os.name == "nt":
        anchor = _open_windows_directory(directory, expected)
        close = _close_windows_handle
    else:
        raise CheckpointDocumentError(
            "secure checkpoint directory operations are unsupported on this platform"
        )
    try:
        yield anchor
    finally:
        close(anchor)


def _validate_checkpoint_directory(metadata: os.stat_result) -> None:
    if _is_symbolic_link(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise CheckpointDocumentError(
            "checkpoint directory must be a real directory, "
            "not a symbolic link or reparse point"
        )


def _open_posix_directory(directory: Path, expected: os.stat_result) -> int:
    flags = (
        os.O_RDONLY
        | _required_posix_flag("O_DIRECTORY")
        | _required_posix_flag("O_NOFOLLOW")
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(directory, flags)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise CheckpointDocumentError(
                "checkpoint directory must not be a symbolic link or reparse point"
            ) from error
        raise
    try:
        metadata = os.fstat(descriptor)
        _validate_checkpoint_directory(metadata)
        if not os.path.samestat(expected, metadata):
            raise CheckpointDocumentError(
                "checkpoint directory changed while it was being opened"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _required_posix_flag(name: str) -> int:
    flag = getattr(os, name, None)
    if not isinstance(flag, int):
        raise CheckpointDocumentError(
            f"secure checkpoint directory operations require {name}"
        )
    return flag


def _open_windows_directory(directory: Path, expected: os.stat_result) -> int:
    handle = _create_windows_directory_handle(directory)
    try:
        attributes = _windows_directory_attributes(handle)
        if attributes & 0x400 or not attributes & 0x10:
            raise CheckpointDocumentError(
                "checkpoint directory must be a real directory, not a reparse point"
            )
        current = directory.lstat()
        _validate_checkpoint_directory(current)
        if not os.path.samestat(expected, current):
            raise CheckpointDocumentError(
                "checkpoint directory changed while it was being opened"
            )
    except BaseException:
        _close_windows_handle(handle)
        raise
    return handle


def _create_windows_directory_handle(directory: Path) -> int:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    path = os.fspath(directory.absolute())
    handle = create_file(
        path,
        0x0080,
        0x0001,
        None,
        3,
        0x02000000 | 0x00200000,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle)


def _windows_directory_attributes(handle: int) -> int:
    import ctypes
    from ctypes import wintypes

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("reparse_tag", wintypes.DWORD),
        ]

    get_information = ctypes.WinDLL(
        "kernel32", use_last_error=True
    ).GetFileInformationByHandleEx
    get_information.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    get_information.restype = wintypes.BOOL
    information = FileAttributeTagInfo()
    if not get_information(
        handle,
        9,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(information.file_attributes)


def _close_windows_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    if not close_handle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _sync_directory(descriptor: int) -> None:
    if os.name != "posix":
        return
    os.fsync(descriptor)
