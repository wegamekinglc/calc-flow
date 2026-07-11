from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from calc_flow.batch import JSONValue

CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """Base class for checkpoint persistence and validation failures."""


class CheckpointFormatError(CheckpointError):
    """Raised when persisted checkpoint data is corrupt or unsupported."""


class CheckpointMismatchError(CheckpointError):
    """Raised when a checkpoint belongs to a different compiled pipeline."""


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        msg = f"checkpoint {label} must be JSON-compatible"
        raise TypeError(msg) from error


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Versioned state committed after a source batch reaches every sink."""

    pipeline_name: str
    pipeline_fingerprint: str
    source_cursor: JSONValue
    sequence: int
    state: Mapping[str, Mapping[str, Any]]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    format_version: int = CHECKPOINT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != CHECKPOINT_FORMAT_VERSION:
            msg = f"unsupported checkpoint format version {self.format_version}"
            raise CheckpointFormatError(msg)
        if not self.pipeline_name:
            msg = "checkpoint pipeline_name must not be empty"
            raise ValueError(msg)
        if not self.pipeline_fingerprint:
            msg = "checkpoint pipeline_fingerprint must not be empty"
            raise ValueError(msg)
        if self.sequence < 0:
            msg = "checkpoint sequence must be greater than or equal to 0"
            raise ValueError(msg)
        if self.created_at.tzinfo is None:
            msg = "checkpoint created_at must include timezone information"
            raise ValueError(msg)
        object.__setattr__(
            self, "source_cursor", _json_copy(self.source_cursor, label="cursor")
        )
        object.__setattr__(self, "state", _json_copy(self.state, label="state"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "pipeline_name": self.pipeline_name,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "source_cursor": self.source_cursor,
            "sequence": self.sequence,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Checkpoint:
        required = {
            "format_version",
            "pipeline_name",
            "pipeline_fingerprint",
            "source_cursor",
            "sequence",
            "state",
            "created_at",
        }
        missing = required - set(data)
        unknown = set(data) - required
        if missing or unknown:
            msg = (
                "invalid checkpoint fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
            raise CheckpointFormatError(msg)
        try:
            created_at = datetime.fromisoformat(data["created_at"])
            return cls(
                pipeline_name=data["pipeline_name"],
                pipeline_fingerprint=data["pipeline_fingerprint"],
                source_cursor=data["source_cursor"],
                sequence=data["sequence"],
                state=data["state"],
                created_at=created_at,
                format_version=data["format_version"],
            )
        except CheckpointFormatError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            msg = "checkpoint contains invalid values"
            raise CheckpointFormatError(msg) from error


class CheckpointStore(Protocol):
    """Persistence contract for versioned pipeline checkpoints."""

    def load(self, pipeline_name: str) -> Checkpoint | None: ...

    def save(self, checkpoint: Checkpoint) -> None: ...

    def delete(self, pipeline_name: str) -> None: ...


class FileCheckpointStore:
    """Atomic JSON checkpoint storage rooted in one directory."""

    def __init__(self, directory: str | Path = ".calc-flow-checkpoints") -> None:
        self._directory = Path(directory)

    def load(self, pipeline_name: str) -> Checkpoint | None:
        path = self._path_for(pipeline_name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise TypeError
            checkpoint = Checkpoint.from_dict(data)
        except CheckpointFormatError:
            raise
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            msg = f"could not read checkpoint for {pipeline_name!r}"
            raise CheckpointFormatError(msg) from error
        if checkpoint.pipeline_name != pipeline_name:
            msg = "checkpoint pipeline name does not match its storage key"
            raise CheckpointFormatError(msg)
        return checkpoint

    def save(self, checkpoint: Checkpoint) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._path_for(checkpoint.pipeline_name)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(
                    checkpoint.to_dict(),
                    stream,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def delete(self, pipeline_name: str) -> None:
        self._path_for(pipeline_name).unlink(missing_ok=True)

    def _path_for(self, pipeline_name: str) -> Path:
        digest = hashlib.sha256(pipeline_name.encode()).hexdigest()
        return self._directory / f"{digest}.json"


def validate_checkpoint(
    checkpoint: Checkpoint, *, pipeline_name: str, fingerprint: str
) -> None:
    if (
        checkpoint.pipeline_name != pipeline_name
        or checkpoint.pipeline_fingerprint != fingerprint
    ):
        msg = (
            "checkpoint fingerprint does not match the compiled pipeline; "
            "reset or migrate the checkpoint before recovery"
        )
        raise CheckpointMismatchError(msg)
