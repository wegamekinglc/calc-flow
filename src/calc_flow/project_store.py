from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from pydantic import ValidationError

from calc_flow.config import MAX_PREVIEW_BYTES, ProjectConfig


class ProjectStoreError(RuntimeError):
    """Base class for project persistence failures."""


class ProjectNotFoundError(ProjectStoreError):
    """Raised when a project ID does not exist."""


class ProjectConflictError(ProjectStoreError):
    """Raised when creating a project whose ID already exists."""


class ProjectFormatError(ProjectStoreError):
    """Raised when imported or persisted project data is invalid."""


def canonical_project_json(project: ProjectConfig) -> str:
    return (
        json.dumps(
            project.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def load_project_document(
    content: str | bytes,
    *,
    format: str,
    max_bytes: int = MAX_PREVIEW_BYTES,
) -> ProjectConfig:
    raw = content.encode() if isinstance(content, str) else content
    if len(raw) > max_bytes:
        msg = f"project document exceeds {max_bytes} bytes"
        raise ProjectFormatError(msg)
    try:
        if format == "json":
            data = json.loads(raw)
        elif format == "yaml":
            try:
                import yaml
            except ImportError as error:
                msg = "YAML import requires the 'web' extra"
                raise ProjectFormatError(msg) from error
            try:
                data = yaml.safe_load(raw)
            except yaml.YAMLError as error:
                msg = "project YAML is invalid or unsafe"
                raise ProjectFormatError(msg) from error
        else:
            msg = "project document format must be 'json' or 'yaml'"
            raise ProjectFormatError(msg)
        if not isinstance(data, dict):
            msg = "project document must contain an object"
            raise ProjectFormatError(msg)
        return ProjectConfig.model_validate(data)
    except ProjectFormatError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        msg = "project document is invalid"
        raise ProjectFormatError(msg) from error


def export_project_document(project: ProjectConfig, *, format: str) -> str:
    if format == "json":
        return canonical_project_json(project)
    if format == "yaml":
        try:
            import yaml
        except ImportError as error:
            msg = "YAML export requires the 'web' extra"
            raise ProjectFormatError(msg) from error
        return yaml.safe_dump(
            project.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            allow_unicode=True,
        )
    msg = "project document format must be 'json' or 'yaml'"
    raise ProjectFormatError(msg)


class FileProjectStore:
    """Thread-safe atomic storage for canonical project JSON documents."""

    def __init__(self, directory: str | Path = ".calc-flow-projects") -> None:
        self._directory = Path(directory)
        self._lock = RLock()

    def list(self) -> tuple[ProjectConfig, ...]:
        with self._lock:
            if not self._directory.exists():
                return ()
            projects = [
                self._load_path(path) for path in self._directory.glob("*.json")
            ]
        return tuple(sorted(projects, key=lambda project: project.id))

    def get(self, project_id: str) -> ProjectConfig:
        with self._lock:
            path = self._path_for(project_id)
            if not path.exists():
                raise ProjectNotFoundError(f"project {project_id!r} does not exist")
            project = self._load_path(path)
            if project.id != project_id:
                msg = "stored project ID does not match its storage key"
                raise ProjectFormatError(msg)
            return project

    def create(self, project: ProjectConfig) -> None:
        with self._lock:
            path = self._path_for(project.id)
            if path.exists():
                raise ProjectConflictError(f"project {project.id!r} already exists")
            self._write(path, project)

    def put(self, project: ProjectConfig) -> None:
        with self._lock:
            self._write(self._path_for(project.id), project)

    def delete(self, project_id: str) -> None:
        with self._lock:
            path = self._path_for(project_id)
            if not path.exists():
                raise ProjectNotFoundError(f"project {project_id!r} does not exist")
            path.unlink()

    def _write(self, path: Path, project: ProjectConfig) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(canonical_project_json(project))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_path(self, path: Path) -> ProjectConfig:
        try:
            return load_project_document(path.read_bytes(), format="json")
        except OSError as error:
            msg = f"could not read project file {path.name!r}"
            raise ProjectStoreError(msg) from error

    def _path_for(self, project_id: str) -> Path:
        digest = hashlib.sha256(project_id.encode()).hexdigest()
        return self._directory / f"{digest}.json"
