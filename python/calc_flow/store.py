from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from calc_flow import _native
from calc_flow.config import ProjectDocument, _validate_json_value


def _run_blocking[T](factory: Callable[[], Awaitable[T]], async_method: str) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:

        async def invoke() -> T:
            return await factory()

        return asyncio.run(invoke())
    message = (
        "blocking store operation cannot run inside an event loop; "
        f"await {async_method}()"
    )
    raise RuntimeError(message)


def _copy_json_value(value: object, *, root_mapping: bool, label: str) -> Any:
    if root_mapping and type(value) is not dict:
        raise TypeError(f"{label} must be a JSON-compatible dict")
    _validate_json_value(value)
    try:
        encoded = json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(f"{label} must contain strict JSON-compatible data") from error
    return json.loads(encoded)


def _project_document(
    project: ProjectDocument | Mapping[str, object],
) -> ProjectDocument:
    if isinstance(project, ProjectDocument):
        return ProjectDocument.model_validate(project.root)
    if type(project) is not dict:
        raise TypeError("project must be a ProjectDocument or strict dict")
    return ProjectDocument.model_validate(project)


def _document_bytes(document: str | bytes) -> bytes:
    if type(document) is str:
        return document.encode("utf-8")
    if type(document) is bytes:
        return document
    raise TypeError("project document must be bytes or str")


def import_project_json(document: str | bytes) -> ProjectDocument:
    imported = _native.import_project_json(_document_bytes(document))
    return ProjectDocument.model_validate_json(imported)


def import_project_yaml(document: str | bytes) -> ProjectDocument:
    imported = _native.import_project_yaml(_document_bytes(document))
    return ProjectDocument.model_validate_json(imported)


def export_project_json(
    project: ProjectDocument | Mapping[str, object],
) -> str:
    document = _project_document(project)
    return _native.export_project_json(document.canonical_json())


def export_project_yaml(
    project: ProjectDocument | Mapping[str, object],
) -> str:
    document = _project_document(project)
    return _native.export_project_yaml(document.canonical_json())


class FileProjectStore:
    __slots__ = ("_inner",)

    def __init__(self, directory: os.PathLike[str] | str) -> None:
        self._inner = _native._FileProjectStore(os.fspath(directory))

    def create(
        self, project: ProjectDocument | Mapping[str, object]
    ) -> Awaitable[None]:
        document = _project_document(project)
        encoded = document.canonical_json()

        async def create() -> None:
            await self._inner.create(encoded)

        return create()

    def put(self, project: ProjectDocument | Mapping[str, object]) -> Awaitable[None]:
        document = _project_document(project)
        encoded = document.canonical_json()

        async def put() -> None:
            await self._inner.put(encoded)

        return put()

    async def get(self, project_id: str) -> ProjectDocument:
        if not isinstance(project_id, str):
            raise TypeError("project_id must be a string")
        document = await self._inner.get(project_id)
        return ProjectDocument.model_validate_json(document)

    async def list(self) -> list[ProjectDocument]:
        documents = await self._inner.list()
        return [ProjectDocument.model_validate_json(document) for document in documents]

    async def delete(self, project_id: str) -> None:
        if not isinstance(project_id, str):
            raise TypeError("project_id must be a string")
        await self._inner.delete(project_id)

    def create_blocking(self, project: ProjectDocument | Mapping[str, object]) -> None:
        return _run_blocking(lambda: self.create(project), "create")

    def put_blocking(self, project: ProjectDocument | Mapping[str, object]) -> None:
        return _run_blocking(lambda: self.put(project), "put")

    def get_blocking(self, project_id: str) -> ProjectDocument:
        return _run_blocking(lambda: self.get(project_id), "get")

    def list_blocking(self) -> list[ProjectDocument]:
        return _run_blocking(self.list, "list")

    def delete_blocking(self, project_id: str) -> None:
        return _run_blocking(lambda: self.delete(project_id), "delete")
