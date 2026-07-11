from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import pyarrow as pa
from pyarrow.interchange import from_dataframe

type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)


class BatchKind(StrEnum):
    TABLE = "table"
    ARRAY = "array"


def _freeze_json(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            msg = f"{path} must contain finite JSON numbers"
            raise ValueError(msg)
        return value
    if isinstance(value, list):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                msg = f"{path} keys must be strings"
                raise TypeError(msg)
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    msg = f"{path} contains non-JSON value {type(value).__name__}"
    raise TypeError(msg)


def _thaw_json(value: Any) -> JSONValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class BatchMetadata:
    """Immutable provenance and ordering metadata carried with a batch."""

    batch_id: str = field(default_factory=lambda: uuid4().hex)
    sequence: int | None = None
    source_id: str | None = None
    cursor: JSONValue = None
    event_time: datetime | None = None
    watermark: datetime | None = None
    attributes: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.batch_id:
            msg = "batch_id must not be empty"
            raise ValueError(msg)
        if self.sequence is not None and self.sequence < 0:
            msg = "sequence must be greater than or equal to 0"
            raise ValueError(msg)

        object.__setattr__(self, "cursor", _freeze_json(self.cursor, path="cursor"))
        frozen_attributes = _freeze_json(self.attributes, path="attributes")
        object.__setattr__(self, "attributes", frozen_attributes)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "batch_id": self.batch_id,
            "sequence": self.sequence,
            "source_id": self.source_id,
            "cursor": _thaw_json(self.cursor),
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "watermark": self.watermark.isoformat() if self.watermark else None,
            "attributes": _thaw_json(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class Batch:
    """Immutable envelope for an Arrow table or Python Array API object."""

    payload: pa.Table | Any
    kind: BatchKind
    metadata: BatchMetadata = field(default_factory=BatchMetadata)

    def __post_init__(self) -> None:
        if self.kind is BatchKind.TABLE and not isinstance(self.payload, pa.Table):
            msg = "table batches must contain a pyarrow.Table"
            raise TypeError(msg)
        if self.kind is BatchKind.ARRAY and not callable(
            getattr(self.payload, "__array_namespace__", None)
        ):
            msg = "array batches must implement __array_namespace__"
            raise TypeError(msg)

    @classmethod
    def table(
        cls,
        data: pa.Table | pa.RecordBatch,
        *,
        metadata: BatchMetadata | None = None,
    ) -> Batch:
        if isinstance(data, pa.RecordBatch):
            data = pa.Table.from_batches([data], schema=data.schema)
        if not isinstance(data, pa.Table):
            msg = "Batch.table accepts only pyarrow.Table or pyarrow.RecordBatch"
            raise TypeError(msg)
        return cls(data, BatchKind.TABLE, metadata or BatchMetadata())

    @classmethod
    def from_tabular_protocol(
        cls,
        data: object,
        *,
        metadata: BatchMetadata | None = None,
    ) -> Batch:
        if isinstance(data, pa.Table | pa.RecordBatch):
            return cls.table(data, metadata=metadata)

        if callable(getattr(data, "__arrow_c_stream__", None)):
            table = pa.RecordBatchReader.from_stream(data).read_all()
            return cls.table(table, metadata=metadata)

        if callable(getattr(data, "__dataframe__", None)):
            table = from_dataframe(data)
            return cls.table(table, metadata=metadata)

        msg = (
            "tabular inputs must be Arrow objects or implement "
            "__arrow_c_stream__ or __dataframe__"
        )
        raise TypeError(msg)

    @classmethod
    def array(
        cls,
        data: Any,
        *,
        metadata: BatchMetadata | None = None,
    ) -> Batch:
        return cls(data, BatchKind.ARRAY, metadata or BatchMetadata())

    @property
    def table_payload(self) -> pa.Table:
        if self.kind is not BatchKind.TABLE:
            msg = "batch does not contain a table"
            raise TypeError(msg)
        return self.payload

    @property
    def array_payload(self) -> Any:
        if self.kind is not BatchKind.ARRAY:
            msg = "batch does not contain an array"
            raise TypeError(msg)
        return self.payload

    @property
    def schema(self) -> pa.Schema | None:
        if self.kind is BatchKind.TABLE:
            return self.table_payload.schema
        return None

    @property
    def num_rows(self) -> int:
        if self.kind is BatchKind.TABLE:
            return self.table_payload.num_rows
        shape = getattr(self.array_payload, "shape", ())
        return int(shape[0]) if shape else 1

    def with_payload(self, payload: pa.Table | Any) -> Batch:
        if self.kind is BatchKind.TABLE:
            return Batch.table(payload, metadata=self.metadata)
        return Batch.array(payload, metadata=self.metadata)

    def with_metadata(self, **changes: Any) -> Batch:
        return replace(self, metadata=replace(self.metadata, **changes))

    def __len__(self) -> int:
        return self.num_rows
