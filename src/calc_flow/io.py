from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol

import pyarrow as pa

from calc_flow.batch import Batch, BatchMetadata, JSONValue


class Source(Protocol):
    """A replayable producer of formed batches starting at a source cursor."""

    def read(self, cursor: JSONValue = None) -> Iterator[Batch]: ...


class Sink(Protocol):
    """A destination that accepts one graph output batch at a time."""

    def write(self, batch: Batch) -> None: ...


class BatchingSource:
    """Group an in-memory record sequence by Arrow row and byte limits."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        source_id: str,
        max_rows: int = 10_000,
        max_bytes: int = 10 * 1024 * 1024,
        schema: pa.Schema | None = None,
    ) -> None:
        if not source_id:
            msg = "source_id must not be empty"
            raise ValueError(msg)
        if max_rows <= 0:
            msg = "max_rows must be greater than 0"
            raise ValueError(msg)
        if max_bytes <= 0:
            msg = "max_bytes must be greater than 0"
            raise ValueError(msg)
        self._records = records
        self.source_id = source_id
        self.max_rows = max_rows
        self.max_bytes = max_bytes
        self.schema = schema

    def read(self, cursor: JSONValue = None) -> Iterator[Batch]:
        start = 0 if cursor is None else cursor
        if isinstance(start, bool) or not isinstance(start, int):
            msg = "BatchingSource cursor must be an integer record offset"
            raise TypeError(msg)
        if start < 0 or start > len(self._records):
            msg = "BatchingSource cursor is outside the record sequence"
            raise ValueError(msg)

        rows: list[Mapping[str, Any]] = []
        batch_start = start
        for index in range(start, len(self._records)):
            candidate_rows = [*rows, self._records[index]]
            candidate = pa.Table.from_pylist(candidate_rows, schema=self.schema)
            exceeds_limit = (
                len(candidate_rows) > self.max_rows or candidate.nbytes > self.max_bytes
            )
            if rows and exceeds_limit:
                yield self._batch(rows, batch_start=batch_start, cursor=index)
                rows = [self._records[index]]
                batch_start = index
            else:
                rows = candidate_rows

        if rows:
            yield self._batch(
                rows,
                batch_start=batch_start,
                cursor=len(self._records),
            )

    def _batch(
        self,
        rows: list[Mapping[str, Any]],
        *,
        batch_start: int,
        cursor: int,
    ) -> Batch:
        table = pa.Table.from_pylist(rows, schema=self.schema)
        return Batch.table(
            table,
            metadata=BatchMetadata(
                sequence=batch_start,
                source_id=self.source_id,
                cursor=cursor,
            ),
        )
