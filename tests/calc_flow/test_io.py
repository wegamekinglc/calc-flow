from __future__ import annotations

import pyarrow as pa
import pytest

from calc_flow.io import BatchingSource


def test_batching_source_groups_records_by_row_limit() -> None:
    source = BatchingSource(
        [{"value": index} for index in range(5)],
        source_id="records",
        max_rows=2,
    )

    batches = list(source.read())

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert [batch.metadata.cursor for batch in batches] == [2, 4, 5]
    assert [batch.metadata.sequence for batch in batches] == [0, 2, 4]
    assert all(batch.metadata.source_id == "records" for batch in batches)


def test_batching_source_groups_records_by_byte_limit() -> None:
    records = [{"value": "x" * 100}, {"value": "y" * 100}]
    one_row_bytes = pa.Table.from_pylist(records[:1]).nbytes
    source = BatchingSource(
        records,
        source_id="records",
        max_rows=10,
        max_bytes=one_row_bytes,
    )

    batches = list(source.read())

    assert [len(batch) for batch in batches] == [1, 1]


def test_batching_source_resumes_at_record_cursor() -> None:
    source = BatchingSource(
        [{"value": index} for index in range(5)],
        source_id="records",
        max_rows=2,
    )

    batches = list(source.read(2))

    assert batches[0].table_payload["value"].to_pylist() == [2, 3]
    assert batches[-1].metadata.cursor == 5


@pytest.mark.parametrize("cursor", [-1, 4])
def test_batching_source_rejects_cursor_outside_records(cursor: int) -> None:
    source = BatchingSource([{"value": 1}], source_id="records")

    with pytest.raises(ValueError, match="outside"):
        list(source.read(cursor))


def test_batching_source_rejects_non_integer_cursor() -> None:
    source = BatchingSource([{"value": 1}], source_id="records")

    with pytest.raises(TypeError, match="integer"):
        list(source.read({"offset": 0}))


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("source_id", "", "source_id"),
        ("max_rows", 0, "max_rows"),
        ("max_bytes", 0, "max_bytes"),
    ],
)
def test_batching_source_rejects_invalid_settings(
    setting: str, value: object, message: str
) -> None:
    options = {"source_id": "records", "max_rows": 10, "max_bytes": 1000}
    options[setting] = value

    with pytest.raises(ValueError, match=message):
        BatchingSource([{"value": 1}], **options)
