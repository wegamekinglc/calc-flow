from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
import pytest

from calc_flow.batch import Batch, BatchKind, BatchMetadata


def test_table_batch_preserves_table_and_metadata() -> None:
    table = pa.table({"x": [1, 2]})
    metadata = BatchMetadata(sequence=3, source_id="source")

    batch = Batch.table(table, metadata=metadata)

    assert batch.kind is BatchKind.TABLE
    assert batch.table_payload is table
    assert batch.metadata is metadata
    assert batch.schema == table.schema
    assert len(batch) == 2


def test_record_batch_conversion_reuses_arrow_buffers() -> None:
    record_batch = pa.RecordBatch.from_pydict({"x": [1, 2, 3]})
    source_buffer = record_batch.column(0).buffers()[1]

    batch = Batch.table(record_batch)
    result_buffer = batch.table_payload.column(0).chunk(0).buffers()[1]

    assert source_buffer is not None
    assert result_buffer is not None
    assert result_buffer.address == source_buffer.address


def test_arrow_c_stream_protocol_ingestion() -> None:
    table = pa.table({"x": [1, 2]})
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())

    batch = Batch.from_tabular_protocol(reader)

    assert batch.table_payload.equals(table)


def test_dataframe_interchange_protocol_ingestion() -> None:
    table = pa.table({"x": [1, 2]})

    class InterchangeOnly:
        def __dataframe__(self, **kwargs: object):
            return table.__dataframe__(**kwargs)

    batch = Batch.from_tabular_protocol(InterchangeOnly())

    assert batch.table_payload.equals(table)


def test_array_batch_requires_array_api_object() -> None:
    array = np.asarray([1, 2, 3])

    batch = Batch.array(array)

    assert batch.kind is BatchKind.ARRAY
    assert batch.array_payload is array
    assert batch.schema is None
    assert len(batch) == 3

    with pytest.raises(TypeError, match="__array_namespace__"):
        Batch.array([1, 2, 3])


def test_metadata_is_deeply_immutable_and_serializable() -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    metadata = BatchMetadata(
        sequence=1,
        cursor={"partition": 2, "offsets": [3, 4]},
        event_time=now,
        attributes={"labels": ["a", "b"]},
    )

    assert metadata.cursor["offsets"] == (3, 4)  # type: ignore[index]
    assert metadata.attributes["labels"] == ("a", "b")
    assert metadata.to_dict() == {
        "batch_id": metadata.batch_id,
        "sequence": 1,
        "source_id": None,
        "cursor": {"partition": 2, "offsets": [3, 4]},
        "event_time": now.isoformat(),
        "watermark": None,
        "attributes": {"labels": ["a", "b"]},
    }

    with pytest.raises(TypeError):
        metadata.attributes["new"] = True  # type: ignore[index]


@pytest.mark.parametrize("value", [float("inf"), object()])
def test_metadata_rejects_non_json_values(value: object) -> None:
    error = ValueError if isinstance(value, float) else TypeError
    with pytest.raises(error):
        BatchMetadata(attributes={"bad": value})  # type: ignore[dict-item]


def test_batch_with_payload_preserves_metadata() -> None:
    batch = Batch.table(
        pa.table({"x": [1]}), metadata=BatchMetadata(source_id="source")
    )

    result = batch.with_payload(pa.table({"x": [2]}))

    assert result.metadata is batch.metadata
    assert result.table_payload.to_pylist() == [{"x": 2}]


@pytest.mark.parametrize(
    "changes",
    (
        {"batch_id": ""},
        {"sequence": -1},
    ),
)
def test_metadata_rejects_invalid_identity(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BatchMetadata(**changes)


def test_metadata_rejects_non_string_mapping_keys() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        BatchMetadata(cursor={1: "invalid"})  # type: ignore[dict-item]


def test_batch_rejects_invalid_table_and_protocol_inputs() -> None:
    with pytest.raises(TypeError, match="table batches"):
        Batch("not-arrow", BatchKind.TABLE)
    with pytest.raises(TypeError, match="accepts only"):
        Batch.table("not-arrow")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tabular inputs"):
        Batch.from_tabular_protocol(object())


def test_batch_kind_specific_accessors_reject_the_other_payload() -> None:
    table_batch = Batch.table(pa.table({"x": [1]}))
    array_batch = Batch.array(np.asarray([1]))

    with pytest.raises(TypeError, match="does not contain an array"):
        _ = table_batch.array_payload
    with pytest.raises(TypeError, match="does not contain a table"):
        _ = array_batch.table_payload


def test_batch_with_metadata_and_scalar_array_shape() -> None:
    batch = Batch.table(pa.table({"x": [1]}))

    updated = batch.with_metadata(sequence=4, source_id="updated")
    scalar = Batch.array(np.asarray(3))

    assert updated.metadata.sequence == 4
    assert updated.metadata.source_id == "updated"
    assert scalar.num_rows == 1
