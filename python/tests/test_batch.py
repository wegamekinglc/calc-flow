from __future__ import annotations

import gc
import sys
import weakref
from types import MappingProxyType

import pyarrow as pa
import pytest

from calc_flow import Batch, ConfigError


def _data_buffer_addresses(table: pa.Table) -> tuple[tuple[int | None, ...], ...]:
    return tuple(
        tuple(
            buffer.address if buffer is not None else None for buffer in chunk.buffers()
        )
        for chunk in table.column(0).chunks
    )


def test_table_batch_reuses_arrow_buffers_and_copies_metadata() -> None:
    table = pa.table({"value": [1, 2, 3]})
    metadata = {"nested": {"enabled": True}}

    batch = Batch.from_pyarrow(table, metadata=metadata)
    metadata["nested"]["enabled"] = False
    result = batch.to_pyarrow()

    assert _data_buffer_addresses(result) == _data_buffer_addresses(table)
    assert batch.metadata == {"nested": {"enabled": True}}
    assert batch.kind == "table"
    assert batch.num_rows == 3


def test_multi_chunk_table_reuses_every_arrow_buffer() -> None:
    table = pa.table(
        {
            "value": pa.chunked_array(
                [pa.array([1, 2], type=pa.int64()), pa.array([3, 4], type=pa.int64())]
            )
        }
    )

    result = Batch.from_pyarrow(table).to_pyarrow()

    assert result.column(0).num_chunks == 2
    assert _data_buffer_addresses(result) == _data_buffer_addresses(table)


def test_zero_row_table_preserves_its_exact_schema() -> None:
    schema = pa.schema(
        [pa.field("value", pa.int64(), nullable=False)],
        metadata={b"owner": b"calc-flow"},
    )
    table = pa.Table.from_batches([], schema=schema)

    batch = Batch.from_pyarrow(table)
    result = batch.to_pyarrow()

    assert batch.num_rows == 0
    assert result.num_rows == 0
    assert result.schema.equals(schema, check_metadata=True)


def test_zero_row_table_with_no_columns_round_trips() -> None:
    table = pa.Table.from_batches([], schema=pa.schema([]))

    result = Batch.from_pyarrow(table).to_pyarrow()

    assert result.num_rows == 0
    assert result.num_columns == 0
    assert result.schema == table.schema


def test_table_batch_owns_buffers_after_caller_references_are_dropped() -> None:
    table = pa.table({"value": [1, 2, 3]})
    expected_addresses = _data_buffer_addresses(table)
    batch = Batch.from_pyarrow(table)

    del table
    gc.collect()

    result = batch.to_pyarrow()
    assert result["value"].to_pylist() == [1, 2, 3]
    assert _data_buffer_addresses(result) == expected_addresses


def test_metadata_accessor_returns_a_defensive_copy() -> None:
    batch = Batch.from_pyarrow(pa.table({"value": [1]}), {"nested": {"value": 1}})

    first = batch.metadata
    first["nested"]["value"] = 2

    assert batch.metadata == {"nested": {"value": 1}}


def test_metadata_accepts_mappings_without_mutating_them() -> None:
    nested = {"enabled": True}
    metadata = MappingProxyType({"nested": nested})

    batch = Batch.from_pyarrow(pa.table({"value": [1]}), metadata)

    assert batch.metadata == {"nested": {"enabled": True}}
    assert nested == {"enabled": True}


def test_metadata_source_and_sequence_names_remain_plain_attributes() -> None:
    batch = Batch.from_pyarrow(
        pa.table({"value": [1]}),
        {"source": "caller-value", "sequence": 42},
    )

    assert batch.metadata == {"source": "caller-value", "sequence": 42}


def test_metadata_preserves_portable_integer_boundaries_exactly() -> None:
    metadata = {
        "minimum": -(2**63),
        "maximum": 2**64 - 1,
        "nested": [True, {"value": 2**64 - 1}],
    }

    batch = Batch.from_pyarrow(pa.table({"value": [1]}), metadata)

    assert batch.metadata == metadata
    assert isinstance(batch.metadata["nested"][0], bool)


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**64, 10**100])
def test_metadata_rejects_integers_outside_the_portable_range(value: int) -> None:
    with pytest.raises(
        TypeError,
        match="metadata integers must be in the portable JSON range",
    ):
        Batch.from_pyarrow(
            pa.table({"value": [1]}),
            {"nested": [{"value": value}]},
        )


@pytest.mark.parametrize("metadata", [[], "metadata", 1, True])
def test_metadata_rejects_non_mappings(metadata: object) -> None:
    with pytest.raises(TypeError, match="metadata must be a JSON-compatible mapping"):
        Batch.from_pyarrow(pa.table({"value": [1]}), metadata)  # type: ignore[arg-type]


def test_metadata_rejects_non_json_values() -> None:
    with pytest.raises(TypeError):
        Batch.from_pyarrow(pa.table({"value": [1]}), {"items": {1, 2}})


def test_metadata_rejects_executable_values() -> None:
    def callback() -> None:
        pass

    with pytest.raises(TypeError):
        Batch.from_pyarrow(pa.table({"value": [1]}), {"callback": callback})


@pytest.mark.parametrize("metadata", [{1: "value"}, {"nested": {1: "value"}}])
def test_metadata_rejects_non_string_mapping_keys(metadata: object) -> None:
    with pytest.raises(TypeError, match="metadata JSON object keys must be strings"):
        Batch.from_pyarrow(pa.table({"value": [1]}), metadata)  # type: ignore[arg-type]


def test_metadata_rejects_circular_mappings() -> None:
    metadata: dict[str, object] = {}
    metadata["self"] = metadata

    with pytest.raises(ValueError, match="acyclic JSON-compatible mapping"):
        Batch.from_pyarrow(pa.table({"value": [1]}), metadata)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_metadata_rejects_non_json_floats(value: float) -> None:
    with pytest.raises(ValueError, match="JSON compliant"):
        Batch.from_pyarrow(pa.table({"value": [1]}), {"value": value})


def test_metadata_rejects_values_beyond_the_supported_json_depth() -> None:
    value: object = None
    for _ in range(32):
        value = [value]

    with pytest.raises(TypeError, match="metadata must be a JSON-compatible mapping"):
        Batch.from_pyarrow(pa.table({"value": [1]}), {"value": value})


def test_metadata_accepts_values_at_the_supported_json_depth() -> None:
    value: object = None
    for _ in range(31):
        value = [value]

    batch = Batch.from_pyarrow(pa.table({"value": [1]}), {"value": value})

    assert batch.metadata == {"value": value}


def test_batch_is_frozen() -> None:
    batch = Batch.from_pyarrow(pa.table({"value": [1]}))

    with pytest.raises(AttributeError):
        batch.kind = "array"  # type: ignore[misc]


def test_from_pyarrow_requires_an_arrow_table() -> None:
    with pytest.raises(
        TypeError,
        match="table must implement the Arrow C stream interface",
    ):
        Batch.from_pyarrow(object())


def test_from_pyarrow_preserves_arrow_producer_exceptions() -> None:
    failure = RuntimeError("Arrow producer failed")

    class BrokenArrowProducer:
        def __arrow_c_stream__(self) -> object:
            raise failure

    with pytest.raises(RuntimeError) as raised:
        Batch.from_pyarrow(BrokenArrowProducer())  # type: ignore[arg-type]

    assert raised.value is failure
    assert raised.tb is not None
    assert raised.tb.tb_next is not None
    assert raised.tb.tb_next.tb_frame.f_code.co_name == "__arrow_c_stream__"


def test_table_accessor_rejects_array_payload() -> None:
    batch = Batch._from_external(object(), "test", 1, {})

    assert batch.kind == "array"
    assert batch.num_rows == 1
    with pytest.raises(TypeError, match="array batches do not contain a PyArrow table"):
        batch.to_pyarrow()


def test_external_batch_validates_backend_and_length() -> None:
    with pytest.raises(ConfigError, match="invalid backend: must not be empty"):
        Batch._from_external(object(), "", 1, {})
    with pytest.raises(OverflowError):
        Batch._from_external(object(), "test", -1, {})


def test_external_batch_owns_the_python_object() -> None:
    class Payload:
        pass

    payload = Payload()
    payload_ref = weakref.ref(payload)
    batch = Batch._from_external(payload, "test", 1, {})

    del payload
    gc.collect()
    assert payload_ref() is not None

    del batch
    gc.collect()
    assert payload_ref() is None


def test_external_batch_payload_cycle_is_collected() -> None:
    class Payload:
        batch: Batch | None = None

    payload = Payload()
    payload_ref = weakref.ref(payload)
    batch = Batch._from_external(payload, "test", 1, {})
    payload.batch = batch

    assert gc.is_tracked(batch)
    del batch
    del payload
    gc.collect()

    assert payload_ref() is None


def test_repeated_external_batch_destruction_does_not_grow_references() -> None:
    payload = object()
    baseline = sys.getrefcount(payload)

    for _ in range(10_000):
        batch = Batch._from_external(payload, "test", 1, {})
        del batch
    gc.collect()

    assert sys.getrefcount(payload) == baseline
