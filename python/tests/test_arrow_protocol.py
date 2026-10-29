from __future__ import annotations

import gc
import sys
from types import ModuleType

import pyarrow as pa
import pytest

import calc_flow as cf


def test_protocol_batch_preserves_buffers_schema_and_repeated_exports() -> None:
    schema = pa.schema(
        [pa.field("a", pa.int64(), metadata={"unit": "count"})],
        metadata={"owner": "caller"},
    )
    table = pa.Table.from_batches(
        [pa.record_batch([pa.array([0, None, 2]).slice(1)], schema=schema)],
        schema=schema,
    )
    batch = cf.Batch.from_arrow(table, {"nested": [1]})
    original = table[0].chunk(0)
    logical_data_address = original.buffers()[1].address + original.offset * 8
    del table, original
    gc.collect()
    for _ in range(3):
        result = pa.table(batch)
        assert result.schema.equals(schema, check_metadata=True)
        assert result.to_pydict() == {"a": [None, 2]}
        column = result[0].chunk(0)
        # Arrow export realigns a non-byte-aligned null bitmap in the baseline too.
        assert column.buffers()[1].address + column.offset * 8 == logical_data_address
        assert column.null_count == 1
    assert cf.Batch.from_arrow(batch).metadata == {}


def test_compute_batch_result_preserves_compatibility_and_input() -> None:
    table = pa.table({"a": [1, 2], "b": [3, 4]}).replace_schema_metadata({"x": "y"})
    batch = cf.Batch.from_pyarrow(table, {"caller": True})

    def build(t):
        return t.select(total=t["a"] + t["b"])

    result = cf.compute(batch, build, output_format="batch")
    assert isinstance(result, cf.Batch)
    assert result.to_pyarrow().to_pydict() == {"total": [4, 6]}
    assert isinstance(cf.compute(table, build), pa.Table)
    assert batch.to_pyarrow().schema.equals(table.schema, check_metadata=True)
    assert batch.metadata == {"caller": True}
    with pytest.raises(ValueError, match="output_format"):
        cf.compute(table, build, output_format="invalid")


def test_arrow_array_table_requires_struct_root() -> None:
    record = pa.record_batch({"a": [1, 2]})

    class Producer:
        def __arrow_c_array__(self, requested_schema=None):
            return record.__arrow_c_array__(requested_schema)

    assert cf.Batch.from_arrow(Producer()).to_pyarrow().to_pydict() == {"a": [1, 2]}
    for column in (pa.array([1, 2]), pa.chunked_array([[1], [2]])):
        with pytest.raises(TypeError, match="struct root"):
            cf.Batch.from_arrow(column)


def test_unsafe_pyarrow_version_is_rejected_before_export(monkeypatch) -> None:
    table = pa.table({"a": [1]})
    monkeypatch.setattr(pa, "__version__", "25.0.0")
    with pytest.raises(ImportError, match="PyArrow"):
        cf.Batch.from_arrow(table)


@pytest.mark.parametrize("base", [pa.Table, pa.RecordBatch, pa.RecordBatchReader])
def test_unsafe_pyarrow_subclass_is_rejected_before_ffi(monkeypatch, base) -> None:
    class Producer(base):
        def __init__(self):
            self.exports = 0

        def __arrow_c_stream__(self):
            self.exports += 1
            raise AssertionError("unsafe PyArrow producer reached FFI")

    producer = Producer()
    monkeypatch.setattr(pa, "__version__", "25.0.0")
    with pytest.raises(ImportError, match="PyArrow"):
        cf.Batch.from_arrow(producer)
    assert producer.exports == 0


@pytest.mark.parametrize("changed", ["version", "module"])
def test_pyarrow_guard_revalidates_live_module_and_version(monkeypatch, changed):
    table = pa.table({"a": [1]})
    schema = table.schema
    batch = cf.Batch.from_arrow(table)
    assert batch.to_pyarrow().to_pydict() == {"a": [1]}
    if changed == "version":
        monkeypatch.setattr(pa, "__version__", "25.0.0")
    else:
        replacement = ModuleType("pyarrow")
        replacement.__version__ = "25.0.0"
        monkeypatch.setitem(sys.modules, "pyarrow", replacement)
    for call in (
        lambda: cf.Batch.from_arrow(table),
        batch.to_pyarrow,
        lambda: cf._native._schema_fields(schema),
    ):
        with pytest.raises(ImportError, match="PyArrow"):
            call()


def test_stream_protocol_rejects_schema_capsule_as_stream() -> None:
    class Producer:
        def __arrow_c_stream__(self):
            return pa.schema([("a", pa.int64())]).__arrow_c_schema__()

    with pytest.raises(ValueError, match="name|arrow_array_stream"):
        cf.Batch.from_arrow(Producer())


def test_stream_protocol_attribute_error_propagates_without_array_fallback() -> None:
    failure = RuntimeError("stream attribute failed")

    class Producer:
        @property
        def __arrow_c_stream__(self):
            raise failure

        def __arrow_c_array__(self):
            raise AssertionError("broken stream must not fall back to array")

    with pytest.raises(RuntimeError) as caught:
        cf.Batch.from_arrow(Producer())
    assert caught.value is failure


@pytest.mark.parametrize("named", [False, True])
def test_batch_stream_results_close_owned_sources(named) -> None:
    import asyncio

    data = pa.table({"a": [1, 2]})
    source = cf.table_input("events", schema=data.schema)
    output = source.select(value=source["a"] + 1)
    owner = (
        cf.Program("native-output", engine="streaming", outputs={"values": output})
        if named
        else output
    )

    async def run():
        closed = asyncio.Event()

        async def feed():
            try:
                yield cf.Batch.from_arrow(data)
                await asyncio.Event().wait()
            finally:
                closed.set()

        inputs = {"events": feed()} if named else feed()
        results = owner.stream(
            inputs,
            output_format="batch",
            config=cf.StreamRuntimeConfig(checkpointing=False),
        )
        async with results:
            event = await asyncio.wait_for(results.__anext__(), 5)
            batch = event.table if named else event
            assert isinstance(batch, cf.Batch)
            assert batch.to_pyarrow().to_pydict() == {"value": [2, 3]}
        assert closed.is_set()
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


def test_non_pyarrow_producer_and_schema() -> None:
    na = pytest.importorskip("nanoarrow")
    producer = na.Array(pa.table({"a": [1, 2]}).to_batches()[0])
    batch = cf.Batch.from_arrow(producer)
    source = cf.table_input("input", schema=na.Schema(pa.schema([("a", pa.int64())])))
    assert source.collect(batch, output_format="batch").to_pyarrow().to_pydict() == {
        "a": [1, 2]
    }


@pytest.mark.parametrize(
    "table",
    [
        pa.table({"a": pa.array([], type=pa.int64())}),
        pa.record_batch({"a": [1, 2]}).select([]),
    ],
)
def test_protocol_preserves_empty_tables_and_zero_column_row_counts(table) -> None:
    batch = cf.Batch.from_arrow(table)
    assert batch.num_rows == table.num_rows
    assert batch.to_pyarrow().num_columns == table.num_columns
    assert pa.table(cf.Batch.from_arrow(batch)).num_rows == table.num_rows


def test_producer_exceptions_propagate_without_fallback() -> None:
    failure = RuntimeError("producer failed")

    class Producer:
        def __arrow_c_stream__(self):
            raise failure

    with pytest.raises(RuntimeError) as caught:
        cf.Batch.from_arrow(Producer())
    assert caught.value is failure


@pytest.mark.parametrize("stream", [False, True])
def test_nullable_struct_root_is_rejected_without_panicking(stream) -> None:
    values = pa.StructArray.from_arrays(
        [pa.array([1, 2])], names=["a"], mask=pa.array([False, True])
    )
    with pytest.raises(TypeError, match="non-null struct"):
        cf.Batch.from_arrow(pa.chunked_array([values]) if stream else values)
