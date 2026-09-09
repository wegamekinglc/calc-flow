from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

import calc_flow as cf


def _buffer_addresses(table: pa.Table) -> list[list[int | None]]:
    return [
        [None if buffer is None else buffer.address for buffer in chunk.buffers()]
        for chunk in table["x"].chunks
    ]


def _metadata_table(metadata_kind):
    field = pa.field(
        "x",
        pa.int64(),
        metadata={b"unit": b"quotes"} if metadata_kind == "field" else None,
    )
    schema = pa.schema(
        [field], metadata={b"source": b"quotes"} if metadata_kind == "schema" else None
    )
    return pa.table({"x": pa.chunked_array([[1, None], [3]])}, schema=schema)


def _supplied_table(data, input_kind, envelope):
    if input_kind == "record_batch":
        supplied = data.to_batches()[0]
        return supplied, pa.Table.from_batches([supplied])
    if input_kind == "batch":
        supplied = cf.Batch.from_pyarrow(data, metadata=envelope)
        return supplied, supplied.to_pyarrow()
    return data, data


@pytest.mark.parametrize("metadata_kind", ["schema", "field"])
@pytest.mark.parametrize("input_kind", ["table", "record_batch", "batch"])
def test_convenience_normalizes_arrow_metadata_without_copying_buffers(
    metadata_kind, input_kind, monkeypatch
):
    data = _metadata_table(metadata_kind)
    schema = data.schema
    envelope = {"source": "quotes", "sequence": 7, "nested": {"enabled": True}}
    supplied, original = _supplied_table(data, input_kind, envelope)
    before = original.to_pydict()
    buffers = _buffer_addresses(original)
    captured = []
    execute = cf.BatchExecutionPlan.execute

    def record(self, inputs, *, options=None):
        captured.append(inputs["input"])
        return execute(self, inputs, options=options)

    monkeypatch.setattr(cf.BatchExecutionPlan, "execute", record)
    t = cf.table_input("quotes", schema=original.schema)
    expression = t.select(y=t["x"] + 1)
    program = cf.Program("metadata", outputs={"answer": expression})
    expected = {"y": [None if value is None else value + 1 for value in before["x"]]}
    assert (
        cf.compute(supplied, lambda q: q.select(y=q["x"] + 1)).to_pydict() == expected
    )
    assert expression.collect(supplied).to_pydict() == expected
    assert program.collect({"quotes": supplied})["answer"].to_pydict() == expected
    assert original.schema.equals(schema, check_metadata=True)
    assert original.to_pydict() == before
    assert _buffer_addresses(original) == buffers
    assert len(captured) == 3
    for batch in captured:
        normalized = batch.to_pyarrow()
        assert normalized.schema.equals(
            pa.schema([pa.field("x", pa.int64())]), check_metadata=True
        )
        assert _buffer_addresses(normalized) == buffers
        assert batch.metadata == (envelope if input_kind == "batch" else {})
    if input_kind == "batch":
        assert supplied.to_pyarrow().schema.equals(schema, check_metadata=True)
        assert supplied.metadata == envelope


@pytest.mark.parametrize("entry", ["compute", "table", "program"])
def test_async_convenience_accepts_arrow_metadata(entry):
    schema = pa.schema(
        [pa.field("x", pa.int64(), metadata={b"unit": b"quotes"})],
        metadata={b"source": b"quotes"},
    )
    data = pa.table({"x": [1]}, schema=schema)
    batch = cf.Batch.from_pyarrow(data, metadata={"sequence": 3})
    t = cf.table_input("quotes", schema=schema)
    expression = t.select(y=t["x"] + 1)

    async def run():
        if entry == "compute":
            result = await cf.compute_async(batch, lambda q: q.select(y=q["x"] + 1))
        elif entry == "table":
            result = await expression.collect_async(batch)
        else:
            program = cf.Program("metadata", outputs={"answer": expression})
            result = (await program.collect_async({"quotes": batch}))["answer"]
        assert result.to_pydict() == {"y": [2]}

    asyncio.run(run())
    assert batch.to_pyarrow().schema.equals(schema, check_metadata=True)
    assert batch.metadata == {"sequence": 3}


@pytest.mark.parametrize("metadata", [None, {}, {b"source": b"quotes"}])
@pytest.mark.parametrize("row_count", [0, 1, 3])
@pytest.mark.parametrize("input_kind", ["table", "record_batch", "batch"])
def test_compute_preserves_zero_column_metadata_rows(
    metadata, row_count, input_kind, monkeypatch
):
    record = (
        pa.record_batch([pa.array(range(row_count), type=pa.int64())], names=["x"])
        .replace_schema_metadata(metadata)
        .select([])
    )
    table = pa.Table.from_batches([record])
    envelope = {"source": "quotes", "sequence": 7}
    supplied = table
    if input_kind == "record_batch":
        supplied = record
    elif input_kind == "batch":
        supplied = cf.Batch.from_pyarrow(table, metadata=envelope)
    original = supplied.to_pyarrow() if input_kind == "batch" else supplied
    schema = original.schema
    captured = []
    execute = cf.BatchExecutionPlan.execute

    def capture(self, inputs, *, options=None):
        captured.append(inputs["input"])
        return execute(self, inputs, options=options)

    monkeypatch.setattr(cf.BatchExecutionPlan, "execute", capture)
    result = cf.compute(supplied, lambda t: t.select(value=cf.lit(1)))
    assert result.to_pydict() == {"value": [1] * row_count}
    assert len(captured) == 1
    assert captured[0].num_rows == row_count
    assert captured[0].to_pyarrow().schema.metadata is None
    assert captured[0].metadata == (envelope if input_kind == "batch" else {})
    assert supplied.num_rows == table.num_rows == record.num_rows == row_count
    original = supplied.to_pyarrow() if input_kind == "batch" else supplied
    assert original.schema.equals(schema, check_metadata=True)
    assert original.schema.metadata == schema.metadata
    assert table.schema.metadata == record.schema.metadata == metadata
    if input_kind == "batch":
        assert supplied.metadata == envelope


@pytest.mark.parametrize("entry", ["compute", "table", "program"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_collection_preserves_zero_column_metadata_rows(entry, asynchronous):
    data = (
        pa.table({"x": pa.chunked_array([[1], [2, 3]])})
        .replace_schema_metadata({b"source": b"quotes"})
        .select([])
    )
    table = cf.table_input("quotes", schema=data.schema)
    expression = table.select(value=cf.lit(1))
    program = cf.Program("metadata", outputs={"answer": expression})
    if entry == "compute":
        collect = cf.compute_async if asynchronous else cf.compute
        result = collect(data, lambda t: t.select(value=cf.lit(1)))
    elif entry == "table":
        collect = expression.collect_async if asynchronous else expression.collect
        result = collect(data)
    else:
        collect = program.collect_async if asynchronous else program.collect
        result = collect({"quotes": data})
    if asynchronous:

        async def run():
            return await result

        result = asyncio.run(run())
    if entry == "program":
        result = result["answer"]
    assert result.to_pydict() == {"value": [1, 1, 1]}
    assert data.num_rows == 3
    assert data.schema.metadata == {b"source": b"quotes"}
