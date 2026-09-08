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


@pytest.mark.parametrize("metadata_kind", ["schema", "field"])
@pytest.mark.parametrize("input_kind", ["table", "record_batch", "batch"])
def test_convenience_normalizes_arrow_metadata_without_copying_buffers(
    metadata_kind, input_kind, monkeypatch
):
    field = pa.field(
        "x",
        pa.int64(),
        metadata={b"unit": b"quotes"} if metadata_kind == "field" else None,
    )
    schema = pa.schema(
        [field], metadata={b"source": b"quotes"} if metadata_kind == "schema" else None
    )
    data = pa.table({"x": pa.chunked_array([[1, None], [3]])}, schema=schema)
    envelope = {"source": "quotes", "sequence": 7, "nested": {"enabled": True}}
    if input_kind == "record_batch":
        supplied = data.to_batches()[0]
        original = pa.Table.from_batches([supplied])
    elif input_kind == "batch":
        supplied = cf.Batch.from_pyarrow(data, metadata=envelope)
        original = supplied.to_pyarrow()
    else:
        supplied = data
        original = data
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
