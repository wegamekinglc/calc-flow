from __future__ import annotations

import asyncio
import gc
import weakref
from dataclasses import replace

import pyarrow as pa
import pytest

import calc_flow as cf
import calc_flow.stream as stream_module
from calc_flow.symbolic.nodes import build


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("projected", [False, True])
def test_materialized_chunks_form_one_complete_snapshot(wrapped, projected):
    data = pa.Table.from_batches(
        [pa.record_batch({"value": [1, 2]}), pa.record_batch({"value": [3]})]
    )
    source = cf.table_input("values", schema=data.schema)
    if projected:
        source = source.select("value")
    output = cf.snapshot_sql("SELECT SUM(value) AS total FROM p", p=source)

    async def run():
        bound = cf.Batch.from_pyarrow(data) if wrapped else data
        async with output.stream(bound) as results:
            tables = [table async for table in results]
        assert results.job.status()["task_count"] == 0
        return tables

    tables = asyncio.run(run())
    assert [table.to_pydict() for table in tables] == [{"total": [6]}]


def _source(name="values"):
    return cf.table_input(name, schema=pa.schema([("value", pa.int64())]))


def _program(output):
    return cf.Program("snapshots", engine="streaming", outputs={"result": output})


def test_declaration_validates_query_aliases_and_table_types():
    with pytest.raises(TypeError, match="snapshot_sql.query"):
        cf.snapshot_sql(1, p=_source())
    with pytest.raises(ValueError, match="snapshot_sql.query"):
        cf.snapshot_sql("", p=_source())
    with pytest.raises(ValueError, match="snapshot_sql.tables"):
        cf.snapshot_sql("SELECT 1")
    with pytest.raises(TypeError, match="snapshot_sql.tables.p"):
        cf.snapshot_sql("SELECT * FROM p", p=object())
    with pytest.raises(ValueError, match="snapshot_sql.tables.bad-name"):
        cf.snapshot_sql("SELECT 1", **{"bad-name": _source()})
    with pytest.raises(ValueError, match="aliases conflict"):
        cf.snapshot_sql("SELECT 1", p=_source(), P=_source("other"))


@pytest.mark.parametrize(
    "query",
    [
        " ",
        "DELETE FROM p",
        "SELECT * FROM p; SELECT * FROM p",
        "SELECT missing FROM p",
        "SELECT value AS x, value AS x FROM p",
        "SELECT value + true FROM p",
    ],
)
def test_invalid_queries_fail_during_schema_planning(query):
    with pytest.raises(cf.CompileError, match=r"outputs.result.snapshot_sql.schema"):
        _program(cf.snapshot_sql(query, p=_source())).to_project()


def test_snapshot_is_stream_only_and_old_sql_still_has_one_stream_alias():
    source = _source()
    output = cf.snapshot_sql("SELECT * FROM p", p=source)
    with pytest.raises(ValueError, match="snapshot_sql requires the streaming engine"):
        cf.Program("batch", engine="sql", outputs={"result": output})
    with pytest.raises(ValueError, match="snapshot_sql.*streaming engine"):
        output.collect(pa.table({"value": [1]}))
    program = _program(output)
    with pytest.raises(RuntimeError, match="engine"):
        program.compile_batch()
    with pytest.raises(RuntimeError, match="engine"):
        program.collect({"values": pa.table({"value": [1]})})
    assert any(
        issue.code == "unsupported_mode"
        for issue in program.analyze(mode="batch").issues
    )
    old = cf.sql("SELECT * FROM p", p=source, q=_source("other"))
    with pytest.raises(ValueError, match="one table alias"):
        _program(old)


def test_schema_only_planning_preserves_native_types_names_and_nullability():
    left = cf.table_input(
        "left", schema=pa.schema([pa.field("value", pa.int64(), nullable=False)])
    )
    right = _source("right")
    query = (
        "SELECT p.value, q.value AS optional, coalesce(q.value, 0) AS filled "
        "FROM p LEFT JOIN q ON p.value = q.value"
    )
    program = _program(cf.snapshot_sql(query, p=left, q=right))
    document = program.to_project().root
    node = next(
        n for n in document["graph"]["nodes"] if n["operator"]["kind"] == "snapshot_sql"
    )
    assert node["output_ports"][0]["schema"] == [
        {"name": "value", "data_type": "int64", "nullable": False},
        {"name": "optional", "data_type": "int64", "nullable": True},
        {"name": "filled", "data_type": "int64", "nullable": False},
    ]
    assert program.compile_stream().fingerprint


def test_project_roundtrip_canonical_alias_order_and_unknown_kind_rejection():
    left, right = _source("left"), _source("right")
    query = "SELECT SUM(p.value + q.value) AS total FROM p CROSS JOIN q"
    first = _program(cf.snapshot_sql(query, q=right, p=left))
    second = _program(cf.snapshot_sql(query, p=left, q=right))
    assert first.fingerprint == second.fingerprint
    document = first.to_project()
    encoded = document.model_dump_json()
    assert cf.ProjectDocument.model_validate_json(encoded).model_dump_json() == encoded
    assert (
        first.compile_stream().fingerprint
        == cf.Runtime()._compile_stream_graph_project(encoded).fingerprint
    )
    broken = document.model_dump()
    snapshot = next(
        n for n in broken["graph"]["nodes"] if n["operator"]["kind"] == "snapshot_sql"
    )
    assert snapshot["operator"]["aliases"] == ["p", "q"]
    snapshot["operator"]["kind"] = "future_snapshot_sql"
    with pytest.raises(ValueError, match="kind|operator|variant"):
        cf.ProjectDocument.model_validate(broken)
    with pytest.raises(ValueError, match="unknown_primitive_version"):
        build("snapshot_sql", (), {}, version=2)


def test_snapshot_capability_is_required_before_lowering(monkeypatch):
    runtime = cf.Runtime()
    capabilities = runtime.capabilities()
    monkeypatch.setattr(
        cf.Runtime,
        "capabilities",
        lambda _self: replace(
            capabilities,
            operators=tuple(
                op for op in capabilities.operators if op.kind != "snapshot_sql"
            ),
        ),
    )
    with pytest.raises(cf.CompileError, match="capability_mismatch.*snapshot_sql"):
        _program(cf.snapshot_sql("SELECT * FROM p", p=_source())).to_project(runtime)


def test_snapshot_successors_and_exactly_once_are_rejected():
    output = cf.snapshot_sql("SELECT * FROM p", p=_source())
    for successor in (output.select("value"), output.sql("SELECT * FROM input")):
        with pytest.raises(ValueError, match="snapshot_sql.*directly to sinks"):
            _program(successor)
    project = _program(output).to_project().model_dump_json()
    with pytest.raises(cf.CompileError, match="snapshot_sql.*exactly-once"):
        cf.Runtime()._compile_stream_graph_project(
            project,
            requirements=cf.StreamRequirements(
                delivery={"output": cf.DeliveryGuarantee.EXACTLY_ONCE}
            ),
        )


async def _tables(*tables):
    for table in tables:
        yield table


def test_empty_initial_snapshot_emits_and_later_updates_replace():
    output = cf.snapshot_sql(
        "SELECT COUNT(*) AS n, SUM(value) AS total FROM p", p=_source()
    )
    data = pa.table({"value": [2, 3]})

    async def run():
        async with output.stream(
            _tables(data.slice(0, 0), data, data.slice(0, 0))
        ) as results:
            return [table.to_pydict() async for table in results]

    assert asyncio.run(run()) == [
        {"n": [0], "total": [None]},
        {"n": [2], "total": [5]},
        {"n": [0], "total": [None]},
    ]


def test_materialized_empty_relation_emits_a_typed_empty_result():
    source = _source()
    output = cf.snapshot_sql("SELECT * FROM p", p=source)
    data = pa.table({"value": pa.array([], type=pa.int64())})

    async def run():
        async with output.stream(data) as results:
            return [table async for table in results]

    tables = asyncio.run(run())
    assert len(tables) == 1
    assert tables[0].equals(data)


def test_uninitialized_alias_at_eof_is_not_an_empty_relation():
    output = cf.snapshot_sql(
        "SELECT COUNT(*) AS n FROM p CROSS JOIN q", p=_source(), q=_source("other")
    )

    async def run():
        async with output.stream(
            {"values": _tables(), "other": pa.table({"value": [1]})}
        ) as results:
            return [table async for table in results]

    assert asyncio.run(run()) == []


@pytest.mark.parametrize("first_alias", ["p", "q"])
def test_latest_relations_match_batch_sql_after_each_update_and_one_eof(first_alias):
    query = "SELECT SUM(p.value * q.value) AS total FROM p CROSS JOIN q"
    p = pa.table({"value": [1, 2]})
    q = pa.table({"value": [10]})
    updates = [q, pa.table({"value": [20, 30]}), q.slice(0, 0)]
    oracle = (
        cf.PipelineBuilder("oracle")
        .sql("query", query, aliases=("p", "q"))
        .compile_batch()
    )
    expected = [
        oracle.execute(
            {"p": cf.Batch.from_pyarrow(p), "q": cf.Batch.from_pyarrow(table)}
        )
        .outputs["output"]
        .to_pyarrow()
        for table in updates
    ]
    output = cf.snapshot_sql(query, p=_source("p"), q=_source("q"))

    async def run():
        queues = {name: asyncio.Queue() for name in ("p", "q")}
        polled = {name: asyncio.Event() for name in queues}

        async def feed(name):
            while True:
                polled[name].set()
                data = await queues[name].get()
                if data is None:
                    return
                yield data

        async with output.stream({name: feed(name) for name in queues}) as results:
            pending = asyncio.create_task(results.__anext__())
            await queues[first_alias].put(p if first_alias == "p" else q)
            await asyncio.wait_for(polled[first_alias].wait(), 5)
            assert not pending.done()
            other = "q" if first_alias == "p" else "p"
            await queues[other].put(q if other == "q" else p)
            actual = [await asyncio.wait_for(pending, 5)]
            await queues["p"].put(None)
            for table in updates[1:]:
                await queues["q"].put(table)
                actual.append(await asyncio.wait_for(results.__anext__(), 5))
            await queues["q"].put(None)
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(results.__anext__(), 5)
        return actual

    actual = asyncio.run(run())
    assert all(table.equals(wanted) for table, wanted in zip(actual, expected))


@pytest.mark.parametrize("kind", ["materialized", "iterable"])
@pytest.mark.parametrize("invalid", ["schema", "rows", "bytes"])
def test_invalid_snapshot_inputs_fail_without_slicing_or_payload_leaks(kind, invalid):
    data = pa.table({"value": [1, 2]})
    budget = cf.EdgeBudget(
        1 if invalid == "rows" else 10, 8 if invalid == "bytes" else 1024
    )
    if invalid == "schema":
        data = pa.table({"value": ["private-payload-sentinel"]})
    output = cf.snapshot_sql("SELECT * FROM p", p=_source())
    config = cf.StreamRuntimeConfig(edge_budget=budget, checkpointing=False)

    async def run():
        bound = data if kind == "materialized" else _tables(data)
        error = ValueError if kind == "materialized" else cf.StreamingRuntimeError
        with pytest.raises(
            error,
            match=r"stream.inputs.values.*schema|stream.inputs.values.*edge_budget",
        ) as caught:
            async with output.stream(bound, config=config) as results:
                async for _ in results:
                    raise AssertionError(
                        "invalid input must not emit a partial relation"
                    )
        assert "private-payload-sentinel" not in str(caught.value)

    asyncio.run(run())


@pytest.mark.parametrize("limited", ["state", "output"])
def test_native_snapshot_state_and_whole_output_budgets_reach_lowered_nodes(limited):
    source = _source()
    output = cf.snapshot_sql(
        "SELECT p.value AS a, q.value AS b FROM p CROSS JOIN q", p=source, q=source
    )
    config = cf.StreamRuntimeConfig(
        checkpointing=False,
        edge_budget=cf.EdgeBudget(2, 1024),
        sql_snapshot_state_budget=cf.StateBudget(1, 1 << 20)
        if limited == "state"
        else None,
    )

    async def run():
        results = output.stream(pa.table({"value": [1, 2]}), config=config)
        with pytest.raises(cf.StreamingRuntimeError, match='operator "cf_sql_'):
            async with results:
                async for _ in results:
                    raise AssertionError("over-budget state/result must not emit")
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())


def test_arrow_source_rejection_keeps_append_path_slicing():
    source = _source()
    row = source.select(incremented=source["value"] + 1)
    output = cf.snapshot_sql("SELECT SUM(incremented) AS total FROM p", p=row)
    data = pa.table({"value": [1, 2, 3]})
    binding = cf.SourceBinding.from_arrow(data, budget=cf.EdgeBudget(1, 1024))

    async def run():
        with pytest.raises(
            ValueError, match="source .*snapshot_sql.*ArrowSource slices relations"
        ):
            async with output.stream(binding):
                pass
        aggregate = source.sql("SELECT SUM(value) AS total FROM input")
        appended = cf.snapshot_sql("SELECT total FROM p", p=aggregate)
        async with appended.stream(binding) as results:
            return [table.to_pydict() async for table in results]

    assert asyncio.run(run()) == [{"total": [1]}, {"total": [3]}, {"total": [6]}]


@pytest.mark.parametrize("reloaded", [False, True])
def test_compiled_snapshot_plan_rejects_sliced_source_before_connectors_open(reloaded):
    class Sink:
        async def open(self):
            raise AssertionError("rejected binding must not open connectors")

        async def write(self, batch):
            raise AssertionError("rejected binding must not deliver data")

        async def close(self):
            pass

    output = cf.snapshot_sql("SELECT * FROM p", p=_source())
    program = _program(output)
    plan = (
        cf.Runtime()._compile_stream_graph_project(
            program.to_project().model_dump_json()
        )
        if reloaded
        else program.compile_stream()
    )
    binding = cf.SourceBinding.from_arrow(
        pa.table({"value": [1, 2]}), budget=cf.EdgeBudget(1, 1024)
    )
    with pytest.raises(ValueError, match="snapshot_sql.*ArrowSource slices relations"):
        cf.StreamingRunner(
            plan,
            {"input": binding},
            {"output": [cf.SinkBinding.ordinary("sink", Sink())]},
            config=cf.StreamRuntimeConfig(checkpointing=False),
        )


def test_shared_aggregate_and_named_snapshot_outputs_have_one_state_owner():
    aggregate = _source().sql("SELECT SUM(value) AS total FROM input")
    output = cf.snapshot_sql(
        "SELECT p.total + q.total AS doubled FROM p CROSS JOIN q",
        p=aggregate,
        q=aggregate,
    )
    program = cf.Program(
        "shared", engine="streaming", outputs={"a": output, "b": output}
    )
    nodes = program.to_project().root["graph"]["nodes"]
    assert sum(n["operator"]["kind"] == "sql" for n in nodes) == 1
    assert sum(n["operator"]["kind"] == "snapshot_sql" for n in nodes) == 1

    async def run():
        found = {"a": [], "b": []}
        async with program.stream({"values": pa.table({"value": [1, 2]})}) as results:
            async for event in results:
                found[event.name].append(event.table.to_pydict())
        return found

    assert asyncio.run(run()) == {"a": [{"doubled": [6]}], "b": [{"doubled": [6]}]}


def test_materialized_ordered_source_is_validated_once_and_released_after_close():
    data = pa.table(
        {"value": [1, 2], "ts": pa.array([1, 2], type=pa.timestamp("us", tz="UTC"))}
    )
    reference = weakref.ref(data)
    source = cf.table_input(
        "values", schema=data.schema, event_time="ts", sequence_by=("ts",)
    )
    output = cf.snapshot_sql("SELECT SUM(value) AS total FROM p", p=source)
    results = output.stream(data)
    del data

    async def run():
        async with results:
            assert (await results.__anext__()).to_pydict() == {"total": [3]}
        assert results.job.status()["task_count"] == 0

    asyncio.run(run())
    gc.collect()
    assert reference() is None


def test_cancellation_closes_snapshot_feeds_tasks_and_temporary_checkpoints(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(stream_module.tempfile, "tempdir", str(tmp_path))
    output = cf.snapshot_sql("SELECT * FROM p", p=_source())

    async def run():
        closed = asyncio.Event()
        reading = asyncio.Event()

        async def feed():
            try:
                yield pa.table({"value": [1]})
                reading.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

        before = asyncio.all_tasks()
        results = output.stream(
            feed(),
            config=cf.StreamRuntimeConfig(
                sql_snapshot_state_budget=cf.StateBudget(10, 1 << 20)
            ),
        )
        async with results:
            assert (await results.__anext__()).to_pydict() == {"value": [1]}
            await asyncio.wait_for(reading.wait(), 5)
            pending = asyncio.create_task(results.__anext__())
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert closed.is_set()
        assert results.job.status()["task_count"] == 0
        assert asyncio.all_tasks() == before

    asyncio.run(run())
    assert list(tmp_path.iterdir()) == []
