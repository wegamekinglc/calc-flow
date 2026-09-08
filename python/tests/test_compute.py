from __future__ import annotations

import pyarrow as pa
import pytest

import calc_flow as cf


def test_compute_quickstart():
    data = pa.table({"a": [1, 3], "b": [2, 4]})
    result = cf.compute(data, lambda t: t.select(total=t["a"] + t["b"]))
    assert result.to_pydict() == {"total": [3, 7]}


@pytest.mark.parametrize("entry", [cf.compute, cf.compute_async])
@pytest.mark.parametrize(
    "ordering",
    [{"entity_by": ("symbol",)}, {"event_time": "ts"}, {"sequence_by": ("ts",)}],
)
def test_compute_rejects_declaration_ordering_before_building(entry, ordering):
    data, _ = _rolling_program()

    def build(t):
        pytest.fail("unexpected ordering keyword reached the expression builder")

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        entry(data, build, **ordering)


def test_collection_binds_logical_names_across_distinct_inputs_and_shared_outputs():
    left_data = pa.table({"x": [2, 3]})
    right_data = pa.table({"x": [20, 30]})
    left = cf.table_input("left_input", schema=left_data.schema)
    right = cf.table_input("right", schema=right_data.schema)
    shared = (left["x"] + 1) * (left["x"] + 1)
    program = cf.Program(
        "p",
        outputs={
            "right-output": right.select("x"),
            "first": left.select(v=shared + 1),
            "second": left.select(v=shared + 2),
        },
    )
    result = program.collect({"right": right_data, "left_input": left_data})
    assert list(result) == ["right-output", "first", "second"]
    assert {name: table.to_pydict() for name, table in result.items()} == {
        "right-output": {"x": [20, 30]},
        "first": {"v": [10, 17]},
        "second": {"v": [11, 18]},
    }
    assert left.select("x").collect(left_data).equals(left_data)


def test_async_collection_snapshots_mapping_and_forwards_options(monkeypatch):
    import asyncio
    from datetime import UTC, datetime, timedelta

    data = pa.table({"x": [3]})
    t = cf.table_input("quotes", schema=data.schema)
    p = cf.Program("p", outputs={"answer": t.select(v=t["x"] + 1)})
    inputs = {"quotes": data}
    options = cf.ExecutionOptions(
        settings={"label": "public"}, deadline=datetime.now(UTC) + timedelta(minutes=1)
    )
    seen = []
    execute = cf.BatchExecutionPlan.execute_async

    def record(self, inputs, *, options=None):
        seen.append(options)
        return execute(self, inputs, options=options)

    monkeypatch.setattr(cf.BatchExecutionPlan, "execute_async", record)
    pending = p.collect_async(inputs, options=options)
    inputs.clear()

    async def run():
        assert (await pending)["answer"].to_pydict() == {"v": [4]}
        assert (
            await cf.compute_async(data, lambda q: q.select(v=q["x"] + 1))
        ).to_pydict() == {"v": [4]}
        assert (await t.select("x").collect_async(data)).equals(data)

    asyncio.run(run())
    assert seen[0] is options


def test_compute_rejects_invalid_builders_and_preserves_original_exception():
    import pytest

    data = pa.table({"x": [1]})
    with pytest.raises(TypeError, match=r"compute.build.*expected TableExpr.*int"):
        cf.compute(data, lambda t: 1)

    async def async_build(t):
        return t

    with pytest.raises(TypeError, match=r"compute.build.*TableExpr"):
        cf.compute(data, async_build)
    error = LookupError("builder failed")

    def broken(t):
        raise error

    with pytest.raises(LookupError) as caught:
        cf.compute(data, broken)
    assert caught.value is error


def test_blocking_convenience_rejects_loop_before_builder():
    import asyncio

    import pytest

    calls = []
    data = pa.table({"x": [1]})

    async def run():
        with pytest.raises(RuntimeError, match=r"compute_async"):
            cf.compute(data, lambda t: calls.append(t))
        t = cf.table_input("q", schema=data.schema)
        with pytest.raises(RuntimeError, match=r"collect_async"):
            t.collect(data)
        with pytest.raises(RuntimeError, match=r"collect_async"):
            cf.Program("p", outputs={"o": t}).collect({"q": data})
        assert calls == []

    asyncio.run(run())


def test_orders_literals_and_input_immutability():
    data = pa.table(
        {"order_id": ["A", "B", "C"], "quantity": [3, 1, 4], "unit_price": [10, 12, 10]}
    )
    before = data.to_pydict()
    calls = []

    def build(t):
        calls.append(t)
        gross = t["quantity"] * t["unit_price"]
        enriched = t.with_columns(gross=gross, fee=cf.row.cast(gross, "float64") / 10.0)
        return enriched.filter(
            (enriched["gross"] >= 20) & ~(enriched["quantity"] == 1)
        ).select("order_id", "gross", label=cf.lit("x'; DROP TABLE t; --"))

    result = cf.compute(data, build)
    assert result.to_pydict() == {
        "order_id": ["A", "C"],
        "gross": [30, 40],
        "label": ["x'; DROP TABLE t; --"] * 2,
    }
    assert len(calls) == 1 and isinstance(calls[0], cf.TableExpr)
    assert data.to_pydict() == before
    assert cf.compute(data.to_batches()[0], lambda t: t.select("quantity"))[
        "quantity"
    ].to_pylist() == [3, 1, 4]


def test_collect_checks_names_kinds_lineage_and_ordering():
    import numpy as np
    import pytest

    data = pa.table({"x": [1]})
    t = cf.table_input("q", schema=data.schema)
    p = cf.Program("p", outputs={"o": t})
    with pytest.raises(ValueError, match=r"inputs.q.*missing table input"):
        p.collect({})
    with pytest.raises(ValueError, match=r"inputs.typo.*unexpected input"):
        p.collect({"q": data, "typo": data})
    with pytest.raises(TypeError, match=r"inputs.q.*table input"):
        p.collect({"q": cf.Batch.from_array(np.array([1]), backend="numpy")})
    with pytest.raises(TypeError, match="Arrow Table"):
        cf.compute({"x": [1]}, lambda q: q)
    with pytest.raises(cf.CompileError, match="duplicate|collision"):
        t.with_columns(x=t["x"] + 1).collect(data)
    other = cf.table_input("other", schema=data.schema)
    with pytest.raises(cf.CompileError, match="lineage|schema_mismatch"):
        t.select(v=t["x"] + other["x"]).collect({"q": data, "other": data})
    with pytest.raises(cf.CompileError, match="ordering|sequence|event_time"):
        t.select(v=cf.ts.lag(t["x"])).collect(data)
    with pytest.raises(TypeError, match="&|identical"):
        bool(t["x"] > 0)


def _rolling_program():
    data = pa.table(
        {
            "symbol": ["a", "a", "a"],
            "ts": pa.array([1, 2, 3], type=pa.timestamp("us", tz="UTC")),
            "price": [2.0, 4.0, 8.0],
        },
        schema=pa.schema(
            [
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("price", pa.float64()),
            ]
        ),
    )
    t = cf.table_input(
        "quotes",
        schema=data.schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )
    output = t.select("price", previous=cf.ts.lag(t["price"]))
    return data, cf.Program("rolling", outputs={"signals": output})


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("temporal", [False, True], ids=["compute", "ordered-collect"])
def test_convenience_forwards_runtime_and_options(asynchronous, temporal, monkeypatch):
    import asyncio
    from datetime import UTC, datetime, timedelta

    data, program = _rolling_program()
    expression = program.outputs[0][1]
    runtime = cf.Runtime()
    options = cf.ExecutionOptions(deadline=datetime.now(UTC) + timedelta(minutes=1))
    runtimes = []
    executions = []
    compile_project = cf.Runtime.compile_batch_project
    execute_name = "execute_async" if asynchronous else "execute"
    execute = getattr(cf.BatchExecutionPlan, execute_name)

    def record_compile(self, document):
        runtimes.append(self)
        return compile_project(self, document)

    def record_execute(self, inputs, *, options=None):
        executions.append(options)
        return execute(self, inputs, options=options)

    monkeypatch.setattr(cf.Runtime, "compile_batch_project", record_compile)
    monkeypatch.setattr(cf.BatchExecutionPlan, execute_name, record_execute)

    def collect():
        if temporal:
            entry = expression.collect_async if asynchronous else expression.collect
            return entry(data, runtime=runtime, options=options)
        entry = cf.compute_async if asynchronous else cf.compute
        return entry(
            data, lambda t: t.select("price"), runtime=runtime, options=options
        )

    async def run():
        return await collect()

    result = asyncio.run(run()) if asynchronous else collect()
    assert result["price"].to_pylist() == [2.0, 4.0, 8.0]
    if temporal:
        assert result["previous"].to_pylist() == [None, 2.0, 4.0]
    assert len(runtimes) == 1 and runtimes[0] is runtime
    assert len(executions) == 1 and executions[0] is options


def test_collect_owns_fresh_state_and_does_not_reset_cached_plan():
    import asyncio

    data, program = _rolling_program()
    runtime = cf.Runtime()
    cached = program.compile_batch(runtime)
    cached.execute({"input": cf.Batch.from_pyarrow(data)})
    snapshot = cached.snapshot()
    for _ in range(2):
        assert program.collect({"quotes": data}, runtime=runtime)["signals"][
            "previous"
        ].to_pylist() == [None, 2.0, 4.0]
    assert cached.snapshot() == snapshot
    assert program.compile_batch(runtime) is cached

    async def run():
        first, second = await asyncio.gather(
            program.collect_async({"quotes": data}, runtime=runtime),
            program.collect_async({"quotes": data}, runtime=runtime),
        )
        assert first["signals"].equals(second["signals"])
        assert first["signals"]["previous"].to_pylist() == [None, 2.0, 4.0]

    asyncio.run(run())
    assert cached.snapshot() == snapshot


def test_matrix_collect_discovers_and_binds_static_parameter():
    import numpy as np
    import pytest

    data = pa.table({"x": [1.0, 2.0], "y": [10.0, 20.0]})
    t = cf.table_input("quotes", schema=data.schema)
    weights = cf.parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(2, 1)
    )
    matrix = cf.linalg.from_columns(t, columns=("y", "x"), backend="numpy")
    output = cf.table.attach_columns(
        t, cf.linalg.matmul(matrix * 2.0 + 1.0, weights), names=("score",)
    )
    program = cf.Program("matrix", outputs={"scores": output})
    runtime = cf.Runtime()
    cf.register_numpy(runtime)
    array = np.array([[1.0], [10.0]])
    inputs = {"quotes": data, "weights": cf.Batch.from_array(array, backend="numpy")}
    result = program.collect(inputs, runtime=runtime)
    assert result["scores"]["score"].to_pylist() == [51.0, 91.0]
    assert array.tolist() == [[1.0], [10.0]]
    with pytest.raises(TypeError, match=r"inputs.weights.*array Batch"):
        program.collect({"quotes": data, "weights": data}, runtime=runtime)
    with pytest.raises(ValueError, match="mapping"):
        output.collect(data, runtime=runtime)


def test_unsupported_schema_names_preserve_native_validation():
    import pytest

    data = pa.table({'order "id"': [1]})
    with pytest.raises(cf.ConfigError, match="portable SQL identifier"):
        cf.compute(data, lambda t: t.select('order "id"'))


def test_async_cancellation_before_start_does_not_leak_coroutines():
    import asyncio
    import gc
    import warnings
    from contextlib import suppress

    data = pa.table({"x": [1]})
    t = cf.table_input("q", schema=data.schema)

    async def run():
        for pending in (cf.compute_async(data, lambda t: t), t.collect_async(data)):
            task = asyncio.create_task(pending)
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(run())
        gc.collect()
    assert not [str(w.message) for w in caught if "was never awaited" in str(w.message)]


def test_async_cancellation_drains_native_bridge(monkeypatch):
    import asyncio

    import pytest

    async def run():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        cleaned = asyncio.Event()

        class Cancellation:
            def cancel(self):
                cancelled.set()

        class NativePlan:
            def _execute_async_cancellable(self, inputs, *, options):
                async def work():
                    started.set()
                    await cancelled.wait()
                    cleaned.set()
                    raise cf.CancelledError("cancelled")

                return asyncio.create_task(work()), Cancellation()

        monkeypatch.setattr(
            cf.Runtime,
            "compile_batch_project",
            lambda self, document: cf.BatchExecutionPlan(NativePlan()),
        )
        task = asyncio.create_task(cf.compute_async(pa.table({"x": [1]}), lambda t: t))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()

    asyncio.run(run())


def test_compute_preserves_batch_identity_metadata_and_sync_options(monkeypatch):
    from datetime import UTC, datetime, timedelta

    batch = cf.Batch.from_pyarrow(
        pa.table({"x": [1.0]}), metadata={"source": "quotes", "sequence": 7}
    )
    options = cf.ExecutionOptions(deadline=datetime.now(UTC) + timedelta(minutes=1))
    captured = []
    execute = cf.BatchExecutionPlan.execute

    def record(self, inputs, *, options=None):
        captured.append((inputs["input"], options))
        return execute(self, inputs, options=options)

    monkeypatch.setattr(cf.BatchExecutionPlan, "execute", record)
    result = cf.compute(
        batch,
        lambda t: t.select(v=10.0 - t["x"], ratio=2.0 / t["x"], neg=-t["x"]),
        options=options,
    )
    assert result.to_pydict() == {"v": [9.0], "ratio": [2.0], "neg": [-1.0]}
    assert captured[0][0] is batch
    assert captured[0][1] is options
    assert batch.metadata["sequence"] == 7


def test_compute_expired_deadline_uses_native_cancellation():
    from datetime import UTC, datetime, timedelta

    import pytest

    options = cf.ExecutionOptions(deadline=datetime.now(UTC) - timedelta(seconds=1))
    with pytest.raises(cf.CancelledError):
        cf.compute(pa.table({"x": [1]}), lambda t: t, options=options)


def test_collect_rejects_unconsumed_explicit_input():
    import pytest

    data = pa.table({"x": [1]})
    t = cf.table_input("q", schema=data.schema)
    unused = cf.table_input("unused", schema=data.schema)
    program = cf.Program("p", inputs=(t, unused), outputs={"answer": t})
    with pytest.raises(cf.CompileError, match=r"inputs.unused.*unconsumed"):
        program.collect({"q": data, "unused": data})
