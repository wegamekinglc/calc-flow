from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pytest

import calc_flow as cf
from calc_flow import _native
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    parameter,
    rows,
    table_input,
    ts,
)

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
CONFIG = cf.StreamRuntimeConfig(checkpointing=False)
SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("seq", pa.uint64(), nullable=False),
        pa.field("x", pa.float64()),
    ]
)


def _quotes() -> cf.TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )


def _signals(quotes: cf.TableExpr | None = None) -> cf.TableExpr:
    quotes = _quotes() if quotes is None else quotes
    return quotes.with_columns(
        FeatureSet([("avg", ts.mean(quotes["x"], window=rows(2)))])
    )


def _at(seconds: int) -> datetime:
    return BASE + timedelta(seconds=seconds)


def _rows(*entries: tuple[int, str, int, float]) -> pa.Table:
    return pa.table(
        {
            "ts": pa.array(
                [_at(seconds) for seconds, *_ in entries],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "symbol": [symbol for _, symbol, _, _ in entries],
            "seq": pa.array([seq for _, _, seq, _ in entries], type=pa.uint64()),
            "x": pa.array([x for *_, x in entries], type=pa.float64()),
        },
        schema=SCHEMA,
    )


def _explicit_session() -> cf.StreamSession[pa.Table]:
    return _signals().session(config=CONFIG, watermarks=cf.SourceProvidedWatermarks())


def test_durable_session_checkpoint_restore(tmp_path) -> None:
    storage = cf.ManagedCheckpointRuntime(tmp_path / "managed")
    config = cf.StreamRuntimeConfig(checkpointing=True)
    with _signals().session(
        config=config,
        watermarks=cf.SourceProvidedWatermarks(),
        checkpoints=storage,
    ) as session:
        session.push(_rows((0, "a", 0, 1.0), (0, "b", 1, 2.0)), watermark=_at(0))
        assert session.checkpoint() == 1

    with _signals().session(
        config=config,
        watermarks=cf.SourceProvidedWatermarks(),
        checkpoints=cf.ManagedCheckpointRuntime(tmp_path / "managed"),
        restore=True,
    ) as restored:
        emitted = restored.push(_rows((1, "a", 2, 3.0)), watermark=_at(1))
        assert emitted["avg"][0].as_py() == 2.0
        assert restored.checkpoint() == 2


def test_checkpoint_rejects_running_loop(tmp_path) -> None:
    session = _signals().session(
        config=cf.StreamRuntimeConfig(checkpointing=True),
        checkpoints=cf.ManagedCheckpointRuntime(tmp_path / "managed"),
    )

    async def run() -> None:
        with pytest.raises(RuntimeError, match="cannot run inside an event loop"):
            session.checkpoint()

    asyncio.run(run())
    session.close()


def test_table_session_matches_stream_under_source_provided_watermarks() -> None:
    pushes = [
        (_rows((0, "a", 0, 1.0), (0, "b", 1, 2.0)), _at(0)),
        (_rows((1, "a", 2, 3.0)), _at(1)),
        (_rows((2, "b", 3, 4.0), (3, "a", 4, 5.0)), _at(2)),
    ]
    with _explicit_session() as session:
        emitted = [session.push(table, watermark=at) for table, at in pushes]
        emitted.append(session.finish())

    async def feed():
        for table, at in pushes:
            yield table
            yield cf.Watermark(at)

    async def collect() -> list[pa.Table]:
        async with _signals().stream(
            feed(), config=CONFIG, watermarks=cf.SourceProvidedWatermarks()
        ) as results:
            return [table async for table in results]

    expected = asyncio.run(collect())
    assert [table.num_rows for table in emitted] == [2, 1, 1, 1]
    assert [
        row
        for table in emitted
        for row in table.select(["symbol", "seq", "avg"]).to_pylist()
    ] == [
        row
        for table in expected
        for row in table.select(["symbol", "seq", "avg"]).to_pylist()
    ]


def test_indicator_session_matches_stream_under_source_provided_watermarks() -> None:
    quotes = _quotes()
    macd = ts.macd_components(quotes["x"], fast_span=1, slow_span=2, signal_span=1)
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("rsi", ts.rsi(quotes["x"], period=2)),
                ("histogram", macd.histogram),
            ]
        )
    )
    pushes = [
        (_rows((0, "a", 0, 1.0)), _at(0)),
        (_rows((1, "a", 1, 2.0)), _at(1)),
        (_rows((2, "a", 2, 4.0)), _at(2)),
    ]
    with signals.session(
        config=CONFIG, watermarks=cf.SourceProvidedWatermarks()
    ) as session:
        emitted = [session.push(table, watermark=at) for table, at in pushes]
        emitted.append(session.finish())

    async def feed():
        for table, at in pushes:
            yield table
            yield cf.Watermark(at)

    async def collect() -> list[pa.Table]:
        async with signals.stream(
            feed(), config=CONFIG, watermarks=cf.SourceProvidedWatermarks()
        ) as results:
            return [table async for table in results]

    expected = asyncio.run(collect())
    assert [
        row
        for table in emitted
        for row in table.select(["symbol", "seq", "rsi", "histogram"]).to_pylist()
    ] == [
        row
        for table in expected
        for row in table.select(["symbol", "seq", "rsi", "histogram"]).to_pylist()
    ]


def test_native_array_sequence_session_accepts_native_and_numpy_backed_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    import calc_flow.array as cfa

    runtime = cf.Runtime()
    cfa.register_native(runtime)
    plan = (
        cf.PipelineBuilder("native-sequence-session")
        .external(
            "sequence",
            "native",
            "array_sequence",
            "1",
            {
                "ir_version": 1,
                "input_dtype": "float64",
                "kernel": {"function": "rolling_mean", "window": 2},
            },
        )
        .compile_stream(runtime=runtime)
    )
    session = _native._StreamSession(plan._inner, CONFIG._native(), {})
    (input_id,) = session.input_ids
    (output_id,) = session.output_ids
    wrapped = cf.Batch.from_array(
        cfa.from_numpy(np.asarray([3.0], dtype=np.float64)), backend="native"
    )

    def unexpected_numpy_conversion(values: object) -> None:
        raise AssertionError(f"unexpected Python NumPy conversion: {values}")

    monkeypatch.setattr(cfa, "from_numpy", unexpected_numpy_conversion)
    for value, expected, wrap in [
        (1.0, 1.0, False),
        (3.0, 2.0, True),
        (5.0, 4.0, False),
    ]:
        values = np.asarray([value], dtype=np.float64)
        batch = wrapped if wrap else values
        emitted = session.push(input_id, batch)
        assert len(emitted[output_id]) == 1
        assert emitted[output_id][0].to_numpy().tolist() == [expected]
    reversed = np.asarray([3.0, 4.0])[::-1]
    emitted = session.push(input_id, reversed)
    assert emitted[output_id][0].to_numpy().tolist() == [4.5, 3.5]
    for invalid in (
        np.asarray([1.0], dtype=">f8"),
        np.asarray([1], dtype=np.int64),
        np.ones((1, 1, 1)),
        np.asarray([1.0]).view(type("ArraySubclass", (np.ndarray,), {})),
    ):
        with pytest.raises((cf.ConfigError, TypeError)):
            session.push(input_id, invalid)
    assert session.finish() == {}
    session.close()


def test_program_session_returns_every_declared_output() -> None:
    quotes = _quotes()
    program = Program(
        "outputs",
        engine="streaming",
        inputs=[quotes],
        outputs=[
            ("signals", _signals(quotes)),
            ("large", quotes.filter(quotes["x"] > 100.0)),
        ],
    )
    schemas = program.collect_schema()
    with program.session(config=CONFIG) as session:
        first = session.push({"quotes": _rows((0, "a", 0, 1.0))})
        second = session.push(
            {"quotes": _rows((1, "a", 1, 2.0))}, watermark={"quotes": _at(1)}
        )
    assert set(first) == set(second) == {"signals", "large"}
    for name, table in first.items():
        assert table.num_rows == 0
        assert table.schema.equals(schemas[name])
    assert second["signals"].column("avg").to_pylist() == [1.0, 1.5]
    assert second["large"].num_rows == 0
    assert second["large"].schema.equals(schemas["large"])


def test_table_output_session_requires_a_result_binding() -> None:
    program = Program("named", mode="stream", outputs={"signals": _signals()})
    with program.session(config=CONFIG) as named:
        table_session = cf.StreamSession(
            named._native, named._input, named._outputs, named._empty, table_output=True
        )
        with pytest.raises(KeyError) as error:
            table_session.finish()
        assert error.value.args == ("result",)


def test_program_session_preserves_mapping_lookup_and_input_errors() -> None:
    class CustomMapping(dict):
        def get(self, key, default=None):
            raise AssertionError("a custom mapping must retain indexed lookup")

    source = table_input("quotes", schema=SCHEMA)
    program = Program("input_errors", mode="stream", outputs={"out": source})
    table = _rows((0, "a", 0, 1.0))
    with program.session(config=CONFIG) as session:
        with pytest.raises(ValueError, match="quotes: missing table input"):
            session.push({})
        with pytest.raises(ValueError, match="bad: unexpected input name"):
            session.push({"extra": table, "bad": table})
        with pytest.raises(TypeError, match="session.inputs.quotes: expected"):
            session.push({"quotes": None})
        assert session.push(CustomMapping(quotes=table))["out"].equals(table)


def test_single_program_input_looks_up_its_declared_name_once() -> None:
    comparisons = 0

    class SuppliedName(str):
        __hash__ = str.__hash__

        def __eq__(self, other: object) -> bool:
            nonlocal comparisons
            comparisons += 1
            return super().__eq__(other)

    source = table_input("quotes", schema=SCHEMA)
    program = Program("one_input", mode="stream", outputs={"out": source})
    table = _rows((0, "a", 0, 1.0))
    supplied = {SuppliedName("quotes"): cf.Batch.from_pyarrow(table)}
    with program.session(config=CONFIG) as session:
        result = session.push(supplied)
    assert result["out"].equals(table)
    assert comparisons == 1


def test_single_program_output_uses_one_physical_name_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quotes = _quotes()
    program = Program("named", mode="stream", outputs={"signals": _signals(quotes)})
    hashes = 0

    class CountedName(str):
        def __hash__(self) -> int:
            nonlocal hashes
            hashes += 1
            return super().__hash__()

    with program.session(config=CONFIG) as session:
        ((logical, physical),) = session._outputs
        monkeypatch.setattr(session, "_outputs", ((logical, CountedName(physical)),))
        result = session.push({"quotes": _rows((0, "a", 0, 1.0))}, watermark=_at(0))
        assert list(result) == ["signals"]
        assert result["signals"].column("avg").to_pylist() == [1.0]
        assert hashes <= 1
        assert session.finish() is not result


@pytest.mark.parametrize("named", [False, True])
def test_single_session_output_keeps_emitted_empty_batch(named: bool) -> None:
    quotes = _quotes()
    expr = quotes.filter(quotes["x"] > 100.0)
    owner = (
        Program("empty", mode="stream", outputs={"filtered": expr}) if named else expr
    )
    data = _rows((0, "a", 0, 1.0))
    with owner.session(config=CONFIG) as session:
        waiting = session.advance(_at(-1))
        output = session.push({"quotes": data} if named else data)
        finished = session.finish()
    emitted = output["filtered"] if named else output
    absent = finished["filtered"] if named else finished
    assert emitted.num_rows == absent.num_rows == 0
    assert emitted.schema.equals(absent.schema, check_metadata=True)
    assert absent is (waiting["filtered"] if named else waiting)
    assert emitted is not absent


def test_default_ordered_watermark_keeps_latest_timestamp_open() -> None:
    with _signals().session(config=CONFIG) as session:
        assert session.push(_rows((0, "a", 0, 1.0))).num_rows == 0
        closed = session.push(_rows((1, "a", 1, 2.0)), watermark=_at(1))
        assert closed.column("avg").to_pylist() == [1.0, 1.5]
        with pytest.raises(cf.ConfigError, match="event_time"):
            session.push(_rows((1, "a", 2, 3.0)))
        assert session.push(_rows((2, "a", 2, 3.0))).num_rows == 0
        assert session.finish().column("avg").to_pylist() == [2.5]


def test_concurrent_push_raises_runtime_error_while_a_call_runs() -> None:
    entered = threading.Event()
    release = threading.Event()

    def slow(values: pa.Array) -> pa.Array:
        entered.set()
        release.wait(10)
        return values

    runtime = cf.Runtime()
    runtime.register_scalar_udf(
        provider="python",
        name="slow",
        version="1",
        input_types=("int64",),
        return_type="int64",
        volatility="immutable",
        function=slow,
    )
    plan = (
        cf.PipelineBuilder("slow")
        .expression("query", "result = slow(value)", udfs=(("python", "slow", "1"),))
        .compile_stream(runtime=runtime)
    )
    session = _native._StreamSession(plan._inner, CONFIG._native(), {})
    (input_id,) = session.input_ids
    (output_id,) = session.output_ids
    batch = pa.table({"value": pa.array([1, 2], type=pa.int64())})
    outcome: dict[str, object] = {}

    def first_push() -> None:
        outcome["first"] = session.push(input_id, batch)

    worker = threading.Thread(target=first_push)
    worker.start()
    try:
        assert entered.wait(10)
        with pytest.raises(RuntimeError, match="busy"):
            session.push(input_id, batch)
    finally:
        release.set()
        worker.join(10)
    first = outcome["first"]
    assert isinstance(first, dict)
    assert [record.num_rows for record in first[output_id]] == [2]
    assert session.push(input_id, batch)[output_id][0].num_rows == 2
    session.close()


def test_close_is_idempotent_and_context_manager_does_not_flush() -> None:
    session = _explicit_session()
    session.close()
    session.close()
    for call in (
        lambda: session.push(_rows((0, "a", 0, 1.0))),
        lambda: session.advance(_at(5)),
        session.finish,
    ):
        with pytest.raises(RuntimeError, match="closed"):
            call()
    with _explicit_session() as session:
        assert session.push(_rows((0, "a", 0, 1.0))).num_rows == 0
    with pytest.raises(RuntimeError, match="closed"):
        session.finish()


def test_exact_session_input_uses_arrow_schema_comparison() -> None:
    from unittest.mock import MagicMock

    from calc_flow import session as session_module

    batch = pa.record_batch({"value": [7]})
    schema = MagicMock(wraps=batch.schema)
    schema.metadata = None
    arrow_input = MagicMock(spec=pa.RecordBatch)
    arrow_input.schema = schema
    schema.__iter__.side_effect = AssertionError("exact schema iterated Python fields")
    result = session_module._session_batch(
        arrow_input, batch.schema, "session.inputs.events"
    )
    assert result is arrow_input
    schema.equals.assert_called_once_with(batch.schema, check_metadata=True)


@pytest.mark.parametrize(
    ("as_table", "schema_metadata"), [(False, None), (True, {b"origin": b"caller"})]
)
def test_session_normalizes_input_and_preserves_caller_metadata(
    as_table: bool, schema_metadata: dict[bytes, bytes] | None
) -> None:
    source = _rows((0, "a", 0, 1.0))
    fields = [field.with_metadata({b"unit": b"sample"}) for field in source.schema]
    schema = pa.schema(fields, metadata=schema_metadata)
    record = pa.RecordBatch.from_arrays(
        [col.combine_chunks() for col in source.columns], schema=schema
    )
    data = pa.Table.from_batches([record]) if as_table else record
    with _explicit_session() as session:
        output = session.push(data, watermark=_at(0))
    assert output.column("avg").to_pylist() == [1.0]
    assert data.schema.equals(schema, check_metadata=True)


def test_schema_mismatch_raises_value_error_and_session_stays_usable() -> None:
    wrong = pa.table({"x": pa.array([1.0], type=pa.float64())})
    with _explicit_session() as session:
        with pytest.raises(ValueError, match=r"session\.inputs\.quotes\.schema"):
            session.push(wrong)
        with pytest.raises(ValueError, match=r"session\.inputs\.quotes\.schema"):
            session.push(cf.Batch.from_pyarrow(wrong))
        prepared = cf.Batch.from_pyarrow(_rows((0, "a", 0, 1.0)))
        emitted = session.push(prepared, watermark=_at(0))
        assert emitted.column("avg").to_pylist() == [1.0]


def test_push_runs_inside_event_loop_but_construction_does_not() -> None:
    session = _explicit_session()

    async def run() -> pa.Table:
        with pytest.raises(RuntimeError, match="event loop"):
            _explicit_session()
        return session.push(_rows((0, "a", 0, 1.0)), watermark=_at(0))

    try:
        assert asyncio.run(run()).column("avg").to_pylist() == [1.0]
    finally:
        session.close()


def test_static_parameters_and_idle_timeouts_are_rejected() -> None:
    quotes = _quotes()
    reference = parameter(
        "reference",
        kind="table",
        schema=[Field("symbol", "string", nullable=False)],
    )
    program = Program(
        "static",
        engine="streaming",
        inputs=[quotes, reference],
        outputs=[("signals", _signals(quotes))],
    )
    with pytest.raises(ValueError, match=r"session\.inputs\.reference: static"):
        program.session(config=CONFIG)
    for policy in (
        cf.BoundedOutOfOrderness(
            "ts",
            timedelta(seconds=1),
            timedelta(seconds=1),
            idle_timeout=timedelta(seconds=5),
        ),
        cf.DisabledWatermarks(idle_timeout=timedelta(seconds=5)),
    ):
        with pytest.raises(ValueError, match="idle_timeout"):
            _signals().session(config=CONFIG, watermarks=policy)


def test_operator_failure_raises_execution_error_then_fails_session() -> None:
    with _explicit_session() as session:
        session.push(_rows((0, "a", 0, 1.0)), watermark=_at(0))
        with pytest.raises(cf.ExecutionError, match="late"):
            session.push(_rows((0, "a", 1, 2.0)))
        for call in (
            lambda: session.push(_rows((1, "a", 2, 3.0))),
            lambda: session.advance(_at(5)),
            session.finish,
        ):
            # The binding's failed state names the original failure.
            with pytest.raises(RuntimeError, match=r"failed earlier \(.*late"):
                call()


def test_session_returns_arrow_output_without_generic_record_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import Mock

    data = cf.Batch.from_pyarrow(_rows((0, "a", 0, 1.0)))
    factory = Mock(wraps=pa.record_batch)
    with _explicit_session() as session:
        monkeypatch.setattr(pa, "record_batch", factory)
        emitted = session.push(data, watermark=_at(0))
        assert emitted["avg"].to_pylist() == [1.0]
        assert session.finish().num_rows == 0
    assert emitted["avg"].to_pylist() == [1.0]
    factory.assert_not_called()


def test_empty_program_results_copy_declared_tables_without_physical_lookups() -> None:
    class EmptyEmissions(dict):
        def get(self, *args):
            raise AssertionError("looked up an output in an empty native emission")

    quotes = _quotes()
    outputs = [("later", _signals(quotes)), ("first", quotes)]
    program = Program("empty", mode="stream", inputs=[quotes], outputs=outputs)
    schemas = program.collect_schema()
    with program.session(
        config=CONFIG, watermarks={"quotes": cf.SourceProvidedWatermarks()}
    ) as session:
        first = session.advance({"quotes": _at(0)})
        second = session._result(EmptyEmissions())
        assert list(first) == [name for name, _ in session._outputs]
        assert first is not second
        for name, table in second.items():
            assert table is first[name]
            assert table.num_rows == 0
            assert table.schema.equals(schemas[name], check_metadata=True)
        first.clear()
        assert list(second) == [name for name, _ in session._outputs]

        reordered = dict(reversed(session._empty.items()))
        reordered["extra"] = second["first"]
        manual = cf.StreamSession(
            session._native,
            session._input,
            session._outputs,
            reordered,
            table_output=False,
        )
        assert list(manual._result({})) == list(second)
