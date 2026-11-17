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
        assert emitted.to_pylist()[0]["avg"] == 2.0
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
    assert [row for table in emitted for row in table.to_pylist()] == [
        row for table in expected for row in table.to_pylist()
    ]


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
