import asyncio
from functools import wraps

import pyarrow as pa
import pytest

import calc_flow as cf


def _sync(test):
    @wraps(test)
    def run(*args, **kwargs):
        return asyncio.run(test(*args, **kwargs))

    return run


@pytest.mark.parametrize("syntax", ["df", "sql"])
@_sync
async def test_stream_two_real_data_batches_finalize_once_and_legacy_stays_per_batch(
    syntax,
):
    events = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
    out = (
        events.distinct(
            ["x"], spec=cf.DistinctSpec.bounded_source(cf.StateBudget(2, 8 << 20))
        )
        if syntax == "df"
        else cf.sql(
            "SELECT DISTINCT x FROM CF_DISTINCT_BOUNDED('a', 2, 8388608)", a=events
        )
    )
    legacy = cf.sql("SELECT DISTINCT x FROM a", a=events)
    program = cf.Program(
        "two_events",
        engine="streaming",
        inputs=[events],
        outputs={"out": out, "legacy": legacy},
    )
    source = cf.SourceBinding.from_arrow(
        pa.table({"x": [1, 1]}), budget=cf.EdgeBudget(max_rows=1, max_bytes=1024)
    )
    seen = []
    async with program.stream(
        {"events": source}, config=cf.StreamRuntimeConfig(checkpointing=False)
    ) as results:
        async for event in results:
            seen.append((event.name, event.table.to_pydict()))
    assert seen.count(("out", {"x": [1]})) == 1
    assert seen.count(("legacy", {"x": [1]})) == 2


@_sync
async def test_unknown_iterator_rejected_before_iterator_acquisition_and_temp_root(
    monkeypatch,
):
    import importlib

    stream_module = importlib.import_module("calc_flow.stream")
    roots = []
    original = stream_module.tempfile.mkdtemp

    def traced_root(*args, **kwargs):
        roots.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(stream_module.tempfile, "mkdtemp", traced_root)
    touched = []

    class Unknown:
        def __aiter__(self):
            touched.append("iterator")
            raise AssertionError("iterator was acquired")

    events = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
    out = events.distinct(
        ["x"], spec=cf.DistinctSpec.bounded_source(cf.StateBudget(2, 8 << 20))
    )
    program = cf.Program(
        "early_gate", engine="streaming", inputs=[events], outputs={"out": out}
    )
    with pytest.raises(
        cf.StreamingRuntimeError, match="source_boundedness_unproved"
    ) as rejected:
        async with program.stream({"events": Unknown()}):
            pass
    assert "stream.inputs.events.boundedness" in str(rejected.value)
    assert rejected.value.component_id == "events"
    assert touched == []
    assert roots == []


@pytest.mark.parametrize("syntax", ["df", "sql"])
def test_finite_distinct_v2_literal_and_export(syntax):
    events = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
    spec = cf.DistinctSpec.bounded_source(cf.StateBudget(5, 8 << 20))
    out = (
        events.distinct(["x"], spec=spec)
        if syntax == "df"
        else cf.sql(
            "SELECT DISTINCT x FROM CF_DISTINCT_BOUNDED('a', 5, 8388608)", a=events
        )
    )
    program = cf.Program("unique", engine="sql", inputs=[events], outputs={"out": out})
    runtime = cf.Runtime()
    plan = program.compile_batch(runtime)
    assert all(value is None for value in plan.snapshot().values())
    for values, expected in [
        ([1, None, 1, None, 2], {None, 1, 2}),
        ([7], {7}),
        ([], set()),
    ]:
        result = program.collect(
            {"events": pa.table({"x": pa.array(values, pa.int64())})}, runtime=runtime
        )
        assert set(result["out"]["x"].to_pylist()) == expected
    document = program.to_project(runtime).model_dump()
    operator = next(
        n["operator"]
        for n in document["graph"]["nodes"]
        if n["operator"]["kind"] == "distinct"
    )
    assert operator["spec"]["ingress_policy"] == "copy_visible_v2"


@pytest.mark.parametrize("syntax", ["df", "sql"])
@_sync
async def test_window_literal_before_projection_and_real_watermark(syntax):
    from datetime import datetime, timedelta, timezone

    schema = pa.schema([pa.field("ts", pa.timestamp("us")), pa.field("q", pa.int64())])
    events = cf.table_input("events", schema=schema)
    spec = cf.DistinctSpec.closed_window(
        "ts", start_micros=0, end_micros=10, state_budget=cf.StateBudget(3, 8 << 20)
    )
    out = (
        events.distinct(["q"], spec=spec)
        if syntax == "df"
        else cf.sql(
            "SELECT DISTINCT q FROM CF_DISTINCT_WINDOW('a','ts',0,10,3,8388608)",
            a=events,
        )
    )
    program = cf.Program(
        "window", engine="streaming", inputs=[events], outputs={"out": out}
    )

    def batch(times, values):
        return cf.Batch.from_pyarrow(
            pa.Table.from_arrays(
                [pa.array(times, pa.timestamp("us")), pa.array(values, pa.int64())],
                schema=schema,
            )
        )

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    records = [
        cf.Data(batch([-1, 0, 9, 10, None], [9, 1, 1, 2, 3]), cf.Cursor(b"1", {})),
        cf.Watermark(epoch + timedelta(microseconds=9)),
        cf.Data(batch([0], [7]), cf.Cursor(b"2", {})),
        cf.Watermark(epoch + timedelta(microseconds=10)),
    ]

    class Source:
        def __init__(self):
            self.calls = 0
            self.records = iter(records)

        def capabilities(self):
            self.calls += 1
            return cf.SourceCapabilities(
                cf.ReplayPositioning.UNSUPPORTED,
                cf.SourceDeliveryCapability.LOSSLESS,
                5,
                4096,
                schema,
                cf.NativeWatermarkCapability.EMITS_NATIVE,
                cf.SourceBoundedness.UNBOUNDED,
            )

        async def open(self, cursor):
            pass

        async def next(self):
            return next(self.records, None)

        async def close(self):
            pass

    source = Source()
    result = []
    async with program.stream(
        {"events": cf.SourceBinding(source)},
        config=cf.StreamRuntimeConfig(checkpointing=False),
    ) as outputs:
        async for event in outputs:
            assert event.table.schema == pa.schema([pa.field("q", pa.int64())])
            result.append(set(event.table["q"].to_pylist()))
    assert result == [{1, 7}]
    assert source.calls == 1


def test_mixed_dag_shared_owner_schema_export_and_r9_analysis():
    events = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
    before = cf.sql("SELECT x AS y FROM a", a=events)
    unique = before.distinct(
        ["y"], spec=cf.DistinctSpec.bounded_source(cf.StateBudget(4, 8 << 20))
    )
    after = cf.sql("SELECT y AS z FROM a", a=unique)
    program = cf.Program(
        "mixed",
        mode="batch",
        outputs={"unique": unique, "after": after, "unrelated": events},
    )
    assert after.collect_schema() == pa.schema([pa.field("z", pa.int64())])
    report = program.analyze(cf.Runtime()).query
    assert (report.schema_version, report.analysis_revision) == (2, 9)
    document = program.to_project().model_dump()
    assert (
        len(
            [
                n
                for n in document["graph"]["nodes"]
                if n["operator"]["kind"] == "distinct"
            ]
        )
        == 1
    )
    result = program.collect({"events": pa.table({"x": [1, None, 1, 2]})})
    assert set(result["unique"]["y"].to_pylist()) == {1, None, 2}
    assert set(result["after"]["z"].to_pylist()) == {1, None, 2}
    assert result["unrelated"]["x"].to_pylist() == [1, None, 1, 2]


def test_distinct_expression_explain_states_owned_peak_and_borrowed_ingress():
    events = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
    unique = events.distinct(
        ["x"], spec=cf.DistinctSpec.bounded_source(cf.StateBudget(2, 8 << 20))
    )
    text = unique.explain()
    assert "copy_visible_v2" in text
    assert "owned participating peak" in text
    assert "borrowed ingress" in text


def test_distinct_spec_and_keys_are_immutable_and_validate_literals():
    from dataclasses import FrozenInstanceError

    budget = cf.StateBudget(2, 8 << 20)
    for start, end in [(True, 10), (0, False), (10, 0), (0, 1 << 63)]:
        with pytest.raises(ValueError):
            cf.DistinctSpec.closed_window(
                "ts", start_micros=start, end_micros=end, state_budget=budget
            )
    with pytest.raises(TypeError):
        cf.DistinctSpec()
    spec = cf.DistinctSpec.bounded_source(budget)
    with pytest.raises(FrozenInstanceError):
        spec._wire = "changed"
    events = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
    keys = ["x"]
    value = events.distinct(keys, spec=spec)
    identity = value.digest
    keys.append("changed")
    assert value.digest == identity
    for invalid in [[], ["x", "x"], [1], "x", b"x"]:
        with pytest.raises((TypeError, ValueError)):
            events.distinct(invalid, spec=spec)


def test_native_distinct_schema_diagnostic_keeps_literal_key_path_and_code():
    events = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
    output = cf.sql(
        "SELECT DISTINCT missing FROM CF_DISTINCT_BOUNDED('a',2,8388608)", a=events
    )
    report = cf.Program("schema_error", mode="batch", outputs={"out": output}).analyze()
    assert any(
        issue.path == "outputs.out.sql.distinct.columns[0]"
        and issue.code == "unresolved_type"
        for issue in report.issues
    )


def test_global_drop_without_temporal_consumer_is_rejected_before_export():
    events = cf.table_input(
        "events",
        schema=pa.schema(
            [pa.field("ts", pa.timestamp("us")), pa.field("q", pa.int64())]
        ),
    )
    selected = events.distinct(
        ["q"],
        spec=cf.DistinctSpec.closed_window(
            "ts", start_micros=0, end_micros=10, state_budget=cf.StateBudget(2, 8 << 20)
        ),
    )
    program = cf.Program("late_policy", mode="stream", outputs={"out": selected})
    with pytest.raises(cf.CompileError, match="late_policy"):
        program.to_project(late_policy="drop")


@_sync
async def test_stream_budget_error_exposes_only_owned_policy_and_numeric_counters():
    events = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
    out = events.distinct(
        ["x"], spec=cf.DistinctSpec.bounded_source(cf.StateBudget(1, 8 << 20))
    )
    program = cf.Program("budget_error", mode="stream", outputs={"out": out})
    source = cf.SourceBinding.from_arrow(
        pa.table({"x": [1, 1]}), budget=cf.EdgeBudget(1, 1024)
    )
    with pytest.raises(cf.StreamingRuntimeError) as rejected:
        async with program.stream(
            {"events": source}, config=cf.StreamRuntimeConfig(checkpointing=False)
        ) as outputs:
            async for _ in outputs:
                pytest.fail("budget failure must not emit a final relation")
    for text in [
        "state_budget_exceeded",
        "copy_visible_v2",
        "history",
        "required=2",
        "limit=1",
    ]:
        assert text in rejected.value.message
    assert rejected.value.component_kind == "operator"
    assert rejected.value.reason_code == "state_budget_exceeded"
    assert rejected.value.component_id.startswith("cf_distinct_")


@pytest.mark.parametrize(
    "query",
    [
        "SELECT x FROM CF_DISTINCT_BOUNDED('a',2,8388608)",
        "SELECT DISTINCT x AS q FROM CF_DISTINCT_BOUNDED('a',2,8388608)",
        "SELECT DISTINCT x+1 FROM CF_DISTINCT_BOUNDED('a',2,8388608)",
        "SELECT DISTINCT x FROM CF_DISTINCT_BOUNDED('a',2,8388608) WHERE x>0",
        "SELECT DISTINCT x FROM CF_DISTINCT_BOUNDED('a',2,8388608) LIMIT 1",
        "SELECT DISTINCT x FROM CF_DISTINCT_UNKNOWN('a',2,8388608)",
    ],
)
def test_distinct_macro_malformed_forms_never_fall_back_to_per_batch(query):
    events = cf.table_input("events", schema=pa.schema([pa.field("x", pa.int64())]))
    program = cf.Program(
        "malformed", mode="batch", outputs={"out": cf.sql(query, a=events)}
    )
    with pytest.raises(cf.CompileError):
        program.compile_batch()


class _ReplaySource:
    def __init__(self, hold, schema, opened):
        self.schema = schema
        self.opened = opened
        self.hold = hold
        self.position = 0
        self.ready = asyncio.Event()

    def capabilities(self):
        return cf.SourceCapabilities(
            cf.ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            cf.SourceDeliveryCapability.LOSSLESS,
            1,
            1024,
            self.schema,
            cf.NativeWatermarkCapability.NEVER_EMITS,
            cf.SourceBoundedness.BOUNDED,
        )

    async def open(self, cursor):
        self.position = 0 if cursor is None else cursor.payload["position"]
        self.opened.append(self.position)

    async def next(self):
        if self.hold and self.position == 1:
            self.ready.set()
            await asyncio.sleep(0.001)
            return cf.Idle()
        if self.position == 2:
            return None
        self.position += 1
        return cf.Data(
            cf.Batch.from_pyarrow(pa.table({"x": [1]})),
            cf.Cursor(bytes([self.position]), {"position": self.position}),
        )

    async def close(self):
        pass


@_sync
async def test_durable_checkpoint_continues_duplicate_history(tmp_path):
    schema = pa.schema([pa.field("x", pa.int64())])
    events = cf.table_input("events", schema=schema)
    unique = events.distinct(
        ["x"], spec=cf.DistinctSpec.bounded_source(cf.StateBudget(2, 8 << 20))
    )
    program = cf.Program("durable_unique", mode="stream", outputs={"out": unique})
    runtime = cf.Runtime()
    written = []
    opened = []

    class Sink:
        async def open(self):
            pass

        async def write(self, batch):
            written.append(batch.to_pyarrow().to_pydict())

        async def close(self):
            pass

    async def start(source):
        plan = await asyncio.to_thread(program.compile_stream, runtime)
        return await cf.StreamingRunner(
            plan,
            {
                plan.source_binding_ids[0]: cf.SourceBinding(
                    source, watermark_policy=cf.DisabledWatermarks()
                )
            },
            {plan.sink_binding_ids[0]: [cf.SinkBinding.ordinary("archive", Sink())]},
            cf.ManagedCheckpointRuntime(tmp_path),
        ).start_async()

    first_source = _ReplaySource(True, schema, opened)
    first = await start(first_source)
    try:
        await asyncio.wait_for(first_source.ready.wait(), 5)
        assert await asyncio.wait_for(first.trigger_checkpoint_async(), 5) == 1
        assert written == []
    finally:
        await first.cancel_async()
    second = await start(_ReplaySource(False, schema, opened))
    outcome = await asyncio.wait_for(second.wait_async(), 10)
    assert outcome.state == "completed"
    assert opened == [0, 1]
    assert written == [{"x": [1]}]


def test_supported_scalar_keys_copy_slices_and_preserve_null_tuple_and_schema():
    from datetime import date, datetime, timezone

    types = [
        pa.bool_(),
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
        pa.string(),
        pa.large_string(),
        pa.date32(),
        pa.date64(),
        pa.timestamp("us"),
        pa.timestamp("us", "UTC"),
    ]
    values = [
        True,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        "x",
        "x",
        date(1970, 1, 1),
        date(1970, 1, 1),
        datetime(1970, 1, 1),
        datetime(1970, 1, 1, tzinfo=timezone.utc),
    ]
    names = [f"k{index}" for index in range(len(types))]
    schema = pa.schema([pa.field(name, dtype) for name, dtype in zip(names, types)])
    raw = pa.Table.from_arrays(
        [
            pa.array([value, value, None, value], type=dtype).slice(1, 3)
            for dtype, value in zip(types, values)
        ],
        schema=schema,
    )
    events = cf.table_input("events", schema=schema)
    out = events.distinct(
        names, spec=cf.DistinctSpec.bounded_source(cf.StateBudget(3, 64 << 20))
    )
    result = cf.Program("types", mode="batch", outputs={"out": out}).collect(
        {"events": raw}
    )["out"]
    assert result.schema == schema
    # UTC epoch microseconds do not require a host timezone database.
    comparable = result.set_column(
        len(names) - 1, names[-1], result[names[-1]].cast(pa.int64())
    )
    rows = {tuple(row[name] for name in names) for row in comparable.to_pylist()}
    assert rows == {(*values[:-1], 0), (None,) * 15}


@_sync
async def test_source_error_and_cancel_do_not_become_successful_eof():
    schema = pa.schema([pa.field("x", pa.int64())])
    events = cf.table_input("events", schema=schema)
    out = events.distinct(
        ["x"], spec=cf.DistinctSpec.bounded_source(cf.StateBudget(2, 8 << 20))
    )
    program = cf.Program("no_false_eof", mode="stream", outputs={"out": out})

    class Disconnect(_ReplaySource):
        async def next(self):
            if self.position:
                raise ConnectionError("source disconnected")
            return await super().next()

    seen = []
    with pytest.raises(cf.StreamingRuntimeError):
        async with program.stream(
            {
                "events": cf.SourceBinding(
                    Disconnect(False, schema, []),
                    watermark_policy=cf.DisabledWatermarks(),
                )
            },
            config=cf.StreamRuntimeConfig(checkpointing=False),
        ) as results:
            async for event in results:
                seen.append(event)
    assert seen == []
    source = _ReplaySource(True, schema, [])
    async with program.stream(
        {"events": cf.SourceBinding(source, watermark_policy=cf.DisabledWatermarks())},
        config=cf.StreamRuntimeConfig(checkpointing=False),
    ) as results:
        await asyncio.wait_for(source.ready.wait(), 5)
        outcome = await results.job.cancel_async()
        assert outcome.state == "cancelled"
        async for event in results:
            seen.append(event)
    assert seen == []


@_sync
async def test_empty_history_emits_one_exact_nonnullable_final():
    schema = pa.schema([pa.field("x", pa.int64(), nullable=False)])
    events = cf.table_input("events", schema=schema)
    out = events.distinct(
        ["x"], spec=cf.DistinctSpec.bounded_source(cf.StateBudget(1, 8 << 20))
    )
    program = cf.Program("empty", mode="batch", outputs={"out": out})
    raw = pa.Table.from_arrays([pa.array([], pa.int64())], schema=schema)
    result = await asyncio.to_thread(program.collect, {"events": raw})
    assert result["out"].schema == schema
    assert result["out"].num_rows == 0

    class Empty(_ReplaySource):
        async def open(self, cursor):
            self.position = 2

    finals = []
    source = cf.SourceBinding(
        Empty(False, schema, []), watermark_policy=cf.DisabledWatermarks()
    )
    async with program.for_mode("stream").stream(
        {"events": source}, config=cf.StreamRuntimeConfig(checkpointing=False)
    ) as outputs:
        async for event in outputs:
            finals.append(event.table)
    assert len(finals) == 1
    assert finals[0].schema == schema
    assert finals[0].num_rows == 0
