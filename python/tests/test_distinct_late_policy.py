import json

import pyarrow as pa
import pytest

import calc_flow as cf


def window(policy="error"):
    return cf.DistinctSpec.closed_window(
        "ts",
        start_micros=0,
        end_micros=10,
        state_budget=cf.StateBudget(2, 8 << 20),
        late_policy=policy,
    )


def test_local_policy_keyword_is_strict_and_canonical():
    assert "late_policy" not in json.loads(window()._wire)
    assert json.loads(window("drop")._wire)["late_policy"] == {
        "kind": "drop",
        "metrics_version": 1,
    }
    for value in [None, True, 1]:
        with pytest.raises(TypeError):
            window(value)
    for value in ["DROP", "side_output", ""]:
        with pytest.raises(ValueError):
            window(value)


@pytest.mark.parametrize("policy", ["error", "drop"])
def test_expression_explain_names_local_policy_and_layout(policy):
    events = cf.table_input(
        "events",
        schema=pa.schema(
            [pa.field("ts", pa.timestamp("us")), pa.field("q", pa.int64())]
        ),
    )
    text = events.distinct(["q"], spec=window(policy)).explain()
    assert f"local DISTINCT late_policy={policy}" in text
    assert f"layout={3 if policy == 'drop' else 2}" in text


@pytest.mark.parametrize("syntax", ["df", "sql"])
def test_local_drop_survives_default_global_error_and_export(syntax):
    events = cf.table_input(
        "events",
        schema=pa.schema(
            [pa.field("ts", pa.timestamp("us")), pa.field("q", pa.int64())]
        ),
    )
    output = (
        events.distinct(["q"], spec=window("drop"))
        if syntax == "df"
        else cf.sql(
            "SELECT DISTINCT q FROM CF_DISTINCT_WINDOW('a','ts',0,10,2,8388608,'drop')",
            a=events,
        )
    )
    program = cf.Program("local", mode="stream", outputs={"out": output})
    document = program.to_project().model_dump()
    distinct = next(
        n["operator"]
        for n in document["graph"]["nodes"]
        if n["operator"]["kind"] == "distinct"
    )
    assert distinct["spec"]["late_policy"] == {"kind": "drop", "metrics_version": 1}
    program.compile_stream()
    for options in [dict(late_policy="drop"), dict(allowed_lateness_micros=1)]:
        with pytest.raises(cf.CompileError, match="no applicable.*consumer"):
            program.to_project(**options)


def declared(syntax="df", policy="drop"):
    events = cf.table_input(
        "events",
        schema=pa.schema(
            [pa.field("ts", pa.timestamp("us")), pa.field("q", pa.int64())]
        ),
    )
    output = (
        events.distinct(["q"], spec=window(policy))
        if syntax == "df"
        else cf.sql(
            "SELECT DISTINCT q FROM CF_DISTINCT_WINDOW("
            f"'a','ts',0,10,2,8388608,'{policy}')",
            a=events,
        )
    )
    return cf.Program("local", mode="stream", outputs={"out": output})


def operators(program, **options):
    return [
        n["operator"]
        for n in program.to_project(**options).model_dump()["graph"]["nodes"]
    ]


@pytest.mark.parametrize("consumer", ["rolling", "cross_section"])
@pytest.mark.parametrize("marked", [False, True])
def test_mixed_local_policies_shared_owner_and_independent_global_consumer(
    consumer, marked
):
    events = cf.table_input(
        "events",
        schema=pa.schema(
            [pa.field("ts", pa.timestamp("us")), pa.field("q", pa.int64())]
        ),
    )
    drop = events.distinct(["q"], spec=window("drop"))
    error = events.distinct(["q"], spec=window())
    quotes = cf.table_input(
        "quotes",
        schema=[
            cf.Field("ts", "timestamp[us, UTC]", nullable=False),
            cf.Field("symbol", "string", nullable=False),
            cf.Field("seq", "uint64", nullable=False),
            cf.Field("x", "float64"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )
    feature = (
        cf.ts.mean(quotes["x"], window=cf.rows(2))
        if consumer == "rolling"
        else cf.cs.rank(quotes["x"], group=cf.exact_time(quotes["ts"]))
    )
    temporal = quotes.with_columns(avg=feature)
    outputs = {"drop": drop, "alias": drop, "error": error, "temporal": temporal}
    if marked:
        pair = cf.with_late_output(temporal)
        outputs.update(temporal=pair.output, late=pair.late)
    program = cf.Program(
        "mixed",
        mode="stream",
        outputs=outputs,
    )
    if marked:
        with pytest.raises(cf.CompileError, match="no applicable.*consumer"):
            program.to_project(late_policy="drop", allowed_lateness_micros=2)
        assert_local_pair(operators(program))
        return
    ops = operators(program, late_policy="drop", allowed_lateness_micros=2)
    assert_local_pair(ops)
    rolling = next(o for o in ops if o["kind"] == consumer)
    assert rolling["spec"]["late_policy"] == {"kind": "drop", "metrics_version": 1}
    program.compile_stream(late_policy="drop", allowed_lateness_micros=2)


def assert_local_pair(ops):
    distinct = [o for o in ops if o["kind"] == "distinct"]
    assert len(distinct) == 2
    assert sum("late_policy" in o["spec"] for o in distinct) == 1


@pytest.mark.parametrize("syntax", ["df", "sql"])
def test_finite_reuse_and_policy_cache_isolation(syntax):
    runtime = cf.Runtime()
    drop = declared(syntax, "drop")
    error = declared(syntax, "error")
    dp = drop.compile_stream(runtime)
    ep = error.compile_stream(runtime)
    assert dp.fingerprint != ep.fingerprint
    assert drop.compile_stream(runtime).fingerprint == dp.fingerprint
    for chosen in [drop, error]:
        finite = chosen.for_mode("batch")
        plan = finite.compile_batch(runtime)
        over = pa.table(
            {
                "ts": pa.array([0, 0, 9], pa.timestamp("us")),
                "q": pa.array([1, 1, 1], pa.int64()),
            }
        )
        with pytest.raises(cf.CalcFlowError, match="state_budget_exceeded"):
            finite.collect({"events": over}, runtime=runtime)
        for times, values, expected in [
            ([0, 9], [1, 1], {1}),
            ([], [], set()),
            ([0], [7], {7}),
        ]:
            table = pa.table(
                {
                    "ts": pa.array(times, pa.timestamp("us")),
                    "q": pa.array(values, pa.int64()),
                }
            )
            assert (
                set(
                    finite.collect({"events": table}, runtime=runtime)["out"][
                        "q"
                    ].to_pylist()
                )
                == expected
            )
        assert all(value is None for value in plan.snapshot().values())


def test_layout2_runtime_accepts_error_and_rejects_drop(monkeypatch):
    from dataclasses import replace

    runtime = cf.Runtime()
    base = runtime.capabilities()
    old = replace(
        base,
        operators=tuple(
            replace(o, state_layouts=(2,)) if o.kind == "distinct" else o
            for o in base.operators
        ),
    )
    monkeypatch.setattr(cf.Runtime, "capabilities", lambda _self: old)
    declared(policy="error").compile_stream(runtime)
    with pytest.raises(cf.CompileError, match="capability_mismatch"):
        declared().compile_stream(runtime)


def test_unknown_distinct_state_version_is_rejected(monkeypatch):
    from dataclasses import replace

    runtime = cf.Runtime()
    base = runtime.capabilities()
    offered = replace(
        base,
        operators=tuple(
            replace(o, state_version=2) if o.kind == "distinct" else o
            for o in base.operators
        ),
    )
    monkeypatch.setattr(cf.Runtime, "capabilities", lambda _self: offered)
    with pytest.raises(cf.CompileError, match="capability_mismatch"):
        declared(policy="error").compile_stream(runtime)


class ReplayWindow:
    def __init__(self, hold, opened):
        import asyncio

        self.hold = hold
        self.opened = opened
        self.position = 0
        self.ready = asyncio.Event()
        self.release = asyncio.Event()
        self.schema = pa.schema(
            [pa.field("ts", pa.timestamp("us")), pa.field("q", pa.int64())]
        )

    def capabilities(self):
        return cf.SourceCapabilities(
            cf.ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            cf.SourceDeliveryCapability.LOSSLESS,
            4,
            4096,
            self.schema,
            cf.NativeWatermarkCapability.EMITS_NATIVE,
            cf.SourceBoundedness.UNBOUNDED,
        )

    async def open(self, cursor):
        self.position = 0 if cursor is None else cursor.payload["position"]
        self.opened.append(self.position)

    def batch(self, records):
        tables = [
            pa.RecordBatch.from_arrays(
                [pa.array(t, pa.timestamp("us")), pa.array(q, pa.int64())],
                schema=self.schema,
            )
            for t, q in records
        ]
        return cf.Batch.from_pyarrow(pa.Table.from_batches(tables))

    async def next(self):
        import asyncio
        from datetime import UTC, datetime, timedelta

        if self.position == 4 and not self.release.is_set():
            self.ready.set()
            await asyncio.sleep(0.001)
            return cf.Idle()
        position = self.position
        self.position += 1
        if position in [1, 2, 4]:
            return cf.Watermark(
                datetime(1970, 1, 1, tzinfo=UTC)
                + timedelta(microseconds={1: 9, 2: 10, 4: 12}[position])
            )
        records = {
            0: [([0, 9], [1, 1])],
            3: [([0, 9], [2, 3]), ([10, None], [4, 5])],
            5: [([1], [8])],
        }
        if position not in records:
            return None
        return cf.Data(
            self.batch(records[position]),
            cf.Cursor(bytes([self.position]), {"position": self.position}),
        )

    async def close(self):
        pass


class Sink:
    def __init__(self, written):
        self.written = written

    async def open(self):
        pass

    async def write(self, batch):
        self.written.append(batch.to_pyarrow().to_pydict())

    async def close(self):
        pass


def metrics(job, operator_id):
    value = job.status()["operators"][operator_id]
    return (
        value["late_rows"],
        value["late_affected_batches"],
        value["max_lateness_micros"],
    )


async def start(program, source, root, written):
    import asyncio

    plan = await asyncio.to_thread(program.compile_stream)
    return await cf.StreamingRunner(
        plan,
        {plan.source_binding_ids[0]: cf.SourceBinding(source)},
        {plan.sink_binding_ids[0]: [cf.SinkBinding.ordinary("archive", Sink(written))]},
        cf.ManagedCheckpointRuntime(root),
    ).start_async()


@pytest.mark.parametrize("syntax", ["df", "sql"])
def test_managed_drop_metrics_restore_without_recount_or_second_final(tmp_path, syntax):
    import asyncio
    import shutil

    async def run():
        program = declared(syntax)
        document = program.to_project().model_dump()
        ids = [
            n["id"]
            for n in document["graph"]["nodes"]
            if n["operator"]["kind"] == "distinct"
        ]
        assert len(ids) == 1
        operator_id = ids[0]
        opened = []
        written = []
        root = tmp_path / "first"
        source = ReplayWindow(True, opened)
        first = await start(program, source, root, written)
        try:
            await asyncio.wait_for(source.ready.wait(), 5)
            assert await asyncio.wait_for(first.trigger_checkpoint_async(), 5) == 1
            assert metrics(first, operator_id) == (2, 1, 10)
            assert written == [{"q": [1]}]
        finally:
            await first.cancel_async()
        replay = tmp_path / "replay"
        shutil.copytree(root, replay)
        bad = ReplayWindow(False, [])
        with pytest.raises(cf.StreamingRuntimeError):
            await start(declared(syntax, "error"), bad, root, written)
        assert bad.opened == []
        for cut in [root, replay]:
            resumed = ReplayWindow(False, opened)
            job = await start(program, resumed, cut, written)
            try:
                await asyncio.wait_for(resumed.ready.wait(), 5)
                assert metrics(job, operator_id) == (2, 1, 10)
                resumed.release.set()
                assert (
                    await asyncio.wait_for(job.wait_async(), 10)
                ).state == "completed"
                assert metrics(job, operator_id) == (3, 2, 11)
            finally:
                if job.status()["state"] == "running":
                    await job.cancel_async()
        assert opened == [0, 4, 4]
        assert written == [{"q": [1]}]

    asyncio.run(run())
