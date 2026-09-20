from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

import calc_flow as cf
from calc_flow import symbolic


def quotes() -> cf.TableExpr:
    return cf.table_input(
        "quotes",
        schema=[
            cf.Field("ts", "timestamp[us, UTC]", nullable=False),
            cf.Field("symbol", "string", nullable=False),
            cf.Field("seq", "uint64", nullable=False),
            cf.Field("x", "float64"),
            cf.Field("label", "string"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )


def rolling(value: cf.TableExpr) -> cf.TableExpr:
    return value.with_columns(avg=cf.ts.mean(value["x"], window=cf.rows(2)))


def test_late_outputs_declaration_is_immutable_exported_and_content_addressed() -> None:
    source = quotes()
    value = rolling(source)
    before = value.digest
    pair = cf.with_late_output(value, allowed_lateness_micros=2)
    assert isinstance(pair, cf.LateOutputs)
    assert symbolic.LateOutputs is cf.LateOutputs
    assert symbolic.with_late_output is cf.with_late_output
    assert isinstance(pair.output, cf.TableExpr)
    assert isinstance(pair.late, cf.TableExpr)
    assert value.digest == before
    assert pair.output.digest != value.digest
    assert pair.output.digest != pair.late.digest
    again = cf.with_late_output(value, allowed_lateness_micros=2)
    assert again.output.digest == pair.output.digest
    assert again.late.digest == pair.late.digest
    assert cf.with_late_output(value).output.digest != pair.output.digest
    assert "allowed_lateness_micros=2" in pair.output.explain()
    with pytest.raises(FrozenInstanceError):
        pair.output = value
    assert not hasattr(pair, "__dict__")


@pytest.mark.parametrize("branch", ["output", "late"])
def test_program_requires_explicit_consumers_for_both_outputs(branch: str) -> None:
    pair = cf.with_late_output(rolling(quotes()))
    value = getattr(pair, branch)
    program = cf.Program("missing", outputs={"only": value})
    with pytest.raises(
        cf.CompileError, match=r"outputs.only.*unconsumed_output.*Program"
    ):
        program.compile_stream(cf.Runtime())


def test_paired_outputs_are_stream_only() -> None:
    pair = cf.with_late_output(rolling(quotes()))
    program = cf.Program("batch", outputs={"normal": pair.output, "late": pair.late})
    with pytest.raises(cf.CompileError, match=r"unsupported_mode.*stream"):
        program.compile_batch(cf.Runtime())


@pytest.mark.parametrize("kind", ["rolling", "cross_section"])
def test_paired_lowering_shares_state_and_preserves_stage_input(kind: str) -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = quotes()
    value = (
        rolling(source)
        if kind == "rolling"
        else source.with_columns(
            rank=cf.cs.rank(source["x"], group=cf.exact_time(source["ts"]))
        )
    )
    pair = cf.with_late_output(value, allowed_lateness_micros=3)
    program = cf.Program(
        "paired", outputs={"normal": pair.output, "diagnostics": pair.late}
    )
    runtime = cf.Runtime()
    document = lower_program_document(
        program, runtime, "stream", allowed_lateness_micros=99, late_policy="error"
    )
    states = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] in {"rolling", "cross_section"}
    ]
    assert len(states) == 1
    state = states[0]
    assert state["operator"]["spec"]["allowed_lateness_micros"] == 3
    assert state["operator"]["spec"]["late_policy"] == {
        "kind": "side_output",
        "metrics_version": 1,
        "schema_version": 1,
    }
    ports = {port["name"]: port for port in state["output_ports"]}
    assert [field["name"] for field in ports["late"]["schema"]] == [
        "ts",
        "symbol",
        "seq",
        "x",
        "label",
        "_cf_late_node",
        "_cf_late_input_port",
        "_cf_late_event_time_micros",
        "_cf_late_closing_time_micros",
        "_cf_late_watermark_micros",
        "_cf_late_reason",
        "_cf_late_source",
        "_cf_late_sequence",
        "_cf_late_row_index",
    ]
    assert {
        edge["source_port"]
        for edge in document["graph"]["edges"]
        if edge["source_node"] == state["id"]
    } == {"output", "late"}
    plan = program.compile_stream(runtime)
    assert set(plan.sink_binding_ids) == {"normal.output", "diagnostics.output"}


def test_fragment_lowering_rejects_ambiguous_state_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calc_flow.symbolic.lower import program as lower_program

    pair = cf.with_late_output(rolling(quotes()), allowed_lateness_micros=3)
    program = cf.Program("paired", outputs={"normal": pair.output, "late": pair.late})
    original = lower_program.lower_program_document

    def duplicate_fragment_state(*args, **kwargs):
        document = original(*args, **kwargs)
        is_late_fragment = kwargs.get("late_policy") == "drop" and any(
            str(name).startswith("cf_late_output_") for name, _ in args[0].outputs
        )
        if is_late_fragment:
            nodes = document["graph"]["nodes"]
            states = [
                node
                for node in nodes
                if node["operator"]["kind"] in {"rolling", "cross_section"}
            ]
            nodes.extend(states)
        return document

    monkeypatch.setattr(
        lower_program, "lower_program_document", duplicate_fragment_state
    )
    with pytest.raises(cf.CompileError, match=r"ambiguous_late_stage"):
        program.compile_stream(cf.Runtime())


@pytest.mark.parametrize(
    "variant", ["none", "nested", "mixed", "operand", "different_groups", "after_stage"]
)
def test_stage_selection_rejects_ambiguous_or_hidden_materialization(
    variant: str,
) -> None:
    source = quotes()
    values = {
        "none": source,
        "nested": source.with_columns(
            v=cf.ts.mean(cf.ts.lag(source["x"]), window=cf.rows(2))
        ),
        "mixed": source.with_columns(
            a=cf.ts.lag(source["x"]),
            b=cf.cs.rank(source["x"], group=cf.exact_time(source["ts"])),
        ),
        "operand": source.with_columns(v=cf.ts.lag(source["x"] + 1.0)),
        "different_groups": source.with_columns(
            a=cf.cs.rank(source["x"], group=cf.exact_time(source["ts"])),
            b=cf.cs.rank(
                source["x"],
                group=cf.exact_time(source["ts"], partition_by=[source["symbol"]]),
            ),
        ),
        "after_stage": rolling(source).select("x"),
    }
    pair = cf.with_late_output(values[variant])
    program = cf.Program("stage", outputs={"normal": pair.output, "late": pair.late})
    with pytest.raises(cf.CompileError, match=r"outputs.*ambiguous_late_stage.*stage"):
        program.compile_stream(cf.Runtime())


@pytest.mark.parametrize("lateness", [-1, True, 1 << 64, 0.5])
def test_late_output_rejects_invalid_lateness(lateness: object) -> None:
    with pytest.raises((TypeError, ValueError), match="allowed_lateness_micros"):
        cf.with_late_output(rolling(quotes()), allowed_lateness_micros=lateness)


def test_late_output_rejects_non_table_value() -> None:
    with pytest.raises(TypeError, match="TableExpr"):
        cf.with_late_output(quotes()["x"])


def test_late_output_rejects_reserved_input_field_before_native_compile() -> None:
    source = cf.table_input(
        "quotes",
        schema=[
            cf.Field("ts", "timestamp[us, UTC]", nullable=False),
            cf.Field("symbol", "string", nullable=False),
            cf.Field("seq", "uint64", nullable=False),
            cf.Field("x", "float64"),
            cf.Field("_cf_late_note", "string"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )
    pair = cf.with_late_output(rolling(source))
    program = cf.Program("reserved", outputs={"normal": pair.output, "late": pair.late})
    with pytest.raises(
        cf.CompileError,
        match=r"outputs\..*reserved_field.*input field \"_cf_late_note\" is reserved"
        r" by late schema version 1",
    ):
        program.compile_stream(cf.Runtime())


def test_named_input_materialization_and_late_sql_preserve_boundary() -> None:
    from calc_flow.symbolic.analyzer import _run
    from calc_flow.symbolic.lower import lower_program_document

    source = quotes()
    prepared = source.with_columns(logged=cf.row.log(source["x"]))
    value = prepared.with_columns(
        a=cf.ts.lag(prepared["logged"]),
        b=cf.ts.mean(prepared["logged"], window=cf.rows(2)) + 1.0,
    )
    pair = cf.with_late_output(value)
    derived = pair.late.with_columns(twice=pair.late["x"] * 2.0).select("x", "twice")
    diagnostics = cf.sql("SELECT * FROM rows WHERE twice > 0", rows=derived)
    program = cf.Program(
        "prepared",
        outputs={"normal": pair.output.select("a", "b"), "diagnostics": diagnostics},
    )
    runtime = cf.Runtime()
    analyzer, _ = _run(program, runtime, "stream")
    facts = analyzer.table(pair.late._node, "outputs.diagnostics")
    assert facts.event_time is None and facts.entity_by == facts.sequence_by == ()
    assert facts.lineage != analyzer.table(prepared._node, "inputs.quotes").lineage
    document = lower_program_document(program, runtime, "stream")
    (state,) = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    ]
    late_fields = next(
        port["schema"] for port in state["output_ports"] if port["name"] == "late"
    )
    assert [field["name"] for field in late_fields[:6]] == [
        "ts",
        "symbol",
        "seq",
        "x",
        "label",
        "logged",
    ]
    assert len(late_fields) == 15
    assert isinstance(program.compile_stream(runtime), cf.StreamExecutionPlan)


@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("successor", ["rolling", "array", "merge"])
def test_late_successors_reject_even_after_allowed_transform(
    transformed: bool, successor: str
) -> None:
    source = quotes()
    pair = cf.with_late_output(rolling(source))
    late = cf.sql("SELECT * FROM rows", rows=pair.late) if transformed else pair.late
    if successor == "rolling":
        rejected = rolling(late)
    elif successor == "array":
        rejected = cf.linalg.from_columns(late, columns=["x"], backend="numpy")
    else:
        rejected = cf.sql("SELECT * FROM a UNION ALL SELECT * FROM b", a=late, b=late)
    program = cf.Program(
        "invalid", outputs={"normal": pair.output, "rejected": rejected}
    )
    with pytest.raises(
        cf.CompileError, match=r"outputs.rejected.*unsupported_mode.*late"
    ):
        program.compile_stream(cf.Runtime())


def _program_for_successor(successor: str) -> cf.Program:
    pair = cf.with_late_output(rolling(quotes()))
    late = pair.late
    derived = {
        "row_local": lambda: late.with_columns(
            twice=late["x"] * 2.0,
            tagged=cf.row.coalesce(late["label"], "n/a"),
        ),
        "filter": lambda: late.filter(late["x"] > 0.0),
        "project": lambda: late.select("x", "_cf_late_reason"),
        "sql": lambda: cf.sql("SELECT * FROM rows WHERE x > 0", rows=late),
    }[successor]()
    return cf.Program("mapped", outputs={"normal": pair.output, "late": derived})


def _rolling_state_id(nodes: dict[str, dict[str, object]]) -> str:
    (state,) = [
        node for node in nodes.values() if node["operator"]["kind"] == "rolling"
    ]
    return state["id"]


def _downstream_ids(edges: list[dict[str, object]], roots: list[str]) -> list[str]:
    pending = list(roots)
    visited: set[str] = set()
    ordered: list[str] = []
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        ordered.append(node_id)
        pending.extend(
            edge["target_node"] for edge in edges if edge["source_node"] == node_id
        )
    return ordered


def _late_successor_nodes(program: cf.Program) -> list[dict[str, object]]:
    from calc_flow.symbolic.lower import lower_program_document

    document = lower_program_document(program, cf.Runtime(), "stream")
    nodes = {node["id"]: node for node in document["graph"]["nodes"]}
    edges = document["graph"]["edges"]
    roots = [
        edge["target_node"]
        for edge in edges
        if edge["source_node"] == _rolling_state_id(nodes)
        and edge["source_port"] == "late"
    ]
    assert roots
    return [nodes[node_id] for node_id in _downstream_ids(edges, roots)]


def _assert_native_acceptance(program: cf.Program) -> None:
    assert isinstance(program.compile_stream(cf.Runtime()), cf.StreamExecutionPlan)


@pytest.mark.parametrize("successor", ["row_local", "filter", "project", "sql"])
def test_allowed_late_successors_lower_to_single_input_expression_or_sql(
    successor: str,
) -> None:
    program = _program_for_successor(successor)
    for node in _late_successor_nodes(program):
        assert node["operator"]["kind"] in {"expression", "sql"}
        assert len(node["input_ports"]) == 1
    _assert_native_acceptance(program)


def batch(times: list[int], values: list[float]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("x", pa.float64()),
            pa.field("label", pa.string()),
        ]
    )
    return pa.table(
        {
            "ts": times,
            "symbol": ["a"] * len(times),
            "seq": list(range(len(times))),
            "x": values,
            "label": ["raw"] * len(times),
        },
        schema=schema,
    )


class ScriptedSource:
    def __init__(self, events: list[int | pa.Table]) -> None:
        self.events = events
        self.index = 0
        self.opened = 0
        self.closed = 0

    def capabilities(self) -> cf.SourceCapabilities:
        return cf.SourceCapabilities(
            cf.ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            cf.SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=2,
            max_batch_bytes=1024 * 1024,
            native_watermarks=cf.NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor: cf.Cursor | None) -> None:
        self.opened += 1
        self.index = 0 if cursor is None else cursor.payload["index"]

    async def next(self) -> cf.Data | cf.Watermark | None:
        if self.index == len(self.events):
            return None
        event = self.events[self.index]
        self.index += 1
        if isinstance(event, int):
            return cf.Watermark(
                datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=event)
            )
        return cf.Data(
            cf.Batch.from_pyarrow(event),
            cf.Cursor(self.index.to_bytes(8, "big"), {"index": self.index}),
        )

    async def close(self) -> None:
        self.closed += 1


def source_binding(source: ScriptedSource) -> cf.SourceBinding:
    return cf.SourceBinding(source, watermark_policy=cf.SourceProvidedWatermarks())


@pytest.mark.parametrize("kind", ["rolling", "cross_section"])
@pytest.mark.parametrize("transformed", [False, True])
def test_stateful_stage_input_rejects_before_source_open(
    kind: str, transformed: bool
) -> None:
    source = quotes()
    first = (
        rolling(source)
        if kind == "rolling"
        else source.with_columns(
            rank=cf.cs.rank(source["x"], group=cf.exact_time(source["ts"]))
        )
    )
    prepared = first.with_columns(named=first["x"] + 1.0) if transformed else first
    pair = cf.with_late_output(
        prepared.with_columns(second=cf.ts.mean(prepared["x"], window=cf.rows(2)))
    )
    program = cf.Program(
        "multiple_stages", outputs={"normal": pair.output, "late": pair.late}
    )
    feed = ScriptedSource([])

    async def run() -> None:
        with pytest.raises(cf.CompileError, match="ambiguous_late_stage"):
            async with program.stream({"quotes": source_binding(feed)}):
                pass
        assert feed.opened == feed.closed == 0

    asyncio.run(asyncio.wait_for(run(), 15))


def _column_values(tables: list[pa.Table], name: str) -> list[object]:
    return [value for table in tables for value in table[name].to_pylist()]


def _rows_with_timestamp_micros(table: pa.Table) -> list[dict[str, object]]:
    assert table.schema.field("ts").type == pa.timestamp("us", tz="UTC")
    index = table.schema.get_field_index("ts")
    return table.set_column(index, "ts", table["ts"].cast(pa.int64())).to_pylist()


def _assert_late_diagnostic(tables: list[pa.Table]) -> None:
    (row,) = [row for table in tables for row in _rows_with_timestamp_micros(table)]
    assert row["ts"] == 0
    assert row["label"] == "raw"
    assert row["_cf_late_event_time_micros"] == 0
    assert row["_cf_late_watermark_micros"] == 1
    assert row["_cf_late_reason"] == "late_row"
    assert row["_cf_late_row_index"] == 0


@pytest.mark.parametrize("kind", ["rolling", "cross_section"])
@pytest.mark.parametrize("late_rows", [False, True])
def test_one_owned_iterator_routes_scripted_watermarks_and_closes(
    kind: str, late_rows: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import calc_flow.stream as stream_module

    monkeypatch.setattr(stream_module.tempfile, "tempdir", str(tmp_path))
    source = quotes()
    value = (
        rolling(source)
        if kind == "rolling"
        else source.with_columns(
            rank=cf.cs.rank(source["x"], group=cf.exact_time(source["ts"]))
        )
    )
    pair = cf.with_late_output(value)
    program = cf.Program("scripted", outputs={"normal": pair.output, "late": pair.late})
    feed = ScriptedSource([1 if late_rows else -1, batch([0, 2], [10.0, 20.0]), 2])
    results = program.stream({"quotes": source_binding(feed)})
    seen: dict[str, list[pa.Table]] = {"normal": [], "late": []}

    async def run() -> None:
        before = asyncio.all_tasks()
        async with results:
            async for output in results:
                assert isinstance(output, cf.StreamOutput)
                seen[output.name].append(output.table)
        assert results.job.status()["state"] == "completed"
        assert results.job.status()["task_count"] == 0
        assert asyncio.all_tasks() == before
        assert feed.opened == feed.closed == 1

    asyncio.run(asyncio.wait_for(run(), 15))
    values = {name: _column_values(tables, "x") for name, tables in seen.items()}
    assert values == {
        "normal": [20.0] if late_rows else [10.0, 20.0],
        "late": [10.0] if late_rows else [],
    }
    if late_rows:
        _assert_late_diagnostic(seen["late"])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("branch", ["output", "late", "indirect"])
def test_missing_consumer_and_standalone_entrypoints_never_open_source(
    branch: str,
) -> None:
    pair = cf.with_late_output(rolling(quotes()))
    value = pair.output if branch == "output" else pair.late
    if branch == "indirect":
        value = cf.sql("SELECT * FROM rows", rows=value.select("x"))
    feed = ScriptedSource([])
    program = cf.Program("missing", outputs={"only": value})

    async def run() -> None:
        for result in (
            program.stream({"quotes": source_binding(feed)}),
            value.stream(source_binding(feed)),
        ):
            with pytest.raises(cf.CompileError, match="unconsumed_output.*Program"):
                async with result:
                    pass
        assert feed.opened == feed.closed == 0

    asyncio.run(run())
    with pytest.raises(cf.CompileError, match="unconsumed_output.*Program"):
        value.collect({"quotes": batch([0], [1.0])})


def test_local_policy_cache_and_export_do_not_change_unmarked_stage() -> None:
    source = quotes()
    value = rolling(source)
    pair = cf.with_late_output(value, allowed_lateness_micros=3)
    program = cf.Program(
        "policies",
        outputs={"normal": pair.output, "late": pair.late, "ordinary": value},
    )
    runtime = cf.Runtime()
    project = program.to_project(
        runtime, mode="stream", allowed_lateness_micros=7, late_policy="drop"
    )
    specs = [
        node["operator"]["spec"]
        for node in project.root["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    ]
    assert len(specs) == 2
    assert sorted(
        (spec["allowed_lateness_micros"], spec["late_policy"]["kind"]) for spec in specs
    ) == [(3, "side_output"), (7, "drop")]
    plan = program.compile_stream(
        runtime, allowed_lateness_micros=7, late_policy="drop"
    )
    cached_count = len(runtime._symbolic_compile_cache)
    repeated = program.compile_stream(
        runtime, allowed_lateness_micros=7, late_policy="drop"
    )
    assert repeated is not plan
    assert repeated.fingerprint == plan.fingerprint
    assert len(runtime._symbolic_compile_cache) == cached_count
    changed = cf.with_late_output(value, allowed_lateness_micros=4)
    other = cf.Program(
        "policies",
        outputs={"normal": changed.output, "late": changed.late, "ordinary": value},
    )
    assert (
        other.compile_stream(
            runtime, allowed_lateness_micros=7, late_policy="drop"
        ).fingerprint
        != plan.fingerprint
    )
    assert "allowed_lateness_micros=3" in program.explain(runtime, mode="stream")
    with pytest.raises(ValueError, match="late_policy"):
        program.compile_stream(runtime, late_policy="side_output")


def test_native_schema_error_retains_real_machine_path() -> None:
    import json

    pair = cf.with_late_output(rolling(quotes()))
    program = cf.Program("native", outputs={"normal": pair.output, "late": pair.late})
    document = program.to_project(mode="stream").root
    state_index = next(
        index
        for index, node in enumerate(document["graph"]["nodes"])
        if node["operator"]["kind"] == "rolling"
    )
    document["graph"]["nodes"][state_index]["operator"]["spec"]["late_policy"][
        "schema_version"
    ] = 2
    with pytest.raises(
        cf.ConfigError,
        match=rf"graph.nodes\[{state_index}\].operator.spec.late_policy.schema_version.*unsupported_version",
    ):
        cf.Runtime()._compile_stream_graph_project(
            json.dumps(document), requirements=cf.StreamRequirements()
        )


async def _assert_paired_lifecycle(
    results: cf.StreamResults[cf.StreamOutput],
    feed: ScriptedSource,
    blocked: asyncio.Event,
    failure: str,
    failures: list[str],
) -> None:
    before = asyncio.all_tasks()
    if failure == "cancel":
        async with results:
            await asyncio.wait_for(blocked.wait(), 5)
            assert feed.index < len(feed.events)
        assert results.job.status()["state"] == "cancelled"
    else:
        with pytest.raises(cf.StreamingRuntimeError) as caught:
            async with results:
                pass
        assert "private-" not in str(caught.value)
        assert failures == [failure]
    assert asyncio.all_tasks() == before
    assert feed.closed == feed.opened


@pytest.mark.parametrize("failure", ["cancel", "sink_open", "source_open"])
def test_paired_lifecycle_closes_resources_before_temporary_state(
    failure: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import calc_flow.stream as stream_module

    monkeypatch.setattr(stream_module.tempfile, "tempdir", str(tmp_path))
    pair = cf.with_late_output(rolling(quotes()))
    program = cf.Program(
        "lifecycle", outputs={"normal": pair.output, "late": pair.late}
    )
    opened_sinks: set[str] = set()
    closed_sinks: set[str] = set()
    blocked = asyncio.Event()
    failures: list[str] = []
    write = stream_module._QueueSink.write
    remove = stream_module.shutil.rmtree

    class Feed(ScriptedSource):
        async def open(self, cursor: cf.Cursor | None) -> None:
            if failure == "source_open":
                self.opened += 1
                failures.append("source_open")
                raise RuntimeError("private-source-sentinel")
            await super().open(cursor)

    feed = Feed([1, *[batch([0], [10.0]) for _ in range(40)], 2])

    async def open_sink(sink) -> None:
        if failure == "sink_open" and sink._name == "normal":
            failures.append("sink_open")
            raise RuntimeError("private-sink-sentinel")
        opened_sinks.add(sink._name)

    async def close_sink(sink) -> None:
        closed_sinks.add(sink._name)

    async def write_sink(sink, value) -> None:
        if sink._queue.full():
            blocked.set()
        await write(sink, value)

    results = program.stream(
        {"quotes": source_binding(feed)},
        config=cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(2, 1024 * 1024)),
    )

    def remove_state(path: str) -> None:
        assert feed.closed == feed.opened
        assert opened_sinks <= closed_sinks
        if failure == "cancel":
            assert results.job.status()["task_count"] == 0
        remove(path)

    monkeypatch.setattr(stream_module._QueueSink, "open", open_sink)
    monkeypatch.setattr(stream_module._QueueSink, "close", close_sink)
    monkeypatch.setattr(stream_module._QueueSink, "write", write_sink)
    monkeypatch.setattr(stream_module.shutil, "rmtree", remove_state)

    asyncio.run(
        asyncio.wait_for(
            _assert_paired_lifecycle(results, feed, blocked, failure, failures), 15
        )
    )
    assert opened_sinks <= closed_sinks
    assert list(tmp_path.iterdir()) == []


class CollectSink:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.closed = 0

    async def open(self) -> None:
        pass

    async def write(self, value: cf.Batch) -> None:
        self.rows.extend(_rows_with_timestamp_micros(value.to_pyarrow()))

    async def close(self) -> None:
        self.closed += 1


def _assert_single_late_state_owner(root: Path) -> None:
    import json

    manifests = list((root / "manifests").glob("manifest-*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    state_owners = [
        key
        for key, entry in manifest["operators"].items()
        if "late_output" in entry["inline_metadata"]
    ]
    assert len(state_owners) == 1


@pytest.mark.parametrize("kind", ["rolling", "cross_section"])
def test_explicit_runner_restores_both_outputs_at_same_control_cut(
    kind: str, tmp_path: Path
) -> None:
    source = quotes()
    value = (
        rolling(source)
        if kind == "rolling"
        else source.with_columns(
            rank=cf.cs.rank(source["x"], group=cf.exact_time(source["ts"]))
        )
    )
    pair = cf.with_late_output(value)
    program = cf.Program("recovery", outputs={"normal": pair.output, "late": pair.late})
    events = [1, batch([0, 2], [10.0, 20.0]), 2, batch([0, 3], [11.0, 30.0]), 3]

    class PausingSource(ScriptedSource):
        def __init__(self, pause: bool) -> None:
            super().__init__(events)
            self.pause = pause
            self.paused = asyncio.Event()
            self.offsets: list[int] = []

        async def open(self, cursor: cf.Cursor | None) -> None:
            await super().open(cursor)
            self.offsets.append(self.index)

        async def next(self):
            if self.pause and self.index == 2:
                self.paused.set()
                await asyncio.sleep(0)
                return cf.Idle()
            return await super().next()

    def runner(
        feed: PausingSource, sinks: dict[str, CollectSink], root: Path
    ) -> cf.StreamingRunner:
        plan = program.compile_stream()
        assert set(plan.sink_binding_ids) == {"normal.output", "late.output"}
        return cf.StreamingRunner(
            plan,
            {plan.source_binding_ids[0]: source_binding(feed)},
            {
                name: [cf.SinkBinding.ordinary(name.split(".")[0] + "_archive", sink)]
                for name, sink in sinks.items()
            },
            cf.ManagedCheckpointRuntime(root),
        )

    async def execute(recover: bool) -> dict[str, list[dict[str, object]]]:
        root = tmp_path / ("recovered" if recover else "uninterrupted")
        sinks = {name: CollectSink() for name in ("normal.output", "late.output")}
        feed = PausingSource(True)
        job = await runner(feed, sinks, root).start_async()
        await asyncio.wait_for(feed.paused.wait(), 5)
        assert await job.trigger_checkpoint_async() == 1
        _assert_single_late_state_owner(root)
        if recover:
            assert (await job.cancel_async()).state == "cancelled"
            assert feed.closed == 1
            feed = PausingSource(False)
            job = await runner(feed, sinks, root).start_async()
            assert feed.offsets == [2]
        else:
            feed.pause = False
        outcome = await job.wait_async()
        assert outcome.state == "completed", outcome.errors
        assert job.status()["task_count"] == 0
        assert feed.closed == 1
        assert all(sink.closed == (2 if recover else 1) for sink in sinks.values())
        return {name: sink.rows for name, sink in sinks.items()}

    async def run() -> None:
        uninterrupted = await execute(False)
        recovered = await execute(True)
        assert recovered == uninterrupted
        assert [row["x"] for row in recovered["late.output"]] == [10.0, 11.0]
        assert [row["_cf_late_sequence"] for row in recovered["late.output"]] == [0, 1]
        assert [row["x"] for row in recovered["normal.output"]] == [20.0, 30.0]

    asyncio.run(asyncio.wait_for(run(), 30))


def test_named_cross_section_operand_and_normal_temporal_successor() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = quotes()
    prepared = source.with_columns(adjusted=cf.row.coalesce(source["x"], 0.0))
    pair = cf.with_late_output(
        prepared.with_columns(
            rank=cf.cs.rank(prepared["adjusted"], group=cf.exact_time(prepared["ts"]))
        )
    )
    normal = pair.output.with_columns(previous=cf.ts.lag(pair.output["rank"]))
    program = cf.Program("normal_chain", outputs={"normal": normal, "late": pair.late})
    document = lower_program_document(program, cf.Runtime(), "stream")
    kinds = [node["operator"]["kind"] for node in document["graph"]["nodes"]]
    assert kinds.count("rolling") == kinds.count("cross_section") == 1
    state = next(
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "cross_section"
    )
    fields = next(
        port["schema"] for port in state["output_ports"] if port["name"] == "late"
    )
    assert (
        next(field for field in fields if field["name"] == "adjusted")["nullable"]
        is False
    )
    assert isinstance(program.compile_stream(), cf.StreamExecutionPlan)


def test_explicit_named_group_column_is_stage_input() -> None:
    source = quotes()
    prepared = source.with_columns(
        bucket=cf.row.where(source["x"] > 0.0, source["symbol"], "other")
    )
    pair = cf.with_late_output(
        prepared.with_columns(
            rank=cf.cs.rank(
                prepared["x"],
                group=cf.exact_time(prepared["ts"], partition_by=[prepared["bucket"]]),
            )
        )
    )
    program = cf.Program(
        "named_group", outputs={"normal": pair.output, "late": pair.late}
    )
    assert isinstance(program.compile_stream(), cf.StreamExecutionPlan)


@pytest.mark.parametrize("split", [False, True])
def test_late_sql_uses_its_own_per_batch_contract(split: bool) -> None:
    pair = cf.with_late_output(rolling(quotes()))
    counts = cf.sql("SELECT COUNT(*) AS n FROM rejected", rejected=pair.late)
    program = cf.Program(
        "sql_counts", outputs={"normal": pair.output, "counts": counts}
    )
    payloads = (
        [batch([0], [10.0]), batch([0], [11.0])]
        if split
        else [batch([0, 0], [10.0, 11.0])]
    )
    feed = ScriptedSource([1, *payloads, 2])

    async def run() -> None:
        totals = []
        async with program.stream({"quotes": source_binding(feed)}) as results:
            async for output in results:
                assert output.name == "counts" or output.table.num_rows == 0
                if output.name == "counts":
                    totals.extend(output.table["n"].to_pylist())
        assert totals == ([1, 1] if split else [2])
        assert feed.opened == feed.closed == 1

    asyncio.run(asyncio.wait_for(run(), 15))


def test_native_runner_missing_sink_rejects_before_source_open(tmp_path: Path) -> None:
    pair = cf.with_late_output(rolling(quotes()))
    program = cf.Program("bindings", outputs={"normal": pair.output, "late": pair.late})
    plan = program.compile_stream()
    feed = ScriptedSource([])
    with pytest.raises(
        cf.StreamingRuntimeError, match=r"late.output.*sink|sink.*late.output"
    ):
        cf.StreamingRunner(
            plan,
            {plan.source_binding_ids[0]: source_binding(feed)},
            {"normal.output": [cf.SinkBinding.ordinary("archive", CollectSink())]},
            cf.ManagedCheckpointRuntime(tmp_path),
        )
    assert feed.opened == feed.closed == 0


def test_native_compiler_rejects_temporal_reentry_after_late_projection() -> None:
    import copy
    import json

    pair = cf.with_late_output(rolling(quotes()))
    program = cf.Program(
        "native_guard",
        outputs={
            "normal": pair.output,
            "late": pair.late.select("ts", "symbol", "seq", "x", "label"),
        },
    )
    document = program.to_project(mode="stream").root
    state = copy.deepcopy(
        next(
            node
            for node in document["graph"]["nodes"]
            if node["operator"]["kind"] == "rolling"
        )
    )
    projection = next(
        node for node in document["graph"]["nodes"] if node["id"] == "late"
    )
    projection["output_ports"] = [
        {**copy.deepcopy(state["input_ports"][0]), "name": "output"}
    ]
    state["id"] = "forbidden"
    state["operator"]["spec"]["late_policy"] = {"kind": "drop", "metrics_version": 1}
    state["output_ports"] = [state["output_ports"][0]]
    document["graph"]["nodes"].append(state)
    edge_index = len(document["graph"]["edges"])
    document["graph"]["edges"].append(
        {
            "source_node": "late",
            "source_port": "output",
            "target_node": "forbidden",
            "target_port": "input",
        }
    )
    with pytest.raises(
        cf.CompileError,
        match=rf"graph.edges\[{edge_index}\].*temporal_output_unavailable",
    ):
        cf.Runtime()._compile_stream_graph_project(json.dumps(document))


def test_late_provider_and_temporal_stream_reject_before_source_open() -> None:
    pair = cf.with_late_output(rolling(quotes()))
    array = cf.linalg.from_columns(quotes(), columns=["x"], backend="numpy")
    provider = cf.table.attach_columns(pair.late, array, names=["p"])
    with pytest.raises(cf.CompileError, match="unsupported_mode.*late"):
        cf.Program(
            "provider", outputs={"normal": pair.output, "late": provider}
        ).compile_stream()
    feed = ScriptedSource([])
    program = cf.Program(
        "temporal", outputs={"normal": pair.output, "late": rolling(pair.late)}
    )

    async def run() -> None:
        with pytest.raises(cf.CompileError, match="unsupported_mode.*late"):
            async with program.stream({"quotes": source_binding(feed)}):
                pass
        assert feed.opened == feed.closed == 0

    asyncio.run(run())


def test_private_analysis_boundary_does_not_replace_a_declared_input() -> None:
    from calc_flow.symbolic.analyzer import _run

    source = quotes()
    pair = cf.with_late_output(rolling(source))
    collision_name = f"cf_late_stage_{pair.output.digest[:24]}"
    collision = cf.table_input(
        collision_name,
        schema=[
            cf.Field("ts", "timestamp[us, UTC]", nullable=False),
            cf.Field("symbol", "string", nullable=False),
            cf.Field("seq", "uint64", nullable=False),
            cf.Field("x", "float64"),
            cf.Field("label", "string"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )
    program = cf.Program(
        "names", outputs={"normal": pair.output, "late": pair.late, "other": collision}
    )
    analyzer, _ = _run(program, cf.Runtime(), "stream")
    assert analyzer.table(collision._node, "outputs.other").lineage == collision_name
