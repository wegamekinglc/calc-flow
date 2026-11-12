from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

import calc_flow as cf


def _program(
    *, chain=False, transform=False, query="SELECT SUM(total) AS total FROM input"
):
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    total = source.sql("SELECT SUM(x) AS total FROM input")
    if transform:
        total = total.filter(total["total"] >= cf.lit(0)).select("total")
    output = total.sql(query) if chain else total
    return cf.Program("updates", mode="stream", outputs={"result": output})


def test_update_choice_is_immutable_and_preserved_by_copies():
    program = _program()
    latest = program.with_update_semantics("latest_relation")
    assert program.update_semantics == "legacy_append"
    assert latest.update_semantics == "latest_relation"
    assert latest.fingerprint != program.fingerprint
    assert latest.for_mode("batch").fingerprint == latest.fingerprint
    assert (
        latest.with_update_semantics("legacy_append").fingerprint == program.fingerprint
    )
    assert (
        latest.output("copy", latest.outputs[0][1]).update_semantics
        == "latest_relation"
    )
    extra = cf.table_input("extra", schema=pa.schema([("x", pa.int64())]))
    assert latest.with_input(extra).update_semantics == "latest_relation"
    with pytest.raises((AttributeError, TypeError)):
        latest.update_semantics = "legacy_append"
    with pytest.raises(TypeError, match="Program.update_semantics"):
        program.with_update_semantics(None)
    with pytest.raises(ValueError, match="invalid_literal"):
        program.with_update_semantics("latest")


@pytest.mark.parametrize("transform", [False, True])
def test_latest_gate_reports_consumer_before_source_or_job(monkeypatch, transform):
    program = _program(chain=True, transform=transform)
    runtime = cf.Runtime()
    program.compile_stream(runtime)
    latest = program.with_update_semantics("latest_relation")
    calls = []

    class Unread:
        def __aiter__(self):
            calls.append("source")
            raise AssertionError("source acquired before gate")

    async def forbid_job(*args, **kwargs):
        calls.append("job")
        raise AssertionError("job started before gate")

    monkeypatch.setattr(cf.StreamingRunner, "start_async", forbid_job)
    report = latest.analyze(runtime).query
    assert report.update_semantics == "latest_relation"
    assert (report.schema_version, report.analysis_revision) == (2, 4)
    edge = next(
        edge for edge in report.edges if edge.path == "outputs.result.sql.tables.input"
    )
    assert edge.stream.producer_update == "complete_snapshot"
    assert edge.stream.consumer_interpretation == "complete_replacement"
    assert edge.stream.support == "supported"

    async def run():
        async with latest.stream({"events": Unread()}, runtime=runtime):
            raise AssertionError("stream started")

    for _ in range(2):
        latest.compile_stream(runtime)
        with pytest.raises(
            (ValueError, cf.ConfigError), match="sql_replacement_state_budget"
        ):
            asyncio.run(run())
    assert calls == []
    latest.for_mode("batch").compile_batch(runtime)


def _observe(program, data):
    async def run():
        async with program.stream({"events": data}) as events:
            return [event.table.to_pydict() async for event in events]

    return asyncio.run(run())


def test_legacy_snapshot_append_chain_keeps_seven():
    data = [pa.table({"x": [1]}), pa.table({"x": [2, 3]})]
    for program in (
        _program(chain=True),
        _program(chain=True).with_update_semantics("legacy_append"),
    ):
        assert _observe(program, data) == [{"total": [1]}, {"total": [7]}]


def test_latest_filter_delivers_empty_replacement_and_no_event_stays_absent():
    latest = _program(transform=True).with_update_semantics("latest_relation")
    assert _observe(latest, [pa.table({"x": [1]}), pa.table({"x": [-2]})]) == [
        {"total": [1]},
        {"total": []},
    ]

    async def empty_events():
        for batch in ():
            yield batch

    assert _observe(latest, empty_events()) == []


def test_latest_sql_cte_preserves_snapshot_facts_and_empty_events():
    program = _program(transform=True)
    output = program.outputs[0][1].sql(
        "WITH visible AS (SELECT total FROM input) SELECT total FROM visible"
    )
    latest = cf.Program(
        "sql-events", mode="stream", outputs={"result": output}
    ).with_update_semantics("latest_relation")
    report = latest.analyze().query
    terminal = next(node for node in report.nodes if node.path == "outputs.result")
    assert terminal.stream.update == "complete_snapshot"
    assert terminal.stream.history == "latest_relation"
    edge = next(
        edge for edge in report.edges if edge.path == "outputs.result.sql.tables.input"
    )
    assert edge.stream.consumer_interpretation == "complete_replacement"
    assert _observe(latest, [pa.table({"x": [1]}), pa.table({"x": [-2]})]) == [
        {"total": [1]},
        {"total": []},
    ]


@pytest.mark.parametrize("entry", ["collect", "collect_async", "execute", "stream"])
@pytest.mark.parametrize("strict", [False, True])
def test_unknown_latest_gates_before_rows_bindings_and_job(monkeypatch, entry, strict):
    import importlib

    compute = importlib.import_module("calc_flow.compute")
    stream = importlib.import_module("calc_flow.stream")
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    output = source.sql("SELECT x FROM input UNION ALL SELECT x FROM input")
    program = cf.Program(
        "unproved",
        mode="stream" if entry == "stream" else "batch",
        outputs={"risk.total": output},
    ).with_update_semantics("latest_relation")
    calls = []

    def forbid(*args, **kwargs):
        calls.append("data")
        raise AssertionError("data or bindings touched before gate")

    async def forbid_job(*args, **kwargs):
        calls.append("job")
        raise AssertionError("job started before gate")

    monkeypatch.setattr(compute, "_collect_batches", forbid)
    monkeypatch.setattr(stream.StreamResults, "_bindings", forbid)
    monkeypatch.setattr(cf.StreamingRunner, "start_async", forbid_job)
    request = (
        cf.PortabilityRequest(
            comparison="per_batch", history="current_batch", cutoff="batch_boundary"
        )
        if strict
        else None
    )

    async def run():
        if entry == "stream":
            async with program.stream({"events": object()}, portability=request):
                raise AssertionError("stream started")
        elif entry == "collect_async":
            await program.collect_async({"events": object()}, portability=request)
        else:
            await asyncio.to_thread(
                getattr(program, entry), {"events": object()}, portability=request
            )

    with pytest.raises(cf.CompileError, match="update_semantics_unproved") as caught:
        asyncio.run(run())
    assert caught.value.query_analysis.update_semantics == "latest_relation"
    assert calls == []


def test_legacy_identity_and_canonical_bytes_match_prechange_baseline():
    import hashlib

    program = _program()
    assert (
        program.fingerprint
        == "3e89831e1d00f2b6ff0e5e6a7a6d65b026890c607dd276e686172245788f5c97"
    )
    assert (
        program.compile_stream().fingerprint
        == "166028f91f81a8db11da25fc289fa26d4f53c689bf92f47a44042d3097bc9f54"
    )
    assert (
        program.for_mode("batch").compile_batch().fingerprint
        == "f8fbe74fb7c56e39e5c0d8bd2c28b249f50ec3d80e201afde3e6a449084d9398"
    )
    encoded = program.to_project().model_dump_json()
    assert (
        hashlib.sha256(encoded.encode()).hexdigest()
        == "03786a8ab7a8797274fc50b19f69ffa57c3600961ef7a35a7bbf670050277bf8"
    )
    assert "update_semantics" not in encoded
    explicit = program.with_update_semantics("legacy_append").to_project()
    assert explicit.model_dump_json() == encoded
    latest = program.with_update_semantics("latest_relation")
    document = latest.to_project()
    assert document.model_dump()["graph"]["update_semantics"] == "latest_relation"
    assert latest.compile_stream().fingerprint != program.compile_stream().fingerprint
    runtime = cf.Runtime()
    assert (
        runtime._compile_stream_graph_project(document.model_dump_json()).fingerprint
        == latest.compile_stream(runtime).fingerprint
    )


def test_batch_latest_same_plan_keeps_independent_state():
    program = _program().with_update_semantics("latest_relation").for_mode("batch")
    runtime = cf.Runtime()
    plan = program.compile_batch(runtime)
    for data, expected in [(pa.table({"x": [1]}), 1), (pa.table({"x": [2, 3]}), 5)]:
        assert program.compile_batch(runtime) is plan
        result = plan.execute({plan.external_inputs[0]: cf.Batch.from_pyarrow(data)})
        assert result.outputs[plan.external_outputs[0]].to_pyarrow().to_pydict() == {
            "total": [expected]
        }


@pytest.mark.parametrize(
    "query",
    [
        "SELECT DISTINCT total FROM input",
        "SELECT total FROM input ORDER BY total",
        "SELECT total FROM input LIMIT 1",
    ],
)
def test_latest_rejects_unproved_snapshot_transform(query):
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    total = source.sql("SELECT SUM(x) AS total FROM input")
    program = cf.Program(
        "unproved-transform", mode="stream", outputs={"risk.total": total.sql(query)}
    ).with_update_semantics("latest_relation")
    report = program.analyze().query
    edge = next(
        edge
        for edge in report.edges
        if edge.path == 'outputs["risk.total"].sql.tables.input'
    )
    assert edge.stream.support == "unknown"
    with pytest.raises(cf.CompileError, match="update_semantics_unproved"):
        program.compile_stream()


def test_exported_latest_gate_preserves_native_error_report():
    latest = _program(
        chain=True, query="SELECT SUM(DISTINCT total) AS total FROM input"
    ).with_update_semantics("latest_relation")
    runtime = cf.Runtime()
    project = latest.to_project(runtime)
    with pytest.raises(
        cf.CompileError, match="update_interpretation_mismatch"
    ) as caught:
        runtime.compile_stream_project(project.model_dump_json())
    report = caught.value.query_analysis
    assert report.update_semantics == "latest_relation"
    assert report.request is None
    assert any(
        issue.path.startswith("graph.edges[")
        and issue.code == "update_interpretation_mismatch"
        for issue in report.diagnostics
    )
    with pytest.raises((AttributeError, TypeError)):
        report.update_semantics = "legacy_append"


def test_report_wire_accepts_legacy_display_and_rejects_unknown_semantics():
    import json

    from calc_flow.symbolic.query_analysis import _from_dict

    encoded = cf.Runtime()._inner._analyze_query_graph(
        json.dumps({"nodes": [], "edges": [], "options": {"portability": None}})
    )
    wire = json.loads(encoded)
    assert _from_dict(wire).update_semantics == "legacy_append"
    wire["schema_version"] = wire["analysis_revision"] = 1
    del wire["update_semantics"]
    assert _from_dict(wire).update_semantics == "legacy_append"
    wire["update_semantics"] = "latest_relation"
    with pytest.raises(ValueError, match="invalid_literal"):
        _from_dict(wire)
    wire["schema_version"] = wire["analysis_revision"] = 2
    wire["update_semantics"] = "future"
    with pytest.raises(ValueError, match="invalid_literal"):
        _from_dict(wire)


@pytest.mark.parametrize("version,revision", [(1, 1), (2, 2), (2, 3)])
def test_latest_rejects_stale_report_before_preparation_cache(
    monkeypatch, version, revision
):
    from dataclasses import replace

    program = _program()
    runtime = cf.Runtime()
    program.compile_stream(runtime)
    analysis = program.analyze(runtime)
    stale = replace(analysis.query, schema_version=version, analysis_revision=revision)
    monkeypatch.setattr(
        cf.Program,
        "analyze",
        lambda *args, **kwargs: replace(analysis, query=stale),
    )
    with pytest.raises(cf.CompileError, match="matching v2 analysis"):
        program.with_update_semantics("latest_relation").compile_stream(runtime)


def test_shared_snapshot_keeps_each_offending_consumption_edge():
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    total = source.sql("SELECT SUM(x) AS total FROM input")
    query = "SELECT SUM(DISTINCT total) AS total FROM input"
    program = cf.Program(
        "shared",
        mode="stream",
        outputs={
            "ok": total.select("total"),
            "bad.one": total.sql(query),
            "bad.two": total.filter(total["total"] >= cf.lit(0)).sql(query),
        },
    ).with_update_semantics("latest_relation")
    runtime = cf.Runtime()
    first = program.analyze(runtime).query
    assert first == program.analyze(runtime).query
    assert {
        issue.path
        for issue in first.diagnostics
        if issue.code == "update_interpretation_mismatch"
    } == {
        'outputs["bad.one"].sql.tables.input',
        'outputs["bad.two"].sql.tables.input',
    }
    with pytest.raises(cf.CompileError, match="update_interpretation_mismatch"):
        program.compile_stream(runtime)


def test_latest_preserves_existing_snapshot_sql_terminal_consumer():
    source = cf.table_input("events", schema=pa.schema([("x", pa.int64())]))
    total = source.sql("SELECT SUM(x) AS total FROM input")
    output = cf.snapshot_sql("SELECT total FROM s", s=total)
    latest = cf.Program(
        "snapshot", mode="stream", outputs={"result": output}
    ).with_update_semantics("latest_relation")
    assert _observe(latest, [pa.table({"x": [1]}), pa.table({"x": [2, 3]})]) == [
        {"total": [1]},
        {"total": [6]},
    ]


def test_lowering_rewrite_preserves_update_choice(monkeypatch):
    import importlib

    forwarder = importlib.import_module("calc_flow.symbolic.lower.macd_forward")

    def copied_declarations(program, analyzer):
        return cf.Program(
            program.name,
            mode=program.mode,
            inputs=program.inputs,
            outputs=program.outputs,
        )

    monkeypatch.setattr(forwarder, "_forward_native_macd", copied_declarations)
    latest = _program().with_update_semantics("latest_relation")
    assert (
        latest.to_project().model_dump()["graph"]["update_semantics"]
        == "latest_relation"
    )
