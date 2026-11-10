from __future__ import annotations

import pyarrow as pa
import pytest

import calc_flow as cf


def _request(history="accepted_history"):
    return cf.PortabilityRequest(
        comparison="final_relation", history=history, cutoff="all_sources_eof"
    )


def test_row_local_report_and_checked_compile_preserve_identity():
    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    output = source.filter(source["x"] > cf.lit(0)).select("x")
    program = cf.Program("positive", engine="sql", outputs={"result": output})
    runtime = cf.Runtime()
    before = program.fingerprint
    ordinary = program.compile_batch(runtime)
    report = program.analyze(
        runtime, options=cf.AnalysisOptions(portability=_request())
    )
    assert report.issues == ()
    assert report.query.portability == "supported"
    assert report.query.schema_version == report.query.analysis_revision == 1
    assert report.query.nodes[-1].stream.history == "current_batch"
    checked = program.compile_batch(runtime, portability=_request())
    assert checked.fingerprint == ordinary.fingerprint
    assert program.fingerprint == before
    assert program.collect(
        {"values": pa.table({"x": [-1, 2]})}, runtime=runtime, portability=_request()
    )["result"].to_pydict() == {"x": [2]}


@pytest.mark.parametrize("engine", ["sql", "streaming"])
def test_snapshot_append_edge_rejects_even_after_warming_compile_cache(engine):
    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    first = source.sql("SELECT SUM(x) AS total FROM input")
    output = first.select("total").sql("SELECT SUM(total) AS total FROM input")
    program = cf.Program("sum_chain", engine=engine, outputs={"total": output})
    runtime = cf.Runtime()
    ordinary = (
        program.compile_batch(runtime)
        if engine == "sql"
        else program.compile_stream(runtime)
    )
    report = program.analyze(runtime).query
    assert report.portability is None
    first_facts = next(
        node for node in report.nodes if node.path.endswith(".project.value")
    )
    assert first_facts.stream.history == "accepted_history"
    assert first_facts.stream.update == "complete_snapshot"
    edge = next(
        edge for edge in report.edges if edge.path == "outputs.total.sql.tables.input"
    )
    assert edge.stream.producer_update == "complete_snapshot"
    assert edge.stream.consumer_interpretation == "append"
    for history in ("accepted_history", "latest_relation"):
        with pytest.raises(
            cf.CompileError, match="update_interpretation_mismatch"
        ) as caught:
            (
                program.compile_batch(runtime, portability=_request(history))
                if engine == "sql"
                else program.compile_stream(runtime, portability=_request(history))
            )
        assert caught.value.query_analysis.portability == "unsupported"
        assert any(
            d.path == edge.path and d.code == "update_interpretation_mismatch"
            for d in caught.value.query_analysis.diagnostics
        )
    cached = (
        program.compile_batch(runtime)
        if engine == "sql"
        else program.compile_stream(runtime)
    )
    if engine == "sql":
        assert cached is ordinary
    else:
        assert cached is not ordinary
        assert cached.fingerprint == ordinary.fingerprint


def test_public_facts_copy_caller_sequences():
    entities = ["symbol"]
    identity = cf.QueryTemporalIdentity(entities, "ts", ["seq"])
    entities.append("mutated")
    assert identity.entity_by == ("symbol",)
    assert identity.sequence_by == ("seq",)
    with pytest.raises(AttributeError):
        identity.event_time = "other"


@pytest.mark.parametrize("entry", ["collect_async", "stream", "execute"])
def test_strict_execution_gates_before_wrapping_inputs_or_starting_jobs(
    monkeypatch, entry
):
    import asyncio
    import importlib
    import threading

    compute = importlib.import_module("calc_flow.compute")
    stream = importlib.import_module("calc_flow.stream")
    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    output = source.sql("SELECT DISTINCT x FROM input")
    program = cf.Program(
        "strict",
        engine="sql" if entry != "stream" else "streaming",
        outputs={"result": output},
    )
    calls = {"rows": 0, "bindings": 0, "jobs": 0, "source": 0}
    threads = []
    original = cf.Program.analyze

    def observe(self, *args, **kwargs):
        threads.append(threading.get_ident())
        return original(self, *args, **kwargs)

    def forbid_rows(*args, **kwargs):
        calls["rows"] += 1
        raise AssertionError("input wrapped before strict check")

    def forbid_bindings(*args, **kwargs):
        calls["bindings"] += 1
        raise AssertionError("bindings created before strict check")

    async def forbid_job(*args, **kwargs):
        calls["jobs"] += 1
        raise AssertionError("job started before strict check")

    class Unread:
        def __aiter__(self):
            calls["source"] += 1
            raise AssertionError("source opened during analysis")

    monkeypatch.setattr(cf.Program, "analyze", observe)
    monkeypatch.setattr(compute, "_collect_batches", forbid_rows)
    monkeypatch.setattr(stream.StreamResults, "_bindings", forbid_bindings)
    monkeypatch.setattr(cf.StreamingRunner, "start_async", forbid_job)

    async def run():
        if entry == "collect_async":
            await program.collect_async({"values": object()}, portability=_request())
        elif entry == "stream":
            async with program.stream({"values": Unread()}, portability=_request()):
                raise AssertionError("strict stream started")
        else:
            await asyncio.to_thread(
                program.execute, {"values": object()}, portability=_request()
            )

    main = threading.get_ident()
    with pytest.raises(cf.CompileError):
        asyncio.run(run())
    assert calls == {"rows": 0, "bindings": 0, "jobs": 0, "source": 0}
    assert threads and all(value != main for value in threads)


def test_explain_accepts_options_and_renders_the_same_query_report():
    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    program = cf.Program("scope", engine="sql", outputs={"result": source})
    text = program.explain(options=cf.AnalysisOptions(portability=_request()))
    assert "schema_version 1 analysis_revision 1" in text
    assert "current_batch" in text
    assert "same accepted" in text


@pytest.mark.parametrize("kind", ["column", "sql"])
def test_existing_schema_errors_block_strict_collect_before_input_wrapping(
    monkeypatch, kind
):
    import importlib

    compute = importlib.import_module("calc_flow.compute")
    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    output = (
        source.select("missing")
        if kind == "column"
        else source.sql("SELECT missing FROM input")
    )
    program = cf.Program("bad", engine="sql", outputs={"result": output})
    report = program.analyze(options=cf.AnalysisOptions(portability=_request()))
    assert report.query.portability == "unsupported"
    assert report.issues
    assert any(node.batch.support == "unsupported" for node in report.query.nodes)

    def forbidden(*args, **kwargs):
        raise AssertionError("wrapped data before schema rejection")

    monkeypatch.setattr(compute, "_collect_batches", forbidden)
    with pytest.raises(cf.CompileError) as caught:
        program.collect({"values": object()}, portability=_request())
    assert caught.value.query_analysis.portability == "unsupported"


def test_shared_and_special_name_paths_are_stable_and_unique():
    source = cf.table_input('event.s"\n', schema=pa.schema([("x", pa.int64())]))
    shared = source.sql("SELECT x FROM input")
    program = cf.Program(
        "paths", engine="sql", outputs={"z": shared.select("x"), "a.b": shared}
    )
    runtime = cf.Runtime()
    first = program.analyze(runtime).query
    second = program.analyze(runtime).query
    assert first == second
    paths = [node.path for node in first.nodes]
    assert paths == sorted(paths, key=lambda path: path.encode())
    assert len(paths) == len(set(paths))
    assert 'inputs["event.s\\"\\n"]' in paths
    assert len(first.edges) == 2
    assert any(edge.source_path == 'inputs["event.s\\"\\n"]' for edge in first.edges)
    assert all(
        edge.source_path in paths and edge.target_path in paths for edge in first.edges
    )


def test_legacy_results_and_native_identities_survive_analysis():
    import asyncio

    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    first = source.sql("SELECT SUM(x) AS total FROM input")
    runtime = cf.Runtime()
    batch = cf.Program("legacy", engine="sql", outputs={"total": first})
    streaming = cf.Program("legacy", engine="streaming", outputs={"total": first})
    assert (
        batch.fingerprint
        == streaming.fingerprint
        == "20a49fa51a1c423a8bd4cadbb6b95bc142292f21f3294e4970efe9b45eddb7de"
    )
    plan = batch.compile_batch(runtime)
    before = batch.to_project(runtime).model_dump_json()
    assert (
        plan.fingerprint
        == "0044b2322eff73664b78a51f577af6293c86bed2e4d2cd6ba93b1fc214f733f6"
    )
    assert (
        streaming.compile_stream(runtime).fingerprint
        == "3bb9b534577cd7034059bcb870d041aff073e87bdc4888e49731ea06e78a39d5"
    )
    report = batch.analyze(runtime).query
    root = next(node for node in report.nodes if node.path == "outputs.total")
    assert root.batch.history == "current_batch"
    assert root.batch.finality == "per_batch"
    assert root.stream.history == "accepted_history"
    assert root.stream.finality == "revisable_snapshot"
    assert batch.to_project(runtime).model_dump_json() == before
    assert batch.compile_batch(runtime) is plan
    values = [
        plan.execute({"input": cf.Batch.from_pyarrow(pa.table({"x": xs}))})
        .outputs["output"]
        .to_pyarrow()
        .to_pydict()
        for xs in ([1], [2, 3])
    ]
    assert values == [{"total": [1]}, {"total": [5]}]

    async def collect(program, batches):
        async def feed():
            for value in batches:
                yield value

        async with program.stream({"values": feed()}) as results:
            return [item.table.to_pydict() async for item in results]

    tables = [pa.table({"x": [1]}), pa.table({"x": [2, 3]})]
    assert asyncio.run(collect(streaming, tables)) == [{"total": [1]}, {"total": [6]}]
    chain = cf.Program(
        "legacy_chain",
        engine="streaming",
        outputs={"total": first.sql("SELECT SUM(total) AS total FROM input")},
    )
    chain.analyze(runtime)
    assert asyncio.run(collect(chain, tables)) == [{"total": [1]}, {"total": [7]}]

    for query, expected in [
        ("SELECT x FROM input ORDER BY x LIMIT 1", [{"x": [1]}, {"x": [2]}]),
        ("SELECT DISTINCT x FROM input ORDER BY x", [{"x": [1, 3]}, {"x": [2]}]),
        (
            "SELECT row_number() OVER (ORDER BY x) AS n FROM input",
            [{"n": [1, 2]}, {"n": [1]}],
        ),
    ]:
        program = cf.Program(
            "per_batch", engine="streaming", outputs={"result": source.sql(query)}
        )
        program.analyze(runtime)
        assert (
            asyncio.run(
                collect(program, [pa.table({"x": [3, 1]}), pa.table({"x": [2]})])
            )
            == expected
        )


def test_query_diagnostic_codes_are_in_the_public_symbolic_vocabulary():
    from calc_flow.symbolic.errors import ISSUE_CODES

    assert {
        "history_scope_mismatch",
        "update_interpretation_mismatch",
        "sql_semantics_unproven",
        "query_semantics_unproven",
        "finality_unproven",
        "row_order_unproven",
    } <= set(ISSUE_CODES)


def test_strict_stream_does_not_iterate_a_sequence_source_before_gate():
    import asyncio
    from collections.abc import Sequence

    reads = []

    class SourceSequence(Sequence):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            reads.append(index)
            raise AssertionError("source sequence pulled before strict check")

    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    program = cf.Program(
        "sequence",
        engine="streaming",
        outputs={"result": source.sql("SELECT DISTINCT x FROM input")},
    )

    async def run():
        async with program.stream({"values": SourceSequence()}, portability=_request()):
            raise AssertionError("strict stream started")

    with pytest.raises(cf.CompileError):
        asyncio.run(run())
    assert reads == []


def test_unknown_upstream_updates_are_not_defaulted_to_append():
    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    output = source.sql(
        "WITH chosen AS (SELECT x FROM input) SELECT x FROM chosen"
    ).select("x")
    report = (
        cf.Program("unknown", engine="sql", outputs={"result": output}).analyze().query
    )
    root = next(node for node in report.nodes if node.path == "outputs.result")
    assert root.stream.update == "unknown"
    assert root.stream.finality == "unknown"


def test_per_batch_sql_on_snapshots_does_not_invent_a_raw_history_counterexample():
    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    output = source.sql("SELECT SUM(x) AS total FROM input").sql(
        "SELECT DISTINCT total FROM input"
    )
    program = cf.Program("snapshot_distinct", engine="sql", outputs={"result": output})
    report = program.analyze(options=cf.AnalysisOptions(portability=_request())).query
    root = next(node for node in report.nodes if node.path == "outputs.result")
    assert root.stream.update == "complete_snapshot"
    assert report.portability == "unknown"
    assert not any(
        issue.code == "history_scope_mismatch" for issue in report.diagnostics
    )


def test_snapshot_facts_keep_latest_relation_even_in_rejected_batch_mode():
    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    output = cf.snapshot_sql("SELECT SUM(x) AS total FROM input", input=source)
    program = cf.Program("latest", engine="streaming", outputs={"result": output})
    report = program.analyze(mode="batch").query
    root = next(node for node in report.nodes if node.path == "outputs.result")
    assert root.batch.support == "unsupported"
    assert root.batch.history == root.stream.history == "latest_relation"
    assert root.stream.update == "complete_snapshot"
    assert root.schema == (cf.Field("total", "int64", True),)


def test_successful_strict_async_execution_and_ordering_proof_boundaries():
    import asyncio

    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    output = source.sql("SELECT x FROM input WHERE x > 0 ORDER BY x")
    runtime = cf.Runtime()
    batch = cf.Program("rows", engine="sql", outputs={"result": output})
    streaming = cf.Program("rows", engine="streaming", outputs={"result": output})
    request = cf.PortabilityRequest(
        comparison="per_batch", history="current_batch", cutoff="batch_boundary"
    )

    async def run():
        actual = await batch.collect_async(
            {"values": pa.table({"x": [-1, 2]})}, runtime=runtime, portability=request
        )
        assert actual["result"].to_pydict() == {"x": [2]}

        async def feed():
            yield pa.table({"x": [-1, 2]})

        async with streaming.execute(
            {"values": feed()}, runtime=runtime, portability=request
        ) as results:
            assert [item.table.to_pydict() async for item in results] == [{"x": [2]}]

    asyncio.run(run())
    facts = next(
        node
        for node in batch.analyze(runtime).query.nodes
        if node.path == "outputs.result"
    )
    assert facts.stream.ordering.keys == (cf.QuerySortKey("x", "ascending", "last"),)
    assert facts.stream.ordering.temporal_identity is None
    declared = cf.PortabilityRequest(
        comparison="per_batch",
        history="current_batch",
        cutoff="batch_boundary",
        ordering="declared",
    )
    with pytest.raises(cf.CompileError, match="row_order_unproven") as caught:
        batch.compile_batch(runtime, portability=declared)
    assert caught.value.query_analysis.portability == "unknown"


@pytest.mark.parametrize(
    "query,comparison",
    [
        ("SELECT x FROM input ORDER BY x LIMIT 1", "per_batch"),
        ("SELECT x FROM input LIMIT 0", "final_relation"),
    ],
)
def test_unproved_selection_is_not_granted_or_given_an_invented_counterexample(
    query, comparison
):
    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    program = cf.Program(
        "selection", engine="sql", outputs={"result": source.sql(query)}
    )
    portability = (
        cf.PortabilityRequest(
            comparison="per_batch", history="current_batch", cutoff="batch_boundary"
        )
        if comparison == "per_batch"
        else _request()
    )
    report = program.analyze(options=cf.AnalysisOptions(portability=portability)).query
    assert report.portability == "unknown"
    assert any(issue.code == "sql_semantics_unproven" for issue in report.diagnostics)


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"comparison": True}, TypeError),
        ({"history": "unknown"}, ValueError),
        ({"cutoff": "all_sources_eof"}, ValueError),
        ({"ordering": "automatic"}, ValueError),
    ],
)
def test_invalid_requests_are_configuration_errors(changes, error):
    fields = dict(
        comparison="per_batch", history="current_batch", cutoff="batch_boundary"
    )
    with pytest.raises(error, match="options.portability"):
        cf.PortabilityRequest(**{**fields, **changes})


def test_old_analysis_result_constructor_remains_compatible():
    old = cf.AnalysisResult("batch", "fingerprint", "session", 1, ())
    assert old.query is None
    with pytest.raises(TypeError, match="PortabilityRequest"):
        cf.AnalysisOptions(portability=old)


@pytest.mark.parametrize(
    "dtype,spelling", [(pa.time32("s"), "time32[s]"), (pa.time64("us"), "time64[us]")]
)
def test_query_schema_keeps_portable_time_type_spellings(dtype, spelling):
    source = cf.table_input(
        "times", schema=pa.schema([pa.field("clock", dtype, nullable=False)])
    )
    report = (
        cf.Program("schema", engine="sql", outputs={"result": source}).analyze().query
    )
    root = next(node for node in report.nodes if node.path == "inputs.times")
    assert root.schema == (cf.Field("clock", spelling, False),)
