from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

import calc_flow as cf
from calc_flow.capabilities import ProviderPort
from calc_flow.symbolic.lower import lower_program_document


def _input(
    name: str, *, key_type: str = "string", nullable: bool = False
) -> cf.TableExpr:
    return cf.table_input(
        name,
        schema=[
            cf.Field("symbol", key_type, nullable),
            cf.Field("time", "timestamp[us, UTC]", False),
            cf.Field("sequence", "uint64", False),
            cf.Field("price", "float64", False),
        ],
        entity_by=["symbol"],
        event_time="time",
        sequence_by=["sequence"],
    )


def _join(left=None, right=None, **options):
    return cf.table.stream_asof_join(
        _input("trades") if left is None else left,
        _input("quotes") if right is None else right,
        tolerance=timedelta(microseconds=10),
        limits=cf.AsofStateLimits(100, 1_000_000),
        **options,
    )


@pytest.fixture
def runtime(monkeypatch):
    value = cf.Runtime()
    snapshot = value.capabilities()
    inner = next(
        operator for operator in snapshot.operators if operator.kind == "stream_join"
    )
    asof = replace(
        inner,
        kind="stream_asof_join",
        version="1",
        modes=("stream",),
        finality="group_final_append_only",
        requires_datafusion=True,
        stateful=True,
        microbatch_invariant=True,
        requires_watermark=True,
        checkpoint_support="checkpointed_stateful",
        state_version=1,
        state_layouts=(1,),
        deterministic=True,
        replay_safe=True,
    )
    capabilities = replace(
        snapshot,
        operators=(
            *(
                operator
                for operator in snapshot.operators
                if operator.kind != "stream_asof_join"
            ),
            asof,
        ),
    )
    monkeypatch.setattr(cf.Runtime, "capabilities", lambda _self: capabilities)
    return value


def test_asof_namespace_and_fluent_forms_share_one_immutable_primitive() -> None:
    left, right = _input("trades"), _input("quotes")
    declared = _join(left, right)
    fluent = left.stream_asof_join(
        right,
        tolerance=timedelta(microseconds=10),
        limits=cf.AsofStateLimits(100, 1_000_000),
    )
    assert declared.identical(fluent)
    assert "stream_asof_join@1" in declared.explain()
    assert "tolerance_micros=10" in declared.explain()
    keys = (["sequence"], ["sequence"])
    overridden = _join(left, right, keys=keys)
    keys[0].clear()
    assert overridden.identical(_join(left, right, keys=(["sequence"], ["sequence"])))
    assert not declared.identical(overridden)


def test_asof_analysis_preserves_left_identity_and_makes_all_right_fields_nullable(
    runtime,
) -> None:
    from calc_flow.symbolic.analyzer import _run

    result = _join(prefixes=("trade", "quote"))
    program = cf.Program("matching", outputs={"matched": result})
    analyzer, _ = _run(program, runtime, "stream")
    assert analyzer.issues == ()
    facts = analyzer.table(result._node, "outputs.matched")
    assert facts.event_time == "trade__time"
    assert facts.entity_by == ("trade__symbol",)
    assert facts.sequence_by == ("trade__sequence",)
    assert facts.state == frozenset({"stream_asof_join"})
    assert [field.nullable for field in facts.schema] == [False] * 4 + [True] * 4


@pytest.mark.parametrize(
    "options",
    [
        {"keys": ([], ["symbol"])},
        {"keys": "ab"},
        {"keys": (["symbol"],)},
        {"prefixes": ("only",)},
        {"prefixes": "ab"},
    ],
)
def test_asof_declaration_rejects_malformed_side_settings(options) -> None:
    with pytest.raises((TypeError, ValueError)):
        _join(**options)


@pytest.mark.parametrize(
    "columns", [("symbol", "sequence", "price"), ("symbol", "time", "price")]
)
def test_asof_declaration_requires_current_temporal_metadata(columns) -> None:
    with pytest.raises((TypeError, ValueError), match="left.*(event_time|sequence_by)"):
        _join(_input("trades").select(*columns))


@pytest.mark.parametrize("key_type", ["float64", "time32[s]"])
def test_asof_analysis_rejects_unsupported_identity_key_types(
    runtime, key_type
) -> None:
    program = cf.Program(
        "bad",
        outputs={
            "matched": _join(
                _input("trades", key_type=key_type), _input("quotes", key_type=key_type)
            )
        },
    )
    assert any(
        issue.code == "type_mismatch" and ".keys[0]" in issue.path
        for issue in program.analyze(runtime, mode="stream").issues
    )


def test_asof_analysis_rejects_nullable_or_mismatched_keys(runtime) -> None:
    nullable = cf.Program(
        "nullable", outputs={"matched": _join(_input("trades", nullable=True))}
    )
    assert any(
        issue.code == "type_mismatch" and "left.keys[0]" in issue.path
        for issue in nullable.analyze(runtime, mode="stream").issues
    )
    mismatch = cf.Program(
        "mismatch", outputs={"matched": _join(_input("trades", key_type="int64"))}
    )
    assert any(
        issue.code == "type_mismatch"
        for issue in mismatch.analyze(runtime, mode="stream").issues
    )


def test_asof_analysis_rejects_batch_and_wrong_temporal_type(runtime) -> None:
    batch = cf.Program("batch", outputs={"matched": _join()}).analyze(
        runtime, mode="batch"
    )
    assert any(issue.code == "unsupported_mode" for issue in batch.issues)
    bad = cf.table_input(
        "trades",
        schema=[
            cf.Field("symbol", "string", False),
            cf.Field("time", "timestamp[ms]", False),
            cf.Field("sequence", "uint64", False),
        ],
        entity_by=["symbol"],
        event_time="time",
        sequence_by=["sequence"],
    )
    result = cf.Program("bad", outputs={"matched": _join(bad)}).analyze(
        runtime, mode="stream"
    )
    assert any(
        issue.code == "ordering_required" and "left.event_time" in issue.path
        for issue in result.issues
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"version": "2"},
        {"modes": ("batch", "stream")},
        {"finality": "unproven"},
        {"requires_datafusion": False},
        {"stateful": False},
        {"microbatch_invariant": False},
        {"requires_watermark": False},
        {"checkpoint_support": "unproven", "state_version": None, "state_layouts": ()},
        {"state_version": 2, "state_layouts": (1, 2)},
        {"state_layouts": (1, 2)},
        {"deterministic": False},
        {"replay_safe": False},
        {"input_ports": (ProviderPort("input", "table", True),)},
        {"output_ports": (ProviderPort("output", "table", False),)},
    ],
)
def test_asof_capability_gate_checks_every_independent_fact(
    runtime, monkeypatch, changes
) -> None:
    snapshot = runtime.capabilities()
    altered = replace(
        snapshot,
        operators=tuple(
            replace(operator, **changes)
            if operator.kind == "stream_asof_join"
            else operator
            for operator in snapshot.operators
        ),
    )
    monkeypatch.setattr(cf.Runtime, "capabilities", lambda _self: altered)
    result = cf.Program("gate", outputs={"matched": _join()}).analyze(
        runtime, mode="stream"
    )
    assert any(issue.code == "capability_mismatch" for issue in result.issues)


def test_asof_forged_temporal_metadata_is_not_trusted(runtime) -> None:
    from calc_flow.symbolic.nodes import CMap, CStr, build

    joined = _join()
    spec = joined._node.attr("spec")
    left = spec.get("left")
    forged_left = CMap.from_mapping(
        dict(left.entries) | {"event_time": CStr("sequence")}
    )
    forged = cf.TableExpr(
        build(
            "stream_asof_join",
            joined._node.args,
            {"spec": CMap.from_mapping(dict(spec.entries) | {"left": forged_left})},
        )
    )
    result = cf.Program("forged", outputs={"matched": forged}).analyze(
        runtime, mode="stream"
    )
    assert any(issue.code == "ordering_required" for issue in result.issues)


def test_asof_lowering_uses_independent_native_kind_and_logical_stream_bindings(
    runtime,
) -> None:
    from calc_flow.symbolic.lower.bindings import _BatchBindings

    bindings = _BatchBindings()
    result = _join(prefixes=("trade", "quote")).select("trade__symbol", "quote__price")
    document = lower_program_document(
        cf.Program("trades", outputs={"matched": result}),
        runtime,
        "stream",
        _bindings=bindings,
    )
    asof = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "stream_asof_join"
    ]
    assert len(asof) == 1
    assert asof[0]["operator"]["spec"]["left"]["keys"] == ["symbol"]
    assert asof[0]["operator"]["spec"]["right"]["sequence_by"] == ["sequence"]
    assert asof[0]["operator"]["spec"]["tolerance_micros"] == 10
    assert list(bindings.names()[0]) == ["quotes", "trades"] or set(
        bindings.names()[0]
    ) == {"quotes", "trades"}
    assert bindings.names()[1] == {"matched": "output"}


def test_asof_dag_shares_one_owner_per_digest_and_keeps_chains_independent(
    runtime,
) -> None:
    left, right, metadata = _input("trades"), _input("quotes"), _input("metadata")
    joined = _join(left, right)
    chained = _join(joined, metadata)
    distinct = _join(left, right, prefixes=("trade", "quote"))
    program = cf.Program(
        "dag",
        outputs={
            "one": joined,
            "two": joined.select("left__symbol"),
            "chained": chained,
            "distinct": distinct,
            "bypass": left,
        },
    )
    document = lower_program_document(program, runtime, "stream")
    nodes = document["graph"]["nodes"]
    assert (
        len([node for node in nodes if node["operator"]["kind"] == "stream_asof_join"])
        == 3
    )
    assert any(
        edge["source_node"].startswith("cf_stream_asof_join_")
        and edge["target_node"].startswith("cf_stream_asof_join_")
        for edge in document["graph"]["edges"]
    )
    assert {node["id"] for node in nodes} >= {
        "one",
        "two",
        "chained",
        "distinct",
        "bypass",
    }


def test_asof_post_join_rolling_lowers_after_finality_boundary(runtime) -> None:
    joined = _join()
    result = joined.with_columns(previous=cf.ts.lag(joined["left__price"]))
    document = lower_program_document(
        cf.Program("rolling", outputs={"result": result}), runtime, "stream"
    )
    nodes = document["graph"]["nodes"]
    assert (
        len([node for node in nodes if node["operator"]["kind"] == "stream_asof_join"])
        == 1
    )
    assert len([node for node in nodes if node["operator"]["kind"] == "rolling"]) == 1


def test_asof_and_inner_can_lower_in_both_directions(runtime) -> None:
    left, right = _input("trades"), _input("quotes")
    joined = cf.table.stream_join(
        left,
        right,
        left_keys=["symbol"],
        right_keys=["symbol"],
        left_event_time="time",
        right_event_time="time",
        bounds=cf.JoinTimeBounds(timedelta(microseconds=10), timedelta()),
        limits=cf.JoinStateLimits(100, 1_000_000, 1_000),
        output_entity_by=["left__symbol"],
        output_event_time="left__time",
        output_sequence_by=["left__sequence", "right__sequence"],
    )
    asof = _join(joined, _input("reference"))
    inner_after = cf.table.stream_join(
        _join(left, right),
        _input("reference"),
        left_keys=["left__symbol"],
        right_keys=["symbol"],
        left_event_time="left__time",
        right_event_time="time",
        bounds=cf.JoinTimeBounds(timedelta(microseconds=10), timedelta()),
        limits=cf.JoinStateLimits(100, 1_000_000, 1_000),
    )
    for output in (asof, inner_after):
        document = lower_program_document(
            cf.Program("mixed", outputs={"result": output}), runtime, "stream"
        )
        kinds = [node["operator"]["kind"] for node in document["graph"]["nodes"]]
        assert kinds.count("stream_asof_join") == 1
        assert kinds.count("stream_join") == 1


def test_asof_explain_states_matching_finality_resources_and_unique_ownership(
    runtime,
) -> None:
    joined = _join()
    explanation = cf.Program("explain", outputs={"a": joined, "b": joined}).explain(
        runtime, mode="stream"
    )
    for fact in (
        "stream_asof_join state_stages 1",
        "backward",
        "tolerance_micros=10",
        "left.keys=symbol",
        "right.sequence_by=sequence",
        "late_policy=error",
        "left_preserving=true",
        "right_nullable=true",
        "watermarks>left_time",
        "output_time=left.time",
        "frontier_lag_micros=1",
        "state_version=1",
        "state_layout=1",
        "max_state_rows=100",
        "max_state_bytes=1000000",
        "workspace_bytes=1000000",
        "delivery=source_and_sink_dependent",
    ):
        assert fact in explanation


def test_asof_rejects_stateful_input_before_join(runtime) -> None:
    left = _input("trades")
    left = left.with_columns(previous=cf.ts.lag(left["price"]))
    result = cf.Program("unsupported", outputs={"result": _join(left)}).analyze(
        runtime, mode="stream"
    )
    assert any(issue.code == "capability_mismatch" for issue in result.issues)


def test_asof_rejects_event_window_chains_but_accepts_independent_branch(
    runtime,
) -> None:
    joined = _join()
    event = cf.window.tumbling(joined, event_time="left__time", size_micros=10)
    issues = (
        cf.Program("window_after", outputs={"result": event})
        .analyze(runtime, mode="stream")
        .issues
    )
    assert any(issue.code == "capability_mismatch" for issue in issues)
    source = _input("window_input")
    separate = cf.window.tumbling(
        source,
        event_time="time",
        size_micros=10,
        aggregates=[cf.window.count("price", output="count")],
    )
    program = cf.Program(
        "independent", outputs={"joined": joined, "windowed": separate}
    )
    document = lower_program_document(program, runtime, "stream")
    kinds = [node["operator"]["kind"] for node in document["graph"]["nodes"]]
    assert kinds.count("stream_asof_join") == 1
    assert kinds.count("window") == 1


def test_asof_sql_output_does_not_regain_temporal_metadata(runtime) -> None:
    sql = _join().sql("SELECT left__symbol, left__time, left__sequence FROM input")
    result = cf.Program("sql", outputs={"result": sql}).analyze(runtime, mode="stream")
    assert result.issues == ()
    with pytest.raises(ValueError, match="event_time"):
        _join(sql)


def test_asof_cross_section_compiles_after_finality_boundary(runtime) -> None:
    from calc_flow.symbolic.windows import CrossSectionGroup

    joined = _join()
    transformed = joined.with_columns(
        mean=cf.cs.demean(
            joined["left__price"],
            group=CrossSectionGroup(
                event_time=joined["left__time"], bucket=None, partition_by=()
            ),
        )
    )
    document = lower_program_document(
        cf.Program("cs", outputs={"result": transformed}), runtime, "stream"
    )
    assert {node["operator"]["kind"] for node in document["graph"]["nodes"]} >= {
        "stream_asof_join",
        "cross_section",
    }


def test_asof_unknown_options_are_rejected_by_both_entry_points() -> None:
    for call in (
        lambda: _join(direction="forward"),
        lambda: _input("trades").stream_asof_join(
            _input("quotes"),
            tolerance=timedelta(),
            limits=cf.AsofStateLimits(1, 1),
            allowed_lateness=1,
        ),
    ):
        with pytest.raises(TypeError, match="unexpected keyword"):
            call()


@pytest.mark.parametrize("version", [0, 2, True])
def test_asof_primitive_rejects_unknown_versions(version) -> None:
    from calc_flow.symbolic.nodes import build

    joined = _join()
    with pytest.raises(ValueError, match="unknown_primitive_version"):
        build(
            "stream_asof_join",
            joined._node.args,
            dict(joined._node.attrs.entries),
            version=version,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown": 1},
        {"tolerance_micros": True},
        {"tolerance_micros": 1.5},
        {"tolerance_micros": -1},
        {"late_policy": "ignore"},
    ],
)
def test_asof_analysis_reports_forged_invalid_spec_as_structured_issue(
    runtime, changes
) -> None:
    from calc_flow.symbolic.asof import _canonical, _wire
    from calc_flow.symbolic.nodes import CBool, CFloat, build

    joined = _join()
    wire = _wire(joined._node.attr("spec"))
    wire.update(
        {name: value for name, value in changes.items() if type(value) in (int, str)}
    )
    attrs = _canonical(wire)
    if (
        changes.get("tolerance_micros") is True
        or changes.get("tolerance_micros") == 1.5
    ):
        from calc_flow.symbolic.nodes import CMap

        value = CBool(True) if changes["tolerance_micros"] is True else CFloat(1.5)
        attrs = CMap.from_mapping(dict(attrs.entries) | {"tolerance_micros": value})
    forged = cf.TableExpr(build("stream_asof_join", joined._node.args, {"spec": attrs}))
    issues = (
        cf.Program("forged", outputs={"result": forged})
        .analyze(runtime, mode="stream")
        .issues
    )
    assert any(
        issue.code == "invalid_literal" and "spec" in issue.path for issue in issues
    )


def test_asof_sequence_requires_non_null_integer_or_string_total_order(runtime) -> None:
    for kind, nullable in (("bool", False), ("float64", False), ("uint64", True)):
        source = cf.table_input(
            "trades",
            schema=[
                cf.Field("symbol", "string", False),
                cf.Field("time", "timestamp[us, UTC]", False),
                cf.Field("sequence", kind, nullable),
            ],
            entity_by=["symbol"],
            event_time="time",
            sequence_by=["sequence"],
        )
        issues = (
            cf.Program("bad", outputs={"result": _join(source)})
            .analyze(runtime, mode="stream")
            .issues
        )
        assert any(
            issue.code == "type_mismatch" and "sequence_by" in issue.path
            for issue in issues
        )


def test_asof_capability_cannot_be_inferred_from_inner_offer(
    runtime, monkeypatch
) -> None:
    snapshot = runtime.capabilities()
    inner_only = replace(
        snapshot,
        operators=tuple(
            operator
            for operator in snapshot.operators
            if operator.kind != "stream_asof_join"
        ),
    )
    monkeypatch.setattr(cf.Runtime, "capabilities", lambda _self: inner_only)
    result = cf.Program("missing", outputs={"result": _join()}).analyze(
        runtime, mode="stream"
    )
    assert any(issue.code == "capability_mismatch" for issue in result.issues)


def test_asof_sql_lowering_preserves_shared_native_owner_and_input_bindings(
    runtime,
) -> None:
    from calc_flow.symbolic.lower.bindings import _BatchBindings

    joined = _join()
    sql = joined.select("left__symbol", "right__price").sql(
        "SELECT left__symbol, right__price FROM input"
    )
    program = cf.Program("sql_asof", outputs={"raw": joined, "sql": sql})
    bindings = _BatchBindings()
    document = lower_program_document(program, runtime, "stream", _bindings=bindings)
    kinds = [node["operator"]["kind"] for node in document["graph"]["nodes"]]
    assert kinds.count("stream_asof_join") == 1
    assert kinds.count("sql") == 1
    assert set(bindings.names()[0]) == {"trades", "quotes"}
    assert set(bindings.names()[1]) == {"raw", "sql"}


def test_asof_sql_and_independent_join_branches_keep_one_owner_each(runtime) -> None:
    first = _join()
    second = _join(_input("orders"), _input("reference"))
    program = cf.Program(
        "sql_branches",
        outputs={
            "first": first.sql("SELECT * FROM input"),
            "second": second,
            "shared": first.select("left__price"),
        },
    )
    document = lower_program_document(program, runtime, "stream")
    assert (
        len(
            [
                node
                for node in document["graph"]["nodes"]
                if node["operator"]["kind"] == "stream_asof_join"
            ]
        )
        == 2
    )


def test_asof_analysis_rejects_cross_prefix_output_name_collision(runtime) -> None:
    left = _input("trades")
    left = left.with_columns(y__symbol=left["symbol"])
    joined = _join(left, prefixes=("x", "x__y"))
    issues = (
        cf.Program("collision", outputs={"result": joined})
        .analyze(runtime, mode="stream")
        .issues
    )
    assert any(
        issue.code == "duplicate_name" and "schema" in issue.path for issue in issues
    )


def test_asof_explicit_keys_work_without_entity_grouping(runtime) -> None:
    schema = [
        cf.Field("symbol", "string", False),
        cf.Field("time", "timestamp[us, UTC]", False),
        cf.Field("sequence", "uint64", False),
    ]
    left = cf.table_input(
        "trades", schema=schema, event_time="time", sequence_by=["sequence"]
    )
    right = cf.table_input(
        "quotes", schema=schema, event_time="time", sequence_by=["sequence"]
    )
    result = _join(left, right, keys=(["symbol"], ["symbol"]))
    assert (
        cf.Program("explicit", outputs={"result": result})
        .analyze(runtime, mode="stream")
        .issues
        == ()
    )


def test_asof_forged_primitive_requires_exactly_two_table_operands(runtime) -> None:
    from calc_flow.symbolic.nodes import build

    joined = _join()
    for operands in (
        (),
        joined._node.args[:1],
        (*joined._node.args, _input("other")._node),
    ):
        malformed = cf.TableExpr(
            build("stream_asof_join", operands, dict(joined._node.attrs.entries))
        )
        result = cf.Program("malformed", outputs={"result": malformed}).analyze(
            runtime, mode="stream"
        )
        assert any(issue.code == "invalid_literal" for issue in result.issues)


def test_asof_logical_binding_names_survive_physical_name_collisions(runtime) -> None:
    from calc_flow.symbolic.lower.bindings import _BatchBindings

    joined = _join(_input("result"), _input("quotes"))
    bindings = _BatchBindings()
    document = lower_program_document(
        cf.Program("collision", outputs={"result": joined}),
        runtime,
        "stream",
        _bindings=bindings,
    )
    assert set(bindings.names()[0]) == {"result", "quotes"}
    source_endpoint = next(iter(bindings.inputs["result"]))
    assert source_endpoint[0] != "result"
    assert any(node["id"] == source_endpoint[0] for node in document["graph"]["nodes"])


def test_asof_builtin_capability_is_an_independent_complete_contract() -> None:
    from dataclasses import asdict

    operators = cf.Runtime().capabilities().operators
    offered = [
        operator for operator in operators if operator.kind == "stream_asof_join"
    ]
    assert len(offered) == 1
    assert asdict(offered[0]) == {
        "kind": "stream_asof_join",
        "version": "1",
        "input_ports": (
            {"name": "left", "kind": "table", "required": True},
            {"name": "right", "kind": "table", "required": True},
        ),
        "output_ports": ({"name": "output", "kind": "table", "required": True},),
        "modes": ("stream",),
        "finality": "group_final_append_only",
        "requires_datafusion": True,
        "stateful": True,
        "microbatch_invariant": True,
        "requires_watermark": True,
        "checkpoint_support": "checkpointed_stateful",
        "state_version": 1,
        "state_layouts": (1,),
        "deterministic": True,
        "replay_safe": True,
    }
    inner = next(operator for operator in operators if operator.kind == "stream_join")
    assert inner.finality == "unproven"
    assert inner.microbatch_invariant is False


@pytest.mark.parametrize("asof_side", ["left", "right"])
def test_inner_consumer_cannot_reinterpret_asof_output_event_time(runtime, asof_side):
    source = cf.table_input(
        "trades",
        schema=[
            cf.Field("symbol", "string", False),
            cf.Field("time", "timestamp[us, UTC]", False),
            cf.Field("other_time", "timestamp[us, UTC]", False),
            cf.Field("sequence", "uint64", False),
        ],
        entity_by=["symbol"],
        event_time="time",
        sequence_by=["sequence"],
    )
    asof = _join(source)
    reference = _input("reference")
    left, right = (asof, reference) if asof_side == "left" else (reference, asof)
    output = cf.table.stream_join(
        left,
        right,
        left_keys=["left__symbol" if asof_side == "left" else "symbol"],
        right_keys=["left__symbol" if asof_side == "right" else "symbol"],
        left_event_time="left__other_time" if asof_side == "left" else "time",
        right_event_time="left__other_time" if asof_side == "right" else "time",
        bounds=cf.JoinTimeBounds(timedelta(microseconds=10), timedelta()),
        limits=cf.JoinStateLimits(100, 1_000_000, 1_000),
    )
    result = cf.Program("wrong_time", outputs={"result": output}).analyze(
        runtime, mode="stream"
    )
    assert any(
        issue.code == "ordering_required"
        and issue.path.endswith(f"{asof_side}_event_time")
        for issue in result.issues
    )


@pytest.mark.parametrize(
    "removed", [None, "left__time", "left__symbol", "left__sequence"]
)
@pytest.mark.parametrize("operation", ["with_columns", "filter"])
def test_stateful_consumers_require_all_left_asof_ordering_metadata(
    runtime, removed, operation
):
    selected = _join().select(
        *(
            name
            for name in ["left__time", "left__symbol", "left__sequence", "left__price"]
            if name != removed
        )
    )
    lagged = cf.ts.lag(selected["left__price"])
    result = (
        selected.with_columns(previous=lagged)
        if operation == "with_columns"
        else selected.filter(lagged > 0.0)
    )
    program = cf.Program("missing_order", outputs={"result": result})
    analysis = program.analyze(runtime, mode="stream")
    if removed is None:
        assert analysis.issues == ()
        program.compile_stream(runtime)
        return
    assert any(issue.code == "ordering_required" for issue in analysis.issues)
    with pytest.raises(cf.CompileError, match="ordering_required"):
        program.compile_stream(runtime)
