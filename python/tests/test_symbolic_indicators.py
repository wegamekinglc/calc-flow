from __future__ import annotations

import pytest

import calc_flow as cf
from calc_flow.symbolic.nodes import CInt, build


def _quotes(*, ordered: bool = True) -> cf.TableExpr:
    return cf.table_input(
        "quotes",
        schema=(
            cf.Field("entity", "string", nullable=False),
            cf.Field("time", "timestamp[us, UTC]", nullable=False),
            cf.Field("seq", "uint64", nullable=False),
            cf.Field("high", "float32"),
            cf.Field("low", "float64"),
            cf.Field("close", "float64"),
        ),
        entity_by=("entity",) if ordered else (),
        event_time="time" if ordered else None,
        sequence_by=("seq",) if ordered else (),
    )


def _arguments(kind: str, source: cf.TableExpr) -> tuple[cf.ColumnExpr, ...]:
    return (
        (source["close"],)
        if kind == "rsi"
        else (source["high"], source["low"], source["close"])
    )


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_indicator_declarations_preserve_full_u64_period_and_input_nodes(
    kind: str,
) -> None:
    arguments = _arguments(kind, _quotes())
    function = getattr(cf.ts, kind)
    declaration = function(*arguments, period=2**64 - 1)
    assert isinstance(declaration, cf.ColumnExpr)
    assert declaration._node.op.name == kind
    assert declaration._node.op.version == 1
    assert declaration._node.args == tuple(argument._node for argument in arguments)
    assert declaration._node.attr("period") == CInt(2**64 - 1)
    assert declaration.identical(function(*arguments, period=2**64 - 1))


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_indicator_declarations_require_an_explicit_period(kind: str) -> None:
    with pytest.raises(TypeError, match="period"):
        getattr(cf.ts, kind)(*_arguments(kind, _quotes()))


@pytest.mark.parametrize("kind", ["rsi", "atr"])
@pytest.mark.parametrize("period", [True, 1.5, "2"])
def test_indicator_declarations_reject_non_integer_periods(
    kind: str, period: object
) -> None:
    with pytest.raises(TypeError, match="period.*integer"):
        getattr(cf.ts, kind)(*_arguments(kind, _quotes()), period=period)


@pytest.mark.parametrize("kind", ["rsi", "atr"])
@pytest.mark.parametrize("period", [0, -1, 2**64])
def test_indicator_declarations_reject_periods_outside_positive_u64(
    kind: str, period: int
) -> None:
    with pytest.raises(ValueError, match="period.*positive u64"):
        getattr(cf.ts, kind)(*_arguments(kind, _quotes()), period=period)


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_indicator_primitives_reject_unknown_semantic_versions(kind: str) -> None:
    arguments = _arguments(kind, _quotes())
    with pytest.raises(ValueError, match="unknown_primitive_version"):
        build(
            kind,
            tuple(value._node for value in arguments),
            {"period": CInt(2)},
            version=2,
        )


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_indicator_analysis_accepts_float_inputs_and_mixed_atr_dtypes(
    kind: str,
) -> None:
    source = _quotes()
    value = getattr(cf.ts, kind)(*_arguments(kind, source), period=2)
    program = cf.Program(
        "indicators", engine="sql", outputs={"out": source.with_columns(signal=value)}
    )
    assert program.analyze(cf.Runtime()).issues == ()


@pytest.mark.parametrize(
    "kind,operand", [("rsi", 0), ("atr", 0), ("atr", 1), ("atr", 2)]
)
def test_indicator_analysis_rejects_integer_operands(kind: str, operand: int) -> None:
    source = _quotes()
    arguments = list(_arguments(kind, source))
    arguments[operand] = cf.row.cast(arguments[operand], "int64")
    value = getattr(cf.ts, kind)(*arguments, period=2)
    program = cf.Program(
        "invalid", engine="sql", outputs={"out": source.with_columns(signal=value)}
    )
    issues = program.analyze(cf.Runtime()).issues
    assert any(
        issue.code == "unsupported_type" and "float32 or float64" in issue.message
        for issue in issues
    )


@pytest.mark.parametrize("kind", ["rsi", "atr"])
@pytest.mark.parametrize("mode", ["batch", "stream"])
def test_indicator_analysis_requires_declared_temporal_order(
    kind: str, mode: str
) -> None:
    source = _quotes(ordered=False)
    value = getattr(cf.ts, kind)(*_arguments(kind, source), period=2)
    program = cf.Program(
        "unordered", engine="sql", outputs={"out": source.with_columns(signal=value)}
    )
    issues = program.analyze(cf.Runtime(), mode=mode).issues
    assert any(
        issue.code == "ordering_required" and "event_time" in issue.path
        for issue in issues
    )


def test_atr_analysis_rejects_differently_filtered_operands() -> None:
    source = _quotes()
    filtered = source.filter(source["close"] > 0.0)
    value = cf.ts.atr(filtered["high"], source["low"], filtered["close"], period=2)
    program = cf.Program(
        "row_sets", engine="sql", outputs={"out": filtered.with_columns(signal=value)}
    )
    assert any(
        issue.code == "schema_mismatch" and "same row set" in issue.message
        for issue in program.analyze(cf.Runtime()).issues
    )


def test_atr_analysis_accepts_a_common_filtered_row_set() -> None:
    source = _quotes()
    filtered = source.filter(source["close"] > 0.0)
    value = cf.ts.atr(filtered["high"], filtered["low"], filtered["close"], period=2)
    program = cf.Program(
        "row_set", engine="sql", outputs={"out": filtered.with_columns(signal=value)}
    )
    assert program.analyze(cf.Runtime()).issues == ()


def test_atr_analysis_checks_new_derived_columns_against_their_row_sets() -> None:
    source = _quotes()
    filtered = source.filter(source["close"] > 0.0)
    derived = filtered.with_columns(
        adjusted=cf.row.cast(source["high"], "float64") + 1.0
    )
    value = cf.ts.atr(derived["adjusted"], filtered["low"], filtered["close"], period=2)
    program = cf.Program(
        "derived_rows",
        engine="sql",
        outputs={"out": derived.with_columns(signal=value)},
    )
    assert any(
        issue.code == "schema_mismatch" and "same row set" in issue.message
        for issue in program.analyze(cf.Runtime()).issues
    )


def test_atr_analysis_accepts_equivalent_row_preserving_declarations() -> None:
    first, second = _quotes(), _quotes()
    filtered = first.filter(first["close"] > 0.0)
    equivalent = second.filter(second["close"] > 0.0)
    derived = filtered.with_columns(adjusted=cf.row.cast(filtered["high"], "float64"))
    value = cf.ts.atr(
        derived["adjusted"], equivalent["low"], filtered["close"], period=2
    )
    program = cf.Program(
        "equivalent", engine="sql", outputs={"out": derived.with_columns(signal=value)}
    )
    assert program.analyze(cf.Runtime()).issues == ()


def test_indicator_analysis_requires_ordering_on_the_actual_projected_source() -> None:
    projected = _quotes().select("high", "low", "close")
    value = cf.ts.rsi(projected["close"], period=2)
    program = cf.Program(
        "projected", engine="sql", outputs={"out": projected.with_columns(signal=value)}
    )
    assert any(
        issue.code == "ordering_required" and "event_time" in issue.path
        for issue in program.analyze(cf.Runtime()).issues
    )


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_indicator_lowering_emits_the_exact_layout_five_primitive(kind: str) -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _quotes()
    value = getattr(cf.ts, kind)(*_arguments(kind, source), period=2**64 - 1)
    program = cf.Program(
        "lower", engine="sql", outputs={"out": source.with_columns(indicator=value)}
    )
    document = lower_program_document(program, cf.Runtime(), "batch")
    rolling = [
        node["operator"]["spec"]
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    ]
    assert len(rolling) == 1
    assert rolling[0]["state_layout_version"] == 5
    expected = (
        {"input": "close"}
        if kind == "rsi"
        else {"high": "high", "low": "low", "close": "close"}
    )
    assert rolling[0]["outputs"] == [
        {
            "kind": kind,
            "primitive_version": 1,
            "period": 2**64 - 1,
            "output": "indicator",
            **expected,
        }
    ]


def test_atr_lowering_materializes_row_local_arguments_before_state() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _quotes()
    high = cf.row.cast(source["high"], "float64") + 1.0
    value = cf.ts.atr(high, source["low"], source["close"], period=2)
    program = cf.Program(
        "row_local", engine="sql", outputs={"out": source.with_columns(indicator=value)}
    )
    document = lower_program_document(program, cf.Runtime(), "batch")
    rolling = next(
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    )
    spec = rolling["operator"]["spec"]["outputs"][0]
    assert spec["high"] not in {"high", "low", "close"}
    assert (spec["low"], spec["close"]) == ("low", "close")
    schema = rolling["input_ports"][0]["schema"]
    assert (
        next(field for field in schema if field["name"] == spec["high"])["data_type"]
        == "float64"
    )


def test_mixed_indicator_and_ewma_stage_uses_layout_five_without_rewriting_ewma() -> (
    None
):
    from calc_flow.symbolic.lower import lower_program_document

    source = _quotes()
    value = source.with_columns(
        rsi=cf.ts.rsi(source["close"], period=2), ema=cf.ts.ema(source["close"], span=3)
    )
    document = lower_program_document(
        cf.Program("mixed", engine="sql", outputs={"out": value}), cf.Runtime(), "batch"
    )
    rolling = next(
        node["operator"]["spec"]
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    )
    assert rolling["state_layout_version"] == 5
    assert next(item for item in rolling["outputs"] if item["kind"] == "ewma") == {
        "kind": "ewma",
        "primitive_version": 1,
        "input": "close",
        "output": "ema",
        "span": 3,
        "min_periods": 1,
    }


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_indicator_group_identity_ignores_alias_and_retains_period_and_inputs(
    kind: str,
) -> None:
    from calc_flow.symbolic.optimizer import _rolling_group_key, _rolling_input_columns

    inputs = (
        {"input": "close"}
        if kind == "rsi"
        else {"high": "high", "low": "low", "close": "close"}
    )
    output = {
        "kind": kind,
        "primitive_version": 1,
        "output": "first",
        "period": 2,
        **inputs,
    }
    key = _rolling_group_key(output)
    assert key is not None
    assert key == _rolling_group_key({**output, "output": "second"})
    assert key != _rolling_group_key({**output, "period": 3})
    operand = "input" if kind == "rsi" else "high"
    assert key != _rolling_group_key({**output, operand: "different"})
    assert _rolling_input_columns(output, "indicator") == tuple(inputs.values())


def test_indicator_explain_counts_one_shared_group_for_two_program_outputs() -> None:
    source = _quotes()
    value = cf.ts.atr(source["high"], source["low"], source["close"], period=2)
    program = cf.Program(
        "shared",
        engine="sql",
        outputs={
            "first": source.with_columns(a=value),
            "second": source.with_columns(b=value),
        },
    )
    assert "shared_state_groups=1" in program.explain(cf.Runtime())


@pytest.mark.parametrize(
    "primitive,options,layout",
    [("lag", {}, 1), ("ema", {"span": 3}, 2), ("cumulative_sum", {}, 4)],
)
def test_old_only_rolling_layout_selection_is_unchanged(
    primitive: str, options: dict[str, int], layout: int
) -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _quotes()
    value = getattr(cf.ts, primitive)(source["close"], **options)
    program = cf.Program(
        "legacy", engine="sql", outputs={"out": source.with_columns(signal=value)}
    )
    document = lower_program_document(program, cf.Runtime(), "batch")
    rolling = next(
        node["operator"]["spec"]
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    )
    assert rolling["state_layout_version"] == layout


def _evaluated_output(
    target: cf.TableExpr, value: cf.ColumnExpr, boundary: str
) -> cf.TableExpr:
    if boundary == "select":
        return target.select("entity", "time", "seq", signal=value)
    if boundary == "predicate":
        return target.filter(value > 50.0)
    return target.with_columns(signal=value)


@pytest.mark.parametrize("kind", ["rsi", "atr"])
@pytest.mark.parametrize("boundary", ["with_columns", "select", "predicate"])
def test_indicator_public_boundaries_reject_a_foreign_filtered_row_set(
    kind: str, boundary: str
) -> None:
    source = _quotes()
    foreign = source.filter(source["close"] > 0.0)
    value = getattr(cf.ts, kind)(*_arguments(kind, foreign), period=2)
    program = cf.Program(
        "foreign",
        engine="sql",
        outputs={"out": _evaluated_output(source, value, boundary)},
    )
    assert any(
        issue.code == "schema_mismatch" and "same row set" in issue.message
        for issue in program.analyze(cf.Runtime()).issues
    )
    with pytest.raises(cf.CompileError, match="schema_mismatch.*same row set"):
        program.to_project(cf.Runtime())


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_indicator_public_boundaries_reject_distinct_source_and_target_filters(
    kind: str,
) -> None:
    source = _quotes()
    target = source.filter(source["close"] > 1.0)
    foreign = source.filter(source["close"] > 0.0)
    value = getattr(cf.ts, kind)(*_arguments(kind, foreign), period=2)
    program = cf.Program(
        "different", engine="sql", outputs={"out": target.with_columns(signal=value)}
    )
    assert any(
        issue.code == "schema_mismatch"
        for issue in program.analyze(cf.Runtime()).issues
    )
    with pytest.raises(cf.CompileError, match="schema_mismatch.*same row set"):
        program.to_project(cf.Runtime())


def _derived_indicator_program(kind: str, *, valid: bool) -> cf.Program:
    source = _quotes()
    filtered = source.filter(source["close"] > 0.0)
    values = filtered if valid else source
    derived = filtered.with_columns(
        adjusted=cf.row.cast(values["high"], "float64") + 1.0
    )
    outer = derived.filter(derived["close"] > 1.0)
    arguments = (
        (outer["adjusted"],)
        if kind == "rsi"
        else (outer["adjusted"], outer["low"], outer["close"])
    )
    value = getattr(cf.ts, kind)(*arguments, period=2)
    return cf.Program(
        "derived", engine="sql", outputs={"out": outer.with_columns(signal=value)}
    )


def _filter_sql(document: dict[str, object]) -> str:
    return " ".join(
        node["operator"]["filter"]
        for node in document["graph"]["nodes"]
        if node["operator"].get("filter")
    )


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_indicator_consumed_definitions_are_checked_before_later_filters(
    kind: str,
) -> None:
    program = _derived_indicator_program(kind, valid=False)
    assert any(
        issue.code == "schema_mismatch" and "same row set" in issue.message
        for issue in program.analyze(cf.Runtime()).issues
    )
    with pytest.raises(cf.CompileError, match="schema_mismatch.*same row set"):
        program.to_project(cf.Runtime())


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_indicator_accepts_valid_derived_definitions_before_later_filters(
    kind: str,
) -> None:
    from calc_flow.symbolic.lower import lower_program_document

    program = _derived_indicator_program(kind, valid=True)
    assert program.analyze(cf.Runtime()).issues == ()
    document = lower_program_document(program, cf.Runtime(), "batch")
    assert '("close" > 0.0)' in _filter_sql(document)
    assert '("close" > 1.0)' in _filter_sql(document)


@pytest.mark.parametrize("kind", ["rsi", "atr"])
@pytest.mark.parametrize("boundary", ["with_columns", "select", "predicate"])
def test_indicator_matching_filtered_target_remains_admitted(
    kind: str, boundary: str
) -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _quotes()
    target = source.filter(source["close"] > 0.0)
    value = getattr(cf.ts, kind)(*_arguments(kind, target), period=2)
    program = cf.Program(
        "same",
        engine="sql",
        outputs={"out": _evaluated_output(target, value, boundary)},
    )
    assert program.analyze(cf.Runtime()).issues == ()
    document = lower_program_document(program, cf.Runtime(), "batch")
    assert '("close" > 0.0)' in _filter_sql(document)


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_indicator_lowerer_rejects_hidden_foreign_definitions_before_any_inline(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from calc_flow.symbolic.lower import segments

    def unexpected_inline(*_args: object) -> None:
        raise AssertionError("a foreign derived definition reached inline")

    monkeypatch.setattr(segments, "_inline", unexpected_inline)
    program = _derived_indicator_program(kind, valid=False)
    with pytest.raises(cf.CompileError, match="schema_mismatch.*same row set"):
        segments._resolve_table(program.outputs[0][1]._node, "out")


@pytest.mark.parametrize("kind", ["rsi", "atr"])
@pytest.mark.parametrize("boundary", ["with_columns", "select", "predicate"])
def test_indicator_lowerer_proves_the_target_before_any_inline(
    kind: str, boundary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from calc_flow.symbolic.lower import segments

    def unexpected_inline(*_args: object) -> None:
        raise AssertionError("a foreign indicator source reached inline")

    monkeypatch.setattr(segments, "_inline", unexpected_inline)
    source = _quotes()
    foreign = source.filter(source["close"] > 0.0)
    value = getattr(cf.ts, kind)(*_arguments(kind, foreign), period=2)
    with pytest.raises(cf.CompileError, match="schema_mismatch.*same row set"):
        segments._resolve_table(_evaluated_output(source, value, boundary)._node, "out")


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_indicator_accepts_derivation_before_the_first_filter(kind: str) -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _quotes()
    derived = source.with_columns(adjusted=cf.row.cast(source["high"], "float64") + 1.0)
    filtered = derived.filter(derived["close"] > 1.0)
    arguments = (
        (filtered["adjusted"],)
        if kind == "rsi"
        else (filtered["adjusted"], filtered["low"], filtered["close"])
    )
    value = getattr(cf.ts, kind)(*arguments, period=2)
    program = cf.Program(
        "before_filter",
        engine="sql",
        outputs={"out": filtered.with_columns(signal=value)},
    )
    assert program.analyze(cf.Runtime()).issues == ()
    assert lower_program_document(program, cf.Runtime(), "batch")["graph"]["nodes"]


@pytest.mark.parametrize("wrapper", ["project_and_columns", "attach"])
def test_indicator_target_proof_accepts_row_preserving_boundaries(wrapper: str) -> None:
    from calc_flow.symbolic.analyzer import _indicator_target_error
    from calc_flow.symbolic.lower import lower_program_document

    source = _quotes()
    runtime = cf.Runtime()
    if wrapper == "attach":
        cf.register_numpy(runtime)
        matrix = cf.linalg.from_columns(source, columns=("high",), backend="numpy")
        weights = cf.parameter(
            "weights", kind="array", backend="numpy", dtype="float32", shape=(1, 1)
        )
        target = cf.table.attach_columns(
            source, cf.linalg.matmul(matrix, weights), names=("copy",)
        )
    else:
        target = source.select(
            "entity", "time", "seq", "high", "low", "close"
        ).with_columns(copy=source["close"] + 1.0)
    value = cf.ts.rsi(source["close"], period=2)
    program = cf.Program(
        "preserving", engine="sql", outputs={"out": target.with_columns(signal=value)}
    )
    assert program.analyze(runtime).issues == ()
    assert _indicator_target_error(value._node, target._node) is None
    if wrapper != "attach":
        assert lower_program_document(program, runtime, "batch")["graph"]["nodes"]


def test_indicator_checks_ordering_on_its_actual_evaluation_target() -> None:
    source = _quotes()
    target = source.select("high", "low", "close")
    value = cf.ts.rsi(source["close"], period=2)
    program = cf.Program(
        "target_order", engine="sql", outputs={"out": target.with_columns(signal=value)}
    )
    assert any(
        issue.code == "ordering_required"
        for issue in program.analyze(cf.Runtime()).issues
    )


@pytest.mark.parametrize("primitive", ["ema", "macd"])
def test_indicator_target_proof_keeps_legacy_temporal_admission(primitive: str) -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _quotes()
    foreign = source.filter(source["close"] > 0.0)
    options = {"span": 2} if primitive == "ema" else {"fast_span": 2, "slow_span": 4}
    value = getattr(cf.ts, primitive)(foreign["close"], **options)
    program = cf.Program(
        "legacy_scope", engine="sql", outputs={"out": source.with_columns(signal=value)}
    )
    assert program.analyze(cf.Runtime()).issues == ()
    document = lower_program_document(program, cf.Runtime(), "batch")
    rolling = next(
        node["operator"]["spec"]
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    )
    assert rolling["state_layout_version"] == 2


@pytest.mark.parametrize("kind", ["rsi", "atr"])
@pytest.mark.parametrize(
    "failure", ["different_definition", "missing_alias", "overwritten_base"]
)
def test_indicator_target_preserves_consumed_column_meanings(
    kind: str, failure: str
) -> None:
    source = _quotes()
    declared = source.with_columns(adjusted=source["close"] + 1.0)
    if failure == "different_definition":
        target = source.with_columns(adjusted=source["close"] + 2.0)
        value = declared["adjusted"]
    elif failure == "missing_alias":
        target, value = source, declared["adjusted"]
    else:
        target, value = (
            source.with_columns(close=source["close"] + 1.0),
            source["close"],
        )
    arguments = (value,) if kind == "rsi" else (value, source["low"], source["close"])
    output = target.with_columns(signal=getattr(cf.ts, kind)(*arguments, period=2))
    program = cf.Program("meanings", engine="sql", outputs={"out": output})
    assert any(
        issue.code == "schema_mismatch"
        for issue in program.analyze(cf.Runtime()).issues
    )
    from calc_flow.symbolic.lower.segments import _resolve_table

    with pytest.raises(cf.CompileError, match="schema_mismatch"):
        _resolve_table(output._node, "out")


def test_indicator_target_accepts_canonically_equivalent_consumed_definitions() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    first, second = _quotes(), _quotes()
    declared = first.with_columns(adjusted=first["close"] + 1.0)
    target = second.with_columns(adjusted=second["close"] + 1.0)
    output = target.with_columns(signal=cf.ts.rsi(declared["adjusted"], period=2))
    program = cf.Program("definitions", engine="sql", outputs={"out": output})
    assert program.analyze(cf.Runtime()).issues == ()
    assert lower_program_document(program, cf.Runtime(), "batch")["graph"]["nodes"]


def test_atr_target_can_supply_separately_declared_matching_derived_columns() -> None:
    from calc_flow.symbolic.lower import lower_program_document

    source = _quotes()
    high, low = cf.row.cast(source["high"], "float64") + 1.0, source["low"] - 1.0
    high_source, low_source = source.with_columns(h=high), source.with_columns(l=low)
    target = source.with_columns(h=high, l=low)
    value = cf.ts.atr(high_source["h"], low_source["l"], source["close"], period=2)
    program = cf.Program(
        "separate", engine="sql", outputs={"out": target.with_columns(signal=value)}
    )
    assert program.analyze(cf.Runtime()).issues == ()
    assert lower_program_document(program, cf.Runtime(), "batch")["graph"]["nodes"]


@pytest.mark.parametrize("failure", ["foreign_rows", "foreign_definition"])
def test_indicator_compositions_preserve_all_consumed_column_bindings(
    failure: str,
) -> None:
    from calc_flow.symbolic.lower.segments import _resolve_table

    source = _quotes()
    if failure == "foreign_rows":
        target = source
        extra = source.filter(source["close"] > 0.0)["close"]
    else:
        target = source.with_columns(adjusted=source["close"] + 2.0)
        extra = source.with_columns(adjusted=source["close"] + 1.0)["adjusted"]
    value = cf.ts.rsi(source["close"], period=2) + extra
    output = target.with_columns(signal=value)
    program = cf.Program("composed", engine="sql", outputs={"out": output})
    assert any(
        issue.code == "schema_mismatch"
        for issue in program.analyze(cf.Runtime()).issues
    )
    with pytest.raises(cf.CompileError, match="schema_mismatch"):
        _resolve_table(output._node, "out")


def _attached_copy(source: cf.TableExpr, scale: float) -> cf.TableExpr:
    matrix = cf.linalg.from_columns(source, columns=("high",), backend="numpy")
    weights = cf.parameter(
        "weights", kind="array", backend="numpy", dtype="float32", shape=(1, 1)
    )
    return cf.table.attach_columns(
        source, cf.linalg.matmul(matrix * scale, weights), names=("copy",)
    )


@pytest.mark.parametrize("equivalent", [True, False])
def test_indicator_target_proof_preserves_attached_column_definitions(
    equivalent: bool,
) -> None:
    from calc_flow.symbolic.analyzer import _indicator_target_error

    source = _quotes()
    declared = _attached_copy(source, 2.0)
    target = _attached_copy(source, 2.0 if equivalent else 3.0)
    value = cf.ts.rsi(declared["copy"], period=2)
    program = cf.Program(
        "attached_meanings",
        engine="sql",
        outputs={"out": target.with_columns(signal=value)},
    )
    runtime = cf.Runtime()
    cf.register_numpy(runtime)
    issues = program.analyze(runtime).issues
    if equivalent:
        assert not any(issue.code == "schema_mismatch" for issue in issues)
        assert _indicator_target_error(value._node, target._node) is None
    else:
        assert any(issue.code == "schema_mismatch" for issue in issues)
        assert _indicator_target_error(value._node, target._node) is not None


def _transitive_target_program(
    kind: str, *, valid: bool, direct: bool = False
) -> cf.Program:
    source = _quotes()
    declared_parent = source.with_columns(x=source["close"] + 1.0)
    target_parent = source.with_columns(
        x=source["close"] + (1.0 if valid else 2.0), unused=source["close"] + 10.0
    )
    expression = declared_parent["x"] + 1.0
    declared = declared_parent.with_columns(y=expression)
    target = target_parent.with_columns(y=expression)
    value_source = target if direct else declared
    arguments = (
        (value_source["y"],)
        if kind == "rsi"
        else (value_source["y"], target["low"], target["close"])
    )
    value = getattr(cf.ts, kind)(*arguments, period=2)
    return cf.Program(
        "transitive", engine="sql", outputs={"out": target.with_columns(signal=value)}
    )


@pytest.mark.parametrize("kind", ["rsi", "atr"])
@pytest.mark.parametrize("direct", [False, True])
def test_indicator_validates_transitive_target_definitions_at_their_own_context(
    kind: str, direct: bool
) -> None:
    program = _transitive_target_program(kind, valid=False, direct=direct)
    assert any(
        issue.code == "schema_mismatch"
        for issue in program.analyze(cf.Runtime()).issues
    )
    with pytest.raises(cf.CompileError, match="schema_mismatch"):
        program.to_project(cf.Runtime())


@pytest.mark.parametrize("kind", ["rsi", "atr"])
@pytest.mark.parametrize("direct", [False, True])
def test_indicator_lowerer_rejects_transitive_target_definitions_before_inline(
    kind: str, direct: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from calc_flow.symbolic.lower import segments

    def unexpected_inline(*_args: object) -> None:
        raise AssertionError("an invalid transitive target reached inline")

    monkeypatch.setattr(segments, "_inline", unexpected_inline)
    program = _transitive_target_program(kind, valid=False, direct=direct)
    with pytest.raises(cf.CompileError, match="schema_mismatch"):
        segments._resolve_table(program.outputs[0][1]._node, "out")


@pytest.mark.parametrize("kind", ["rsi", "atr"])
def test_indicator_accepts_matching_transitive_target_definitions(kind: str) -> None:
    from calc_flow.symbolic.lower import lower_program_document

    program = _transitive_target_program(kind, valid=True)
    assert program.analyze(cf.Runtime()).issues == ()
    document = lower_program_document(program, cf.Runtime(), "batch")
    rolling = next(
        node["operator"]["spec"]
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "rolling"
    )
    assert rolling["state_layout_version"] == 5
