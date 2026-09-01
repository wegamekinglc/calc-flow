from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pytest

from calc_flow.capabilities import RuntimeCapabilities
from calc_flow.errors import CompileError
from calc_flow.pipeline import Runtime
from calc_flow.symbolic import (
    AnalysisResult,
    FeatureSet,
    Field,
    Parameter,
    Program,
    TableExpr,
    cs,
    duration,
    exact_time,
    linalg,
    parameter,
    row,
    rows,
    table,
    table_input,
    ts,
    window,
)


def _quotes_ordered() -> TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64"),
            Field("y", "float64"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )


def _quotes_plain() -> TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("x", "float64"),
            Field("y", "float64"),
        ],
    )


def _trades_plain() -> TableExpr:
    return table_input(
        "trades",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("y", "float64"),
        ],
    )


def _weights() -> Parameter:
    return parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(2, 1),
    )


def _row_local_program() -> Program:
    quotes = _quotes_plain()
    return Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(FeatureSet([("score", quotes["x"] + quotes["y"])])),
            )
        ],
    )


def _issue_paths(result: AnalysisResult) -> tuple[str, ...]:
    return tuple(issue.path for issue in result.issues)


def test_valid_row_local_program_analyzes_cleanly() -> None:
    result = _row_local_program().analyze(Runtime(), mode="batch")

    assert result.issues == ()
    assert result.mode == "batch"
    assert result.program_fingerprint == _row_local_program().fingerprint


def test_analyze_and_explain_require_explicit_runtime_and_mode() -> None:
    program = _row_local_program()
    runtime = Runtime()
    with pytest.raises(TypeError, match="Runtime"):
        program.analyze(object(), mode="batch")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mode"):
        program.analyze(runtime, mode="continuous")  # type: ignore[arg-type]

    snapshot = runtime.capabilities()
    result = program.analyze(runtime, mode="batch")
    assert result.capability_session_id == snapshot.scope.session_id
    assert result.capability_revision == snapshot.scope.revision


def test_unknown_field_reference_fails_from_named_output() -> None:
    quotes = _quotes_plain()
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(FeatureSet([("score", quotes["missing"])])),
            )
        ],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.path == "outputs.signals.score"
    assert issue.code == "unresolved_type"
    assert "missing" in issue.message


def test_nested_unknown_field_reports_operand_path() -> None:
    quotes = _quotes_plain()
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet([("score", quotes["x"] + quotes["missing"])])
                ),
            )
        ],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert result.issues[0].path == "outputs.signals.score.add.right"
    assert result.issues[0].code == "unresolved_type"


def test_column_operators_reject_mixed_input_lineage() -> None:
    quotes = _quotes_plain()
    trades = _trades_plain()
    program = Program(
        "p",
        inputs=[quotes, trades],
        outputs=[
            (
                "signals",
                quotes.with_columns(FeatureSet([("score", quotes["x"] + trades["y"])])),
            )
        ],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert "outputs.signals.score.add.right.lineage" in _issue_paths(result)
    codes = {issue.code for issue in result.issues}
    assert codes == {"schema_mismatch"}


def test_undeclared_input_reference_is_reported_at_input_path() -> None:
    quotes = _quotes_plain()
    trades = _trades_plain()
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[("trades_out", trades.with_columns(FeatureSet()))],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert "inputs.trades" in _issue_paths(result)
    undeclared = next(issue for issue in result.issues if issue.path == "inputs.trades")
    assert undeclared.code == "unresolved_type"
    assert "not declared" in undeclared.message


def test_incompatible_operand_types_require_explicit_cast() -> None:
    quotes = _quotes_plain()
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet([("score", quotes["symbol"] + quotes["x"])])
                ),
            )
        ],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert "outputs.signals.score.add.right.dtype" in _issue_paths(result)
    dtype_issue = next(
        issue
        for issue in result.issues
        if issue.path == "outputs.signals.score.add.right.dtype"
    )
    assert dtype_issue.code == "unsupported_type"

    literal_program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(FeatureSet([("score", quotes["x"] + 1)])),
            )
        ],
    )
    literal_result = literal_program.analyze(Runtime(), mode="batch")
    assert "outputs.signals.score.add.right.dtype" in _issue_paths(literal_result)


def test_matmul_propagates_symbolic_dimensions_and_row_lineage() -> None:
    quotes = _quotes_ordered()
    scores = linalg.matmul(
        linalg.from_columns(quotes, columns=["x", "y"], backend="numpy"),
        _weights(),
    )
    program = Program(
        "p",
        inputs=[quotes, _weights()],
        outputs=[("scores", scores)],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert result.issues == ()
    explanation = program.explain(Runtime(), mode="batch")
    assert (
        "output scores array backend numpy dtype float64"
        " shape (quotes, 1) lineage quotes" in explanation
    )


def test_attached_array_derives_table_fields_with_types() -> None:
    quotes = _quotes_ordered()
    scores = linalg.matmul(
        linalg.from_columns(quotes, columns=["x", "y"], backend="numpy"),
        _weights(),
    )
    signals = table.attach_columns(quotes, scores, names=("score",))
    program = Program(
        "p",
        inputs=[quotes, _weights()],
        outputs=[("signals", signals)],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert result.issues == ()
    explanation = program.explain(Runtime(), mode="batch")
    assert "field score float64 nullable=true" in explanation


def test_matmul_rejects_inner_dimension_mismatch_with_d12_path() -> None:
    quotes = _quotes_ordered()
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(3, 1),
    )
    program = Program(
        "p",
        inputs=[quotes, weights],
        outputs=[
            (
                "scores",
                linalg.matmul(
                    linalg.from_columns(quotes, columns=["x", "y"], backend="numpy"),
                    weights,
                ),
            )
        ],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert "outputs.scores.matmul.right.shape[0]" in _issue_paths(result)
    issue = next(
        issue
        for issue in result.issues
        if issue.path == "outputs.scores.matmul.right.shape[0]"
    )
    assert issue.code == "schema_mismatch"


def test_matmul_rejects_rank_and_backend_mismatch_but_promotes_safe_dtype() -> None:
    quotes = _quotes_ordered()
    flat = parameter(
        "flat",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(2,),
    )
    float32 = parameter(
        "float32",
        kind="array",
        backend="numpy",
        dtype="float32",
        shape=(2, 1),
    )
    jax_weights = parameter(
        "jax_weights",
        kind="array",
        backend="jax",
        dtype="float64",
        shape=(2, 1),
    )
    base = linalg.from_columns(quotes, columns=["x", "y"], backend="numpy")

    for name, weights in (("flat", flat), ("jax", jax_weights)):
        program = Program(
            "p",
            inputs=[quotes, weights],
            outputs=[("scores", linalg.matmul(base, weights))],
        )
        result = program.analyze(Runtime(), mode="batch")
        paths = _issue_paths(result)
        assert paths, name
        assert all(path.startswith("outputs.scores.matmul.right") for path in paths)

    promoted = Program(
        "p",
        inputs=[quotes, float32],
        outputs=[("scores", linalg.matmul(base, float32))],
    )
    promoted_result = promoted.analyze(Runtime(), mode="batch")
    assert promoted_result.issues == ()
    assert "output scores array backend numpy dtype float64" in promoted.explain(
        Runtime(), mode="batch"
    )


def test_attach_rejects_foreign_row_axis_lineage() -> None:
    quotes = _quotes_ordered()
    trades = table_input(
        "trades",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("y", "float64"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["ts"],
    )
    foreign = linalg.from_columns(trades, columns=["y"], backend="numpy")

    program = Program(
        "p",
        inputs=[quotes, trades],
        outputs=[("signals", table.attach_columns(quotes, foreign, names=("v",)))],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert "outputs.signals.attach_columns.array.lineage" in _issue_paths(result)


def test_attach_rejects_width_and_name_collisions() -> None:
    quotes = _quotes_ordered()
    wide = linalg.from_columns(quotes, columns=["x", "y"], backend="numpy")

    width_program = Program(
        "p",
        inputs=[quotes],
        outputs=[("signals", table.attach_columns(quotes, wide, names=("score",)))],
    )
    width_result = width_program.analyze(Runtime(), mode="batch")
    assert "outputs.signals.attach_columns.array.shape[1]" in _issue_paths(width_result)

    collision_program = Program(
        "p",
        inputs=[quotes],
        outputs=[("signals", table.attach_columns(quotes, wide, names=("v", "x")))],
    )
    collision_result = collision_program.analyze(Runtime(), mode="batch")
    assert "outputs.signals.x" in _issue_paths(collision_result)
    collision = next(
        issue for issue in collision_result.issues if issue.path == "outputs.signals.x"
    )
    assert collision.code == "duplicate_name"


def test_with_columns_rejects_foreign_lineage_and_name_collisions() -> None:
    quotes = _quotes_plain()
    trades = _trades_plain()

    lineage_program = Program(
        "p",
        inputs=[quotes, trades],
        outputs=[
            (
                "signals",
                quotes.with_columns(FeatureSet([("score", trades["y"])])),
            )
        ],
    )
    lineage_result = lineage_program.analyze(Runtime(), mode="batch")
    assert "outputs.signals.score.lineage" in _issue_paths(lineage_result)

    collision_program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(FeatureSet([("x", quotes["x"])])),
            )
        ],
    )
    collision_result = collision_program.analyze(Runtime(), mode="batch")
    assert "outputs.signals.x" in _issue_paths(collision_result)


def test_stream_mode_requires_event_time_entity_and_sequence() -> None:
    quotes = _quotes_plain()
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            (
                                "score",
                                ts.mean(quotes["x"], window=rows(3)),
                            )
                        ]
                    )
                ),
            )
        ],
    )

    stream_result = program.analyze(Runtime(), mode="stream")
    paths = _issue_paths(stream_result)
    assert "inputs.quotes.event_time" in paths
    assert "inputs.quotes.entity_by" in paths
    assert "inputs.quotes.sequence_by" in paths
    assert all(issue.code == "ordering_required" for issue in stream_result.issues)

    batch_result = program.analyze(Runtime(), mode="batch")
    assert batch_result.issues == ()


def test_stream_mode_validates_ordering_field_types() -> None:
    quotes = table_input(
        "quotes",
        schema=[
            Field("ts", "string", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "float64", nullable=False),
            Field("x", "float64"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet([("score", ts.mean(quotes["x"], window=rows(3)))])
                ),
            )
        ],
    )
    result = program.analyze(Runtime(), mode="stream")

    paths = _issue_paths(result)
    assert "inputs.quotes.event_time" in paths
    assert "inputs.quotes.sequence_by[0]" in paths
    assert all(issue.code == "ordering_required" for issue in result.issues)


def test_stream_mode_allows_row_local_programs_without_ordering() -> None:
    result = _row_local_program().analyze(Runtime(), mode="stream")

    assert result.issues == ()


def test_stream_mode_rejects_unbounded_table_derived_array_output() -> None:
    quotes = _quotes_ordered()
    scores = linalg.from_columns(quotes, columns=["x", "y"], backend="numpy")
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[("scores", scores)],
    )

    stream_result = program.analyze(Runtime(), mode="stream")
    assert len(stream_result.issues) == 1
    assert stream_result.issues[0].path == "outputs.scores"
    assert stream_result.issues[0].code == "unbounded_state"

    batch_result = program.analyze(Runtime(), mode="batch")
    assert batch_result.issues == ()

    attached = table.attach_columns(
        quotes,
        linalg.matmul(scores, _weights()),
        names=("score",),
    )
    attached_program = Program(
        "p",
        inputs=[quotes, _weights()],
        outputs=[("signals", attached)],
    )
    assert attached_program.analyze(Runtime(), mode="stream").issues == ()


def test_filter_rejects_foreign_lineage_and_non_bool_predicates() -> None:
    quotes = _quotes_plain()
    trades = _trades_plain()
    lineage_program = Program(
        "p",
        inputs=[quotes, trades],
        outputs=[("signals", table.filter(quotes, trades["y"] > 0.0))],
    )
    lineage_result = lineage_program.analyze(Runtime(), mode="batch")
    assert "outputs.signals.filter.predicate.lineage" in _issue_paths(lineage_result)

    type_program = Program(
        "p",
        inputs=[quotes],
        outputs=[("signals", table.filter(quotes, quotes["x"]))],
    )
    type_result = type_program.analyze(Runtime(), mode="batch")
    assert "outputs.signals.filter.predicate.dtype" in _issue_paths(type_result)


def test_window_derives_window_schema_and_checks_event_time_field() -> None:
    quotes = _quotes_ordered()
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "windows",
                window.tumbling(quotes, event_time="ts", size_micros=60_000_000),
            )
        ],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert result.issues == ()
    explanation = program.explain(Runtime(), mode="batch")
    assert "field window_start timestamp[us, UTC] nullable=false" in explanation
    assert "field window_end timestamp[us, UTC] nullable=false" in explanation

    broken = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "windows",
                window.tumbling(quotes, event_time="nope", size_micros=60_000_000),
            )
        ],
    )
    broken_result = broken.analyze(Runtime(), mode="batch")
    assert "outputs.windows.window_tumbling.event_time" in _issue_paths(broken_result)


def test_capability_snapshot_gates_portable_types_and_batch_kinds() -> None:
    quotes = _quotes_plain()

    class StrippedRuntime(Runtime):
        def capabilities(self) -> RuntimeCapabilities:
            base = super().capabilities()
            return dataclasses.replace(
                base,
                portable_arrow_types=tuple(
                    name for name in base.portable_arrow_types if name != "float64"
                ),
            )

    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(FeatureSet([("score", quotes["symbol"])])),
            )
        ],
    )
    result = program.analyze(StrippedRuntime(), mode="batch")

    paths = _issue_paths(result)
    assert "inputs.quotes.schema[2].data_type" in paths
    mismatch = next(
        issue
        for issue in result.issues
        if issue.path == "inputs.quotes.schema[2].data_type"
    )
    assert mismatch.code == "capability_mismatch"

    class TableOnlyRuntime(Runtime):
        def capabilities(self) -> RuntimeCapabilities:
            base = super().capabilities()
            return dataclasses.replace(base, batch_kinds=("table",))

    array_program = Program(
        "p",
        inputs=[quotes, _weights()],
        outputs=[
            (
                "scores",
                linalg.matmul(
                    linalg.from_columns(quotes, columns=["x", "y"], backend="numpy"),
                    _weights(),
                ),
            )
        ],
    )
    array_result = array_program.analyze(TableOnlyRuntime(), mode="batch")
    assert "outputs.scores" in _issue_paths(array_result)
    assert "static_inputs.weights" in _issue_paths(array_result)


def test_explain_reports_state_requirements_per_output() -> None:
    quotes = _quotes_ordered()
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            (
                                "momentum",
                                ts.mean(quotes["x"], window=duration(60_000_000)),
                            ),
                            (
                                "rank",
                                cs.rank(
                                    quotes["x"],
                                    group=exact_time(quotes["ts"]),
                                ),
                            ),
                        ]
                    )
                ),
            )
        ],
    )
    explanation = program.explain(Runtime(), mode="batch")

    assert "state cross_section, duration(60000000)" in explanation


def test_explain_reports_constant_exponential_state() -> None:
    quotes = _quotes_ordered()
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet([("ema", ts.ewma(quotes["x"], span=12))])
                ),
            )
        ],
    )

    explanation = program.explain(Runtime(), mode="stream")

    assert "state constant(span=12)" in explanation


def test_analysis_is_deterministic_and_never_mutates_declarations() -> None:
    program = _row_local_program()
    runtime = Runtime()
    fingerprint_before = program.fingerprint

    first = program.analyze(runtime, mode="batch")
    second = program.analyze(runtime, mode="batch")

    assert first == second
    assert program.fingerprint == fingerprint_before
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.mode = "stream"  # type: ignore[misc]

    explanation_before = program.explain(runtime, mode="batch")
    program.analyze(runtime, mode="stream")
    assert program.explain(runtime, mode="batch") == explanation_before


def test_explain_renders_program_facts_deterministically() -> None:
    program = _row_local_program()
    runtime = Runtime()
    explanation = program.explain(runtime, mode="batch")

    assert explanation == program.explain(runtime, mode="batch")
    assert explanation.startswith("program p\n")
    assert "  mode batch" in explanation
    assert f"  fingerprint {program.fingerprint}" in explanation
    assert "capability session" in explanation
    assert "output signals table" in explanation
    assert "field score float64 nullable=true" in explanation


# --- review blocker regressions (B1-B6, S4) -------------------------------


def _trades_flagged() -> TableExpr:
    return table_input(
        "trades",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("flag", "bool", nullable=False),
            Field("y", "float64"),
        ],
    )


def _feature_program(
    quotes: TableExpr, name: str, value: object, **rest: object
) -> Program:  # type: ignore[override]
    extra_inputs = rest.get("extra_inputs", ())
    return Program(
        "p",
        inputs=[quotes, *extra_inputs],  # type: ignore[list-item]
        outputs=[("signals", quotes.with_columns(FeatureSet([(name, value)])))],  # type: ignore[arg-type]
    )


def test_successful_cast_resolves_derived_field_type() -> None:
    quotes = _quotes_plain()
    program = _feature_program(quotes, "half", row.cast(quotes["x"], "float32"))
    result = program.analyze(Runtime(), mode="batch")

    assert result.issues == ()
    assert "field half float32 nullable=true" in program.explain(
        Runtime(), mode="batch"
    )


def test_stream_mode_requires_ordering_for_lag_and_delta() -> None:
    quotes = _quotes_plain()
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            ("prev", ts.lag(quotes["x"], periods=2)),
                            ("change", ts.delta(quotes["x"])),
                        ]
                    )
                ),
            )
        ],
    )

    stream_result = program.analyze(Runtime(), mode="stream")
    paths = _issue_paths(stream_result)
    assert "inputs.quotes.event_time" in paths
    assert "inputs.quotes.entity_by" in paths
    assert "inputs.quotes.sequence_by" in paths
    assert all(issue.code == "ordering_required" for issue in stream_result.issues)

    assert program.analyze(Runtime(), mode="batch").issues == ()


@pytest.mark.parametrize(
    ("feature_name", "feature"),
    (
        (
            "mean_change",
            lambda quotes: ts.mean(
                ts.delta(quotes["x"]),
                window=rows(3),
            ),
        ),
        (
            "adjusted_rank",
            lambda quotes: cs.rank(
                quotes["x"] + 1.0,
                group=exact_time(quotes["ts"]),
            ),
        ),
    ),
)
def test_analysis_accepts_stageable_stateful_operands(
    feature_name: str,
    feature: Callable[[TableExpr], object],
) -> None:
    quotes = _quotes_ordered()
    program = _feature_program(quotes, feature_name, feature(quotes))

    result = program.analyze(Runtime(), mode="batch")

    assert result.issues == ()
    program.compile_batch(Runtime())


def test_analysis_rejects_cross_section_output_inside_rolling_state() -> None:
    quotes = _quotes_ordered()
    ranked = cs.rank(quotes["x"], group=exact_time(quotes["ts"]))
    program = _feature_program(
        quotes,
        "rank_mean",
        ts.mean(ranked, window=rows(3)),
    )

    result = program.analyze(Runtime(), mode="batch")

    assert any(
        issue.path == "outputs.signals.rank_mean.mean.value"
        and issue.code == "unsupported_type"
        for issue in result.issues
    )
    with pytest.raises(CompileError, match="rolling mean argument"):
        program.compile_batch(Runtime())


def test_analysis_accepts_row_local_derived_columns_and_input_aliases() -> None:
    quotes = _quotes_ordered()
    derived = quotes.with_columns(
        FeatureSet(
            (
                ("adjusted", quotes["x"] + 1.0),
                ("price", quotes["x"]),
            )
        )
    )
    signals = derived.with_columns(
        FeatureSet(
            (
                ("adjusted_previous", ts.lag(derived["adjusted"])),
                ("price_previous", ts.lag(derived["price"])),
            )
        )
    )
    program = Program("p", inputs=(quotes,), outputs=(("signals", signals),))

    result = program.analyze(Runtime(), mode="batch")

    assert result.issues == ()
    program.compile_batch(Runtime())


def test_direct_input_alias_remains_analyzable_and_compilable() -> None:
    quotes = _quotes_ordered()
    aliased = quotes.with_columns(FeatureSet((("price", quotes["x"]),)))
    projected = table.project(aliased, ("ts", "symbol", "seq", "x", "price"))
    filtered = table.filter(projected, projected["x"] > 0.0)
    signals = filtered.with_columns(
        FeatureSet((("price_previous", ts.lag(filtered["price"])),))
    )
    program = Program("p", inputs=(quotes,), outputs=(("signals", signals),))

    assert program.analyze(Runtime(), mode="batch").issues == ()
    program.compile_batch(Runtime())


def test_original_column_after_row_local_derivation_remains_materializable() -> None:
    quotes = _quotes_ordered()
    derived = quotes.with_columns(FeatureSet((("adjusted", quotes["x"] + 1.0),)))
    signals = derived.with_columns(FeatureSet((("previous", ts.lag(derived["x"])),)))
    program = Program("p", inputs=(quotes,), outputs=(("signals", signals),))

    assert program.analyze(Runtime(), mode="batch").issues == ()
    program.compile_batch(Runtime())


def test_non_row_local_table_boundary_is_not_a_rolling_operand() -> None:
    quotes = _quotes_ordered()
    scores = linalg.matmul(
        linalg.from_columns(quotes, columns=("x", "y"), backend="numpy"),
        _weights(),
    )
    attached = table.attach_columns(quotes, scores, names=("score",))
    signals = attached.with_columns(FeatureSet((("previous", ts.lag(attached["x"])),)))
    program = Program(
        "p",
        inputs=(quotes, _weights()),
        outputs=(("signals", signals),),
    )

    result = program.analyze(Runtime(), mode="batch")

    assert any(
        issue.path == "outputs.signals.previous.lag.value"
        and issue.code == "unsupported_type"
        for issue in result.issues
    )


def test_where_condition_and_coalesce_operands_respect_lineage() -> None:
    quotes = _quotes_plain()
    trades = _trades_flagged()

    where_program = Program(
        "p",
        inputs=[quotes, trades],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            (
                                "score",
                                row.where(trades["flag"], quotes["x"], quotes["y"]),
                            )
                        ]
                    )
                ),
            )
        ],
    )
    where_result = where_program.analyze(Runtime(), mode="batch")
    assert any(
        issue.code == "schema_mismatch" and ".lineage" in issue.path
        for issue in where_result.issues
    )

    coalesce_program = Program(
        "p",
        inputs=[quotes, trades],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet([("score", row.coalesce(trades["y"], quotes["x"]))])
                ),
            )
        ],
    )
    coalesce_result = coalesce_program.analyze(Runtime(), mode="batch")
    assert any(
        issue.code == "schema_mismatch" and ".lineage" in issue.path
        for issue in coalesce_result.issues
    )


def test_where_condition_state_propagates_to_result() -> None:
    quotes = _quotes_ordered()
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            (
                                "score",
                                row.where(
                                    ts.mean(quotes["x"], window=rows(3)) > 1.0,
                                    quotes["x"],
                                    quotes["y"],
                                ),
                            )
                        ]
                    )
                ),
            )
        ],
    )
    explanation = program.explain(Runtime(), mode="batch")

    assert "state rows(3)" in explanation


def test_cross_section_group_columns_are_analyzed() -> None:
    quotes = _quotes_ordered()
    trades = _trades_flagged()

    unresolved = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            (
                                "score",
                                cs.rank(
                                    quotes["x"],
                                    group=exact_time(quotes["no_such_field"]),
                                ),
                            )
                        ]
                    )
                ),
            )
        ],
    )
    unresolved_result = unresolved.analyze(Runtime(), mode="batch")
    assert "outputs.signals.score.rank.event_time" in _issue_paths(unresolved_result)
    assert all(issue.code == "unresolved_type" for issue in unresolved_result.issues)

    lineage = Program(
        "p",
        inputs=[quotes, trades],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            (
                                "score",
                                cs.zscore(
                                    quotes["x"],
                                    group=exact_time(
                                        quotes["ts"],
                                        partition_by=[trades["flag"]],
                                    ),
                                ),
                            )
                        ]
                    )
                ),
            )
        ],
    )
    lineage_result = lineage.analyze(Runtime(), mode="batch")
    assert "outputs.signals.score.zscore.partition_by[0].lineage" in _issue_paths(
        lineage_result
    )

    value_mismatch = Program(
        "p",
        inputs=[quotes, trades],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            (
                                "score",
                                cs.zscore(quotes["x"], group=exact_time(trades["ts"])),
                            )
                        ]
                    )
                ),
            )
        ],
    )
    value_mismatch_result = value_mismatch.analyze(Runtime(), mode="batch")
    assert "outputs.signals.score.zscore.event_time.lineage" in _issue_paths(
        value_mismatch_result
    )


def test_window_output_has_no_row_axis_lineage() -> None:
    quotes = _quotes_ordered()
    windows = window.tumbling(quotes, event_time="ts", size_micros=60_000_000)
    derived = linalg.from_columns(windows, columns=["window_start"], backend="numpy")
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[("signals", table.attach_columns(quotes, derived, names=("w",)))],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert "outputs.signals.attach_columns.array.lineage" in _issue_paths(result)


def test_statistics_reject_non_numeric_inputs() -> None:
    quotes = table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("flag", "bool"),
            Field("x", "float64"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )
    cases = [
        ("sum", ts.sum(quotes["symbol"], window=rows(3))),
        ("mean", ts.mean(quotes["flag"], window=rows(3))),
        ("variance", ts.variance(quotes["symbol"], window=rows(3))),
        ("stddev", ts.stddev(quotes["flag"], window=rows(3))),
        (
            "covariance",
            ts.covariance(quotes["x"], quotes["symbol"], window=rows(3)),
        ),
        (
            "correlation",
            ts.correlation(quotes["symbol"], quotes["x"], window=rows(3)),
        ),
        ("delta", ts.delta(quotes["symbol"])),
        (
            "zscore",
            cs.zscore(quotes["symbol"], group=exact_time(quotes["ts"])),
        ),
        (
            "demean",
            cs.demean(quotes["flag"], group=exact_time(quotes["ts"])),
        ),
    ]

    for name, value in cases:
        program = _feature_program(quotes, "score", value)
        result = program.analyze(Runtime(), mode="batch")
        dtype_issues = [
            issue
            for issue in result.issues
            if issue.code == "unsupported_type" and issue.path.endswith(".dtype")
        ]
        assert dtype_issues, name
        assert all(
            issue.path.startswith("outputs.signals.score.") for issue in dtype_issues
        ), name

    numeric_program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            ("total", ts.sum(quotes["seq"], window=rows(3))),
                            ("spread", ts.stddev(quotes["x"], window=rows(3))),
                            ("last", ts.min(quotes["symbol"], window=rows(3))),
                        ]
                    )
                ),
            )
        ],
    )
    assert numeric_program.analyze(Runtime(), mode="batch").issues == ()
    explanation = numeric_program.explain(Runtime(), mode="batch")
    assert "field last string nullable=true" in explanation


def test_elementwise_shape_paths_index_from_the_left() -> None:
    quotes = _quotes_ordered()
    square = parameter(
        "square",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(3, 3),
    )
    left = linalg.from_columns(quotes, columns=["x", "y"], backend="numpy")
    program = Program(
        "p",
        inputs=[quotes, square],
        outputs=[("scores", left * square)],
    )
    result = program.analyze(Runtime(), mode="batch")

    mismatch = next(issue for issue in result.issues if issue.code == "schema_mismatch")
    assert mismatch.path == "outputs.scores.mul.right.shape[1]"
    unresolved = next(
        issue for issue in result.issues if issue.code == "unresolved_type"
    )
    assert unresolved.path == "outputs.scores.mul.right.shape[0]"


# --- coverage-closure behaviors over the public API ------------------------


def _quotes_typed() -> TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("px", "float64", nullable=False),
            Field("x", "float64"),
            Field("qty", "int64"),
            Field("flag", "bool"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )


def test_projection_resolves_columns_and_preserves_field_types() -> None:
    quotes = _quotes_typed()
    narrow = table.project(quotes, ["symbol", "x"])
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[("signals", narrow.with_columns(FeatureSet([("s", narrow["x"])])))],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert result.issues == ()
    explanation = program.explain(Runtime(), mode="batch")
    assert "field symbol string nullable=false" in explanation
    assert "field x float64 nullable=true" in explanation

    broken = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                table.project(quotes, ["x", "missing"]).with_columns(FeatureSet()),
            )
        ],
    )
    broken_result = broken.analyze(Runtime(), mode="batch")
    assert "outputs.signals.with_columns.value.project.columns[1]" in _issue_paths(
        broken_result
    )


def test_row_local_scalar_functions_infer_types() -> None:
    quotes = _quotes_typed()
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            ("logged", row.log(quotes["x"])),
                            ("expanded", row.exp(quotes["x"])),
                            ("rooted", row.sqrt(quotes["px"])),
                            ("absolute", row.abs(quotes["qty"])),
                            ("inverted", -quotes["qty"]),
                            ("negated", -quotes["x"]),
                            ("flagged", ~quotes["flag"]),
                            ("both", quotes["flag"] & quotes["flag"]),
                        ]
                    )
                ),
            )
        ],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert result.issues == ()
    explanation = program.explain(Runtime(), mode="batch")
    for name in ("logged", "expanded", "negated"):
        assert f"field {name} float64 nullable=true" in explanation
    assert "field rooted float64 nullable=false" in explanation
    assert "field absolute int64 nullable=true" in explanation
    assert "field inverted int64 nullable=true" in explanation
    assert "field flagged bool nullable=true" in explanation
    assert "field both bool nullable=true" in explanation


def test_scalar_functions_reject_non_numeric_operands() -> None:
    quotes = _quotes_typed()
    cases = [
        ("log", row.log(quotes["symbol"])),
        ("exp", row.exp(quotes["flag"])),
        ("sqrt", row.sqrt(quotes["symbol"])),
        ("abs", row.abs(quotes["symbol"])),
        ("neg", -quotes["symbol"]),
        ("not", ~quotes["x"]),
        ("and", quotes["x"] & quotes["x"]),
        ("clip", row.clip(quotes["symbol"], lower=0.0, upper=1.0)),
        ("truediv", quotes["qty"] / quotes["qty"]),
    ]
    for name, value in cases:
        program = _feature_program(quotes, "score", value)
        result = program.analyze(Runtime(), mode="batch")
        dtype_issues = [
            issue
            for issue in result.issues
            if issue.code == "unsupported_type" and issue.path.endswith(".dtype")
        ]
        assert dtype_issues, name

    clean = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            ("clipped", row.clip(quotes["x"], lower=0.0, upper=1.0)),
                            ("ratio", quotes["x"] / quotes["px"]),
                        ]
                    )
                ),
            )
        ],
    )
    assert clean.analyze(Runtime(), mode="batch").issues == ()
    explanation = clean.explain(Runtime(), mode="batch")
    assert "field clipped float64 nullable=true" in explanation
    assert "field ratio float64 nullable=true" in explanation


def test_where_and_coalesce_report_operand_type_mismatches() -> None:
    quotes = _quotes_typed()
    mismatched = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            (
                                "score",
                                row.where(
                                    quotes["flag"], quotes["x"], quotes["symbol"]
                                ),
                            ),
                            (
                                "fallback",
                                row.coalesce(quotes["x"], quotes["symbol"]),
                            ),
                        ]
                    )
                ),
            )
        ],
    )
    mismatched_result = mismatched.analyze(Runtime(), mode="batch")
    assert "outputs.signals.score.where.when_false.dtype" in _issue_paths(
        mismatched_result
    )
    assert "outputs.signals.fallback.coalesce.values[1].dtype" in _issue_paths(
        mismatched_result
    )

    non_bool = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [("score", row.where(quotes["x"], quotes["x"], quotes["x"]))]
                    )
                ),
            )
        ],
    )
    non_bool_result = non_bool.analyze(Runtime(), mode="batch")
    assert "outputs.signals.score.where.condition.dtype" in _issue_paths(
        non_bool_result
    )

    null_fallback = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet([("score", row.coalesce(quotes["px"], None))])
                ),
            )
        ],
    )
    null_result = null_fallback.analyze(Runtime(), mode="batch")
    assert null_result.issues == ()
    explanation = null_fallback.explain(Runtime(), mode="batch")
    assert "field score float64 nullable=false" in explanation


def test_from_columns_requires_one_resolvable_dtype() -> None:
    quotes = _quotes_typed()

    mixed = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "scores",
                linalg.from_columns(quotes, columns=["x", "symbol"], backend="numpy"),
            )
        ],
    )
    mixed_result = mixed.analyze(Runtime(), mode="batch")
    assert "outputs.scores.from_columns.columns[1].dtype" in _issue_paths(mixed_result)

    unknown = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "scores",
                linalg.from_columns(quotes, columns=["missing"], backend="numpy"),
            )
        ],
    )
    unknown_result = unknown.analyze(Runtime(), mode="batch")
    assert "outputs.scores.from_columns.columns[0]" in _issue_paths(unknown_result)


def test_window_group_fields_propagate_into_window_schema() -> None:
    quotes = _quotes_typed()
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "windows",
                window.tumbling(
                    quotes,
                    event_time="ts",
                    size_micros=60_000_000,
                    group_by=["symbol"],
                ),
            )
        ],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert result.issues == ()
    explanation = program.explain(Runtime(), mode="batch")
    assert "field symbol string nullable=false" in explanation

    broken = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "windows",
                window.tumbling(
                    quotes,
                    event_time="ts",
                    size_micros=60_000_000,
                    group_by=["nope"],
                ),
            )
        ],
    )
    broken_result = broken.analyze(Runtime(), mode="batch")
    assert "outputs.windows.window_tumbling.group_by[0]" in _issue_paths(broken_result)


def test_matmul_rejects_mixed_row_lineages() -> None:
    quotes = _quotes_typed()
    trades = table_input(
        "trades",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("y", "float64"),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["ts"],
    )
    program = Program(
        "p",
        inputs=[quotes, trades],
        outputs=[
            (
                "scores",
                linalg.matmul(
                    linalg.from_columns(quotes, columns=["x"], backend="numpy"),
                    linalg.from_columns(trades, columns=["y"], backend="numpy"),
                ),
            )
        ],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert "outputs.scores.matmul.right.lineage" in _issue_paths(result)


def test_elementwise_broadcast_expands_unit_dimensions() -> None:
    quotes = _quotes_typed()
    unit = parameter(
        "unit",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(1, 1),
    )
    base = linalg.from_columns(quotes, columns=["px", "x"], backend="numpy")
    program = Program(
        "p",
        inputs=[quotes, unit],
        outputs=[("scores", base * unit), ("flags", base == base)],
    )
    result = program.analyze(Runtime(), mode="batch")

    assert result.issues == ()
    explanation = program.explain(Runtime(), mode="batch")
    assert (
        "output scores array backend numpy dtype float64 shape (quotes, 2)"
        " lineage quotes" in explanation
    )
    assert "output flags array backend numpy dtype bool" in explanation


def test_array_elementwise_primitive_domains_are_checked_statically() -> None:
    values = table_input(
        "values",
        schema=[
            Field("flag", "bool", nullable=False),
            Field("value", "float32", nullable=False),
        ],
    )
    flags = linalg.from_columns(values, columns=("flag",), backend="numpy")
    numbers = linalg.from_columns(values, columns=("value",), backend="numpy")
    bool_weights = parameter(
        "bool_weights",
        kind="array",
        backend="numpy",
        dtype="bool",
        shape=(1, 1),
    )
    valid = Program(
        "valid-array-booleans",
        inputs=(values,),
        outputs=(("flags", (~flags) & flags),),
    ).analyze(Runtime(), mode="batch")

    assert valid.issues == ()

    invalid = Program(
        "invalid-array-domains",
        inputs=(values, bool_weights),
        outputs=(
            ("not_numbers", ~numbers),
            ("and_numbers", numbers & numbers),
            ("bool_matmul", linalg.matmul(flags, bool_weights)),
        ),
    ).analyze(Runtime(), mode="batch")
    paths = _issue_paths(invalid)

    assert "outputs.not_numbers.not.value.dtype" in paths
    assert "outputs.and_numbers.and.left.dtype" in paths
    assert "outputs.bool_matmul.matmul.left.dtype" in paths


def test_array_true_division_and_weak_scalars_follow_provider_dtypes() -> None:
    values = table_input(
        "values",
        schema=[
            Field("floating", "float32", nullable=False),
            Field("integer", "int32", nullable=False),
        ],
    )
    floating = linalg.from_columns(
        values,
        columns=("floating",),
        backend="numpy",
    )
    numpy_integer = linalg.from_columns(
        values,
        columns=("integer",),
        backend="numpy",
    )
    jax_integer = linalg.from_columns(
        values,
        columns=("integer",),
        backend="jax",
    )
    program = Program(
        "provider-array-dtypes",
        inputs=(values,),
        outputs=(
            ("floating", floating * 2.0),
            ("numpy_division", numpy_integer / 2),
            ("jax_division", jax_integer / 2),
        ),
    )

    assert program.analyze(Runtime(), mode="batch").issues == ()
    explanation = program.explain(Runtime(), mode="batch")
    assert "output floating array backend numpy dtype float32" in explanation
    assert "output numpy_division array backend numpy dtype float64" in explanation
    assert "output jax_division array backend jax dtype float32" in explanation


def test_array_unary_and_binary_dtypes_use_provider_or_fail_closed() -> None:
    values = table_input(
        "values",
        schema=[Field("value", "float32", nullable=False)],
    )
    supported = linalg.from_columns(
        values,
        columns=("value",),
        backend="numpy",
    )
    unsupported = linalg.from_columns(
        values,
        columns=("value",),
        backend="unavailable",
    )
    program = Program(
        "provider-array-dtype-proof",
        inputs=(values,),
        outputs=(
            ("negated", -supported),
            ("unsupported_negated", -unsupported),
            ("unsupported_added", unsupported + unsupported),
        ),
    )

    result = program.analyze(Runtime(), mode="batch")
    paths = _issue_paths(result)

    assert "outputs.unsupported_negated.neg.value.dtype" in paths
    assert "outputs.unsupported_added.add.right.dtype" in paths
    assert "output negated array backend numpy dtype float32" in program.explain(
        Runtime(), mode="batch"
    )


def test_matmul_keeps_known_dtype_when_other_operand_is_unresolved() -> None:
    values = table_input(
        "values",
        schema=[Field("value", "float64", nullable=False)],
    )
    numbers = linalg.from_columns(
        values,
        columns=("value",),
        backend="numpy",
    )
    unresolved = ~numbers
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(1, 1),
    )
    program = Program(
        "partial-matmul-dtype-proof",
        inputs=(values, weights),
        outputs=(
            ("left_unresolved", linalg.matmul(unresolved, weights)),
            ("right_unresolved", linalg.matmul(numbers, unresolved)),
        ),
    )

    result = program.analyze(Runtime(), mode="batch")
    explanation = program.explain(Runtime(), mode="batch")
    paths = _issue_paths(result)

    assert "outputs.left_unresolved.matmul.left.not.value.dtype" in paths
    assert "outputs.right_unresolved.matmul.right.shape[0]" in paths
    assert "output left_unresolved array backend numpy dtype float64" in explanation
    assert "output right_unresolved array backend numpy dtype float64" in explanation


def test_stream_ordering_checks_field_existence_and_nullability() -> None:
    quotes = table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64"),
            Field("x", "float64"),
        ],
        entity_by=["symbol", "industry"],
        event_time="ts",
        sequence_by=["seq"],
    )
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet([("score", ts.mean(quotes["x"], window=rows(3)))])
                ),
            )
        ],
    )
    result = program.analyze(Runtime(), mode="stream")

    paths = _issue_paths(result)
    assert "inputs.quotes.entity_by[1]" in paths
    assert "inputs.quotes.sequence_by[0]" in paths
    assert all(issue.code == "ordering_required" for issue in result.issues)


def test_winsorize_requires_floating_input() -> None:
    quotes = _quotes_typed()

    broken = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            (
                                "score",
                                cs.winsorize(
                                    quotes["qty"],
                                    group=exact_time(quotes["ts"]),
                                    lower=0.1,
                                    upper=0.9,
                                ),
                            )
                        ]
                    )
                ),
            )
        ],
    )
    broken_result = broken.analyze(Runtime(), mode="batch")
    assert "outputs.signals.score.winsorize.value.dtype" in _issue_paths(broken_result)

    clean = Program(
        "p",
        inputs=[quotes],
        outputs=[
            (
                "signals",
                quotes.with_columns(
                    FeatureSet(
                        [
                            (
                                "score",
                                cs.winsorize(
                                    quotes["x"],
                                    group=exact_time(quotes["ts"]),
                                    lower=0.1,
                                    upper=0.9,
                                ),
                            )
                        ]
                    )
                ),
            )
        ],
    )
    assert clean.analyze(Runtime(), mode="batch").issues == ()


def test_explain_renders_static_input_declarations() -> None:
    quotes = _quotes_typed()
    weights = parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(2, 1),
    )
    lookup = parameter(
        "lookup",
        kind="table",
        schema=[Field("k", "string", nullable=False), Field("v", "float64")],
    )
    program = Program(
        "p",
        inputs=[quotes, weights, lookup],
        outputs=[("signals", quotes.with_columns(FeatureSet()))],
    )
    explanation = program.explain(Runtime(), mode="batch")

    assert (
        "static_input weights array backend numpy dtype float64 shape (2, 1)"
        in explanation
    )
    assert "static_input lookup table fields 2" in explanation
