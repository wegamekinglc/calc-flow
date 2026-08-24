from __future__ import annotations

import dataclasses

import pytest

from calc_flow.capabilities import RuntimeCapabilities
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


def test_matmul_rejects_rank_dtype_and_backend_mismatch() -> None:
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

    for name, weights in (("flat", flat), ("float32", float32), ("jax", jax_weights)):
        program = Program(
            "p",
            inputs=[quotes, weights],
            outputs=[("scores", linalg.matmul(base, weights))],
        )
        result = program.analyze(Runtime(), mode="batch")
        paths = _issue_paths(result)
        assert paths, name
        assert all(path.startswith("outputs.scores.matmul.right") for path in paths)


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
