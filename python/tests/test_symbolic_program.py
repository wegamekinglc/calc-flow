from __future__ import annotations

import dataclasses

import pytest

from calc_flow.symbolic import (
    AnalysisIssue,
    AnalysisResult,
    ArrayExpr,
    ColumnExpr,
    FeatureSet,
    Field,
    Parameter,
    Program,
    TableExpr,
    linalg,
    parameter,
    table_input,
)
from calc_flow.symbolic.nodes import CSeq, CStr

# Normative v1 program golden vector from
# .codex/artifacts/specs/symbolic-computation-contract.md section 4.4.
PROGRAM_FINGERPRINT = "f09929c7be3d368981565aca0cfd1a3c5becaba3927d06cc25e330912c1e6888"


def _quotes() -> TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("x", "float64"),
            Field("y", "float64"),
        ],
    )


def _weights() -> Parameter[ArrayExpr]:
    return parameter(
        "weights",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=(2, 1),
    )


def test_feature_set_preserves_order_and_copies_caller_sequences() -> None:
    features = [("score", _quotes()["x"]), ("alpha", _quotes()["y"])]
    source = list(features)
    feature_set = FeatureSet(source)
    source.append(("late", _quotes()["x"]))

    assert tuple(name for name, _ in feature_set.features) == ("score", "alpha")
    assert all(
        kept is original
        for (_, kept), (_, original) in zip(feature_set.features, features, strict=True)
    )
    assert type(feature_set.features) is tuple


def test_feature_set_rejects_duplicate_names_with_stable_path() -> None:
    column = _quotes()["x"]
    with pytest.raises(
        ValueError, match=r"^features\[1\].name: duplicate_name:"
    ) as exc:
        FeatureSet([("score", column), ("score", column)])

    assert "score" in str(exc.value)


def test_feature_set_rejects_non_column_values_and_non_string_names() -> None:
    with pytest.raises(TypeError, match=r"^features\[0\].value:"):
        FeatureSet([("score", _quotes())])
    with pytest.raises(TypeError, match=r"^features\[0\].name:"):
        FeatureSet([(1, _quotes()["x"])])


def test_with_feature_returns_new_feature_set_without_mutation() -> None:
    def digests(feature_set: FeatureSet) -> tuple[tuple[str, str], ...]:
        return tuple((name, value.digest) for name, value in feature_set.features)

    first = FeatureSet([("score", _quotes()["x"])])
    grown = first.with_feature("alpha", _quotes()["y"])

    assert digests(first) == (("score", _quotes()["x"].digest),)
    assert digests(grown) == (
        ("score", _quotes()["x"].digest),
        ("alpha", _quotes()["y"].digest),
    )
    with pytest.raises(ValueError, match=r"^features\[2\].name: duplicate_name:"):
        grown.with_feature("alpha", _quotes()["x"])


def test_feature_set_equality_is_object_identity_only() -> None:
    column = _quotes()["x"]
    feature_set = FeatureSet([("score", column)])
    equal_looking = FeatureSet([("score", column)])

    assert feature_set == feature_set
    assert feature_set != equal_looking
    assert feature_set != FeatureSet()


def test_with_columns_builds_table_with_named_feature_children() -> None:
    quotes = _quotes()
    features = FeatureSet([("score", quotes["x"] + quotes["y"])])
    derived = quotes.with_columns(features)

    assert isinstance(derived, TableExpr)
    assert derived.identical(quotes) is False
    node = derived._node
    assert node.op.name == "with_columns"
    assert node.attr("names") == CSeq((CStr("score"),))
    assert len(node.args) == 2
    assert node.args[0].digest == quotes._node.digest


def test_with_columns_rejects_non_feature_set_operand() -> None:
    with pytest.raises(
        TypeError,
        match=r"^calc_flow.symbolic.TableExpr.with_columns.features:"
        " expected FeatureSet; got",
    ):
        _quotes().with_columns([("score", _quotes()["x"])])  # type: ignore[arg-type]


def test_program_copies_inputs_and_outputs_at_construction() -> None:
    quotes = _quotes()
    inputs = [quotes]
    outputs = [("signals", quotes)]
    program = Program("p", inputs=inputs, outputs=outputs)
    inputs.append(_weights())
    outputs.append(("extra", quotes))

    assert program.name == "p"
    assert program.inputs == (quotes,)
    assert program.outputs == (("signals", quotes),)


def test_program_rejects_wrong_host_types() -> None:
    with pytest.raises(TypeError):
        Program("p", inputs=[object()])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        Program("p", outputs=[("signals", object())])  # type: ignore[list-item]
    with pytest.raises(TypeError):
        Program("p", outputs=[(1, _quotes())])  # type: ignore[list-item]
    with pytest.raises(ValueError, match=r"^Program.name:"):
        Program("")


def test_program_rejects_derived_tables_and_wrong_domains_as_inputs() -> None:
    with pytest.raises(TypeError, match=r"^Program.inputs\[0\]:"):
        Program(
            "p", inputs=[linalg.from_columns(_quotes(), columns=["x"], backend="numpy")]
        )  # type: ignore[list-item]
    with pytest.raises(ValueError, match=r"^Program.inputs\[0\]:"):
        Program("p", inputs=[_quotes().with_columns(FeatureSet())])  # type: ignore[list-item]


def test_program_rejects_duplicate_input_names_with_stable_paths() -> None:
    with pytest.raises(ValueError, match=r"^inputs.quotes: duplicate_name:") as exc:
        Program("p", inputs=[_quotes(), _quotes()])
    assert "quotes" in str(exc.value)

    with pytest.raises(ValueError, match=r"^static_inputs.weights: duplicate_name:"):
        Program("p", inputs=[_weights(), _weights()])

    with pytest.raises(ValueError, match=r"^static_inputs.quotes: duplicate_name:"):
        Program("p", inputs=[_quotes(), parameter("quotes", kind="table", schema=[])])


def test_program_rejects_duplicate_output_names_with_stable_paths() -> None:
    quotes = _quotes()
    with pytest.raises(ValueError, match=r"^outputs.signals: duplicate_name:"):
        Program(
            "p", inputs=[quotes], outputs=[("signals", quotes), ("signals", quotes)]
        )


def test_program_equality_is_object_identity_only() -> None:
    quotes = _quotes()
    program = Program("p", inputs=[quotes], outputs=[("signals", quotes)])
    equal_looking = Program("p", inputs=[quotes], outputs=[("signals", quotes)])

    assert program == program
    assert program != equal_looking


def test_with_input_and_output_build_new_programs() -> None:
    quotes = _quotes()
    program = Program("p")
    with_quotes = program.with_input(quotes)
    complete = with_quotes.output("signals", quotes)

    assert program.inputs == ()
    assert program.outputs == ()
    assert with_quotes.inputs == (quotes,)
    assert complete.outputs == (("signals", quotes),)
    assert complete.name == "p"


def test_program_fingerprint_matches_frozen_golden_vector() -> None:
    quotes = table_input("quotes", schema=[Field("x", "float64")])
    program = Program(
        "p",
        inputs=[quotes],
        outputs=[("signals", quotes)],
    )

    assert program.fingerprint == PROGRAM_FINGERPRINT


def test_program_fingerprint_ignores_construction_order_not_declaration_order() -> None:
    first = _quotes()["x"] + _quotes()["y"]
    second = _quotes()["y"] + _quotes()["x"]

    def build(left: ColumnExpr, right: ColumnExpr) -> Program:
        quotes = _quotes()
        return Program(
            "p",
            inputs=[quotes],
            outputs=[
                ("signals", quotes.with_columns(FeatureSet([("score", left + right)])))
            ],
        )

    assert build(first, second).fingerprint == build(first, second).fingerprint
    assert build(first, second).fingerprint != build(second, first).fingerprint

    quotes = _quotes()
    weights = _weights()
    scores = linalg.matmul(
        linalg.from_columns(quotes, columns=["x", "y"], backend="numpy"),
        weights,
    )
    table = quotes.with_columns(FeatureSet())
    direct = Program(
        "p",
        inputs=[quotes, weights],
        outputs=[("signals", table), ("scores", scores)],
    )
    reordered_inputs = Program(
        "p",
        inputs=[weights, quotes],
        outputs=[("signals", table), ("scores", scores)],
    )
    reordered_outputs = Program(
        "p",
        inputs=[quotes, weights],
        outputs=[("scores", scores), ("signals", table)],
    )

    assert direct.fingerprint != reordered_inputs.fingerprint
    assert direct.fingerprint != reordered_outputs.fingerprint


def test_program_fingerprint_needs_no_runtime() -> None:
    quotes = _quotes()
    program = Program("p", inputs=[quotes], outputs=[("signals", quotes)])

    assert program.fingerprint == program.fingerprint
    assert len(program.fingerprint) == 64
    assert program.fingerprint == program.fingerprint.lower()


def test_analysis_issue_and_result_are_frozen_value_objects() -> None:
    issue = AnalysisIssue(
        path="outputs.signals.score",
        code="unresolved_type",
        message="unknown field",
    )
    same = AnalysisIssue(
        path="outputs.signals.score",
        code="unresolved_type",
        message="unknown field",
    )
    assert issue == same
    with pytest.raises(dataclasses.FrozenInstanceError):
        issue.path = "outputs.signals.alpha"  # type: ignore[misc]

    result = AnalysisResult(
        mode="batch",
        program_fingerprint="0" * 64,
        capability_session_id="session",
        capability_revision=1,
        issues=(issue,),
    )
    same_result = AnalysisResult(
        mode="batch",
        program_fingerprint="0" * 64,
        capability_session_id="session",
        capability_revision=1,
        issues=(same,),
    )
    assert result == same_result
    assert result != AnalysisResult(
        mode="stream",
        program_fingerprint="0" * 64,
        capability_session_id="session",
        capability_revision=1,
        issues=(issue,),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.mode = "stream"  # type: ignore[misc]
