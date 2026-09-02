from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest

from calc_flow import JoinStateLimits, JoinTimeBounds, Runtime
from calc_flow.capabilities import RuntimeCapabilities
from calc_flow.errors import CompileError
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    table,
    table_input,
    ts,
    window,
)
from calc_flow.symbolic.lower import lower_program_document


def _left(*, ordered: bool = True):
    return table_input(
        "authorizations",
        schema=[
            Field("account_id", "int64", nullable=False),
            Field("authorized_at", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("amount", "float64"),
        ],
        entity_by=["account_id"] if ordered else (),
        event_time="authorized_at" if ordered else None,
        sequence_by=["sequence"] if ordered else (),
    )


def _right(*, ordered: bool = True, key_type: str = "int64"):
    return table_input(
        "payments",
        schema=[
            Field("account_id", key_type, nullable=False),
            Field("paid_at", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("settled", "bool", nullable=False),
        ],
        entity_by=["account_id"] if ordered else (),
        event_time="paid_at" if ordered else None,
        sequence_by=["sequence"] if ordered else (),
    )


def _bounds() -> JoinTimeBounds:
    return JoinTimeBounds(timedelta(seconds=5), timedelta(seconds=2))


def _limits() -> JoinStateLimits:
    return JoinStateLimits(100, 1_000_000, 1_000)


def _join(left=None, right=None):
    left = _left() if left is None else left
    right = _right() if right is None else right
    return table.stream_join(
        left,
        right,
        left_keys=["account_id"],
        right_keys=["account_id"],
        left_event_time="authorized_at",
        right_event_time="paid_at",
        bounds=_bounds(),
        limits=_limits(),
        left_prefix="authorization",
        right_prefix="payment",
    )


def _program(output=None, *, left=None, right=None) -> Program:
    left = _left() if left is None else left
    right = _right() if right is None else right
    joined = _join(left, right) if output is None else output
    return Program(
        "payment_match",
        inputs=[left, right],
        outputs=[("matches", joined)],
    )


def test_stream_join_declaration_is_immutable_and_deterministic() -> None:
    left = _left()
    right = _right()
    left_keys = ["account_id"]
    right_keys = ["account_id"]
    first = table.stream_join(
        left,
        right,
        left_keys=left_keys,
        right_keys=right_keys,
        left_event_time="authorized_at",
        right_event_time="paid_at",
        bounds=_bounds(),
        limits=_limits(),
        left_prefix="authorization",
        right_prefix="payment",
    )
    second = _join(left, right)

    assert first.identical(second)
    left_keys.append("ignored")
    right_keys.clear()
    assert first.identical(second)
    assert "stream_join@1" in first.explain()


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"left_keys": []}, "non-empty"),
        ({"right_keys": ["account_id", "sequence"]}, "equal length"),
        ({"left_keys": "account_id"}, r"Sequence\[str\]"),
        ({"left_event_time": ""}, "non-empty string"),
        ({"bounds": object()}, "JoinTimeBounds"),
        ({"limits": object()}, "JoinStateLimits"),
        ({"left_prefix": "same", "right_prefix": "same"}, "prefixes"),
    ],
)
def test_stream_join_declaration_rejects_invalid_configuration(
    changes: dict[str, object], match: str
) -> None:
    arguments: dict[str, object] = {
        "left_keys": ["account_id"],
        "right_keys": ["account_id"],
        "left_event_time": "authorized_at",
        "right_event_time": "paid_at",
        "bounds": _bounds(),
        "limits": _limits(),
        "left_prefix": "authorization",
        "right_prefix": "payment",
    }
    arguments.update(changes)

    with pytest.raises((TypeError, ValueError), match=match):
        table.stream_join(_left(), _right(), **arguments)


def test_stream_join_analysis_infers_prefixed_exact_schema() -> None:
    program = _program()

    result = program.analyze(Runtime(), mode="stream")

    assert result.issues == ()
    explanation = program.explain(Runtime(), mode="stream")
    assert "field authorization__account_id int64 nullable=false" in explanation
    assert "field authorization__amount float64 nullable=true" in explanation
    assert "field payment__paid_at timestamp[us, UTC] nullable=false" in explanation
    assert "field payment__settled bool nullable=false" in explanation
    assert "state stream_join" in explanation
    assert "stream_join state_stages 1" in explanation
    assert "max_state_rows_per_side=100" in explanation


def test_stream_join_analysis_rejects_batch_and_missing_ordering() -> None:
    batch = _program().analyze(Runtime(), mode="batch")
    assert any(issue.code == "unsupported_mode" for issue in batch.issues)
    with pytest.raises(CompileError, match="unsupported_mode"):
        _program().compile_batch(Runtime())

    left = _left(ordered=False)
    right = _right(ordered=False)
    unordered = _program(_join(left, right), left=left, right=right).analyze(
        Runtime(), mode="stream"
    )
    paths = {issue.path for issue in unordered.issues}
    assert {
        "inputs.authorizations.event_time",
        "inputs.authorizations.entity_by",
        "inputs.authorizations.sequence_by",
        "inputs.payments.event_time",
        "inputs.payments.entity_by",
        "inputs.payments.sequence_by",
    } <= paths


def test_stream_join_analysis_rejects_schema_mismatches() -> None:
    right = _right(key_type="string")
    key_mismatch = _program(_join(_left(), right), right=right).analyze(
        Runtime(), mode="stream"
    )
    assert any(
        issue.path.endswith("right_keys[0]") and issue.code == "schema_mismatch"
        for issue in key_mismatch.issues
    )

    missing = table.stream_join(
        _left(),
        _right(),
        left_keys=["missing"],
        right_keys=["account_id"],
        left_event_time="authorized_at",
        right_event_time="paid_at",
        bounds=_bounds(),
        limits=_limits(),
    )
    missing_result = _program(missing).analyze(Runtime(), mode="stream")
    assert any(
        issue.path.endswith("left_keys[0]") and issue.code == "unresolved_type"
        for issue in missing_result.issues
    )


def test_stream_join_requires_ordering_for_nested_and_post_join_state() -> None:
    first = _join()
    nested = table.stream_join(
        first,
        _right(),
        left_keys=["authorization__account_id"],
        right_keys=["account_id"],
        left_event_time="authorization__authorized_at",
        right_event_time="paid_at",
        bounds=_bounds(),
        limits=_limits(),
    )
    nested_program = Program(
        "nested",
        inputs=[_left(), _right()],
        outputs=[("matches", nested)],
    )
    assert any(
        issue.code == "ordering_required"
        for issue in nested_program.analyze(Runtime(), mode="stream").issues
    )

    joined = _join()
    stateful = joined.with_columns(
        FeatureSet([("previous", ts.lag(joined["authorization__amount"]))])
    )
    stateful_result = _program(stateful).analyze(Runtime(), mode="stream")
    assert any(issue.code == "ordering_required" for issue in stateful_result.issues)

    windowed = window.tumbling(
        joined,
        event_time="authorization__authorized_at",
        size_micros=1_000_000,
    )
    windowed_result = _program(windowed).analyze(Runtime(), mode="stream")
    assert any(issue.code == "ordering_required" for issue in windowed_result.issues)


def test_stream_join_lowering_supports_multiple_and_unrelated_outputs() -> None:
    left = _left()
    right = _right()
    first = _join(left, right)
    second = table.stream_join(
        left,
        right,
        left_keys=["account_id"],
        right_keys=["account_id"],
        left_event_time="authorized_at",
        right_event_time="paid_at",
        bounds=JoinTimeBounds(timedelta(seconds=1), timedelta(seconds=1)),
        limits=_limits(),
    )
    multiple = Program(
        "multiple",
        inputs=[left, right],
        outputs=[("first", first), ("second", second)],
    )
    multiple_document = lower_program_document(multiple, Runtime(), "stream")
    assert (
        sum(
            node["operator"]["kind"] == "stream_join"
            for node in multiple_document["graph"]["nodes"]
        )
        == 2
    )
    assert multiple.compile_stream(Runtime()).source_binding_ids == (
        "authorizations.input",
        "payments.input",
    )

    unrelated = Program(
        "unrelated",
        inputs=[left, right],
        outputs=[("matches", first), ("left", left)],
    )
    unrelated_document = lower_program_document(unrelated, Runtime(), "stream")
    assert (
        sum(
            node["operator"]["kind"] == "stream_join"
            for node in unrelated_document["graph"]["nodes"]
        )
        == 1
    )
    unrelated.compile_stream(Runtime())


def test_stream_join_direct_root_uses_one_native_node() -> None:
    program = _program()

    document = lower_program_document(program, Runtime(), "stream")
    plan = program.compile_stream(Runtime())

    nodes = document["graph"]["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["operator"]["kind"] == "stream_join"
    assert document["graph"]["edges"] == []
    assert plan.source_binding_ids == ("left", "right")
    assert plan.sink_binding_ids == ("output",)


def test_stream_join_lowers_stateless_input_stages_before_native_join() -> None:
    left = _left()
    right = _right()
    filtered_left = table.filter(left, left["amount"] > 0.0)
    program = _program(_join(filtered_left, right), left=left, right=right)

    document = lower_program_document(program, Runtime(), "stream")
    nodes = document["graph"]["nodes"]
    joins = [node for node in nodes if node["operator"]["kind"] == "stream_join"]

    assert len(joins) == 1
    join_id = joins[0]["id"]
    incoming = [
        edge for edge in document["graph"]["edges"] if edge["target_node"] == join_id
    ]
    assert len(incoming) == 1
    assert incoming[0]["target_port"] == "left"
    assert program.compile_stream(Runtime()).source_binding_ids == ("input", "right")


def test_stream_join_lowers_once_and_shares_post_join_branches() -> None:
    left = _left()
    right = _right()
    joined = _join(left, right)
    settled = table.filter(joined, joined["payment__settled"])
    amounts = table.project(
        joined.with_columns(
            FeatureSet(
                [
                    (
                        "double_amount",
                        joined["authorization__amount"] * 2.0,
                    )
                ]
            )
        ),
        ["authorization__account_id", "double_amount"],
    )
    program = Program(
        "payment_match",
        inputs=[left, right],
        outputs=[("settled", settled), ("amounts", amounts)],
    )

    document = lower_program_document(program, Runtime(), "stream")

    nodes = document["graph"]["nodes"]
    joins = [node for node in nodes if node["operator"]["kind"] == "stream_join"]
    assert len(joins) == 1
    spec = joins[0]["operator"]["spec"]
    assert spec == {
        "join_type": "inner",
        "left_keys": ["account_id"],
        "right_keys": ["account_id"],
        "left_event_time": "authorized_at",
        "right_event_time": "paid_at",
        "bounds": {"before_micros": 5_000_000, "after_micros": 2_000_000},
        "limits": {
            "max_state_rows_per_side": 100,
            "max_state_bytes_per_side": 1_000_000,
            "max_matches_per_input_batch": 1_000,
        },
        "left_prefix": "authorization",
        "right_prefix": "payment",
    }
    join_id = joins[0]["id"]
    outgoing = [
        edge for edge in document["graph"]["edges"] if edge["source_node"] == join_id
    ]
    assert len(outgoing) == 1
    assert not any(
        edge["target_node"] == join_id for edge in document["graph"]["edges"]
    )
    plan = program.compile_stream(Runtime())
    assert plan.source_binding_ids == ("left", "right")


def test_stream_join_capability_gate_rejects_missing_replay_proof() -> None:
    class UnsafeRuntime(Runtime):
        def capabilities(self) -> RuntimeCapabilities:
            base = super().capabilities()
            operators = tuple(
                dataclasses.replace(operator, replay_safe=False)
                if operator.kind == "stream_join"
                else operator
                for operator in base.operators
            )
            return dataclasses.replace(base, operators=operators)

    with pytest.raises(CompileError, match="capability_mismatch"):
        _program().compile_stream(UnsafeRuntime())
