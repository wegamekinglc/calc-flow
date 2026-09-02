from __future__ import annotations

from datetime import timedelta

import pytest

from calc_flow import JoinStateLimits, JoinTimeBounds, Runtime
from calc_flow.errors import CompileError
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    cs,
    exact_time,
    table,
    table_input,
    ts,
)
from calc_flow.symbolic.lower import lower_program_document


def _bounds() -> JoinTimeBounds:
    return JoinTimeBounds(timedelta(seconds=5), timedelta(seconds=2))


def _limits() -> JoinStateLimits:
    return JoinStateLimits(1_000, 16 * 1024 * 1024, 10_000)


def _input(name: str, value_name: str):
    return table_input(
        name,
        schema=[
            Field("key", "int64", nullable=False),
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field(value_name, "float64", nullable=False),
        ],
        entity_by=["key"],
        event_time="ts",
        sequence_by=["sequence"],
    )


def _join(
    left,
    right,
    *,
    left_prefix: str,
    right_prefix: str,
    ordered_output: bool = False,
):
    output_ordering = (
        {
            "output_entity_by": [f"{left_prefix}__key"],
            "output_event_time": f"{left_prefix}__ts",
            "output_sequence_by": [
                f"{left_prefix}__sequence",
                f"{right_prefix}__sequence",
            ],
        }
        if ordered_output
        else {}
    )
    return table.stream_join(
        left,
        right,
        left_keys=["key"],
        right_keys=["key"],
        left_event_time="ts",
        right_event_time="ts",
        bounds=_bounds(),
        limits=_limits(),
        left_prefix=left_prefix,
        right_prefix=right_prefix,
        **output_ordering,
    )


def test_stream_join_v2_output_ordering_is_immutable_and_versioned() -> None:
    left = _input("left_events", "left_value")
    right = _input("right_events", "right_value")
    entity_by = ["left__key"]
    sequence_by = ["left__sequence", "right__sequence"]

    joined = table.stream_join(
        left,
        right,
        left_keys=["key"],
        right_keys=["key"],
        left_event_time="ts",
        right_event_time="ts",
        bounds=_bounds(),
        limits=_limits(),
        output_entity_by=entity_by,
        output_event_time="left__ts",
        output_sequence_by=sequence_by,
    )
    same = _join(
        left,
        right,
        left_prefix="left",
        right_prefix="right",
        ordered_output=True,
    )

    entity_by.append("ignored")
    sequence_by.clear()
    assert joined.identical(same)
    assert "stream_join@2" in joined.explain()


def test_stream_join_v1_identity_remains_frozen() -> None:
    joined = _join(
        _input("left_events", "left_value"),
        _input("right_events", "right_value"),
        left_prefix="left",
        right_prefix="right",
    )

    assert (
        joined._node.digest
        == "b4e1c30a8d73116d951c3e84d8ba3fc01889315da66b22af908b735a8f04eca7"
    )
    assert "stream_join@1" in joined.explain()


@pytest.mark.parametrize(
    "ordering",
    [
        {"output_event_time": "left__ts"},
        {"output_entity_by": ["left__key"]},
        {"output_sequence_by": ["left__sequence"]},
    ],
)
def test_stream_join_v2_requires_complete_output_ordering(
    ordering: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="output ordering"):
        table.stream_join(
            _input("left_events", "left_value"),
            _input("right_events", "right_value"),
            left_keys=["key"],
            right_keys=["key"],
            left_event_time="ts",
            right_event_time="ts",
            bounds=_bounds(),
            limits=_limits(),
            **ordering,
        )


def test_nested_join_requires_and_accepts_explicit_intermediate_ordering() -> None:
    left = _input("left_events", "left_value")
    middle = _input("middle_events", "middle_value")
    right = _input("right_events", "right_value")
    unordered = _join(
        left,
        middle,
        left_prefix="left",
        right_prefix="middle",
    )
    invalid = table.stream_join(
        unordered,
        right,
        left_keys=["left__key"],
        right_keys=["key"],
        left_event_time="left__ts",
        right_event_time="ts",
        bounds=_bounds(),
        limits=_limits(),
        left_prefix="matched",
        right_prefix="right",
    )
    invalid_result = Program(
        "nested-unordered",
        inputs=[left, middle, right],
        outputs=[("matches", invalid)],
    ).analyze(Runtime(), mode="stream")
    assert any(issue.code == "ordering_required" for issue in invalid_result.issues)

    ordered = _join(
        left,
        middle,
        left_prefix="left",
        right_prefix="middle",
        ordered_output=True,
    )
    nested = table.stream_join(
        ordered,
        right,
        left_keys=["left__key"],
        right_keys=["key"],
        left_event_time="left__ts",
        right_event_time="ts",
        bounds=_bounds(),
        limits=_limits(),
        left_prefix="matched",
        right_prefix="right",
    )
    program = Program(
        "nested-ordered",
        inputs=[left, middle, right],
        outputs=[("matches", nested)],
    )

    assert program.analyze(Runtime(), mode="stream").issues == ()
    assert "stream_join state_stages 2" in program.explain(Runtime(), mode="stream")
    document = lower_program_document(program, Runtime(), "stream")
    joins = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "stream_join"
    ]
    assert len(joins) == 2
    assert program.compile_stream(Runtime()).source_binding_ids == (
        "left_events.input",
        "middle_events.input",
        "right_events.input",
    )


def test_stream_join_v2_rejects_unproved_output_ordering_fields() -> None:
    left = _input("left_events", "left_value")
    right = _input("right_events", "right_value")
    joined = table.stream_join(
        left,
        right,
        left_keys=["key"],
        right_keys=["key"],
        left_event_time="ts",
        right_event_time="ts",
        bounds=_bounds(),
        limits=_limits(),
        output_entity_by=["right__key"],
        output_event_time="left__left_value",
        output_sequence_by=["left__sequence"],
    )
    result = Program(
        "invalid-post-join-ordering",
        inputs=[left, right],
        outputs=[("matches", joined)],
    ).analyze(Runtime(), mode="stream")

    paths = {issue.path for issue in result.issues}
    assert "outputs.matches.stream_join.output_entity_by" in paths
    assert "outputs.matches.stream_join.output_event_time" in paths
    assert "outputs.matches.stream_join.output_sequence_by" in paths


def test_nested_join_rejects_mismatched_or_projected_away_ordering() -> None:
    left = _input("left_events", "left_value")
    middle = _input("middle_events", "middle_value")
    right = _input("right_events", "right_value")
    ordered = _join(
        left,
        middle,
        left_prefix="left",
        right_prefix="middle",
        ordered_output=True,
    )
    wrong_time = table.stream_join(
        ordered,
        right,
        left_keys=["left__key"],
        right_keys=["key"],
        left_event_time="middle__ts",
        right_event_time="ts",
        bounds=_bounds(),
        limits=_limits(),
    )
    wrong_result = Program(
        "nested-wrong-time",
        inputs=[left, middle, right],
        outputs=[("matches", wrong_time)],
    ).analyze(Runtime(), mode="stream")
    assert any(
        issue.path.endswith("left_event_time") and issue.code == "ordering_required"
        for issue in wrong_result.issues
    )

    projected = table.project(
        ordered,
        ["left__key", "left__ts", "left__sequence", "middle__middle_value"],
    )
    missing_sequence = table.stream_join(
        projected,
        right,
        left_keys=["left__key"],
        right_keys=["key"],
        left_event_time="left__ts",
        right_event_time="ts",
        bounds=_bounds(),
        limits=_limits(),
    )
    projected_result = Program(
        "nested-projected-ordering",
        inputs=[left, middle, right],
        outputs=[("matches", missing_sequence)],
    ).analyze(Runtime(), mode="stream")
    assert any(issue.code == "ordering_required" for issue in projected_result.issues)


def test_nested_join_lowers_row_local_segment_between_join_boundaries() -> None:
    left = _input("left_events", "left_value")
    middle = _input("middle_events", "middle_value")
    right = _input("right_events", "right_value")
    ordered = _join(
        left,
        middle,
        left_prefix="left",
        right_prefix="middle",
        ordered_output=True,
    )
    positive = table.filter(ordered, ordered["left__left_value"] > 0.0)
    nested = table.stream_join(
        positive,
        right,
        left_keys=["left__key"],
        right_keys=["key"],
        left_event_time="left__ts",
        right_event_time="ts",
        bounds=_bounds(),
        limits=_limits(),
    )
    program = Program(
        "nested-row-local-segment",
        inputs=[left, middle, right],
        outputs=[("matches", nested)],
    )

    document = lower_program_document(program, Runtime(), "stream")
    kinds = [node["operator"]["kind"] for node in document["graph"]["nodes"]]
    assert kinds.count("stream_join") == 2
    assert kinds.count("expression") == 5
    program.compile_stream(Runtime())


def test_independent_joins_and_unrelated_output_share_declared_sources() -> None:
    first = _input("first_events", "first_value")
    second = _input("second_events", "second_value")
    third = _input("third_events", "third_value")
    fourth = _input("fourth_events", "fourth_value")
    first_join = _join(
        first,
        second,
        left_prefix="first",
        right_prefix="second",
    )
    second_join = _join(
        third,
        fourth,
        left_prefix="third",
        right_prefix="fourth",
    )
    passthrough = table.project(first, ["key", "first_value"])
    program = Program(
        "independent-joins",
        inputs=[first, second, third, fourth],
        outputs=[
            ("first_matches", first_join),
            ("second_matches", second_join),
            ("first_values", passthrough),
        ],
    )

    document = lower_program_document(program, Runtime(), "stream")
    joins = [
        node
        for node in document["graph"]["nodes"]
        if node["operator"]["kind"] == "stream_join"
    ]
    assert len(joins) == 2
    assert program.compile_stream(Runtime()).source_binding_ids == (
        "first_events.input",
        "fourth_events.input",
        "second_events.input",
        "third_events.input",
    )


def test_relational_source_id_avoids_output_name_collision() -> None:
    left = _input("matches", "left_value")
    right = _input("right_events", "right_value")
    joined = _join(
        left,
        right,
        left_prefix="left",
        right_prefix="right",
        ordered_output=True,
    )
    program = Program(
        "source-output-collision",
        inputs=[left, right],
        outputs=[("matches", joined)],
    )

    source_ids = program.compile_stream(Runtime()).source_binding_ids
    assert len(source_ids) == 2
    assert any(name.startswith("cf_source_") for name in source_ids)
    assert "right_events.input" in source_ids


def test_relational_source_id_avoids_generated_stage_collision() -> None:
    left = _input("features__cf_rolling", "left_value")
    right = _input("right_events", "right_value")
    joined = _join(
        left,
        right,
        left_prefix="left",
        right_prefix="right",
        ordered_output=True,
    )
    output = joined.with_columns(
        FeatureSet([("previous", ts.lag(joined["left__left_value"]))])
    )
    program = Program(
        "source-generated-stage-collision",
        inputs=[left, right],
        outputs=[("features", output)],
    )

    assert program.analyze(Runtime(), mode="stream").issues == ()
    source_ids = program.compile_stream(Runtime()).source_binding_ids
    assert len(source_ids) == 2
    assert any(name.startswith("cf_source_") for name in source_ids)
    assert "right_events.input" in source_ids


def test_relational_source_id_avoids_join_side_stage_collision() -> None:
    left = _input("left_events", "left_value")
    right = _input("right_events", "right_value")
    left_with_lag = left.with_columns(
        FeatureSet([("previous", ts.lag(left["left_value"]))])
    )
    joined = _join(
        left_with_lag,
        right,
        left_prefix="left",
        right_prefix="right",
    )
    unrelated = _input(
        f"cf_stream_join_{joined._node.digest[:16]}__left__cf_rolling",
        "unrelated_value",
    )
    program = Program(
        "source-join-side-stage-collision",
        inputs=[left, right, unrelated],
        outputs=[
            ("matches", joined),
            ("unrelated", table.project(unrelated, ["key", "unrelated_value"])),
        ],
    )

    assert program.analyze(Runtime(), mode="stream").issues == ()
    source_ids = program.compile_stream(Runtime()).source_binding_ids
    assert len(source_ids) == 3
    assert any(name.startswith("cf_source_") for name in source_ids)
    assert "left_events.input" in source_ids
    assert "right_events.input" in source_ids


def test_post_join_rolling_requires_and_uses_explicit_output_ordering() -> None:
    left = _input("left_events", "left_value")
    right = _input("right_events", "right_value")
    unordered = _join(
        left,
        right,
        left_prefix="left",
        right_prefix="right",
    )
    invalid_output = unordered.with_columns(
        FeatureSet([("previous", ts.lag(unordered["left__left_value"]))])
    )
    invalid = Program(
        "post-join-unordered",
        inputs=[left, right],
        outputs=[("features", invalid_output)],
    )
    with pytest.raises(CompileError, match="ordering_required"):
        invalid.compile_stream(Runtime())

    joined = _join(
        left,
        right,
        left_prefix="left",
        right_prefix="right",
        ordered_output=True,
    )
    output = joined.with_columns(
        FeatureSet([("previous", ts.lag(joined["left__left_value"]))])
    )
    program = Program(
        "post-join-rolling",
        inputs=[left, right],
        outputs=[("features", output)],
    )

    document = lower_program_document(program, Runtime(), "stream")
    kinds = [node["operator"]["kind"] for node in document["graph"]["nodes"]]
    assert kinds.count("stream_join") == 1
    assert kinds.count("rolling") == 1
    assert program.compile_stream(Runtime()).source_binding_ids == (
        "left_events.input",
        "right_events.input",
    )


def test_post_join_ordering_is_required_when_upstream_uses_same_state_kind() -> None:
    left = _input("left_events", "left_value")
    right = _input("right_events", "right_value")
    left_with_lag = left.with_columns(
        FeatureSet([("left_previous", ts.lag(left["left_value"]))])
    )
    joined = _join(
        left_with_lag,
        right,
        left_prefix="left",
        right_prefix="right",
    )
    post_join_lag = joined.with_columns(
        FeatureSet([("previous", ts.lag(joined["left__left_value"]))])
    )
    result = Program(
        "same-state-kind",
        inputs=[left, right],
        outputs=[("features", post_join_lag)],
    ).analyze(Runtime(), mode="stream")

    assert any(issue.code == "ordering_required" for issue in result.issues)


def test_post_join_cross_section_uses_explicit_output_ordering() -> None:
    left = _input("left_events", "left_value")
    right = _input("right_events", "right_value")
    joined = _join(
        left,
        right,
        left_prefix="left",
        right_prefix="right",
        ordered_output=True,
    )
    output = joined.with_columns(
        FeatureSet(
            [
                (
                    "rank",
                    cs.rank(
                        joined["left__left_value"],
                        group=exact_time(joined["left__ts"]),
                    ),
                )
            ]
        )
    )
    program = Program(
        "post-join-cross-section",
        inputs=[left, right],
        outputs=[("features", output)],
    )

    document = lower_program_document(program, Runtime(), "stream")
    kinds = [node["operator"]["kind"] for node in document["graph"]["nodes"]]
    assert kinds.count("stream_join") == 1
    assert kinds.count("cross_section") == 1
    program.compile_stream(Runtime())
