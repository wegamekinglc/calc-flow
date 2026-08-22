from __future__ import annotations

from datetime import timedelta

import pytest

from calc_flow import (
    ArrowFieldSpec,
    JoinStateLimits,
    JoinTimeBounds,
    PipelineBuilder,
)


def test_stream_join_builder_is_immutable_and_serializes_exact_values() -> None:
    left = [
        ArrowFieldSpec("account_id", "int64", False),
        ArrowFieldSpec("authorized_at", "timestamp[us]", False),
    ]
    right = [
        ArrowFieldSpec("account_id", "int64", False),
        ArrowFieldSpec("paid_at", "timestamp[us]", False),
    ]
    original = PipelineBuilder("payments")

    updated = original.stream_join(
        "match",
        left_schema=left,
        right_schema=right,
        left_keys=["account_id"],
        right_keys=["account_id"],
        left_event_time="authorized_at",
        right_event_time="paid_at",
        bounds=JoinTimeBounds(timedelta(minutes=5), timedelta(seconds=30)),
        limits=JoinStateLimits(100, 1_000_000, 1_000),
        left_prefix="authorization",
        right_prefix="payment",
    )

    assert original.project["graph"]["nodes"] == []
    node = updated.project["graph"]["nodes"][0]
    assert node["operator"]["kind"] == "stream_join"
    assert node["operator"]["spec"]["bounds"] == {
        "before_micros": 300_000_000,
        "after_micros": 30_000_000,
    }
    left.append(ArrowFieldSpec("ignored", "string"))
    assert len(node["input_ports"][0]["schema"]) == 2


def test_stream_join_values_reject_unsafe_json_integers_and_bool_limits() -> None:
    with pytest.raises(TypeError, match="exact int"):
        JoinStateLimits(True, 1, 1)
    with pytest.raises(ValueError, match="9007199254740991"):
        JoinStateLimits(9_007_199_254_740_992, 1, 1)
    with pytest.raises(ValueError, match="9007199254740991"):
        JoinTimeBounds(timedelta.max, timedelta())


def test_stream_join_builder_rejects_untyped_schema_values() -> None:
    with pytest.raises(TypeError, match="ArrowFieldSpec"):
        PipelineBuilder("payments").stream_join(
            "match",
            left_schema=[{"name": "id", "data_type": "int64"}],  # type: ignore[list-item]
            right_schema=[ArrowFieldSpec("id", "int64")],
            left_keys=["id"],
            right_keys=["id"],
            left_event_time="ts",
            right_event_time="ts",
            bounds=JoinTimeBounds(timedelta(), timedelta()),
            limits=JoinStateLimits(1, 1, 1),
        )
