from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest

import calc_flow as cf


def _spec() -> cf.AsofJoinSpec:
    return cf.AsofJoinSpec(
        cf.AsofJoinSide(["symbol"], "time", ["sequence"], "trade"),
        cf.AsofJoinSide(["symbol"], "time", ["sequence"], "quote"),
        timedelta(microseconds=10),
        cf.AsofStateLimits(100, 1_000_000),
    )


def test_asof_spec_defensively_copies_identity_and_encodes_exact_nested_wire() -> None:
    keys = ["symbol"]
    sequence = ["sequence"]
    side = cf.AsofJoinSide(keys, "time", sequence, "trade")
    keys.clear()
    sequence.append("ignored")
    assert side.keys == ("symbol",)
    assert side.sequence_by == ("sequence",)
    with pytest.raises(FrozenInstanceError):
        side.event_time = "changed"
    fields = [
        cf.ArrowFieldSpec("symbol", "string", False),
        cf.ArrowFieldSpec("time", "timestamp[us, UTC]", False),
        cf.ArrowFieldSpec("sequence", "uint64", False),
    ]
    original = cf.PipelineBuilder("trades")
    changed = original.stream_asof_join(
        "matched", left_schema=fields, right_schema=fields, spec=_spec()
    )
    fields.clear()
    assert original.project["graph"]["nodes"] == []
    node = changed.project["graph"]["nodes"][0]
    assert node["operator"] == {
        "kind": "stream_asof_join",
        "spec": {
            "left": {
                "keys": ["symbol"],
                "event_time": "time",
                "sequence_by": ["sequence"],
                "prefix": "trade",
            },
            "right": {
                "keys": ["symbol"],
                "event_time": "time",
                "sequence_by": ["sequence"],
                "prefix": "quote",
            },
            "tolerance_micros": 10,
            "limits": {"max_state_rows": 100, "max_state_bytes": 1_000_000},
            "late_policy": "error",
        },
    }
    assert [port["name"] for port in node["input_ports"]] == ["left", "right"]
    assert len(node["input_ports"][0]["schema"]) == 3
    assert node["output_ports"][0]["name"] == "output"
    assert [field["nullable"] for field in node["output_ports"][0]["schema"]] == [
        False
    ] * 3 + [True] * 3


@pytest.mark.parametrize("value", [True, 1.0, "1", None])
def test_asof_limits_reject_non_integer_values(value: object) -> None:
    with pytest.raises(TypeError, match="exact int"):
        cf.AsofStateLimits(value, 1)
    with pytest.raises(TypeError, match="exact int"):
        cf.AsofStateLimits(1, value)


@pytest.mark.parametrize("value", [0, -1, 9_007_199_254_740_992])
def test_asof_limits_reject_values_outside_positive_safe_integer_range(
    value: int,
) -> None:
    with pytest.raises(ValueError, match="9007199254740991"):
        cf.AsofStateLimits(value, 1)


@pytest.mark.parametrize("values", ["symbol", [], ["symbol", "symbol"], [""], [True]])
def test_asof_side_rejects_invalid_identity_lists(values: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        cf.AsofJoinSide(values, "time", ["sequence"], "left")
    with pytest.raises((TypeError, ValueError)):
        cf.AsofJoinSide(["symbol"], "time", values, "left")


@pytest.mark.parametrize("prefix", ["", "1left", "左", "left-right", True])
def test_asof_side_requires_ascii_identifier_prefix(prefix: object) -> None:
    with pytest.raises((TypeError, ValueError), match="prefix"):
        cf.AsofJoinSide(["symbol"], "time", ["sequence"], prefix)


@pytest.mark.parametrize(
    "duration", [1, True, "10us", timedelta(microseconds=-1), timedelta.max]
)
def test_asof_spec_rejects_invalid_tolerance(duration: object) -> None:
    spec = _spec()
    with pytest.raises((TypeError, ValueError), match="tolerance"):
        cf.AsofJoinSpec(spec.left, spec.right, duration, spec.limits)


@pytest.mark.parametrize(
    "changes", [{"right": None}, {"limits": {}}, {"late_policy": "ignore"}]
)
def test_asof_spec_requires_typed_sides_limits_and_explicit_late_policy(
    changes: dict[str, object],
) -> None:
    spec = _spec()
    values = {
        "left": spec.left,
        "right": spec.right,
        "tolerance": spec.tolerance,
        "limits": spec.limits,
    }
    with pytest.raises((TypeError, ValueError)):
        cf.AsofJoinSpec(**(values | changes))


def test_asof_spec_rejects_equal_prefixes_and_unequal_key_counts() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="prefix"):
        cf.AsofJoinSpec(spec.left, spec.left, spec.tolerance, spec.limits)
    different_keys = cf.AsofJoinSide(["symbol", "venue"], "time", ["sequence"], "quote")
    with pytest.raises(ValueError, match="equal length"):
        cf.AsofJoinSpec(spec.left, different_keys, spec.tolerance, spec.limits)


def test_asof_tolerance_preserves_safe_integer_maximum_without_float_rounding() -> None:
    spec = _spec()
    largest = cf.AsofJoinSpec(
        spec.left,
        spec.right,
        timedelta(microseconds=9_007_199_254_740_991),
        spec.limits,
    )
    from calc_flow.asof_join_spec import _asof_wire_spec

    assert _asof_wire_spec(largest)["tolerance_micros"] == 9_007_199_254_740_991


@pytest.mark.parametrize("name", ["", 1, None])
def test_asof_builder_rejects_invalid_node_name(name) -> None:
    with pytest.raises((TypeError, ValueError), match="name"):
        cf.PipelineBuilder("bad").stream_asof_join(
            name, left_schema=[], right_schema=[], spec=_spec()
        )


def test_asof_builder_rejects_untyped_spec_and_prefix_field_collision() -> None:
    with pytest.raises(TypeError, match="AsofJoinSpec"):
        cf.PipelineBuilder("bad").stream_asof_join(
            "node", left_schema=[], right_schema=[], spec={}
        )
    spec = _spec()
    collision = cf.AsofJoinSpec(
        cf.AsofJoinSide(["y__z"], "time", ["sequence"], "x"),
        cf.AsofJoinSide(["z"], "time", ["sequence"], "x__y"),
        spec.tolerance,
        spec.limits,
    )
    with pytest.raises(ValueError, match="collision"):
        cf.PipelineBuilder("bad").stream_asof_join(
            "node",
            left_schema=[cf.ArrowFieldSpec("y__z", "string", False)],
            right_schema=[cf.ArrowFieldSpec("z", "string", False)],
            spec=collision,
        )


def test_asof_builder_compiles_through_native_stream_project_contract() -> None:
    fields = [
        cf.ArrowFieldSpec("symbol", "string", False),
        cf.ArrowFieldSpec("time", "timestamp[us, UTC]", False),
        cf.ArrowFieldSpec("sequence", "uint64", False),
    ]
    builder = cf.PipelineBuilder("advanced_asof").stream_asof_join(
        "matched", left_schema=fields, right_schema=fields, spec=_spec()
    )
    before = builder.project
    plan = builder.compile_stream()
    assert plan.name == "advanced_asof"
    assert len(plan.fingerprint) == 64
    assert builder.project == before
