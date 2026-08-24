"""Retained SCE symbolic acceptance vectors.

The behaviors kept here are the ones the per-module suites under
``python/tests/`` do not already pin: caller-owned value detachment for
parameter shapes, grouping values, and node attribute mappings; frozen
declaration values beyond ``Field``; literal-path rejection of non-JSON clip
bounds; and declaration-order and reflected-operator identity semantics.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from calc_flow.symbolic import (
    ArrayExpr,
    ColumnExpr,
    CrossSectionGroup,
    EventTimeBucket,
    Field,
    Parameter,
    RowFrame,
    TableExpr,
    cs,
    event_time_bucket,
    exact_time,
    linalg,
    parameter,
    row,
    table,
    table_input,
)
from calc_flow.symbolic.nodes import CInt, build


def _ticks() -> TableExpr:
    return table_input(
        "ticks",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("sym", "string", nullable=False),
            Field("px", "float64"),
            Field("qty", "float64"),
        ],
        entity_by=["sym"],
        event_time="ts",
    )


def _prices() -> TableExpr:
    return table_input(
        "prices",
        schema=[Field("a", "float64"), Field("b", "float64")],
    )


def _array_parameter() -> Parameter[ArrayExpr]:
    return parameter(
        "w",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=[2, 2],
    )


def test_parameter_leaves_caller_shape_unchanged_and_detached() -> None:
    shape = [2, 3]
    weights = parameter(
        "w",
        kind="array",
        backend="numpy",
        dtype="float64",
        shape=shape,
    )
    digest_before = weights.digest
    assert shape == [2, 3]
    shape.append(9)
    shape[0] = 7
    assert weights.digest == digest_before


def test_grouping_values_copy_the_caller_partition_sequence() -> None:
    prices = _prices()
    event_time = _ticks()["ts"]
    partition = [prices["a"], prices["b"]]
    group = exact_time(event_time, partition_by=partition)
    bucket = event_time_bucket(event_time, width_micros=5, partition_by=partition)
    partition.append(prices["a"])
    assert isinstance(group, CrossSectionGroup)
    assert isinstance(group.partition_by, tuple)
    assert len(group.partition_by) == 2
    assert isinstance(bucket, CrossSectionGroup)
    assert isinstance(bucket.bucket, EventTimeBucket)
    assert isinstance(bucket.bucket.partition_by, tuple)
    assert len(bucket.bucket.partition_by) == 2


def test_grouping_accepts_lists_and_tuples_with_equal_identity() -> None:
    event_time = _ticks()["ts"]
    prices = _prices()
    columns = [prices["a"], prices["b"]]
    from_list = cs.demean(
        prices["a"],
        group=exact_time(event_time, partition_by=list(columns)),
    )
    from_tuple = cs.demean(
        prices["a"],
        group=exact_time(event_time, partition_by=tuple(columns)),
    )
    assert from_list.identical(from_tuple)


def test_node_attrs_survive_caller_mapping_mutation() -> None:
    attrs = {"periods": CInt(4)}
    node = build("lag", (build("literal", (), {"value": CInt(1)}),), attrs)
    attrs["periods"] = CInt(99)
    attrs["extra"] = None
    assert node.attr("periods") == CInt(4)
    assert node.attr("extra") is None


def test_expression_and_declaration_values_reject_attribute_writes() -> None:
    prices = _prices()
    targets = (
        (prices, "_node"),
        (prices["a"], "_node"),
        (
            linalg.from_columns(prices, columns=["a", "b"], backend="numpy"),
            "_node",
        ),
        (_array_parameter(), "_node"),
        (Field("a", "float64"), "name"),
        (RowFrame(3), "size"),
        (exact_time(_ticks()["ts"]), "event_time"),
        (event_time_bucket(_ticks()["ts"], width_micros=5), "bucket"),
        (EventTimeBucket(_ticks()["ts"], 5), "width_micros"),
    )
    for target, attribute in targets:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(target, attribute, None)


def test_row_clip_bounds_reject_non_json_values_on_the_literal_path() -> None:
    column = _prices()["a"]
    rejected = (
        (lambda value: value, "builtins.function"),
        ({"a": 1}, "builtins.dict"),
        ([0], "builtins.list"),
        (b"x", "builtins.bytes"),
    )
    for bound, type_text in rejected:
        expected = (
            "calc_flow.symbolic.literal.value: invalid_literal: declaration"
            f" literals must be strict JSON scalars; got {type_text}"
        )
        with pytest.raises(ValueError, match=re.escape(expected)):
            row.clip(column, lower=bound, upper=1)


def test_projection_order_carry_identity() -> None:
    prices = _prices()
    assert not table.project(prices, ["a", "b"]).identical(
        table.project(prices, ["b", "a"])
    )


def test_entity_by_and_partition_order_carry_identity() -> None:
    first = table_input(
        "t",
        schema=[Field("a", "float64")],
        entity_by=["x", "y"],
    )
    second = table_input(
        "t",
        schema=[Field("a", "float64")],
        entity_by=["y", "x"],
    )
    assert first.digest != second.digest
    event_time = _ticks()["ts"]
    prices = _prices()
    assert not cs.demean(
        prices["a"],
        group=exact_time(event_time, partition_by=[prices["a"], prices["b"]]),
    ).identical(
        cs.demean(
            prices["a"],
            group=exact_time(event_time, partition_by=[prices["b"], prices["a"]]),
        )
    )


def test_comparison_operators_build_symbolic_nodes_not_booleans() -> None:
    prices = _prices()
    assert isinstance(prices["a"] == prices["b"], ColumnExpr)
    assert isinstance(prices["a"] >= 2.5, ColumnExpr)
    assert not (prices["a"] == prices["b"]).identical(prices["b"] == prices["a"])
    assert (2.0 * prices["a"]).identical(prices["a"] * 2.0)
    assert not (2.0 - prices["a"]).identical(prices["a"] - 2.0)
