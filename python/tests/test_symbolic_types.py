from __future__ import annotations

import re

import numpy as np
import pytest

from calc_flow.symbolic import (
    Field,
    RowFrame,
    cs,
    duration,
    event_time_bucket,
    exact_time,
    parameter,
    row,
    rows,
    table_input,
    ts,
    window,
)


def _column():
    return table_input("q", schema=[Field("x", "float64")])["x"]


def test_field_is_an_immutable_value() -> None:
    first = Field("x", "float64")
    second = Field("x", "float64")
    assert first == second
    assert first != Field("x", "float32")
    assert first != Field("x", "float64", nullable=False)
    with pytest.raises(AttributeError):
        first.name = "y"  # type: ignore[misc]


def test_field_rejects_wrong_host_types() -> None:
    with pytest.raises(TypeError, match="Field.name"):
        Field(1, "float64")
    with pytest.raises(TypeError, match="Field.data_type"):
        Field("x", 3.0)
    with pytest.raises(TypeError, match="Field.nullable"):
        Field("x", "float64", nullable=1)


def test_table_input_rejects_unknown_portable_types() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "inputs.q.schema[0].data_type: unsupported_type: unknown portable"
        ),
    ):
        table_input("q", schema=[Field("x", "decimal128(18, 4)")])


def test_table_input_accepts_timestamp_us_utc() -> None:
    quotes = table_input(
        "q",
        schema=[Field("ts", "timestamp[us, UTC]", nullable=False)],
        event_time="ts",
    )
    assert len(quotes.digest) == 64


def test_table_input_rejects_duplicate_schema_names() -> None:
    with pytest.raises(ValueError, match=re.escape("duplicate_name")):
        table_input("q", schema=[Field("x", "float64"), Field("x", "float32")])


def test_table_input_rejects_wrong_schema_item_types() -> None:
    with pytest.raises(TypeError, match=re.escape("inputs.q.schema[0]")):
        table_input("q", schema=[("x", "float64")])  # type: ignore[list-item]
    with pytest.raises(TypeError, match=re.escape("inputs.table_input.name")):
        table_input(42, schema=[Field("x", "float64")])  # type: ignore[arg-type]


def test_table_input_rejects_non_string_ordering_fields() -> None:
    with pytest.raises(ValueError, match=re.escape("inputs.q.entity_by[0]")):
        table_input("q", schema=[Field("x", "float64")], entity_by=["", "b"])
    with pytest.raises(TypeError, match=re.escape("inputs.q.sequence_by[0]")):
        table_input(
            "q",
            schema=[Field("x", "float64")],
            sequence_by=[1],  # type: ignore[list-item]
        )


def test_rows_and_duration_reject_invalid_frames() -> None:
    with pytest.raises(ValueError, match=re.escape("positive")):
        rows(0)
    with pytest.raises(ValueError, match=re.escape("positive")):
        rows(-3)
    with pytest.raises(ValueError, match=re.escape("positive")):
        duration(0)
    with pytest.raises(TypeError, match=re.escape("calc_flow.symbolic.rows.size")):
        rows(True)  # type: ignore[arg-type]
    with pytest.raises(
        TypeError, match=re.escape("calc_flow.symbolic.duration.micros")
    ):
        duration(1.5)  # type: ignore[arg-type]
    assert rows(5) == RowFrame(5)
    assert duration(1_000) is not None


def test_event_time_bucket_requires_positive_width() -> None:
    column = _column()
    with pytest.raises(ValueError, match=re.escape("positive")):
        event_time_bucket(column, width_micros=0)
    with pytest.raises(
        TypeError, match=re.escape("calc_flow.symbolic.event_time_bucket.width_micros")
    ):
        event_time_bucket(column, width_micros=True)  # type: ignore[arg-type]


def test_parameter_table_kind_rejects_array_fields() -> None:
    with pytest.raises(ValueError, match=re.escape("static_inputs.w.backend")):
        parameter(
            "w",
            kind="table",
            schema=[Field("k", "string")],
            backend="numpy",  # type: ignore[call-overload]
        )
    with pytest.raises(ValueError, match=re.escape("static_inputs.w.dtype")):
        parameter(
            "w",
            kind="table",
            schema=[Field("k", "string")],
            dtype="float64",  # type: ignore[call-overload]
        )
    with pytest.raises(ValueError, match=re.escape("static_inputs.w.shape")):
        parameter(
            "w",
            kind="table",
            schema=[Field("k", "string")],
            shape=(1,),  # type: ignore[call-overload]
        )


def test_parameter_array_kind_rejects_table_fields() -> None:
    with pytest.raises(ValueError, match=re.escape("static_inputs.w.schema")):
        parameter(
            "w",
            kind="array",
            backend="numpy",
            dtype="float64",
            shape=(1,),
            schema=[Field("k", "string")],  # type: ignore[call-overload]
        )


def test_parameter_rejects_unknown_mutability() -> None:
    with pytest.raises(ValueError, match=re.escape("static_inputs.w.mutability")):
        parameter(
            "w",
            kind="array",
            backend="numpy",
            dtype="float64",
            shape=(1,),
            mutability="dynamic",  # type: ignore[call-overload]
        )


def test_parameter_rejects_array_like_host_values_stably() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "static_inputs.p.shape: invalid_literal: table parameters do"
            " not accept shape"
        ),
    ):
        parameter(
            "p",
            kind="table",
            schema=[Field("k", "string")],
            shape=np.array([1, 2]),  # type: ignore[call-overload]
        )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "static_inputs.w.schema: invalid_literal: array parameters do"
            " not accept schema"
        ),
    ):
        parameter(
            "w",
            kind="array",
            backend="numpy",
            dtype="float64",
            shape=(1,),
            schema=np.array([Field("k", "string")]),  # type: ignore[call-overload]
        )
    with pytest.raises(TypeError, match=re.escape("static_inputs.w.mutability")):
        parameter(
            "w",
            kind="array",
            backend="numpy",
            dtype="float64",
            shape=(1,),
            mutability=np.array(["static", "dynamic"]),  # type: ignore[call-overload]
        )


def test_parameter_validates_array_dtype_and_shape() -> None:
    with pytest.raises(
        ValueError, match=re.escape("static_inputs.w.dtype: unsupported_type")
    ):
        parameter("w", kind="array", backend="numpy", dtype="complex64", shape=(1,))
    with pytest.raises(ValueError, match=re.escape("static_inputs.w.shape[0]")):
        parameter("w", kind="array", backend="numpy", dtype="float64", shape=(-1,))
    with pytest.raises(TypeError, match=re.escape("static_inputs.w.shape[0]")):
        parameter(
            "w",
            kind="array",
            backend="numpy",
            dtype="float64",
            shape=(True,),  # type: ignore[call-overload]
        )
    with pytest.raises(TypeError, match=re.escape("static_inputs.w.backend")):
        parameter("w", kind="array", backend=42, dtype="float64", shape=(1,))  # type: ignore[call-overload]


def test_scalar_operands_reject_non_json_values() -> None:
    x = _column()
    with pytest.raises(TypeError, match=re.escape("got builtins.function")):
        x + (lambda value: value)  # type: ignore[operator]
    with pytest.raises(TypeError, match=re.escape("got builtins.list")):
        x + [1.0]  # type: ignore[operator]
    with pytest.raises(TypeError, match=re.escape("got builtins.dict")):
        x + {"a": 1.0}  # type: ignore[operator]
    with pytest.raises(TypeError, match=re.escape("got builtins.bytes")):
        x + b"payload"  # type: ignore[operator]


def test_scalar_operands_reject_non_finite_and_out_of_range_values() -> None:
    x = _column()
    with pytest.raises(ValueError, match=re.escape("invalid_literal")):
        x + float("nan")  # type: ignore[operator]
    with pytest.raises(ValueError, match=re.escape("invalid_literal")):
        x + float("-inf")  # type: ignore[operator]
    with pytest.raises(ValueError, match=re.escape("invalid_literal")):
        x + (2**64)  # type: ignore[operator]
    with pytest.raises(ValueError, match=re.escape("invalid_literal")):
        x + (-(2**63) - 1)  # type: ignore[operator]


def test_ts_primitives_validate_their_declarations() -> None:
    x = _column()
    with pytest.raises(
        ValueError, match=re.escape("calc_flow.symbolic.ts.lag.periods")
    ):
        ts.lag(x, periods=0)
    with pytest.raises(
        ValueError, match=re.escape("calc_flow.symbolic.ts.variance.ddof")
    ):
        ts.variance(x, window=rows(5), ddof=2)
    with pytest.raises(
        ValueError, match=re.escape("calc_flow.symbolic.ts.count.min_periods")
    ):
        ts.count(x, window=rows(5), min_periods=6)
    with pytest.raises(
        ValueError, match=re.escape("calc_flow.symbolic.ts.count.min_periods")
    ):
        ts.count(x, window=rows(5), min_periods=0)


def test_cs_primitives_validate_bounds() -> None:
    x = _column()
    group = exact_time(x)
    with pytest.raises(
        ValueError, match=re.escape("calc_flow.symbolic.cs.winsorize.lower")
    ):
        cs.winsorize(x, group=group, lower=-0.1, upper=0.9)
    with pytest.raises(
        ValueError, match=re.escape("calc_flow.symbolic.cs.winsorize.upper")
    ):
        cs.winsorize(x, group=group, lower=0.1, upper=1.4)
    with pytest.raises(ValueError, match=re.escape("finite")):
        cs.winsorize(x, group=group, lower=float("nan"), upper=1.0)
    with pytest.raises(
        ValueError, match=re.escape("calc_flow.symbolic.cs.zscore.ddof")
    ):
        cs.zscore(x, group=group, ddof=3)
    valid = cs.winsorize(x, group=group, lower=0, upper=1)
    assert len(valid.digest) == 64


def test_cs_grouped_features_validate_selection_and_fill_declarations() -> None:
    x = _column()
    group = exact_time(x)
    with pytest.raises(ValueError, match=re.escape("calc_flow.symbolic.cs.top.count")):
        cs.top(x, group=group, count=0)
    with pytest.raises(
        TypeError, match=re.escape("calc_flow.symbolic.cs.bottom.include_ties")
    ):
        cs.bottom(x, group=group, count=1, include_ties=1)  # type: ignore[arg-type]
    assert len(cs.top(x, group=group, count=2).digest) == 64
    assert len(cs.bottom(x, group=group, count=2, include_ties=False).digest) == 64
    assert len(cs.mean_fill(x, group=group, min_samples=2).digest) == 64


def test_row_primitives_validate_declarations() -> None:
    x = _column()
    with pytest.raises(
        ValueError, match=re.escape("calc_flow.symbolic.row.clip.lower")
    ):
        row.clip(x, lower=0.9, upper=0.1)
    with pytest.raises(ValueError, match=re.escape("finite")):
        row.clip(x, lower=0.0, upper=float("inf"))
    with pytest.raises(
        ValueError,
        match=re.escape("calc_flow.symbolic.row.cast.data_type: unsupported_type"),
    ):
        row.cast(x, "map<string, int64>")
    assert len(row.cast(x, "float32").digest) == 64


def test_window_namespace_validates_declarations() -> None:
    quotes = table_input(
        "q", schema=[Field("ts", "timestamp[us, UTC]", nullable=False)]
    )
    with pytest.raises(
        ValueError,
        match=re.escape("calc_flow.symbolic.window.tumbling.size_micros"),
    ):
        window.tumbling(quotes, event_time="ts", size_micros=0)
    with pytest.raises(
        TypeError,
        match=re.escape("calc_flow.symbolic.window.hopping.slide_micros"),
    ):
        window.hopping(quotes, event_time="ts", size_micros=60, slide_micros=True)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=re.escape("calc_flow.symbolic.window.tumbling.group_by[0]"),
    ):
        window.tumbling(quotes, event_time="ts", size_micros=60, group_by=[""])
    assert len(window.tumbling(quotes, event_time="ts", size_micros=60).digest) == 64
