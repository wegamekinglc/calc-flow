from __future__ import annotations

import pyarrow as pa
import pytest

from calc_flow import Runtime
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    cs,
    exact_time,
    table_input,
    ts,
)


@pytest.mark.parametrize(
    ("kind", "source_name", "fingerprint"),
    [
        ("rolling", "signals__cf_rolling", None),
        ("rolling", "signals__cf_rolling_input", None),
        ("cross_section", "signals__cf_cross_section", None),
        ("cross_section", "signals__cf_cross_section_input", None),
        ("prefilter", "signals__cf_prefilter", None),
        (
            "rolling",
            "ordinary",
            "8373464279ebe0e6f38920f88d3646310b9a29f87f8565e4267161f96c80e26b",
        ),
        (
            "cross_section",
            "ordinary",
            "679f3c4880ce33e25d81da1bc0e8f0499d76d96465d032f8ca2b9451aaa15164",
        ),
        (
            "prefilter",
            "ordinary",
            "113cdf4edb0afbec8fd67bc583f07de30e01f2a1ab15cb9f3c73da4f8a6f4c20",
        ),
    ],
)
def test_batch_source_collision_with_planned_stage_preserves_results(
    kind: str, source_name: str, fingerprint: str | None
) -> None:
    quotes = table_input(
        "quotes",
        schema=[
            Field("time", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64"),
        ],
        entity_by=["symbol"],
        event_time="time",
        sequence_by=["seq"],
    )
    other = table_input(source_name, schema=[Field("value", "int64")])
    source = quotes.filter(quotes["x"] > 0.0) if kind == "prefilter" else quotes
    feature = (
        cs.rank(source["x"] + 1.0, group=exact_time(source["time"]))
        if kind == "cross_section"
        else ts.lag(source["x"] + 1.0)
    )
    program = Program(
        "planned-source-collision",
        inputs=[quotes, other],
        outputs=[
            ("signals", source.with_columns(FeatureSet([("feature", feature)]))),
            ("other", other),
        ],
    )
    if fingerprint is not None:
        assert program.compile_batch(Runtime()).fingerprint == fingerprint
    quotes_data = pa.table(
        {
            "time": pa.array([1, 2], type=pa.timestamp("us", tz="UTC")),
            "symbol": ["a", "a"],
            "seq": pa.array([1, 2], type=pa.uint64()),
            "x": [1.0, 2.0],
        },
        schema=pa.schema(
            [
                pa.field("time", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("x", pa.float64()),
            ]
        ),
    )
    other_data = pa.table({"value": [10, 20]})

    outputs = program.collect({"quotes": quotes_data, source_name: other_data})

    expected = [1.0, 1.0] if kind == "cross_section" else [None, 2.0]
    assert outputs["signals"].column("feature").to_pylist() == expected
    assert outputs["other"].equals(other_data)
    assert quotes_data.column("x").to_pylist() == [1.0, 2.0]


@pytest.mark.parametrize(
    "source_name", ["signals__cf_cse_1", "signals__cf_cse_2", "ordinary"]
)
def test_batch_source_collision_with_cse_preserves_results(source_name: str) -> None:
    quotes = table_input("quotes", schema=[Field("x", "float64")])
    other = table_input(source_name, schema=[Field("value", "int64")])
    inner = quotes["x"] * 2.0
    middle = inner + 1.0
    signals = quotes.select(a=middle * 2.0, b=middle * 3.0, c=inner + 5.0)
    program = Program(
        "cse-source-collision",
        inputs=[quotes, other],
        outputs=[("signals", signals), ("other", other)],
    )

    if source_name == "ordinary":
        assert program.compile_batch(Runtime()).fingerprint == (
            "b049a6fb70bfdc9a6741ade83c9f1eec2a61a8abc3f0188f7129ef0ca0adb9e4"
        )
    outputs = program.collect(
        {"quotes": pa.table({"x": [1.0, 2.0]}), source_name: pa.table({"value": [10]})}
    )

    assert outputs["signals"].to_pydict() == {
        "a": [6.0, 10.0],
        "b": [9.0, 15.0],
        "c": [7.0, 9.0],
    }
    assert outputs["other"].to_pydict() == {"value": [10]}
