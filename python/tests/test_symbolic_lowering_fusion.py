from __future__ import annotations

import pyarrow as pa
import pytest

from calc_flow import Batch, Runtime
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    TableExpr,
    row,
    table,
    table_input,
)
from calc_flow.symbolic.lower import lower_program_document


def _xy() -> TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("x", "float64", nullable=False),
            Field("y", "float64", nullable=False),
            Field("i", "int64", nullable=False),
            Field("b", "bool", nullable=False),
            Field("s", "string"),
        ],
    )


def _xy_batch() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("x", pa.float64(), nullable=False),
            pa.field("y", pa.float64(), nullable=False),
            pa.field("i", pa.int64(), nullable=False),
            pa.field("b", pa.bool_(), nullable=False),
            pa.field("s", pa.string(), nullable=True),
        ]
    )
    return pa.table(
        {
            "x": pa.array([1.0, -4.0, 16.0, 2.0], type=pa.float64()),
            "y": pa.array([2.0, 3.0, 2.0, 0.5], type=pa.float64()),
            "i": pa.array([1, 2, 3, 4], type=pa.int64()),
            "b": pa.array([True, False, True, False], type=pa.bool_()),
            "s": pa.array(["a", None, "c", "d"], type=pa.string()),
        },
        schema=schema,
    )


def _nodes(document: dict) -> list[dict]:
    return document["graph"]["nodes"]


def test_with_columns_lowers_to_one_fused_expression_node() -> None:
    quotes = _xy()
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("add_xy", quotes["x"] + quotes["y"]),
                ("diff", quotes["x"] - quotes["y"]),
                ("score", (quotes["x"] * quotes["y"]) / quotes["y"]),
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _nodes(document)
    assert len(nodes) == 1
    operator = nodes[0]["operator"]
    assert operator["kind"] == "expression"
    assert operator["expression"] == ""
    assert operator["udfs"] == []
    assert operator["filter"] is None
    assert nodes[0]["id"] == "signals"
    assert operator["select"] == [
        '"x"',
        '"y"',
        '"i"',
        '"b"',
        '"s"',
        '("x" + "y") AS "add_xy"',
        '("x" - "y") AS "diff"',
        '(("x" * "y") / "y") AS "score"',
    ]


def test_fused_node_executes_row_local_features() -> None:
    quotes = _xy()
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("add_xy", quotes["x"] + quotes["y"]),
                ("neg_x", -quotes["x"]),
                ("ge", quotes["x"] >= quotes["y"]),
                ("both", (quotes["x"] > quotes["y"]) & quotes["b"]),
                ("inv", ~quotes["b"]),
                ("picked", row.where(quotes["b"], quotes["x"], quotes["y"])),
                ("coa", row.coalesce(quotes["s"], "fallback")),
                ("log_x", row.log(quotes["x"])),
                ("exp_y", row.exp(quotes["y"])),
                ("sqrt_x", row.sqrt(quotes["x"])),
                ("abs_x", row.abs(quotes["x"])),
                ("clip_x", row.clip(quotes["x"], lower=0.0, upper=10.0)),
                ("cast_i", row.cast(quotes["i"], "float64")),
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])
    plan = program.compile_batch(Runtime())

    result = plan.execute({"input": Batch.from_pyarrow(_xy_batch())})
    output = result.outputs["output"].to_pyarrow().to_pydict()

    assert output["add_xy"] == [3.0, -1.0, 18.0, 2.5]
    assert output["neg_x"] == [-1.0, 4.0, -16.0, -2.0]
    assert output["ge"] == [False, False, True, True]
    assert output["both"] == [False, False, True, False]
    assert output["inv"] == [False, True, False, True]
    assert output["picked"] == [1.0, 3.0, 16.0, 0.5]
    assert output["coa"] == ["a", "fallback", "c", "d"]
    assert output["log_x"][0] == pytest.approx(0.0)
    assert output["exp_y"][0] == pytest.approx(7.38905609893065)
    assert output["sqrt_x"][0] == pytest.approx(1.0)
    assert output["sqrt_x"][1] != output["sqrt_x"][1]  # sqrt of a negative is NaN
    assert output["sqrt_x"][2:] == [4.0, pytest.approx(2**0.5)]
    assert output["abs_x"] == [1.0, 4.0, 16.0, 2.0]
    assert output["clip_x"] == [1.0, 0.0, 10.0, 2.0]
    assert output["cast_i"] == [1.0, 2.0, 3.0, 4.0]


def test_feature_may_reference_an_earlier_feature_in_one_node() -> None:
    quotes = _xy()
    derived = quotes.with_columns(FeatureSet([("base2", quotes["x"] * 2.0)]))
    signals = derived.with_columns(FeatureSet([("quad", derived["base2"] * 2.0)]))
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _nodes(document)
    assert [node["id"] for node in nodes] == ["signals__cf_cse_1", "signals"]
    tier, final = nodes
    assert tier["operator"]["select"] == [
        '"x"',
        '"y"',
        '"i"',
        '"b"',
        '"s"',
        '("x" * 2.0) AS "__cf_cse_0"',
    ]
    select = final["operator"]["select"]
    assert '"__cf_cse_0" AS "base2"' in select
    assert '("__cf_cse_0" * 2.0) AS "quad"' in select

    plan = program.compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_xy_batch())})
    output = result.outputs["output"].to_pyarrow().to_pydict()
    assert output["base2"] == [2.0, -8.0, 32.0, 4.0]
    assert output["quad"] == [4.0, -16.0, 64.0, 8.0]


def test_filter_fuses_into_the_same_node_where_clause() -> None:
    quotes = _xy()
    derived = quotes.with_columns(FeatureSet([("score", quotes["x"] + 1.0)]))
    filtered = table.filter(derived, derived["score"] > 0.0)
    program = Program("p", inputs=[quotes], outputs=[("signals", filtered)])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _nodes(document)
    assert [node["id"] for node in nodes] == ["signals__cf_cse_1", "signals"]
    tier, final = nodes
    assert '("x" + 1.0) AS "__cf_cse_0"' in tier["operator"]["select"]
    operator = final["operator"]
    assert operator["filter"] == '("__cf_cse_0" > 0.0)'
    assert '"__cf_cse_0" AS "score"' in operator["select"]

    plan = program.compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_xy_batch())})
    output = result.outputs["output"].to_pyarrow().to_pydict()
    assert output["x"] == [1.0, 16.0, 2.0]
    assert output["score"] == [2.0, 17.0, 3.0]


def test_project_fuses_into_the_select_list() -> None:
    quotes = _xy()
    derived = quotes.with_columns(FeatureSet([("score", quotes["x"] + 1.0)]))
    projected = table.project(derived, ["x", "score"])
    program = Program("p", inputs=[quotes], outputs=[("signals", projected)])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _nodes(document)
    assert len(nodes) == 1
    assert nodes[0]["operator"]["select"] == ['"x"', '("x" + 1.0) AS "score"']


def test_literal_spellings_round_trip_exactly() -> None:
    quotes = _xy()
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("none", row.coalesce(quotes["s"], None)),
                ("yes", quotes["x"] == -0.0),
                (
                    "big",
                    row.cast(quotes["i"], "uint64") + 18446744073709551615,
                ),
                ("text", row.where(quotes["b"], "it's", "other")),
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(program, Runtime(), "batch")

    select = _nodes(document)[0]["operator"]["select"]
    assert 'COALESCE("s", NULL) AS "none"' in select
    assert '("x" = -0.0) AS "yes"' in select
    assert '(CAST("i" AS BIGINT UNSIGNED) + 18446744073709551615) AS "big"' in select
    assert "(CASE WHEN \"b\" THEN 'it''s' ELSE 'other' END) AS \"text\"" in select


def test_multiple_outputs_share_one_input_through_a_fanout_node() -> None:
    quotes = _xy()
    first = quotes.with_columns(FeatureSet([("a", quotes["x"] + 1.0)]))
    second = table.filter(
        quotes.with_columns(FeatureSet([("scaled", quotes["y"] * 2.0)])),
        quotes["x"] > 0.0,
    )
    program = Program(
        "p", inputs=[quotes], outputs=[("first", first), ("second", second)]
    )

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _nodes(document)
    assert [node["id"] for node in nodes] == ["quotes", "first", "second"]
    fanout = nodes[0]["operator"]
    assert fanout["kind"] == "expression"
    assert fanout["select"] == ['"x"', '"y"', '"i"', '"b"', '"s"']
    edges = document["graph"]["edges"]
    assert edges == [
        {
            "source_node": "quotes",
            "source_port": "output",
            "target_node": "first",
            "target_port": "input",
        },
        {
            "source_node": "quotes",
            "source_port": "output",
            "target_node": "second",
            "target_port": "input",
        },
    ]
    assert document["data_sources"] == [
        {"data": [], "format": "inline_json", "id": "source_1", "input": "input"}
    ]


def test_multiple_inputs_lower_with_named_external_inputs() -> None:
    quotes = _xy()
    trades = table_input("trades", schema=[Field("z", "float64", nullable=False)])
    first = quotes.with_columns(FeatureSet([("a", quotes["x"] + 1.0)]))
    second = trades.with_columns(FeatureSet([("b", trades["z"] * 2.0)]))
    program = Program(
        "p",
        inputs=[quotes, trades],
        outputs=[("first", first), ("second", second)],
    )

    document = lower_program_document(program, Runtime(), "batch")

    assert [node["id"] for node in _nodes(document)] == [
        "quotes",
        "trades",
        "first",
        "second",
    ]
    assert document["data_sources"] == [
        {
            "data": [],
            "format": "inline_json",
            "id": "source_1",
            "input": "quotes.input",
        },
        {
            "data": [],
            "format": "inline_json",
            "id": "source_2",
            "input": "trades.input",
        },
    ]


def test_lowered_document_is_byte_deterministic() -> None:
    import json

    def build() -> str:
        quotes = _xy()
        signals = table.filter(
            quotes.with_columns(FeatureSet([("a", quotes["x"] + 1.0)])),
            quotes["x"] > 0.0,
        )
        program = Program("p", inputs=[quotes], outputs=[("signals", signals)])
        return json.dumps(
            lower_program_document(program, Runtime(), "batch"), sort_keys=True
        )

    assert build() == build()
