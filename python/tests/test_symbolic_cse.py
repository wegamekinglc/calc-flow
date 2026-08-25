from __future__ import annotations

import pyarrow as pa

from calc_flow import Batch, Runtime
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    TableExpr,
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
        ],
    )


def _xy_batch() -> pa.Table:
    schema = pa.schema(
        [
            pa.field("x", pa.float64(), nullable=False),
            pa.field("y", pa.float64(), nullable=False),
        ]
    )
    return pa.table(
        {
            "x": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            "y": pa.array([4.0, 5.0, 6.0], type=pa.float64()),
        },
        schema=schema,
    )


def _nodes(document: dict) -> list[dict]:
    return document["graph"]["nodes"]


def test_twenty_independent_outputs_form_one_fused_node() -> None:
    quotes = _xy()
    features = [
        (f"feature_{index}", (quotes["x"] + float(index)) * quotes["y"])
        for index in range(20)
    ]
    signals = quotes.with_columns(FeatureSet(features))
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _nodes(document)
    assert len(nodes) == 1
    operator = nodes[0]["operator"]
    assert operator["kind"] == "expression"
    assert len(operator["select"]) == 22


def test_shared_subexpression_is_computed_once() -> None:
    quotes = _xy()
    shared = quotes["x"] * quotes["y"]
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("a", shared + 1.0),
                ("b", shared + 2.0),
                ("c", shared + 3.0),
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _nodes(document)
    assert [node["id"] for node in nodes] == ["signals__cf_cse_1", "signals"]
    tier, final = nodes
    assert tier["operator"]["select"] == [
        '"x"',
        '"y"',
        '("x" * "y") AS "__cf_cse_0"',
    ]
    assert final["operator"]["select"] == [
        '"x"',
        '"y"',
        '("__cf_cse_0" + 1.0) AS "a"',
        '("__cf_cse_0" + 2.0) AS "b"',
        '("__cf_cse_0" + 3.0) AS "c"',
    ]
    assert document["graph"]["edges"] == [
        {
            "source_node": "signals__cf_cse_1",
            "source_port": "output",
            "target_node": "signals",
            "target_port": "input",
        }
    ]

    plan = program.compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_xy_batch())})
    output = result.outputs["output"].to_pyarrow().to_pydict()
    assert output["a"] == [5.0, 11.0, 19.0]
    assert output["b"] == [6.0, 12.0, 20.0]
    assert output["c"] == [7.0, 13.0, 21.0]


def test_nested_sharing_materializes_deeply_first() -> None:
    quotes = _xy()
    inner = quotes["x"] * quotes["y"]
    middle = inner + 1.0
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("a", middle * 2.0),
                ("b", middle * 3.0),
                ("c", inner + 5.0),
            ]
        )
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _nodes(document)
    assert [node["id"] for node in nodes] == [
        "signals__cf_cse_1",
        "signals__cf_cse_2",
        "signals",
    ]
    first, second, final = nodes
    assert first["operator"]["select"] == [
        '"x"',
        '"y"',
        '("x" * "y") AS "__cf_cse_1"',
    ]
    assert second["operator"]["select"] == [
        '"x"',
        '"y"',
        '"__cf_cse_1"',
        '("__cf_cse_1" + 1.0) AS "__cf_cse_0"',
    ]
    assert final["operator"]["select"] == [
        '"x"',
        '"y"',
        '("__cf_cse_0" * 2.0) AS "a"',
        '("__cf_cse_0" * 3.0) AS "b"',
        '("__cf_cse_1" + 5.0) AS "c"',
    ]

    plan = program.compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_xy_batch())})
    output = result.outputs["output"].to_pyarrow().to_pydict()
    assert output["a"] == [10.0, 22.0, 38.0]
    assert output["b"] == [15.0, 33.0, 57.0]
    assert output["c"] == [9.0, 15.0, 23.0]


def test_filter_predicate_shares_the_materialized_subexpression() -> None:
    quotes = _xy()
    shared = quotes["x"] * quotes["y"]
    derived = quotes.with_columns(
        FeatureSet([("a", shared + 1.0), ("b", shared + 2.0)])
    )
    filtered = table.filter(derived, shared > 5.0)
    program = Program("p", inputs=[quotes], outputs=[("signals", filtered)])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _nodes(document)
    assert [node["id"] for node in nodes] == ["signals__cf_cse_1", "signals"]
    tier, final = nodes
    assert tier["operator"]["select"] == [
        '"x"',
        '"y"',
        '("x" * "y") AS "__cf_cse_0"',
    ]
    assert final["operator"]["filter"] == '("__cf_cse_0" > 5.0)'
    assert final["operator"]["select"] == [
        '"x"',
        '"y"',
        '("__cf_cse_0" + 1.0) AS "a"',
        '("__cf_cse_0" + 2.0) AS "b"',
    ]

    plan = program.compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_xy_batch())})
    output = result.outputs["output"].to_pyarrow().to_pydict()
    assert output["x"] == [2.0, 3.0]
    assert output["a"] == [11.0, 19.0]


def test_trivial_subexpressions_are_never_materialized() -> None:
    quotes = _xy()
    signals = quotes.with_columns(
        FeatureSet([("a", quotes["x"] + quotes["y"]), ("b", quotes["x"] - quotes["y"])])
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _nodes(document)
    assert len(nodes) == 1
    assert nodes[0]["operator"]["select"] == [
        '"x"',
        '"y"',
        '("x" + "y") AS "a"',
        '("x" - "y") AS "b"',
    ]


def test_identical_features_share_one_materialized_column() -> None:
    quotes = _xy()
    signals = quotes.with_columns(
        FeatureSet([("a", quotes["x"] * quotes["y"]), ("b", quotes["x"] * quotes["y"])])
    )
    program = Program("p", inputs=[quotes], outputs=[("signals", signals)])

    document = lower_program_document(program, Runtime(), "batch")

    nodes = _nodes(document)
    assert [node["id"] for node in nodes] == ["signals__cf_cse_1", "signals"]
    tier, final = nodes
    assert tier["operator"]["select"] == [
        '"x"',
        '"y"',
        '("x" * "y") AS "__cf_cse_0"',
    ]
    assert final["operator"]["select"] == [
        '"x"',
        '"y"',
        '"__cf_cse_0" AS "a"',
        '"__cf_cse_0" AS "b"',
    ]
