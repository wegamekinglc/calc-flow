from __future__ import annotations

import pyarrow as pa

from calc_flow import Batch, Runtime, compute
from calc_flow.symbolic import (
    ColumnExpr,
    FeatureSet,
    Field,
    Program,
    TableExpr,
    rows,
    table,
    table_input,
    ts,
)
from calc_flow.symbolic.lower import lower_program_document, segments


def _ordered() -> TableExpr:
    return table_input(
        "quotes",
        schema=[
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64", nullable=True),
        ],
        entity_by=["symbol"],
        event_time="ts",
        sequence_by=["seq"],
    )


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
    program = Program(
        "p", engine="sql", inputs=[quotes], outputs=[("signals", signals)]
    )

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
    program = Program(
        "p", engine="sql", inputs=[quotes], outputs=[("signals", signals)]
    )

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
    program = Program(
        "p", engine="sql", inputs=[quotes], outputs=[("signals", signals)]
    )

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
    program = Program(
        "p", engine="sql", inputs=[quotes], outputs=[("signals", filtered)]
    )

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
    program = Program(
        "p", engine="sql", inputs=[quotes], outputs=[("signals", signals)]
    )

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
    program = Program(
        "p", engine="sql", inputs=[quotes], outputs=[("signals", signals)]
    )

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


def _diamond(expr: ColumnExpr, depth: int) -> ColumnExpr:
    for _ in range(depth):
        expr = expr + expr
    return expr


def test_inlining_builds_each_shared_subexpression_once(monkeypatch) -> None:
    calls = 0
    original = segments.build

    def counting_build(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(segments, "build", counting_build)
    quotes = _xy()
    signals = quotes.with_columns(FeatureSet([("z", _diamond(quotes["x"], 16))]))

    segments._resolve_table(signals._node, "signals")

    assert calls <= 2 * 16


def test_primitive_search_yields_each_shared_subtree_once() -> None:
    quotes = _ordered()
    mean = ts.mean(quotes["x"], window=rows(3))

    found = list(segments._find_rolling(_diamond(mean, 16)._node))

    assert [node.digest for node in found] == [mean._node.digest]


def test_deep_shared_subexpression_diamond_lowers_and_computes() -> None:
    depth = 64
    table = pa.table({"x": [1.0, 2.0]})

    result = compute(
        table, lambda source: source.select(y=_diamond(source["x"], depth))
    )

    assert result.column("y").to_pylist() == [2.0**depth, 2.0 ** (depth + 1)]
