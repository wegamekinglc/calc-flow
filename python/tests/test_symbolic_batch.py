from __future__ import annotations

import pyarrow as pa

from calc_flow import Batch, Runtime
from calc_flow.symbolic import FeatureSet, Field, Program, row, table, table_input


def _program() -> Program:
    quotes = table_input(
        "quotes",
        schema=[
            Field("x", "float64", nullable=False),
            Field("y", "float64"),
            Field("i", "int64", nullable=False),
        ],
    )
    signals = quotes.with_columns(
        FeatureSet(
            [
                ("total", quotes["x"] + quotes["y"]),
                ("ratio", quotes["x"] / quotes["y"]),
                ("label", row.where(quotes["i"] > 1, "big", "small")),
                ("cast_i", row.cast(quotes["i"], "float32")),
                ("coalesced", row.coalesce(quotes["y"], 0.0)),
            ]
        )
    )
    return Program("features", inputs=[quotes], outputs=[("signals", signals)])


def _table(rows: list[tuple[float, float | None, int]]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("x", pa.float64(), nullable=False),
            pa.field("y", pa.float64(), nullable=True),
            pa.field("i", pa.int64(), nullable=False),
        ]
    )
    return pa.table(
        {
            "x": pa.array([row[0] for row in rows], type=pa.float64()),
            "y": pa.array([row[1] for row in rows], type=pa.float64()),
            "i": pa.array([row[2] for row in rows], type=pa.int64()),
        },
        schema=schema,
    )


def test_batch_output_schema_and_values_are_exact() -> None:
    plan = _program().compile_batch(Runtime())
    result = plan.execute(
        {"input": Batch.from_pyarrow(_table([(2.0, 4.0, 3), (1.0, None, 1)]))}
    )
    output = result.outputs["output"].to_pyarrow()

    assert output.schema.names == [
        "x",
        "y",
        "i",
        "total",
        "ratio",
        "label",
        "cast_i",
        "coalesced",
    ]
    types = {field.name: str(field.type) for field in output.schema}
    assert types["total"] == "double"
    assert types["ratio"] == "double"
    assert types["label"] == "string"
    assert types["cast_i"] == "float"
    assert types["coalesced"] == "double"

    values = output.to_pydict()
    assert values["total"] == [6.0, None]
    assert values["ratio"] == [0.5, None]
    assert values["label"] == ["big", "small"]
    assert values["cast_i"] == [3.0, 1.0]
    assert values["coalesced"] == [4.0, 0.0]


def test_batch_handles_a_single_row() -> None:
    plan = _program().compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_table([(8.0, 2.0, 5)]))})
    output = result.outputs["output"].to_pyarrow().to_pydict()
    assert output["total"] == [10.0]
    assert output["label"] == ["big"]


def test_batch_handles_an_empty_input() -> None:
    plan = _program().compile_batch(Runtime())
    result = plan.execute({"input": Batch.from_pyarrow(_table([]))})
    output = result.outputs["output"].to_pyarrow()
    assert output.num_rows == 0
    assert output.schema.names == [
        "x",
        "y",
        "i",
        "total",
        "ratio",
        "label",
        "cast_i",
        "coalesced",
    ]


def test_batch_reexecution_is_deterministic() -> None:
    plan = _program().compile_batch(Runtime())
    table = _table([(2.0, 4.0, 3), (1.0, None, 1), (3.5, 0.5, 2)])
    first = plan.execute({"input": Batch.from_pyarrow(table)})
    second = plan.execute({"input": Batch.from_pyarrow(table)})
    assert (
        first.outputs["output"].to_pyarrow().to_pydict()
        == second.outputs["output"].to_pyarrow().to_pydict()
    )


def test_filter_only_program_lowers_and_executes() -> None:
    quotes = table_input("quotes", schema=[Field("x", "float64", nullable=False)])
    filtered = table.filter(quotes, quotes["x"] > 1.0)
    program = Program("p", inputs=[quotes], outputs=[("signals", filtered)])
    plan = program.compile_batch(Runtime())

    schema = pa.schema([pa.field("x", pa.float64(), nullable=False)])
    batch_table = pa.table(
        {"x": pa.array([0.5, 1.5, 2.5], type=pa.float64())}, schema=schema
    )
    result = plan.execute({"input": Batch.from_pyarrow(batch_table)})
    assert result.outputs["output"].to_pyarrow().to_pydict()["x"] == [1.5, 2.5]
