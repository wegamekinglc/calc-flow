from __future__ import annotations

import inspect

import pyarrow as pa
import pytest

import calc_flow as cf


def test_pipe_composes_tables_columns_and_arguments_once():
    data = pa.table({"x": [2.0, 4.0]})
    source = cf.table_input("values", schema=data.schema)
    calls = []

    def scaled(column, factor, *, offset):
        calls.append(column)
        return column * factor + offset

    output = source.pipe(lambda t: t.select(value=t["x"].pipe(scaled, 2.0, offset=1.0)))
    program = output.pipe(lambda t: cf.Program("scaled", outputs={"result": t}))
    assert len(calls) == 1
    assert program.collect({"values": data})["result"].to_pydict() == {
        "value": [5.0, 9.0]
    }
    assert len(calls) == 1
    assert data.to_pydict() == {"x": [2.0, 4.0]}


def test_pipe_preserves_exceptions_and_closes_async_builders():
    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    error = LookupError("builder failed")

    def broken(t):
        raise error

    with pytest.raises(LookupError) as caught:
        source.pipe(broken)
    assert caught.value is error

    async def async_build(t):
        return t

    pending = async_build(source)
    with pytest.raises(TypeError, match=r"pipe.*synchronous"):
        source.pipe(lambda t: pending)
    assert inspect.getcoroutinestate(pending) == inspect.CORO_CLOSED


def test_sql_and_expression_pipeline_executes_in_one_graph():
    data = pa.table(
        {"id": [1, 2, 3], "quantity": [2, 1, 3], "price": [10.0, 5.0, 10.0]}
    )

    def pipeline(t):
        return (
            t.with_columns(gross=cf.row.cast(t["quantity"], "float64") * t["price"])
            .sql("SELECT id, gross FROM input WHERE gross >= 20 ORDER BY id")
            .pipe(lambda t: t.select("id", net=t["gross"] * 0.9))
        )

    assert cf.compute(data, pipeline).to_pydict() == {"id": [1, 3], "net": [18.0, 27.0]}
    source = cf.table_input("orders", schema=data.schema)
    document = cf.Program("sql", outputs={"output": pipeline(source)}).to_project()
    assert '"kind":"sql"' in document.model_dump_json().replace(" ", "")


def test_sql_join_nested_cte_and_shared_output_keep_logical_bindings():
    orders = pa.table({"id": [1, 2], "amount": [75, 120]})
    fees = pa.table({"id": [1, 2], "fee": [5, 12]})
    source = cf.table_input("orders", schema=orders.schema)
    fee_source = cf.table_input("fees", schema=fees.schema)
    joined = cf.sql(
        "SELECT o.id, o.amount - f.fee AS net "
        "FROM o JOIN f ON o.id = f.id ORDER BY o.id",
        o=source,
        f=fee_source,
    ).sql("WITH selected AS (SELECT * FROM input) SELECT * FROM selected")
    program = cf.Program(
        "branches",
        outputs={"net": joined, "double": joined.select(doubled=joined["net"] * 2)},
    )
    result = program.collect({"fees": fees, "orders": orders})
    assert result["net"].to_pydict() == {"id": [1, 2], "net": [70, 108]}
    assert result["double"].to_pydict() == {"doubled": [140, 216]}


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM input",
        "SELECT * FROM input; SELECT * FROM input",
        "SELECT missing FROM input",
    ],
)
def test_sql_rejects_invalid_queries_during_planning(query):
    data = pa.table({"x": [1]})
    with pytest.raises((cf.CompileError, cf.ConfigError, cf.ExecutionError)):
        cf.compute(data, lambda t: t.sql(query))


def test_sql_validates_explicit_tables_and_duplicate_output_fields():
    source = cf.table_input("values", schema=pa.schema([("x", pa.int64())]))
    with pytest.raises(ValueError, match="sql.tables"):
        cf.sql("SELECT 1")
    with pytest.raises(TypeError, match="sql.tables.bad"):
        cf.sql("SELECT * FROM bad", bad=1)
    with pytest.raises(
        (cf.CompileError, cf.ConfigError, cf.ExecutionError), match="duplicate|unique"
    ):
        source.sql("SELECT x AS value, x AS value FROM input").collect(
            pa.table({"x": [1]})
        )


def test_sql_computed_schema_preserves_native_nullability_through_fragments():
    data = pa.table({"x": [1, None]})
    source = cf.table_input("values", schema=data.schema)
    filled = source.sql("SELECT coalesce(x, 0) AS x FROM input")
    result = (
        filled.with_columns(y=filled["x"] + 1).sql("SELECT y FROM input").collect(data)
    )
    assert result.to_pydict() == {"y": [2, 1]}
    assert result.schema.field("y").nullable is False


def _ordered_source():
    schema = pa.schema(
        [
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("price", pa.float64(), nullable=False),
        ]
    )
    return cf.table_input(
        "quotes",
        schema=schema,
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("ts",),
    )


def test_sql_shared_rolling_ancestor_is_one_native_state_operator():
    source = _ordered_source()
    rolling = source.with_columns(delta=cf.ts.delta(source["price"]))
    first = rolling.select("delta").sql("SELECT delta FROM input")
    second = rolling.select(twice=rolling["delta"] * 2.0).sql("SELECT twice FROM input")
    document = cf.Program(
        "shared-state", outputs={"first": first, "second": second}
    ).to_project(mode="stream")
    nodes = document.root["graph"]["nodes"]
    assert sum(node["operator"]["kind"] == "rolling" for node in nodes) == 1


def test_sql_output_requires_new_ordering_before_temporal_expressions():
    source = _ordered_source().sql("SELECT * FROM input ORDER BY ts")
    with pytest.raises(
        cf.CompileError, match="ordering_required|SQL output has no temporal ordering"
    ):
        cf.Program(
            "invalid",
            outputs={"out": source.select(previous=cf.ts.lag(source["price"]))},
        ).to_project(mode="stream")


def test_sql_composes_with_registered_matrix_and_static_weights():
    import numpy as np

    data = pa.table({"x": [1.0, 3.0]})
    source = cf.table_input("values", schema=data.schema).sql("SELECT x FROM input")
    weights = cf.parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(1, 1)
    )
    matrix = cf.linalg.from_columns(source, columns=["x"], backend="numpy")
    attached = cf.table.attach_columns(
        source, cf.linalg.matmul(matrix, weights), names=["weighted"]
    )
    runtime = cf.Runtime()
    cf.register_numpy(runtime)
    output = attached.sql("SELECT weighted FROM input")
    result = output.collect(
        {
            "values": data,
            "weights": cf.Batch.from_array(np.array([[2.0]]), backend="numpy"),
        },
        runtime=runtime,
    )
    assert result.to_pydict() == {"weighted": [2.0, 6.0]}
