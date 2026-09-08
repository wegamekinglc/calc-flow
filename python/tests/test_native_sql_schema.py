from __future__ import annotations

import asyncio
from types import MappingProxyType

import pyarrow as pa
import pytest

from calc_flow import Batch, ConfigError, ExecutionError, PipelineBuilder, Runtime


def test_native_sql_schema_matches_named_join_and_nullable_aggregate():
    runtime = Runtime()
    tables = {
        "orders": pa.table({"id": [1, 2], "amount": [12, 20]}),
        "fees": pa.table({"id": [1, 2], "fee": [2, 3]}),
    }
    query = (
        "WITH net AS (SELECT o.amount - f.fee AS value "
        "FROM orders o JOIN fees f ON o.id = f.id) "
        "SELECT SUM(value) AS total, COUNT(*) AS count FROM net"
    )
    schemas = MappingProxyType({name: table.schema for name, table in tables.items()})
    planned = runtime._inner._infer_sql_schema(query, schemas)
    actual = (
        PipelineBuilder("schema-parity")
        .sql("query", query, aliases=tuple(tables))
        .compile_batch(runtime)
        .execute({name: Batch.from_pyarrow(table) for name, table in tables.items()})
        .outputs["output"]
        .to_pyarrow()
    )
    assert actual.to_pydict() == {"total": [27], "count": [2]}
    assert planned.equals(actual.schema, check_metadata=True)
    assert planned.field("total").nullable
    assert not planned.field("count").nullable
    assert schemas["orders"].equals(tables["orders"].schema, check_metadata=True)


@pytest.mark.parametrize(
    "query",
    ["DELETE FROM input", "CREATE TABLE x (id INT)", "SELECT 1; SELECT 2"],
)
def test_native_sql_schema_rejects_non_select_statements(query):
    with pytest.raises(ConfigError, match="SELECT|statement"):
        Runtime()._inner._infer_sql_schema(
            query, {"input": pa.schema([("x", pa.int64())])}
        )


@pytest.mark.parametrize(
    "query", ["SELECT missing FROM input", "SELECT x FROM missing"]
)
def test_native_sql_schema_reports_resolution_errors(query):
    with pytest.raises(ExecutionError, match="missing"):
        Runtime()._inner._infer_sql_schema(
            query, {"input": pa.schema([("x", pa.int64())])}
        )


def test_native_sql_schema_rejects_duplicate_output_names():
    with pytest.raises((ConfigError, ExecutionError), match="duplicate|unique"):
        Runtime()._inner._infer_sql_schema(
            "SELECT left_table.x, right_table.x FROM left_table CROSS JOIN right_table",
            {
                name: pa.schema([("x", pa.int64())])
                for name in ("left_table", "right_table")
            },
        )


def test_native_sql_schema_validates_named_inputs():
    runtime = Runtime()
    with pytest.raises(ConfigError, match="alias|input"):
        runtime._inner._infer_sql_schema("SELECT 1", {})
    with pytest.raises((ConfigError, ExecutionError), match="alias|port|identifier"):
        runtime._inner._infer_sql_schema("SELECT 1", {"bad-name": pa.schema([])})
    with pytest.raises(ValueError, match="__arrow_c_schema__"):
        runtime._inner._infer_sql_schema("SELECT 1", {"input": object()})


def test_native_sql_schema_preserves_metadata_without_mutating_input():
    schema = pa.schema(
        [pa.field("x", pa.int64(), nullable=False, metadata={"unit": "USD"})],
        metadata={"origin": "caller"},
    )
    output = Runtime()._inner._infer_sql_schema(
        "SELECT x FROM input", {"input": schema}
    )
    assert output.equals(schema, check_metadata=True)
    assert schema.metadata == {b"origin": b"caller"}


def test_native_sql_schema_is_available_inside_event_loop_without_selecting_callbacks():
    calls = []
    runtime = Runtime()

    def forbidden(value):
        calls.append(value)
        raise AssertionError("schema planning must not invoke Python callbacks")

    runtime.register_scalar_udf(
        provider="python",
        name="forbidden",
        version="1",
        input_types=("int64",),
        return_type="int64",
        volatility="immutable",
        function=forbidden,
    )

    async def plan():
        schemas = {"input": pa.schema([pa.field("x", pa.int64(), nullable=False)])}
        result = runtime._inner._infer_sql_schema(
            "SELECT x + 1 AS value FROM input", schemas
        )
        assert result == pa.schema([pa.field("value", pa.int64(), nullable=False)])
        with pytest.raises(ExecutionError, match="forbidden"):
            runtime._inner._infer_sql_schema("SELECT forbidden(x) FROM input", schemas)

    asyncio.run(plan())
    assert calls == []
