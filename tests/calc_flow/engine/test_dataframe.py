from __future__ import annotations

import pyarrow as pa

from calc_flow.engine.base import Engine
from calc_flow.engine.dataframe import (
    DataFrameEngine,
    DataFusionEngine,
    PandasEngine,
    PolarsEngine,
)


def test_all_dataframe_engines_are_engines() -> None:
    for cls in [PandasEngine, PolarsEngine, DataFusionEngine]:
        engine = cls()
        assert isinstance(engine, Engine)
        assert isinstance(engine, DataFrameEngine)


def test_pandas_engine_evaluate() -> None:
    engine = PandasEngine()
    table = pa.Table.from_pylist([{"a": 1, "b": 10}, {"a": 2, "b": 20}])
    result = engine.evaluate("c = a + b", table)
    assert result.schema.names == ["a", "b", "c"]
    assert result.to_pylist() == [
        {"a": 1, "b": 10, "c": 11},
        {"a": 2, "b": 20, "c": 22},
    ]


def test_pandas_engine_evaluate_expression_result() -> None:
    engine = PandasEngine()
    table = pa.Table.from_pylist([{"a": 1, "b": 10}, {"a": 2, "b": 20}])
    result = engine.evaluate("a + b", table)
    assert result.schema.names == ["result"]
    assert result.to_pylist() == [{"result": 11}, {"result": 22}]


def test_polars_engine_evaluate() -> None:
    engine = PolarsEngine()
    table = pa.Table.from_pylist([{"a": 1, "b": 10}, {"a": 2, "b": 20}])
    result = engine.evaluate("c = a + b", table)
    assert result.to_pylist() == [
        {"a": 1, "b": 10, "c": 11},
        {"a": 2, "b": 20, "c": 22},
    ]


def test_polars_engine_sql() -> None:
    engine = PolarsEngine()
    left = pa.Table.from_pylist([{"id": 1, "a": 10}, {"id": 2, "a": 20}])
    right = pa.Table.from_pylist([{"id": 1, "b": 3}, {"id": 2, "b": 4}])
    result = engine.sql(
        "select l.id, l.a + r.b as total "
        "from left_table l join right_table r on l.id = r.id",
        {"left_table": left, "right_table": right},
    )
    assert result.to_pylist() == [{"id": 1, "total": 13}, {"id": 2, "total": 24}]


def test_datafusion_engine_evaluate() -> None:
    engine = DataFusionEngine()
    table = pa.Table.from_pylist([{"a": 1, "b": 10}, {"a": 2, "b": 20}])
    result = engine.evaluate("c = a + b", table)
    assert result.to_pylist() == [
        {"a": 1, "b": 10, "c": 11},
        {"a": 2, "b": 20, "c": 22},
    ]


def test_datafusion_engine_sql() -> None:
    engine = DataFusionEngine()
    table = pa.Table.from_pylist([{"a": 1, "b": 10}, {"a": 2, "b": 20}])
    result = engine.sql("select a, b, a + b as c from input", {"input": table})
    assert result.to_pylist() == [
        {"a": 1, "b": 10, "c": 11},
        {"a": 2, "b": 20, "c": 22},
    ]


def test_datafusion_engine_sql_accepts_empty_tables() -> None:
    engine = DataFusionEngine()
    table = pa.table({"a": pa.array([], type=pa.int64())})

    result = engine.sql("select count(*) as n from input", {"input": table})

    assert result.to_pylist() == [{"n": 0}]
