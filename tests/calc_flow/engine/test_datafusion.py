from __future__ import annotations

import pyarrow as pa
import pytest

from calc_flow.batch import Batch, BatchKind, BatchMetadata
from calc_flow.engine.base import Engine
from calc_flow.engine.datafusion import (
    DataFusionConfig,
    DataFusionEngine,
    DataFusionRuntime,
)


def test_datafusion_engine_is_table_engine() -> None:
    engine = DataFusionEngine()
    assert isinstance(engine, Engine)
    assert engine.input_kind is BatchKind.TABLE


def test_datafusion_engine_evaluate_assignment() -> None:
    engine = DataFusionEngine()
    batch = Batch.table(pa.table({"a": [1, 2], "b": [10, 20]}))

    result = engine.evaluate("c = a + b", batch)

    assert result.table_payload.to_pylist() == [
        {"a": 1, "b": 10, "c": 11},
        {"a": 2, "b": 20, "c": 22},
    ]


def test_datafusion_engine_evaluate_expression() -> None:
    engine = DataFusionEngine()
    batch = Batch.table(pa.table({"a": [1, 2], "b": [10, 20]}))

    result = engine.evaluate("a + b", batch)

    assert result.table_payload.to_pylist() == [{"result": 11}, {"result": 22}]


def test_datafusion_engine_sql_joins_batches() -> None:
    engine = DataFusionEngine()
    left = Batch.table(pa.table({"id": [1, 2], "a": [10, 20]}))
    right = Batch.table(pa.table({"id": [1, 2], "b": [3, 4]}))

    result = engine.sql(
        "select l.id, l.a + r.b as total "
        "from left_table l join right_table r on l.id = r.id",
        {"left_table": left, "right_table": right},
    )

    assert result.table_payload.to_pylist() == [
        {"id": 1, "total": 13},
        {"id": 2, "total": 24},
    ]
    assert result.metadata.attributes["input_batch_ids"] == (
        left.metadata.batch_id,
        right.metadata.batch_id,
    )


def test_datafusion_engine_preserves_single_input_metadata() -> None:
    metadata = BatchMetadata(source_id="input", sequence=2)
    batch = Batch.table(pa.table({"a": [1]}), metadata=metadata)

    result = DataFusionEngine().sql("select * from input", {"input": batch})

    assert result.metadata is metadata


def test_datafusion_engine_accepts_empty_table() -> None:
    batch = Batch.table(pa.table({"a": pa.array([], type=pa.int64())}))

    result = DataFusionEngine().sql("select count(*) as n from input", {"input": batch})

    assert result.table_payload.to_pylist() == [{"n": 0}]


@pytest.mark.parametrize(
    "query",
    [
        "delete from input",
        "select * from input; select 1",
        "create table output as select * from input",
        "with rows as (select * from input) delete from input",
        "with rows as (select * from input) insert into output select * from rows",
    ],
)
def test_datafusion_engine_rejects_non_select_queries(query: str) -> None:
    batch = Batch.table(pa.table({"a": [1]}))

    with pytest.raises(ValueError):
        DataFusionEngine().sql(query, {"input": batch})


def test_datafusion_engine_rejects_wrong_input_kind() -> None:
    import numpy as np

    with pytest.raises(TypeError, match="requires table batches"):
        DataFusionEngine().evaluate("x", Batch.array(np.asarray([1])))


def test_datafusion_engine_rejects_empty_input_mapping() -> None:
    with pytest.raises(ValueError, match="at least one"):
        DataFusionEngine().sql("select 1", {})


def test_datafusion_runtime_collects_query_metrics_and_reuses_session() -> None:
    batch = Batch.table(pa.table({"a": [1, 2]}))

    with DataFusionRuntime() as runtime:
        session_id = runtime.session_id
        runtime.sql("select a + 1 as a from input", {"input": batch}, node_id="one")
        runtime.sql("select a + 2 as a from input", {"input": batch}, node_id="two")

        assert runtime.session_id == session_id
        assert [metric.node_id for metric in runtime.metrics] == ["one", "two"]
        assert all(metric.planning_ns > 0 for metric in runtime.metrics)
        assert all(metric.execution_ns > 0 for metric in runtime.metrics)
        assert all("Projection" in metric.logical_plan for metric in runtime.metrics)


def test_datafusion_runtime_cleans_up_aliases_after_query_error() -> None:
    batch = Batch.table(pa.table({"a": [1]}))

    with DataFusionRuntime() as runtime:
        with pytest.raises(ValueError, match="missing"):
            runtime.sql(
                "select * from missing",
                {"input": batch},
            )

        result = runtime.sql("select * from input", {"input": batch})

    assert result.table_payload.to_pylist() == [{"a": 1}]


def test_datafusion_runtime_rejects_invalid_alias_and_closed_use() -> None:
    batch = Batch.table(pa.table({"a": [1]}))
    runtime = DataFusionRuntime()

    with pytest.raises(ValueError, match="invalid.*alias"):
        runtime.sql("select 1", {"not-valid": batch})
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        runtime.sql("select * from input", {"input": batch})


@pytest.mark.parametrize("field", ["batch_size", "target_partitions"])
def test_datafusion_config_rejects_non_positive_values(field: str) -> None:
    values = {"batch_size": 8192, "target_partitions": 1}
    values[field] = 0

    with pytest.raises(ValueError, match=field):
        DataFusionConfig(**values)
