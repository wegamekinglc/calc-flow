from __future__ import annotations

import asyncio
import threading

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import calc_flow as cf


def _builder(query="SELECT value + 1 AS result FROM input WHERE value > 0"):
    return cf.PipelineBuilder("reusable-query").sql("query", query)


def _batch(values, *, dtype=None, metadata=None):
    table = pa.table(
        {"value": pa.array(values, type=pa.int64() if dtype is None else dtype)}
    )
    if metadata is not None:
        table = table.replace_schema_metadata(metadata)
    return cf.Batch.from_pyarrow(table), table


def _result(result):
    return result.outputs["output"].to_pyarrow()


def _ipc(table):
    stream = pa.BufferOutputStream()
    with pa.ipc.new_stream(stream, table.schema) as writer:
        writer.write_table(table)
    return stream.getvalue().to_pybytes()


def _planned(result):
    assert len(result.datafusion_metrics) == 1
    return result.datafusion_metrics[0]["physical_planning_count"]


@pytest.mark.parametrize("initial", ([1, None, -2], [1, 2]))
def test_repeated_batch_queries_reuse_planning_with_fresh_inputs_and_metrics(initial):
    plan = _builder().compile_batch()
    for index, values in enumerate((initial, [5, -7, None], [], [9, -1])):
        batch, table = _batch(values)
        before, caller_before = _ipc(batch.to_pyarrow()), _ipc(table)
        result = plan.execute({"input": batch})
        assert _result(result)["result"].to_pylist() == [
            value + 1 for value in values if value is not None and value > 0
        ]
        assert _ipc(batch.to_pyarrow()) == before
        assert _ipc(table) == caller_before
        assert _planned(result) == (1 if index == 0 else 0)
        assert result.datafusion_metrics[0]["output_rows"] == _result(result).num_rows


def test_batch_query_templates_match_exact_schema_and_metadata():
    plan = _builder().compile_batch()
    for dtype, metadata in (
        (pa.int64(), {b"unit": b"a"}),
        (pa.int32(), {b"unit": b"a"}),
        (pa.int32(), {b"unit": b"b"}),
    ):
        batch, _ = _batch([1, 2], dtype=dtype, metadata=metadata)
        for count in (1, 0):
            result = plan.execute({"input": batch})
            assert _planned(result) == count
            assert _result(result)["result"].to_pylist() == [2, 3]


@pytest.mark.parametrize(
    "config",
    ({"target_partitions": 2}, {"parallelism_mode": "auto"}),
)
def test_batch_query_templates_preserve_other_partition_policies(config):
    plan = _builder().with_datafusion_config(**config).compile_batch()
    batch, _ = _batch([1, 2])
    for _ in range(2):
        result = plan.execute({"input": batch})
        assert _planned(result) == 1
        assert _result(result)["result"].to_pylist() == [2, 3]


@pytest.mark.parametrize(
    "query",
    ("SELECT SUM(value) AS result FROM input", "SELECT now() AS result FROM input"),
)
def test_ineligible_batch_queries_keep_planning(query):
    plan = _builder(query).compile_batch()
    batch, _ = _batch([1, 2])
    for _ in range(2):
        assert _planned(plan.execute({"input": batch})) == 1


def _udf_plan(function):
    runtime = cf.Runtime()
    runtime.register_scalar_udf(
        provider="python",
        name="transform",
        version="1",
        input_types=("int64",),
        return_type="int64",
        volatility="immutable",
        function=function,
    )
    return (
        cf.PipelineBuilder("snapshot")
        .expression(
            "query", "result = transform(value)", udfs=(("python", "transform", "1"),)
        )
        .compile_batch(runtime)
    )


def test_batch_templates_are_private_to_graph_config_and_udf_snapshot():
    plans = (
        _udf_plan(lambda values: pc.add(values, 1)),
        _udf_plan(lambda values: pc.add(values, 100)),
        _builder("SELECT value * 3 AS result FROM input").compile_batch(),
        _builder()
        .with_datafusion_config(batch_size=16, collect_diagnostics=False)
        .compile_batch(),
    )
    assert plans[0].fingerprint == plans[1].fingerprint
    batch, _ = _batch([1, 2])
    for count in (1, 0):
        for plan, expected in zip(plans, ([2, 3], [101, 102], [3, 6], [2, 3])):
            result = plan.execute({"input": batch})
            assert _planned(result) == count
            assert _result(result)["result"].to_pylist() == expected


def test_warm_batch_templates_isolate_concurrent_calls_and_cancellation():
    started, release = threading.Event(), threading.Event()

    def transform(values):
        if values[0].as_py() == 99:
            started.set()
            if not release.wait(timeout=10):
                raise RuntimeError("cancelled query did not release its callback")
        return pc.add(values, 1)

    plan = _udf_plan(transform)

    async def execute(value):
        batch, table = _batch([value])
        before, caller_before = _ipc(batch.to_pyarrow()), _ipc(table)
        result = await plan.execute_async({"input": batch})
        assert _ipc(batch.to_pyarrow()) == before
        assert _ipc(table) == caller_before
        assert _planned(result) == 0
        return _result(result)["result"].to_pylist()

    async def run():
        batch, _ = _batch([1])
        assert _planned(await plan.execute_async({"input": batch})) == 1
        results = await asyncio.gather(*(execute(value) for value in (2, 3, 4)))
        assert results == [[3], [4], [5]]
        cancelled = asyncio.create_task(execute(99))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            following = asyncio.create_task(execute(7))
            cancelled.cancel()
            await asyncio.sleep(0)
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert await following == [8]
        assert await execute(8) == [9]

    asyncio.run(run())
