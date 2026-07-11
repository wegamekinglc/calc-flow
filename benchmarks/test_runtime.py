from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

from benchmarks.support import (
    BenchmarkFixture,
    record_benchmark,
    selected_scale,
    table_inputs,
)
from calc_flow import (
    Batch,
    Checkpoint,
    ExpressionOperator,
    FileCheckpointStore,
    Pipeline,
    RunContext,
    StatelessOperator,
)


def identity(inputs: Mapping[str, Batch], _context: RunContext) -> Mapping[str, Batch]:
    return {"output": inputs["input"]}


@pytest.mark.benchmark(group="dag", min_rounds=3, max_time=0.5)
def test_graph_fan_out(benchmark: BenchmarkFixture) -> None:
    batch = table_inputs(selected_scale().table_rows).fact
    plan = (
        Pipeline("benchmark-fan-out")
        .add_node("root", StatelessOperator("root", identity))
        .add_node("gross", ExpressionOperator("gross", "gross = amount * quantity"))
        .add_node("tax", ExpressionOperator("tax", "tax = amount / 5"))
        .add_node("flag", ExpressionOperator("flag", "large = amount >= 5000"))
        .connect("root", "gross")
        .connect("root", "tax")
        .connect("root", "flag")
        .compile()
    )

    result = benchmark(plan.execute, {"input": batch})

    assert len(result.outputs) == 3
    record_benchmark(
        benchmark,
        scenario="dag_three_way_fan_out",
        input_rows=batch.num_rows,
        output_rows=sum(output.num_rows for output in result.outputs.values()),
        metrics=result.datafusion_metrics,
    )


def _checkpoint() -> Checkpoint:
    rows = selected_scale().table_rows
    return Checkpoint(
        pipeline_name="benchmark-checkpoint",
        pipeline_fingerprint="a" * 64,
        source_cursor={"offset": rows},
        sequence=rows,
        state={"counter": {"rows": rows, "partitions": [0, 1, 2, 3]}},
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.mark.benchmark(group="checkpoint", min_rounds=5, max_time=0.5)
def test_checkpoint_json_serialization(benchmark: BenchmarkFixture) -> None:
    checkpoint = _checkpoint()

    document = benchmark(json.dumps, checkpoint.to_dict(), sort_keys=True)

    assert json.loads(document)["sequence"] == checkpoint.sequence
    record_benchmark(
        benchmark,
        scenario="checkpoint_json_serialization",
        input_rows=0,
        output_rows=0,
    )


@pytest.mark.benchmark(group="checkpoint", min_rounds=5, max_time=0.5)
def test_checkpoint_atomic_write(benchmark: BenchmarkFixture, tmp_path) -> None:
    checkpoint = _checkpoint()
    store = FileCheckpointStore(tmp_path)

    benchmark(store.save, checkpoint)

    assert store.load(checkpoint.pipeline_name) == checkpoint
    record_benchmark(
        benchmark,
        scenario="checkpoint_atomic_write",
        input_rows=0,
        output_rows=0,
    )


@pytest.mark.benchmark(group="checkpoint", min_rounds=5, max_time=0.5)
def test_checkpoint_recovery_load(benchmark: BenchmarkFixture, tmp_path) -> None:
    checkpoint = _checkpoint()
    store = FileCheckpointStore(tmp_path)
    store.save(checkpoint)

    recovered = benchmark(store.load, checkpoint.pipeline_name)

    assert recovered == checkpoint
    record_benchmark(
        benchmark,
        scenario="checkpoint_recovery_load",
        input_rows=0,
        output_rows=0,
    )
