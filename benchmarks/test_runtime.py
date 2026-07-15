from __future__ import annotations

import json

import pytest

from benchmarks.support import (
    BenchmarkFixture,
    benchmark_group,
    record_benchmark,
    selected_scale,
    table_inputs,
)
from calc_flow import FileCheckpointStore, PipelineBuilder


@pytest.mark.benchmark(group=benchmark_group("dag"), min_rounds=3, max_time=0.5)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_graph_fan_out(benchmark: BenchmarkFixture, _scale: str) -> None:
    batch = table_inputs(selected_scale().table_rows).fact
    plan = (
        PipelineBuilder("benchmark-fan-out")
        .expression("root", "root_amount = amount")
        .expression("gross", "gross = amount * quantity")
        .expression("tax", "tax = amount / 5")
        .expression("flag", "large = amount >= 5000")
        .connect("root", "gross")
        .connect("root", "tax")
        .connect("root", "flag")
        .compile()
    )

    warm_result = plan.execute({"input": batch})
    record_benchmark(
        benchmark,
        scenario="dag_three_way_fan_out",
        input_rows=batch.num_rows,
        output_rows=sum(output.num_rows for output in warm_result.outputs.values()),
        metrics=warm_result.datafusion_metrics,
    )

    result = benchmark(plan.execute, {"input": batch})

    assert len(result.outputs) == 3


def _checkpoint() -> dict[str, object]:
    rows = selected_scale().table_rows
    return {
        "created_at": "2026-01-01T00:00:00Z",
        "format_version": 2,
        "pipeline_fingerprint": "a" * 64,
        "pipeline_name": "benchmark-checkpoint",
        "sequence": rows,
        "source_cursor": {"offset": rows},
        "state": {"counter": {"partitions": [0, 1, 2, 3], "rows": rows}},
    }


@pytest.mark.benchmark(group=benchmark_group("checkpoint"), min_rounds=5, max_time=0.5)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_checkpoint_json_serialization(
    benchmark: BenchmarkFixture, _scale: str
) -> None:
    checkpoint = _checkpoint()
    record_benchmark(
        benchmark,
        scenario="checkpoint_json_serialization",
        input_rows=0,
        output_rows=0,
    )

    document = benchmark(json.dumps, checkpoint, sort_keys=True)

    assert json.loads(document)["sequence"] == checkpoint["sequence"]


@pytest.mark.benchmark(group=benchmark_group("checkpoint"), min_rounds=5, max_time=0.5)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_checkpoint_atomic_write(
    benchmark: BenchmarkFixture, tmp_path, _scale: str
) -> None:
    checkpoint = _checkpoint()
    store = FileCheckpointStore(tmp_path)
    record_benchmark(
        benchmark,
        scenario="checkpoint_atomic_write",
        input_rows=0,
        output_rows=0,
    )

    benchmark(store.save_blocking, checkpoint)

    assert store.load_blocking(str(checkpoint["pipeline_name"])) == checkpoint


@pytest.mark.benchmark(group=benchmark_group("checkpoint"), min_rounds=5, max_time=0.5)
@pytest.mark.parametrize("_scale", [selected_scale().name])
def test_checkpoint_recovery_load(
    benchmark: BenchmarkFixture, tmp_path, _scale: str
) -> None:
    checkpoint = _checkpoint()
    store = FileCheckpointStore(tmp_path)
    store.save_blocking(checkpoint)
    record_benchmark(
        benchmark,
        scenario="checkpoint_recovery_load",
        input_rows=0,
        output_rows=0,
    )

    recovered = benchmark(
        store.load_blocking,
        str(checkpoint["pipeline_name"]),
    )

    assert recovered == checkpoint
