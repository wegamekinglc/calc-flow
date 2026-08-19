from __future__ import annotations

import pytest

from benchmarks.support import (
    BenchmarkFixture,
    benchmark_group,
    record_benchmark,
    selected_scale,
    table_inputs,
)
from calc_flow import PipelineBuilder


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
        .compile_batch()
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
