"""Identity guards for the maintained streaming operator benchmarks."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pyarrow as pa

from benchmarks import test_stream_operators as operators
from scripts.benchmark_suite.identity import validate_identity


def test_filtered_stream_descriptor_preserves_input_and_output_row_counts(
    monkeypatch,
) -> None:
    scenario = "rolling_arithmetic_filter"
    source, _parts = operators._workload(scenario)
    sequence = source["sequence"].to_numpy()
    selected = sequence % operators.ENTITIES < operators.ENTITIES // 2
    mean = operators.expected_rolling_mean(
        source["price"].to_numpy(), entities=operators.ENTITIES, window=20
    )
    output = pa.table(
        {"sequence": sequence[selected], "value": (mean * 2.0 + 1.0)[selected]}
    )
    monkeypatch.setattr(operators, "_run", AsyncMock(return_value=output))
    benchmark = Mock(side_effect=lambda run: run())
    benchmark.extra_info = {}

    operators.test_stream_operator_lifecycle(benchmark, scenario, "overhead")

    validate_identity(benchmark.extra_info)
    assert benchmark.extra_info["workload_identity"]["input_rows"] == source.num_rows
    assert benchmark.extra_info["workload_identity"]["output_rows"] == output.num_rows
    assert source.num_rows == 2 * output.num_rows
