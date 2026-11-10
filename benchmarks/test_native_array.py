from __future__ import annotations

import pytest

from benchmarks.native_array_cases import (
    EAGER_CASES,
    bridge_case,
    eager_case,
    ownership_case,
    program_case,
    stream_case,
)
from benchmarks.native_array_support import NativeBenchmarkFixture, measure_native
from benchmarks.support import benchmark_group, selected_scale


@pytest.mark.parametrize("scenario", EAGER_CASES)
@pytest.mark.parametrize("_scale", [selected_scale().name])
@pytest.mark.benchmark(group=benchmark_group("native-array-eager-api"))
def test_native_eager(
    benchmark: NativeBenchmarkFixture, scenario: str, _scale: str
) -> None:
    measure_native(benchmark, eager_case(scenario, selected_scale()))


@pytest.mark.parametrize("kind", ("numpy", "arrow"))
@pytest.mark.parametrize("_scale", [selected_scale().name])
@pytest.mark.benchmark(group=benchmark_group("native-array-ingestion"))
def test_native_ingestion(
    benchmark: NativeBenchmarkFixture, kind: str, _scale: str
) -> None:
    case = ownership_case(kind, selected_scale())
    result = measure_native(benchmark, case)
    case.inputs[0][0] = -1
    case.validate(result)


@pytest.mark.parametrize("_scale", [selected_scale().name])
@pytest.mark.benchmark(group=benchmark_group("native-array-collect"))
def test_native_array_expr_collect(
    benchmark: NativeBenchmarkFixture, _scale: str
) -> None:
    measure_native(benchmark, program_case(selected_scale()))


@pytest.mark.parametrize("_scale", [selected_scale().name])
@pytest.mark.benchmark(group=benchmark_group("native-table-array-table"))
def test_native_table_bridge(benchmark: NativeBenchmarkFixture, _scale: str) -> None:
    measure_native(benchmark, bridge_case(selected_scale()))


@pytest.mark.parametrize("_scale", [selected_scale().name])
@pytest.mark.benchmark(group=benchmark_group("native-macd-owned-stream"))
def test_native_macd_stream(benchmark: NativeBenchmarkFixture, _scale: str) -> None:
    measure_native(benchmark, stream_case(selected_scale()))
