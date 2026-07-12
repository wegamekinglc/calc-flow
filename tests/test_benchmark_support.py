from __future__ import annotations

from types import SimpleNamespace

from benchmarks.support import SCALES, record_benchmark


def test_overhead_table_and_array_scales_match() -> None:
    scale = SCALES["overhead"]

    assert scale.table_rows == scale.array_elements == 1_000


def test_record_benchmark_replaces_metadata_without_mutating_existing_mapping() -> None:
    original = {"existing": "value"}
    benchmark = SimpleNamespace(extra_info=original)

    record_benchmark(
        benchmark,
        scenario="metadata",
        input_rows=2,
        output_rows=1,
        backend="numpy",
    )

    assert original == {"existing": "value"}
    assert benchmark.extra_info is not original
    assert benchmark.extra_info["existing"] == "value"
    assert benchmark.extra_info["scenario"] == "metadata"
    assert benchmark.extra_info["backend"] == "numpy"
