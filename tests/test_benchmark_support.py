from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from benchmarks import support
from benchmarks.support import (
    SCALES,
    ArrayBenchmarkRecord,
    BenchmarkScale,
    record_array_benchmark,
    record_benchmark,
)


def _array_record(**changes: object) -> ArrayBenchmarkRecord:
    record = ArrayBenchmarkRecord(
        scenario="array_mean",
        scope="plan_end_to_end",
        backend="jax",
        expression="mean(x)",
        input_dtype="float32",
        output_dtype="float32",
        input_rows=1_000,
        output_rows=1,
    )
    return replace(record, **changes)


def _patch_array_identities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        support,
        "_machine_identity",
        lambda: {
            "operating_system": "linux",
            "architecture": "x86_64",
            "cpu_brand": "example cpu",
            "logical_cpu_count": 8,
            "python_implementation": "cpython",
        },
    )
    monkeypatch.setattr(
        support,
        "_dependency_identity",
        lambda backend: {
            "python_version": "3.13.9",
            "numpy_version": "2.5.1",
            **(
                {"jax_version": "0.10.2", "jaxlib_version": "0.10.2"}
                if backend == "jax"
                else {}
            ),
        },
    )
    monkeypatch.setattr(
        support,
        "_backend_configuration",
        lambda backend: (
            {"jax_platform": "cpu", "jax_enable_x64": False} if backend == "jax" else {}
        ),
    )


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


def test_record_array_benchmark_emits_complete_contract_v2_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_array_identities(monkeypatch)
    benchmark = SimpleNamespace(extra_info={"retained": True})

    record_array_benchmark(benchmark, _array_record())

    assert benchmark.extra_info["retained"] is True
    assert benchmark.extra_info["benchmark_contract_version"] == 2
    assert benchmark.extra_info["workload_version"] == 1
    assert benchmark.extra_info["scenario"] == "array_mean"
    assert benchmark.extra_info["scope"] == "plan_end_to_end"
    assert benchmark.extra_info["backend"] == "jax"
    assert benchmark.extra_info["expression"] == "mean(x)"
    assert benchmark.extra_info["input_dtype"] == "float32"
    assert benchmark.extra_info["output_dtype"] == "float32"
    assert benchmark.extra_info["input_rows"] == 1_000
    assert benchmark.extra_info["output_rows"] == 1
    assert benchmark.extra_info["machine_identity"]["cpu_brand"] == "example cpu"
    assert benchmark.extra_info["dependency_identity"] == {
        "python_version": "3.13.9",
        "numpy_version": "2.5.1",
        "jax_version": "0.10.2",
        "jaxlib_version": "0.10.2",
    }
    assert benchmark.extra_info["python_version"] == "3.13.9"
    assert benchmark.extra_info["numpy_version"] == "2.5.1"
    assert benchmark.extra_info["jax_version"] == "0.10.2"
    assert benchmark.extra_info["jaxlib_version"] == "0.10.2"
    assert benchmark.extra_info["jax_platform"] == "cpu"
    assert benchmark.extra_info["jax_enable_x64"] is False
    assert type(benchmark.extra_info["process_rss_bytes"]) is int
    for name in (
        "machine_fingerprint",
        "dependency_fingerprint",
        "workload_fingerprint",
    ):
        assert len(benchmark.extra_info[name]) == 64
        assert benchmark.extra_info[name] == benchmark.extra_info[name].lower()
        int(benchmark.extra_info[name], 16)


def test_cpu_brand_normalization_is_casefolded_and_whitespace_collapsed() -> None:
    assert support._normalize_cpu_brand("  ExAMPLE\t CPU\nName  ") == "example cpu name"


def test_machine_identity_contains_only_stable_compatibility_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        support,
        "get_cpu_info",
        lambda: {
            "brand_raw": "  ExAMPLE   CPU ",
            "hz_actual_friendly": "4.2 GHz",
            "hostname": "ignored-host",
            "pid": 123,
            "rss": 456,
            "total_memory": 789,
        },
    )
    monkeypatch.setattr(support.platform, "system", lambda: "Linux")
    monkeypatch.setattr(support.platform, "machine", lambda: "X86_64")
    monkeypatch.setattr(support.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(support.os, "cpu_count", lambda: 8)

    assert support._machine_identity() == {
        "operating_system": "linux",
        "architecture": "x86_64",
        "cpu_brand": "example cpu",
        "logical_cpu_count": 8,
        "python_implementation": "cpython",
    }


def test_machine_identity_rejects_an_unavailable_logical_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(support, "get_cpu_info", lambda: {"brand_raw": "CPU"})
    monkeypatch.setattr(support.os, "cpu_count", lambda: None)

    with pytest.raises(RuntimeError, match="logical CPU count is unavailable"):
        support._machine_identity()


def test_canonical_fingerprint_uses_compact_sorted_utf8_json() -> None:
    expected = hashlib.sha256('{"a":"é","z":1}'.encode()).hexdigest()

    fingerprint = support._canonical_fingerprint({"z": 1, "a": "é"})

    assert fingerprint == expected
    assert fingerprint == fingerprint.lower()


def test_dependency_identity_is_backend_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []
    versions = {"numpy": "2.5.1", "jax": "0.10.2", "jaxlib": "0.10.2"}
    monkeypatch.setattr(support.platform, "python_version", lambda: "3.13.9")
    monkeypatch.setattr(
        support,
        "version",
        lambda dependency: requested.append(dependency) or versions[dependency],
    )

    assert support._dependency_identity("numpy") == {
        "python_version": "3.13.9",
        "numpy_version": "2.5.1",
    }
    assert requested == ["numpy"]

    requested.clear()
    assert support._dependency_identity("jax") == {
        "python_version": "3.13.9",
        "numpy_version": "2.5.1",
        "jax_version": "0.10.2",
        "jaxlib_version": "0.10.2",
    }
    assert requested == ["numpy", "jax", "jaxlib"]


def test_numpy_configuration_and_metadata_omit_jax_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_array_identities(monkeypatch)
    benchmark = SimpleNamespace(extra_info={})

    record_array_benchmark(benchmark, _array_record(backend="numpy"))

    assert support._backend_configuration("numpy") == {}
    for name in (
        "jax_platform",
        "jax_enable_x64",
        "jax_version",
        "jaxlib_version",
    ):
        assert name not in benchmark.extra_info


def test_record_array_benchmark_replaces_metadata_without_mutating_previous_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_array_identities(monkeypatch)
    original = {"retained": True}
    benchmark = SimpleNamespace(extra_info=original)

    record_array_benchmark(benchmark, _array_record())

    assert original == {"retained": True}
    assert benchmark.extra_info is not original
    assert benchmark.extra_info["retained"] is True


@pytest.mark.parametrize(
    "changed_record",
    [
        _array_record(scope="provider_boundary"),
        _array_record(input_dtype="float64"),
        _array_record(output_dtype="float64"),
        _array_record(expression="sum(x)"),
        _array_record(input_rows=2_000),
        _array_record(output_rows=2),
    ],
)
def test_workload_fingerprint_changes_with_record_identity(
    monkeypatch: pytest.MonkeyPatch,
    changed_record: ArrayBenchmarkRecord,
) -> None:
    _patch_array_identities(monkeypatch)
    baseline = SimpleNamespace(extra_info={})
    changed = SimpleNamespace(extra_info={})

    record_array_benchmark(baseline, _array_record())
    record_array_benchmark(changed, changed_record)

    assert (
        baseline.extra_info["workload_fingerprint"]
        != changed.extra_info["workload_fingerprint"]
    )


def test_workload_fingerprint_changes_with_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_array_identities(monkeypatch)
    active_scale = BenchmarkScale("overhead", 1_000, 1_000, 16)
    monkeypatch.setattr(support, "selected_scale", lambda: active_scale)
    baseline = SimpleNamespace(extra_info={})
    changed = SimpleNamespace(extra_info={})

    record_array_benchmark(baseline, _array_record())
    active_scale = BenchmarkScale("small", 10_000, 10_000, 64)
    record_array_benchmark(changed, _array_record())

    assert (
        baseline.extra_info["workload_fingerprint"]
        != changed.extra_info["workload_fingerprint"]
    )


def test_workload_fingerprint_changes_with_jax_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_array_identities(monkeypatch)
    configuration: dict[str, object] = {
        "jax_platform": "cpu",
        "jax_enable_x64": False,
    }
    monkeypatch.setattr(
        support, "_backend_configuration", lambda _backend: dict(configuration)
    )
    baseline = SimpleNamespace(extra_info={})
    changed = SimpleNamespace(extra_info={})

    record_array_benchmark(baseline, _array_record())
    configuration["jax_enable_x64"] = True
    record_array_benchmark(changed, _array_record())

    assert (
        baseline.extra_info["workload_fingerprint"]
        != changed.extra_info["workload_fingerprint"]
    )


def test_process_rss_does_not_change_compatibility_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_array_identities(monkeypatch)
    process_rss = 100
    monkeypatch.setattr(
        support.psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=process_rss)),
    )
    baseline = SimpleNamespace(extra_info={})
    changed = SimpleNamespace(extra_info={})

    record_array_benchmark(baseline, _array_record())
    process_rss = 200
    record_array_benchmark(changed, _array_record())

    assert baseline.extra_info["process_rss_bytes"] == 100
    assert changed.extra_info["process_rss_bytes"] == 200
    for name in (
        "machine_fingerprint",
        "dependency_fingerprint",
        "workload_fingerprint",
    ):
        assert baseline.extra_info[name] == changed.extra_info[name]


@pytest.mark.parametrize(
    "record",
    [
        _array_record(scope=cast(Any, "unsupported")),
        _array_record(backend=cast(Any, "unsupported")),
        _array_record(scenario=""),
        _array_record(expression=""),
    ],
)
def test_invalid_array_record_fails_before_metadata_is_attached(
    monkeypatch: pytest.MonkeyPatch,
    record: ArrayBenchmarkRecord,
) -> None:
    original = {"retained": True}
    benchmark = SimpleNamespace(extra_info=original)
    monkeypatch.setattr(
        support,
        "_machine_identity",
        lambda: pytest.fail("metadata collection must not run"),
    )

    with pytest.raises(ValueError):
        record_array_benchmark(benchmark, record)

    assert benchmark.extra_info is original
    assert benchmark.extra_info == {"retained": True}
