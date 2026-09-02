"""Informational Studio backend resource and input-decoding benchmarks."""

from __future__ import annotations

import base64
import importlib
import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
from benchmarks.support import (
    BenchmarkFixture,
    record_benchmark,
    record_comparable_identity,
    selected_scale,
)

pytestmark = pytest.mark.studio_performance


@lru_cache(maxsize=1)
def _run_manager() -> Any:
    return importlib.import_module("calc_flow_studio.run_manager")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _record(
    benchmark: BenchmarkFixture,
    *,
    scenario: str,
    input_rows: int,
    output_rows: int,
    workload: dict[str, object],
) -> None:
    record_benchmark(
        benchmark,
        scenario=scenario,
        input_rows=input_rows,
        output_rows=output_rows,
        backend="studio",
    )
    record_comparable_identity(
        benchmark,
        workload_identity={
            "suite": "studio-observability",
            "workload_version": 1,
            "scenario": scenario,
            **workload,
        },
        # calc-flow and calc-flow-studio are the code under test, not fixed
        # comparison dependencies.
        dependency_packages=("pyarrow",),
    )


@pytest.mark.benchmark(group="studio-resource-monitor", min_rounds=20, max_time=1.0)
@pytest.mark.parametrize(("jobs", "files_per_job"), [(1, 100), (10, 100), (100, 10)])
def test_checkpoint_directory_monitor_scaling(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
    jobs: int,
    files_per_job: int,
) -> None:
    roots = []
    payload = b"x" * 128
    for job_index in range(jobs):
        root = tmp_path / f"job-{job_index}"
        root.mkdir()
        roots.append(root)
        for file_index in range(files_per_job):
            root.joinpath(f"segment-{file_index}.bin").write_bytes(payload)
    expected_bytes = jobs * files_per_job * len(payload)
    directory_size = _run_manager()._directory_size

    _record(
        benchmark,
        scenario=f"studio_checkpoint_scan_{jobs}x{files_per_job}",
        input_rows=jobs * files_per_job,
        output_rows=jobs,
        workload={"jobs": jobs, "checkpoint_files_per_job": files_per_job},
    )

    measured = benchmark(lambda: sum(directory_size(root) for root in roots))

    _require(measured == expected_bytes, "checkpoint scan byte total changed")


def _arrow_ipc_payload(rows: int) -> str:
    table = pa.table({"id": range(rows), "value": ["value"] * rows})
    sink = BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        midpoint = rows // 2
        writer.write_table(table.slice(0, midpoint))
        writer.write_table(table.slice(midpoint))
    return base64.b64encode(sink.getvalue()).decode("ascii")


@pytest.mark.benchmark(group="studio-input-decode", min_rounds=20, max_time=1.0)
@pytest.mark.parametrize("input_format", ["json", "arrow_ipc"])
def test_input_decode_and_combine_chunks(
    benchmark: BenchmarkFixture,
    input_format: str,
) -> None:
    rows = min(selected_scale().table_rows, 10_000)
    payload = (
        json.dumps([{"id": index, "value": "value"} for index in range(rows)])
        if input_format == "json"
        else _arrow_ipc_payload(rows)
    )
    decode_source = _run_manager()._decode_source
    max_bytes = 64 * 1024 * 1024
    warm_table, _encoded, _decoded = decode_source(
        input_format, payload, max_bytes=max_bytes
    )
    _require(warm_table.num_rows == rows, "warm decode row count changed")
    _require(
        all(len(column.chunks) == 1 for column in warm_table.columns),
        "warm decode did not combine Arrow chunks",
    )

    _record(
        benchmark,
        scenario=f"studio_decode_{input_format}_10000",
        input_rows=rows,
        output_rows=rows,
        workload={"format": input_format, "rows": rows, "combine_chunks": True},
    )

    table, _encoded, _decoded = benchmark(
        decode_source, input_format, payload, max_bytes=max_bytes
    )

    _require(table.num_rows == rows, "measured decode row count changed")
    _require(
        all(len(column.chunks) == 1 for column in table.columns),
        "measured decode did not combine Arrow chunks",
    )
