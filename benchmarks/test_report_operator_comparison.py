from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from benchmarks.report_operator_cases import OPERATORS
from benchmarks.report_operator_inputs import OperatorWorkload


@pytest.mark.parametrize("case", OPERATORS)
def test_report_comparison_matches_independent_oracle(tmp_path, case):
    from benchmarks.report_operator_comparison import NativeCase

    native = NativeCase(case, OperatorWorkload(768, 24, 7))
    batch, _ = native.batch()
    stream, _ = asyncio.run(native.stream(tmp_path))

    native.validate(batch)
    native.validate(stream)
    assert [
        event.batch.num_rows for event in native.events if hasattr(event, "batch")
    ] == [168, 168, 168, 168, 96]


def test_operator_validator_rejects_duplicate_sequence():
    from benchmarks.report_operator_comparison import validate_output

    output = pa.table(
        {"sequence": pa.array([0, 0], type=pa.uint64()), "value": [1.0, 2.0]}
    )
    with pytest.raises(AssertionError, match="sequence"):
        validate_output(output, np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    ("sequence", "value", "field"),
    [
        (pa.array([0.0, 1.0]), pa.array([1.0, 2.0]), "sequence"),
        (pa.array([0, 1], type=pa.uint64()), pa.array(["1", "2"]), "value"),
    ],
)
def test_operator_validator_rejects_castable_values_of_wrong_type(
    sequence, value, field
):
    from benchmarks.report_operator_comparison import validate_output

    with pytest.raises(AssertionError, match=field):
        validate_output(
            pa.table({"sequence": sequence, "value": value}), np.array([1.0, 2.0])
        )


def test_boolean_operator_rejects_numeric_substitute_and_normalizes_null():
    from benchmarks.report_operator_comparison import NativeCase, validate_output

    native = NativeCase("all_true20", OperatorWorkload(2, 2, 1))
    sequence = pa.array([0, 1], type=pa.uint64())
    numeric = pa.table({"sequence": sequence, "value": [1.0, 1.0]})
    with pytest.raises(AssertionError, match="value"):
        native.validate(numeric)

    nullable = pa.table({"sequence": sequence, "value": [True, None]})

    np.testing.assert_equal(
        validate_output(nullable, np.array([1.0, np.nan]), value_type=pa.bool_()),
        np.array([1.0, np.nan]),
    )


def test_report_sample_records_explicitly_unsupported_finance(tmp_path):
    from benchmarks.report_operator_comparison import collect_case

    result = collect_case("rsi20", OperatorWorkload(96, 4, 5), tmp_path, samples=1)

    assert result["finance"]["status"] == "unsupported"
    assert result["correctness"] == "passed"
    assert len(result["batch_seconds"]) == len(result["stream_seconds"]) == 1


@pytest.mark.parametrize("case", ["cs_percentile", "quantile20", "where"])
def test_report_finance_worker_compares_fresh_transforms(tmp_path, case):
    from benchmarks.report_operator_comparison import collect_case

    python = os.environ.get("FINANCE_PYTHON_PYTHON")
    if python is None:
        pytest.skip("requires the pinned Finance-Python worker")
    result = collect_case(
        case,
        OperatorWorkload(768, 24, 7),
        tmp_path,
        samples=1,
        finance_python=Path(python),
    )

    assert result["correctness"] == result["finance"]["status"] == "passed"
    assert len(result["finance"]["seconds"]) == 1


@pytest.mark.parametrize(
    "backend", ["calc-flow-batch", "calc-flow-stream", "finance-python"]
)
def test_scheduled_report_worker_validates_and_repeats_outputs(tmp_path, backend):
    from scripts.benchmark_suite.catalog import report_operator_cases
    from scripts.benchmark_suite.worker import dispatch

    if backend == "finance-python" and "FINANCE_PYTHON_PYTHON" not in os.environ:
        pytest.skip("requires the pinned Finance-Python worker")
    case = next(
        case
        for case in report_operator_cases()
        if case["backend"] == backend and case["scenario"] == "cs_percentile"
    )
    fixture = {**case, "rows": 768, "entities": 24, "chunk_ticks": 7}
    warm, active = dispatch({"operation": "prepare", "case": fixture}, None, tmp_path)
    try:
        sample, _ = dispatch({"operation": "sample"}, active, tmp_path)
        assert (
            warm["warmup"]["correctness"]
            == sample["correctness"]
            == {
                "passed": True,
                "rows": 768,
            }
        )
        assert sample["seconds"] > 0.0
    finally:
        complete, active = dispatch({"operation": "finish"}, active, tmp_path)
        assert complete["state"] == "completed"
        assert active is None
