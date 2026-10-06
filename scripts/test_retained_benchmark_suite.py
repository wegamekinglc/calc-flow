from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from scripts.benchmark_suite import catalog, measure, report, validation
from scripts.benchmark_suite.measure import validate_stream_sample


@pytest.mark.parametrize(
    "assignment", ("SMALL_BATCH_ROWS = 2048", "CHECKPOINT_ROW_SCALES = ('1000000',)")
)
def test_baseline_variant_membership_requires_matching_declared_dimensions(
    tmp_path, assignment
):
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/benchmark_suite/catalog.py").read_text()
    name = assignment.split(" = ")[0]
    text = "\n".join(
        assignment if line.startswith(name + " = ") else line
        for line in text.splitlines()
    )
    baseline = tmp_path / "scripts/benchmark_suite/catalog.py"
    baseline.parent.mkdir(parents=True)
    baseline.write_text(text)
    ids = catalog.baseline_case_ids(tmp_path, {"family": "engines"})
    selected = [
        case
        for case in catalog.engine_cases(100_000)
        if case.get("variant")
        and case["batch_rows"] == 1024
        and (
            name != "CHECKPOINT_ROW_SCALES"
            or case["checkpoint_interval_millis"] is not None
        )
    ]
    assert selected
    assert all(catalog.comparison_kind(case, ids) == "new" for case in selected)


def test_scheduled_retained_variants_have_explicit_dimensions_and_cost_caps():
    cases = catalog.engine_cases()
    interval = [case for case in cases if case["scenario"] == "interval_join"]
    assert interval
    assert max(case["rows"] for case in interval) == 1_000_000
    assert {case["backend"] for case in interval} == {
        "calc-flow-stream",
        "calc-flow-sql",
        "datafusion",
        "polars",
        "polars-1t",
    }
    for scenario in ("projection", "join", "interval_join", "asof_join"):
        variants = [
            case
            for case in cases
            if case["scenario"] == scenario and case.get("variant")
        ]
        assert any(
            case["batch_rows"] == 1024 and case["checkpoint_interval_millis"] is None
            for case in variants
        )
        durations = [
            case for case in variants if case["checkpoint_interval_millis"] == 100
        ]
        assert {case["rows"] for case in durations} == {100_000, 1_000_000}
        assert {case["batch_rows"] for case in durations} == {1024, 64_000}
        assert all(
            case["workload"] == "checkpoint-duration"
            and case["replay_mode"] == "exact-cursor"
            for case in durations
        )
    assert len({case["id"] for case in cases}) == len(cases)


def checkpoint_sample(case):
    return {
        "seconds": 0.25,
        "correctness": {"passed": True},
        "stream_evidence": {
            key: case[key]
            for key in (
                "batch_rows",
                "checkpoint_interval_millis",
                "replay_mode",
                "workload",
                "scope",
                "source_mode",
                "source_bindings",
            )
        }
        | {
            "nonterminal_epochs": [1],
            "rows_before_checkpoint": 1024,
            "recovery": "verified",
        },
    }


@pytest.mark.parametrize(
    "dimension",
    (
        "batch_rows",
        "checkpoint_interval_millis",
        "replay_mode",
        "workload",
        "scope",
        "source_mode",
        "source_bindings",
    ),
)
def test_checkpoint_evidence_rejects_missing_or_mislabeled_dimensions(dimension):
    case = next(
        case
        for case in catalog.engine_cases(100_000)
        if case.get("checkpoint_interval_millis") == 100
    )
    sample = checkpoint_sample(case)
    validate_stream_sample(case, sample)
    missing = deepcopy(sample)
    del missing["stream_evidence"][dimension]
    with pytest.raises(ValueError, match="stream evidence"):
        validate_stream_sample(case, missing)
    wrong = deepcopy(sample)
    wrong["stream_evidence"][dimension] = "wrong"
    with pytest.raises(ValueError, match="stream evidence"):
        validate_stream_sample(case, wrong)


@pytest.mark.parametrize(
    "change",
    (
        {"nonterminal_epochs": []},
        {"rows_before_checkpoint": 100_000},
        {"recovery": "configured"},
    ),
)
def test_checkpoint_evidence_requires_durable_nonterminal_recovery_proof(change):
    case = next(
        case
        for case in catalog.engine_cases(100_000)
        if case.get("checkpoint_interval_millis") == 100
    )
    sample = checkpoint_sample(case)
    sample["stream_evidence"].update(change)
    with pytest.raises(ValueError, match="checkpoint"):
        validate_stream_sample(case, sample)


def test_checkpoint_lifecycle_evidence_cannot_omit_the_declared_delay():
    case = next(
        case
        for case in catalog.engine_cases(100_000)
        if case.get("checkpoint_interval_millis") == 100
    )
    sample = checkpoint_sample(case) | {"seconds": 0.01}
    with pytest.raises(ValueError, match="checkpoint"):
        validate_stream_sample(case, sample)


@pytest.mark.parametrize("operation", ("prepare", "sample"))
def test_round_rejects_mislabeled_stream_samples_and_closes_worker(
    monkeypatch, tmp_path, operation
):
    import asyncio

    case = next(
        case
        for case in catalog.engine_cases(100_000)
        if case.get("checkpoint_interval_millis") == 100
    )
    sample = checkpoint_sample(case)
    wrong = deepcopy(sample)
    wrong["stream_evidence"]["batch_rows"] = 17
    release = {"native_sha256": "a" * 64}

    async def request(**message):
        if message["operation"] == "hello":
            return {**release, "polars_threads": 32, "tokio_worker_threads": "32"}
        if message["operation"] == "prepare":
            return {"case": case, "warmup": wrong if operation == "prepare" else sample}
        return wrong

    worker = SimpleNamespace(request=request, close=AsyncMock())
    monkeypatch.setattr(measure.Worker, "start", AsyncMock(return_value=worker))
    with pytest.raises(ValueError, match="stream evidence"):
        asyncio.run(
            measure._round(
                case,
                {"candidate": (tmp_path, tmp_path)},
                {"candidate": release},
                tmp_path,
            )
        )
    worker.close.assert_awaited_once()


def test_aggregation_rejects_mislabeled_original_stream_evidence():
    case = next(
        case
        for case in catalog.engine_cases(100_000)
        if case.get("checkpoint_interval_millis") == 100
    ) | {"comparison": "new"}
    sample = checkpoint_sample(case)
    release = {"native_sha256": "a" * 64}
    evidence = {
        "samples": {"candidate": [sample] * 10},
        "completion": {"candidate": {"state": "completed"}},
        "native_sha256": {"candidate": release["native_sha256"]},
    }
    validation._validate_round(case, evidence, {"candidate": release})
    wrong = deepcopy(evidence)
    wrong["samples"]["candidate"][0]["stream_evidence"]["replay_mode"] = "unsupported"
    with pytest.raises(ValueError, match="stream evidence"):
        validation._validate_round(case, wrong, {"candidate": release})


def test_cross_library_report_keeps_throughput_and_lifecycle_variants_distinct():
    cases = catalog.engine_cases(100_000)
    native = next(
        case
        for case in cases
        if case["backend"] == "calc-flow-stream"
        and case["scenario"] == "projection"
        and not case.get("variant")
    )
    variant = next(
        case for case in cases if case.get("checkpoint_interval_millis") == 100
    )

    def measured(case, seconds):
        return {
            **case,
            "status": "ok",
            "comparison": "new",
            "correctness": True,
            "candidate": [[seconds] * 10, [seconds] * 10],
            "baseline": [],
        }

    text = report.render_report([measured(native, 0.001), measured(variant, 0.2)], [])
    assert "Checkpoint lifecycle variants" in text
    assert "200.000" in text
    assert "1.000" in text


@pytest.mark.parametrize("variant", ("small-batch", "checkpoint-recovery"))
@pytest.mark.parametrize("defect", ("incorrect", "nonfinite"))
def test_invalid_variant_evidence_keeps_the_complete_report(variant, defect):
    case = next(
        case
        for case in catalog.engine_cases(100_000)
        if case.get("variant") == variant and case["scenario"] == "projection"
    ) | {
        "status": "ok",
        "comparison": "new",
        "correctness": defect != "incorrect",
        "candidate": [[float("nan") if defect == "nonfinite" else 0.2] * 10] * 2,
        "baseline": [],
    }
    text = report.render_report([case], [])
    assert case["id"] in text
    assert "Incomplete or invalid evidence" in text
    assert "invalid" in text
    assert (
        "Small-batch native variants"
        if variant == "small-batch"
        else "Checkpoint lifecycle variants"
    ) in text
