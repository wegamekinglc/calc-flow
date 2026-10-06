from __future__ import annotations

import asyncio
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts.benchmark_suite import catalog, measure, report, validation
from scripts.benchmark_suite.measure import validate_stream_sample


def checkpoint_case():
    return next(
        case
        for case in catalog.engine_cases(100_000)
        if case.get("checkpoint_interval_millis") == 100
    )


def checkpoint_sample(case):
    return {
        "seconds": 0.25,
        "correctness": {"passed": True},
        "stream_evidence": {key: case[key] for key in catalog.STREAM_EVIDENCE_FIELDS}
        | {
            "nonterminal_epochs": [1],
            "rows_before_checkpoint": 1024,
            "recovery": "verified",
        },
    }


def measured(case, seconds):
    return {
        **case,
        "status": "ok",
        "comparison": "new",
        "correctness": True,
        "candidate": [[seconds] * 10, [seconds] * 10],
        "baseline": [],
    }


def baseline_assignment_text(text: str, assignment: str) -> str:
    name = assignment.split(" = ")[0]
    return "\n".join(
        assignment if line.startswith(name + " = ") else line
        for line in text.splitlines()
    )


def baseline_selected_variants(name: str) -> list[dict]:
    variants = [case for case in catalog.engine_cases(100_000) if case.get("variant")]
    small = [case for case in variants if case["batch_rows"] == 1024]
    if name != "CHECKPOINT_ROW_SCALES":
        return small
    return [case for case in small if case["checkpoint_interval_millis"] is not None]


class RetainedBenchmarkSuiteTests(unittest.TestCase):
    def test_baseline_variant_membership_requires_matching_declared_dimensions(self):
        assignments = (
            "SMALL_BATCH_ROWS = 2048",
            "CHECKPOINT_ROW_SCALES = ('1000000',)",
        )
        for assignment in assignments:
            with self.subTest(assignment=assignment), TemporaryDirectory() as directory:
                self.check_baseline_variant_assignment(Path(directory), assignment)

    def check_baseline_variant_assignment(self, temporary, assignment):
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts/benchmark_suite/catalog.py").read_text()
        name = assignment.split(" = ")[0]
        text = baseline_assignment_text(text, assignment)
        baseline = temporary / "scripts/benchmark_suite/catalog.py"
        baseline.parent.mkdir(parents=True)
        baseline.write_text(text)
        ids = catalog.baseline_case_ids(temporary, {"family": "engines"})
        selected = baseline_selected_variants(name)
        self.assertTrue(selected)
        for case in selected:
            self.assertEqual(catalog.comparison_kind(case, ids), "new")

    def test_scheduled_interval_references_have_explicit_cost_caps(self):
        interval = [
            case
            for case in catalog.engine_cases()
            if case["scenario"] == "interval_join"
        ]
        self.assertTrue(interval)
        self.assertEqual(max(case["rows"] for case in interval), 1_000_000)
        self.assertEqual(
            {case["backend"] for case in interval},
            {"calc-flow-stream", "calc-flow-sql", "datafusion", "polars", "polars-1t"},
        )

    def test_scheduled_retained_variants_have_explicit_dimensions(self):
        cases = [case for case in catalog.engine_cases() if case.get("variant")]
        for scenario in ("projection", "join", "interval_join", "asof_join"):
            with self.subTest(scenario=scenario):
                variants = [case for case in cases if case["scenario"] == scenario]
                self.check_small_throughput(variants)
                self.check_checkpoint_dimensions(variants)

    def check_small_throughput(self, variants):
        small = [case for case in variants if case["batch_rows"] == 1024]
        self.assertTrue(
            any(case["checkpoint_interval_millis"] is None for case in small)
        )

    def check_checkpoint_dimensions(self, variants):
        durations = [
            case for case in variants if case["checkpoint_interval_millis"] == 100
        ]
        self.assertEqual({case["rows"] for case in durations}, {100_000, 1_000_000})
        self.assertEqual({case["batch_rows"] for case in durations}, {1024, 64_000})
        for case in durations:
            self.assertEqual(case["workload"], "checkpoint-duration")
            self.assertEqual(case["replay_mode"], "exact-cursor")

    def test_scheduled_case_ids_are_unique(self):
        cases = catalog.engine_cases()
        self.assertEqual(len({case["id"] for case in cases}), len(cases))

    def test_checkpoint_evidence_rejects_missing_or_mislabeled_dimensions(self):
        case = checkpoint_case()
        sample = checkpoint_sample(case)
        validate_stream_sample(case, sample)
        for dimension in catalog.STREAM_EVIDENCE_FIELDS:
            with self.subTest(dimension=dimension):
                missing = deepcopy(sample)
                del missing["stream_evidence"][dimension]
                with self.assertRaisesRegex(ValueError, "stream evidence"):
                    validate_stream_sample(case, missing)
                wrong = deepcopy(sample)
                wrong["stream_evidence"][dimension] = "wrong"
                with self.assertRaisesRegex(ValueError, "stream evidence"):
                    validate_stream_sample(case, wrong)

    def test_checkpoint_evidence_requires_durable_nonterminal_recovery_proof(self):
        changes = (
            {"nonterminal_epochs": []},
            {"rows_before_checkpoint": 100_000},
            {"recovery": "configured"},
        )
        case = checkpoint_case()
        for change in changes:
            with self.subTest(change=change):
                sample = checkpoint_sample(case)
                sample["stream_evidence"].update(change)
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    validate_stream_sample(case, sample)

    def test_checkpoint_lifecycle_evidence_cannot_omit_the_declared_delay(self):
        case = checkpoint_case()
        sample = checkpoint_sample(case) | {"seconds": 0.01}
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate_stream_sample(case, sample)

    def test_round_rejects_mislabeled_stream_samples_and_closes_worker(self):
        for operation in ("prepare", "sample"):
            with self.subTest(operation=operation), TemporaryDirectory() as directory:
                self.check_round_mislabel(Path(directory), operation)

    def check_round_mislabel(self, temporary, operation):
        case = checkpoint_case()
        sample = checkpoint_sample(case)
        wrong = deepcopy(sample)
        wrong["stream_evidence"]["batch_rows"] = 17
        release = {"native_sha256": "a" * 64}

        async def request(**message):
            if message["operation"] == "hello":
                return {**release, "polars_threads": 32, "tokio_worker_threads": "32"}
            if message["operation"] == "prepare":
                warmup = wrong if operation == "prepare" else sample
                return {"case": case, "warmup": warmup}
            return wrong

        worker = SimpleNamespace(request=request, close=AsyncMock())
        with (
            patch.object(measure.Worker, "start", AsyncMock(return_value=worker)),
            self.assertRaisesRegex(ValueError, "stream evidence"),
        ):
            asyncio.run(
                measure._round(
                    case,
                    {"candidate": (temporary, temporary)},
                    {"candidate": release},
                    temporary,
                )
            )
        worker.close.assert_awaited_once()

    def test_aggregation_rejects_mislabeled_original_stream_evidence(self):
        case = checkpoint_case() | {"comparison": "new"}
        sample = checkpoint_sample(case)
        release = {"native_sha256": "a" * 64}
        evidence = {
            "samples": {"candidate": [sample] * 10},
            "completion": {"candidate": {"state": "completed"}},
            "native_sha256": {"candidate": release["native_sha256"]},
        }
        validation._validate_round(case, evidence, {"candidate": release})
        wrong = deepcopy(evidence)
        wrong["samples"]["candidate"][0]["stream_evidence"]["replay_mode"] = (
            "unsupported"
        )
        with self.assertRaisesRegex(ValueError, "stream evidence"):
            validation._validate_round(case, wrong, {"candidate": release})

    def test_cross_library_report_keeps_throughput_and_lifecycle_variants_distinct(
        self,
    ):
        cases = catalog.engine_cases(100_000)
        projection = [case for case in cases if case["scenario"] == "projection"]
        native = [case for case in projection if case["backend"] == "calc-flow-stream"]
        throughput = next(case for case in native if not case.get("variant"))
        text = report.render_report(
            [measured(throughput, 0.001), measured(checkpoint_case(), 0.2)], []
        )
        self.assertIn("Checkpoint lifecycle variants", text)
        self.assertIn("200.000", text)
        self.assertIn("1.000", text)

    def test_invalid_variant_evidence_keeps_the_complete_report(self):
        for variant in ("small-batch", "checkpoint-recovery"):
            for defect in ("incorrect", "nonfinite"):
                with self.subTest(variant=variant, defect=defect):
                    self.check_invalid_variant_report(variant, defect)

    def check_invalid_variant_report(self, variant, defect):
        projection = [
            case
            for case in catalog.engine_cases(100_000)
            if case["scenario"] == "projection"
        ]
        case = next(case for case in projection if case.get("variant") == variant) | {
            "status": "ok",
            "comparison": "new",
            "correctness": defect != "incorrect",
            "candidate": [[float("nan") if defect == "nonfinite" else 0.2] * 10] * 2,
            "baseline": [],
        }
        text = report.render_report([case], [])
        self.assertIn(case["id"], text)
        self.assertIn("Incomplete or invalid evidence", text)
        self.assertIn("invalid", text)
        section = (
            "Small-batch native variants"
            if variant == "small-batch"
            else "Checkpoint lifecycle variants"
        )
        self.assertIn(section, text)


if __name__ == "__main__":
    unittest.main()
