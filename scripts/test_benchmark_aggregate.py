from __future__ import annotations

import argparse
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_suite.__main__ import parse_args
from scripts.benchmark_suite.aggregate import (
    case_failures,
    collect,
    summarize,
    validate_fragment,
)
from scripts.benchmark_suite.catalog import CONTRACT, get_shard, shard_cases
from scripts.benchmark_suite.provenance import harness_sha256
from scripts.benchmark_suite.validation import _validate_catalog_cases
from scripts.test_benchmark_suite import measured_case

BASE, HEAD, RUST = "a" * 40, "b" * 40, "c" * 40
TAG = "calc-flow-python-v2026.10.13"


def fragment():
    shard = get_shard("engines-10")
    return {
        "contract": CONTRACT,
        "harness_sha256": harness_sha256(),
        "shard": shard,
        "releases": {
            side: {
                "git_sha": sha,
                "git_clean": True,
                "build_profile": "release",
                "native_sha256": "a" * 64,
            }
            for side, sha in (("baseline", BASE), ("candidate", HEAD))
        },
        "cases": [
            {**case, "status": "error", "error": "fixture failure"}
            for case in shard_cases(shard)
        ],
        "errors": [],
    }


def rust_fragment():
    shard = get_shard("rust")
    case = {
        "id": "rust/stream_join_perf",
        "family": "rust",
        "scope": "criterion-sample-mean",
        "status": "error",
        "error": "fixture failure",
    }
    return {
        "contract": CONTRACT,
        "harness_sha256": harness_sha256(),
        "shard": shard,
        "expected_case_ids": [case["id"]],
        "releases": {
            side: {
                "git_sha": sha,
                "git_clean": True,
                "build_profile": "release",
                "native_sha256": "a" * 64,
            }
            for side, sha in (("baseline", RUST), ("candidate", HEAD))
        },
        "cases": [case],
        "errors": [],
    }


class BenchmarkAggregateTests(unittest.TestCase):
    def collect_one(self, report):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results.json").write_text(json.dumps(report), encoding="utf-8")
            return collect(root, BASE, HEAD)

    def summarize_without_artifacts(self, kind: str, ref: str, rust: str = BASE):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs").mkdir()
            arguments = argparse.Namespace(
                input=root / "inputs",
                output=root / "summary",
                github_summary=None,
                expected_base=BASE,
                expected_head=HEAD,
                baseline_kind=kind,
                baseline_ref=ref,
                expected_rust_base=rust,
            )
            with redirect_stdout(io.StringIO()):
                status = summarize(arguments)
            markdown = (root / "summary/summary.md").read_text(encoding="utf-8")
            errors = json.loads(
                (root / "summary/summary.json").read_text(encoding="utf-8")
            )["errors"]
        return status, markdown.splitlines(), errors

    def test_complete_report_header_states_the_resolved_baseline(self):
        status, lines, errors = self.summarize_without_artifacts("release", TAG)
        self.assertEqual(status, 1)
        self.assertEqual(lines[0], "# Complete benchmark results")
        self.assertEqual(lines[2], f"Baseline: release {TAG} (`{BASE}`)")
        self.assertFalse(any(error.startswith("baseline") for error in errors))

    def test_complete_report_notes_a_first_parent_rust_baseline(self):
        status, lines, errors = self.summarize_without_artifacts("release", TAG, RUST)
        self.assertEqual(status, 1)
        self.assertEqual(lines[2], f"Baseline: release {TAG} (`{BASE}`)")
        self.assertEqual(lines[4], f"Rust shard baseline: first parent (`{RUST}`)")
        self.assertFalse(any(error.startswith("rust baseline") for error in errors))

    def test_matching_rust_baseline_adds_no_note(self):
        _, lines, _ = self.summarize_without_artifacts("release", TAG, BASE)
        self.assertNotIn("Rust shard baseline", "\n".join(lines))

    def test_invalid_rust_baseline_provenance_fails_closed(self):
        _, lines, errors = self.summarize_without_artifacts("release", TAG, "HEAD^")
        self.assertEqual(lines[4], "Rust shard baseline: unresolved")
        self.assertTrue(
            any(error.startswith("rust baseline provenance:") for error in errors)
        )

    def test_unresolved_baseline_provenance_fails_closed(self):
        _, lines, errors = self.summarize_without_artifacts("", "")
        self.assertEqual(lines[2], "Baseline: unresolved")
        self.assertTrue(
            any(error.startswith("baseline provenance:") for error in errors)
        )

    def test_rust_fragments_validate_against_the_rust_baseline(self):
        report = rust_fragment()
        validate_fragment(report, BASE, HEAD, RUST)
        with self.assertRaisesRegex(ValueError, "not a clean release build"):
            validate_fragment(report, BASE, HEAD)

    def test_engines_fragments_stay_anchored_at_the_release_baseline(self):
        validate_fragment(fragment(), BASE, HEAD, RUST)
        report = fragment()
        report["releases"]["baseline"]["git_sha"] = RUST
        with self.assertRaisesRegex(ValueError, "not a clean release build"):
            validate_fragment(report, BASE, HEAD, RUST)

    def test_collect_applies_the_rust_baseline_to_rust_shards_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results.json").write_text(
                json.dumps(rust_fragment()), encoding="utf-8"
            )
            _, errors, _ = collect(root, BASE, HEAD, RUST)
            self.assertFalse(
                any("not a clean release build" in error for error in errors)
            )
            _, errors, _ = collect(root, BASE, HEAD)
            self.assertTrue(
                any("not a clean release build" in error for error in errors)
            )

    def test_summarize_command_requires_baseline_provenance(self):
        command = [
            "benchmark_suite",
            "summarize",
            "--input=in",
            "--output=out",
            f"--expected-base={BASE}",
            f"--expected-head={HEAD}",
            f"--expected-rust-base={RUST}",
        ]
        provenance = ["--baseline-kind=release", f"--baseline-ref={TAG}"]
        with patch("sys.argv", [*command, *provenance]):
            arguments = parse_args()
        self.assertEqual(arguments.baseline_kind, "release")
        self.assertEqual(arguments.baseline_ref, TAG)
        self.assertEqual(arguments.expected_rust_base, RUST)
        with (
            patch("sys.argv", command),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            parse_args()

    def test_malformed_artifact_still_renders_missing_catalog_cases(self):
        for report in ([], {"shard": None}, {"cases": [None]}):
            with self.subTest(report=report):
                cases, errors, _ = self.collect_one(report)
                self.assertTrue(errors)
                self.assertTrue(
                    any(
                        case["id"] == "engines/10/calc-flow-sql/sma20" for case in cases
                    )
                )

    def test_present_but_empty_shard_does_not_hide_missing_cases(self):
        report = fragment()
        report["cases"] = []
        cases, errors, _ = self.collect_one(report)
        self.assertTrue(errors)
        names = {case["id"] for case in cases}
        self.assertTrue({case["id"] for case in shard_cases(report["shard"])} <= names)

    def test_partial_shard_preserves_the_results_that_were_returned(self):
        report = fragment()
        retained = report["cases"][0]
        report["cases"] = [retained]
        cases, errors, _ = self.collect_one(report)
        self.assertTrue(errors)
        actual = next(case for case in cases if case["id"] == retained["id"])
        self.assertEqual(actual, retained)

    def test_nonfinite_artifact_does_not_poison_summary_json(self):
        report = fragment()
        report["cases"][0]["candidate"] = [[float("nan")]]
        cases, errors, _ = self.collect_one(report)
        self.assertTrue(errors)
        json.dumps(cases, allow_nan=False)

    def test_dirty_or_wrong_baseline_is_not_comparable(self):
        for key, value in (("git_clean", False), ("git_sha", HEAD)):
            report = fragment()
            report["releases"]["baseline"][key] = value
            with self.assertRaises(ValueError):
                validate_fragment(report, BASE, HEAD)

    def test_startup_inclusive_stream_evidence_is_not_comparable(self):
        report = fragment()
        case = next(c for c in report["cases"] if c["backend"] == "calc-flow-stream")
        case["scope"] = "runner-start-to-drain-arrow"
        with self.assertRaisesRegex(ValueError, "dimensions"):
            validate_fragment(report, BASE, HEAD)

    def test_older_timing_and_minimum_only_contracts_are_rejected(self):
        for version in (1, 2):
            report = fragment()
            report["contract"] = f"calc-flow-benchmark-suite-v{version}"
            with (
                self.subTest(version=version),
                self.assertRaisesRegex(ValueError, "contract"),
            ):
                validate_fragment(report, BASE, HEAD)

    def test_invalid_allocation_values_fail_the_gate(self):
        for value in (-1, float("nan"), True, "0"):
            row = measured_case(
                kind="metric",
                metric="calls_per_dispatch",
                baseline_value=0,
                candidate_value=value,
            )
            self.assertTrue(case_failures([row]))

    def test_gate_rederives_intervals_instead_of_trusting_stored_verdict(self):
        row = measured_case(
            candidate=[[1.1] * 10] * 2,
            result={"verdict": "no-confirmed-regression", "round_intervals": []},
        )
        self.assertIn("regression", case_failures([row])[0])
        row = measured_case(result={"verdict": "regression"})
        self.assertEqual(case_failures([row]), [])

    def test_wrong_comparison_kind_is_not_a_historical_pass(self):
        report = fragment()
        row = measured_case(
            **shard_cases(report["shard"])[0], comparison="external", baseline=[]
        )
        report["cases"][0] = row
        with self.assertRaisesRegex(ValueError, "comparison"):
            validate_fragment(report, BASE, HEAD)

    def test_duplicate_case_cannot_pass_inventory_validation(self):
        report = fragment()
        report["cases"].append(copy.deepcopy(report["cases"][0]))
        with self.assertRaises(ValueError):
            validate_fragment(report, BASE, HEAD)

    def test_raw_samples_and_release_hashes_are_revalidated(self):
        report = fragment()
        row = measured_case(**shard_cases(report["shard"])[0])
        row["evidence"] = [
            {
                "environment": {"polars_threads": 32, "tokio_worker_threads": "32"},
                "native_sha256": {side: "a" * 64 for side in ("baseline", "candidate")},
                "completion": {
                    side: {"state": "completed"} for side in ("baseline", "candidate")
                },
                "samples": {
                    side: [
                        {"seconds": seconds, "correctness": {"passed": True}}
                        for seconds in row[side][index]
                    ]
                    for side in ("baseline", "candidate")
                },
            }
            for index in range(2)
        ]
        report["cases"][0] = row
        _validate_catalog_cases(report, report["shard"])
        row["evidence"][0]["native_sha256"]["baseline"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "hash"):
            _validate_catalog_cases(report, report["shard"])
        row["evidence"][0]["native_sha256"]["baseline"] = "a" * 64
        row["evidence"][1]["samples"]["candidate"][0]["seconds"] = 9.0
        with self.assertRaisesRegex(ValueError, "samples"):
            _validate_catalog_cases(report, report["shard"])


if __name__ == "__main__":
    unittest.main()
