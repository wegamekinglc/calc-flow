from __future__ import annotations

import copy
import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.benchmark_suite import issue406_harness as harness
from scripts.benchmark_suite.issue406_clock import clock_lock
from scripts.benchmark_suite.issue406_measurement import (
    cpu_evidence_status,
    sample_cpu_evidence,
    validate_cpu_evidence,
)
from scripts.test_issue406_harness import clock_lock_fixture, fixtures


def short_phase() -> dict:
    return {
        "schema": "calc-flow.phase-resources.v2",
        "status": "measured",
        "scope": "handler",
        "wall_seconds": 0.0014,
        "process_cpu_seconds": 0.0015,
        "process_cpu_clock": "CLOCK_PROCESS_CPUTIME_ID",
        "process_cpu_resolution_seconds": 1e-9,
        "threads": [{"tid": 1, "runtime_seconds": 0.001}],
        "departed_threads": ["2/4"],
    }


class ProcessClockEvidenceTests(unittest.TestCase):
    def test_sub_tick_clock_metadata_survives_target_binding_and_budget_check(self):
        original = short_phase()
        before = copy.deepcopy(original)
        evidence = sample_cpu_evidence(
            {"cpu_evidence": original, "handler_seconds": 0.001371196},
            scope="handler",
        )
        self.assertEqual(evidence["process_cpu_resolution_seconds"], 1e-9)
        self.assertEqual(evidence["departed_threads"], ["2/4"])
        self.assertEqual(
            cpu_evidence_status(evidence, threads=1, budget=2.0), "verified"
        )
        self.assertEqual(original, before)

    def test_clock_or_resolution_missing_unknown_or_abnormal_is_rejected(self):
        for field, value in (
            ("status", "unmeasured"),
            ("process_cpu_clock", None),
            ("process_cpu_clock", "thread-total"),
            ("process_cpu_clock", []),
            ("process_cpu_clock", "linux-proc-stat-utime-stime"),
            ("process_cpu_resolution_seconds", None),
            ("process_cpu_resolution_seconds", 0),
            ("process_cpu_resolution_seconds", -1),
            ("process_cpu_resolution_seconds", float("nan")),
            ("process_cpu_seconds", -1),
        ):
            evidence = {**short_phase(), field: value}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_cpu_evidence(evidence, threads=1, budget=2.0)

    def test_old_cpu_refusal_and_zero_tick_uncertainty_are_not_rewritten(self):
        old = {
            **short_phase(),
            "schema": "calc-flow.phase-resources.v1",
            "wall_seconds": 0.001371196,
            "process_cpu_clock": "linux-proc-stat-utime-stime",
            "process_cpu_resolution_seconds": 0.01,
        }
        with self.assertRaisesRegex(ValueError, "CPU budget exceeded"):
            validate_cpu_evidence(
                {**old, "process_cpu_seconds": 0.01}, threads=1, budget=2.0
            )
        self.assertEqual(
            cpu_evidence_status(
                {**old, "process_cpu_seconds": 0}, threads=1, budget=2.0
            ),
            "inconclusive",
        )

    def test_high_precision_clock_still_rejects_actual_excess_cpu(self):
        with self.assertRaisesRegex(ValueError, "CPU budget exceeded"):
            validate_cpu_evidence(
                {**short_phase(), "process_cpu_seconds": 0.003}, threads=1, budget=2.0
            )
        quantized = {**short_phase(), "process_cpu_resolution_seconds": 0.001}
        self.assertEqual(
            cpu_evidence_status(quantized, threads=1, budget=2.0), "inconclusive"
        )


class ClockOverlayTests(unittest.TestCase):
    def test_changed_effective_lock_rejects_build_and_removes_owned_mirror(self):
        with TemporaryDirectory() as raw:
            baseline, candidate = fixtures(Path(raw))
            with (
                patch.object(harness, "ROOT", candidate),
                self.assertRaisesRegex(ValueError, "sealed benchmark dependency lock"),
                harness.fresh_acceptance_harness(
                    baseline, Path(raw) / "output", "stream_join_perf"
                ) as mirror,
            ):
                lock = mirror / "Cargo.lock"
                lock.write_text(lock.read_text().replace("0.27.1", "0.28.0"))
            self.assertFalse(mirror.exists())

    def test_lock_adds_only_the_pinned_benchmark_edge_and_is_idempotent(self):
        original = clock_lock_fixture()
        actual = clock_lock(original)
        expected = tomllib.loads(original)
        expected["package"][0]["dependencies"].insert(0, "nix")
        self.assertEqual(tomllib.loads(actual), expected)
        self.assertEqual(clock_lock(actual), actual)
        self.assertEqual(clock_lock_fixture(), original)

    def test_lock_rejects_missing_forged_or_ambiguous_dependency_identity(self):
        original = clock_lock_fixture()
        for contents in (
            original.replace('name = "nix"', 'name = "other"'),
            original.replace('version = "0.27.1"', 'version = "0.27.2"'),
            original.replace("2eb04e9c", "00000000"),
            original + "[[package]]" + original.split("[[package]]")[2],
            original.replace('name = "calc-flow"', 'name = "unknown"'),
        ):
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                clock_lock(contents)
