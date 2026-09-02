"""Tests for isolated stream-lifecycle benchmark evidence validation."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.verify_stream_lifecycle_evidence import validate_report


def _fingerprint(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _report(*, resumed_batches: int = 0) -> dict[str, object]:
    machine = {"cpu": "stable"}
    dependency = {"calc_flow": "4.0.0"}
    workload = {"rows": 50_000}
    phases = {
        "startup_duration_seconds": 0.01,
        "steady_processing_duration_seconds": 0.02,
        "checkpoint_duration_seconds": 0.03,
        "cancel_duration_seconds": 0.01,
        "recovery_duration_seconds": 0.01,
        "shutdown_duration_seconds": 0.01,
    }
    return {
        "benchmarks": [
            {
                "name": "test_stream_window_checkpoint_and_recovery[standard]",
                "stats": {"mean": 0.1, "stddev": 0.001, "rounds": 20},
                "extra_info": {
                    "scenario": "symbolic_stream_window_checkpoint",
                    "checkpoint_batches": 10,
                    "checkpoint_bytes": 107_141,
                    "checkpoint_bytes_p50": 107_141,
                    "checkpoint_bytes_p95": 107_145,
                    "checkpoint_duration_p50_seconds": 0.02,
                    "checkpoint_duration_p95_seconds": 0.04,
                    "recovery_resumed_batches": resumed_batches,
                    "recovery_duration_p50_seconds": 0.01,
                    "recovery_duration_p95_seconds": 0.02,
                    "diagnostic_samples": 20,
                    "total_duration_seconds": 0.1,
                    "rss_before_bytes": 100_000,
                    "rss_after_bytes": 110_000,
                    "peak_rss_bytes": 120_000,
                    "machine_identity": machine,
                    "dependency_identity": dependency,
                    "workload_identity": workload,
                    "machine_fingerprint": _fingerprint(machine),
                    "dependency_fingerprint": _fingerprint(dependency),
                    "workload_fingerprint": _fingerprint(workload),
                    **phases,
                },
            }
        ]
    }


class StreamLifecycleEvidenceTests(unittest.TestCase):
    def test_accepts_complete_isolated_evidence(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "report.json"
            path.write_text(json.dumps(_report()), encoding="utf-8")

            evidence = validate_report(path, minimum_rounds=20)

        self.assertEqual(evidence["checkpoint_bytes"], 107_141)

    def test_rejects_replayed_batches_or_too_few_rounds(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "report.json"
            path.write_text(json.dumps(_report(resumed_batches=1)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "replayed batches"):
                validate_report(path, minimum_rounds=20)

            document = _report()
            document["benchmarks"][0]["stats"]["rounds"] = 19  # type: ignore[index]
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least 20 rounds"):
                validate_report(path, minimum_rounds=20)

    def test_rejects_missing_phase_or_fingerprint_mismatch(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "report.json"
            document = _report()
            del document["benchmarks"][0]["extra_info"][  # type: ignore[index]
                "shutdown_duration_seconds"
            ]
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shutdown_duration_seconds"):
                validate_report(path, minimum_rounds=20)

            document = _report()
            document["benchmarks"][0]["extra_info"][  # type: ignore[index]
                "machine_fingerprint"
            ] = "not-a-fingerprint"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "machine_fingerprint"):
                validate_report(path, minimum_rounds=20)


if __name__ == "__main__":
    unittest.main()
