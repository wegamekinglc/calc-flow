from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.groupby_diagnostic import dal392


class FixedEvidenceTests(unittest.TestCase):
    def test_large_thread_snapshots_are_preserved_without_overflowing_ipc(self):
        phases = {
            scope: {
                "status": "measured",
                "wall_seconds": 0.1,
                "process_cpu_seconds": 0.1,
                "threads": [{"identity": "thread", "runtime_seconds": 0.01}] * 1000,
            }
            for scope in dal392.plan()["phases"]
        }
        phases["native_status_after_processing"] = {"operators": {}}
        sample = {"seconds": 0.1, "phase_evidence": phases}
        with tempfile.TemporaryDirectory() as directory:
            result = dal392.persist_phases(sample, Path(directory), 1)
            path = Path(directory) / result["phase_evidence"]["file"]
            self.assertEqual(json.loads(path.read_text()), phases)
            self.assertEqual(
                result["phase_evidence"]["sha256"], dal392.sha256_file(path)
            )
            self.assertLess(len(json.dumps(result)), 64 * 1024)
            self.assertEqual(sample["phase_evidence"], phases)

    def test_original_release_mismatch_fails_closed(self):
        with (
            patch.object(dal392, "load_release", return_value=dal392.SEALS["A"]),
            self.assertRaisesRegex(ValueError, "original October 20"),
        ):
            dal392.sealed("B", Path("release.json"))

    def test_changed_common_harness_fails_before_measurement(self):
        cpu = {
            "lscpu": [
                {"field": "Model name:", "data": "AMD EPYC 7763 64-Core Processor"}
            ]
        }
        with (
            patch.object(dal392, "host", return_value={}),
            patch.object(dal392, "require_host"),
            patch.object(dal392, "write_json"),
            patch.object(
                dal392.subprocess,
                "check_output",
                return_value=json.dumps(cpu),
            ),
            patch.object(dal392.platform, "python_version", return_value="3.13.16"),
            patch.object(dal392, "harness_sha256", return_value="changed"),
            patch.object(dal392, "sealed") as load,
        ):
            with self.assertRaisesRegex(ValueError, "common harness"):
                dal392.preflight(Path("output"), Path("releases"))
            load.assert_not_called()


class PhaseBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_phase_probe_preserves_original_timer_and_oracle(self):
        class Clock:
            def __init__(self, scope):
                self.scope = scope

            def finish(self):
                return {"scope": self.scope, "process_cpu_seconds": 1.0}

        async def original(bindings, *, phase_clock):
            clock = phase_clock("processing")
            return "arrow-table", 0.125, {"cpu_evidence": clock.finish()}

        binding = SimpleNamespace(job=SimpleNamespace(status=lambda: {"operators": {}}))
        with patch.object(dal392, "PhaseClock", Clock):
            capture = dal392.PhaseCapture()
            table, seconds, _ = await capture.processing(original, binding)
            checked = capture.validate(
                lambda value: {"passed": value == "arrow-table"}, table
            )
        self.assertEqual(seconds, 0.125)
        self.assertEqual(checked, {"passed": True})
        self.assertTrue(set(dal392.plan()["phases"]) <= capture.phases.keys())


if __name__ == "__main__":
    unittest.main()
