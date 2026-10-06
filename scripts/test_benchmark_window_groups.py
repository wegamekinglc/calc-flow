from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from scripts.benchmark_suite.rust import bench_targets, run_binary
from scripts.benchmark_suite.window_groups import (
    CASES,
    expected_digest,
    window_groups_rows,
)


def report() -> dict:
    cases = []
    for name, config in CASES.items():
        sample = {
            "seconds": 0.01,
            "process_seconds": 0.008,
            "end_seconds": 0.002,
            "checkpoint_bytes": 0,
            "checkpoint_sha256": None,
            "process_output_rows": 0,
            "input_rows": config["rows"],
            "output_rows": config["groups"] * (2 if config["hopping"] else 1),
            "sha256": expected_digest(config),
            "validated_all_rows": True,
            "validated_recovery": False,
        }
        cases.append(
            {
                "name": name,
                "config": config,
                "oracle": sample
                | {
                    "validated_recovery": True,
                    "checkpoint_bytes": 1234,
                    "checkpoint_sha256": "0" * 64,
                },
                "samples": [sample] * 20,
            }
        )
    return {
        "schema": "calc-flow.window-groups.v1",
        "scope": "operator-input-and-finalization",
        "cases": cases,
    }


class WindowGroupsEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def test_full_inventory_preserves_observations(self):
        with TemporaryDirectory() as raw:
            path = Path(raw) / "window.json"
            path.write_text(json.dumps(report()))
            rows = window_groups_rows(path)
        self.assertEqual(len(rows), 24)
        self.assertIn("stream_window_groups", bench_targets(Path.cwd()))
        self.assertEqual(
            rows["stream_window_groups/composite_8192_hopping_five"]["samples"],
            [0.01] * 20,
        )

    def test_invalid_sample_counts_timings_values_and_recovery_fail(self):
        changes = (
            ("seconds", True),
            ("seconds", 0),
            ("process_seconds", -1),
            ("end_seconds", 0),
            ("end_seconds", 0.003),
            ("input_rows", 99999),
            ("input_rows", True),
            ("output_rows", 3),
            ("process_output_rows", 1),
            ("checkpoint_bytes", 1),
            ("checkpoint_sha256", "0" * 64),
            ("sha256", "0" * 64),
            ("validated_all_rows", False),
        )
        with TemporaryDirectory() as raw:
            path = Path(raw) / "window.json"
            for field, value in changes:
                evidence = copy.deepcopy(report())
                evidence["cases"][0]["samples"][0][field] = value
                path.write_text(json.dumps(evidence))
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    window_groups_rows(path)
            evidence = report()
            evidence["cases"][0]["oracle"]["validated_recovery"] = False
            path.write_text(json.dumps(evidence))
            with self.assertRaises(ValueError):
                window_groups_rows(path)

    def test_invalid_inventory_workload_and_minimum_observations_fail(self):
        mutations = (
            lambda cases: cases.pop(),
            lambda cases: cases.append(cases[0]),
            lambda cases: cases[0]["samples"].pop(),
            lambda cases: cases[0]["config"].update({"groups": 5}),
            lambda cases: cases[0].update({"name": "unknown"}),
        )
        with TemporaryDirectory() as raw:
            path = Path(raw) / "window.json"
            for index, mutate in enumerate(mutations):
                evidence = copy.deepcopy(report())
                mutate(evidence["cases"])
                path.write_text(json.dumps(evidence))
                with self.subTest(index=index), self.assertRaises(ValueError):
                    window_groups_rows(path)

    def test_check_mode_accepts_no_measurements_only_when_explicit(self):
        evidence = report()
        for case in evidence["cases"]:
            case["samples"] = []
        with TemporaryDirectory() as raw:
            path = Path(raw) / "window.json"
            path.write_text(json.dumps(evidence))
            self.assertEqual(len(window_groups_rows(path, minimum_samples=0)), 24)
            with self.assertRaises(ValueError):
                window_groups_rows(path)

    async def test_binary_dispatch_uses_native_specialized_loader(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "output"
            output.mkdir()
            (output / "window-groups.json").write_text(json.dumps(report()))
            with patch(
                "scripts.benchmark_suite.rust.command", new_callable=AsyncMock
            ) as command:
                rows = await run_binary(
                    "stream_window_groups", root / "binary", root, output, "candidate"
                )
            self.assertEqual(len(rows), 24)
            self.assertEqual(
                command.call_args.args[0],
                [
                    str(root / "binary"),
                    "--samples",
                    "20",
                    "--output",
                    str(output / "window-groups.json"),
                ],
            )

    def test_contract_and_recovery_snapshot_errors_fail(self):
        mutations = (
            lambda evidence: evidence.update({"schema": "unknown"}),
            lambda evidence: evidence.update({"scope": "runtime"}),
            lambda evidence: evidence["cases"][0]["oracle"].update(
                {"checkpoint_bytes": 0}
            ),
            lambda evidence: evidence["cases"][0]["oracle"].update(
                {"checkpoint_sha256": "Z" * 64}
            ),
            lambda evidence: evidence["cases"][0]["config"].update({"hopping": 0}),
        )
        with TemporaryDirectory() as raw:
            path = Path(raw) / "window.json"
            for index, mutate in enumerate(mutations):
                evidence = copy.deepcopy(report())
                mutate(evidence)
                path.write_text(json.dumps(evidence))
                with self.subTest(index=index), self.assertRaises(ValueError):
                    window_groups_rows(path)


if __name__ == "__main__":
    unittest.main()
