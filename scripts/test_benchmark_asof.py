from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.benchmark_suite.asof import CASES, _valid_sample, asof_rows
from scripts.benchmark_suite.rust import (
    bench_targets,
    run_binary,
)


def report() -> dict:
    cases = []
    for name, config in CASES.items():
        sample = {
            "config": config,
            "seconds": 0.1,
            "output_rows": config["pending"],
            "chunks": [[128, 0.1]] * (config["pending"] // 128),
            "max_chunk_bytes": 8192,
            "rss_available": True,
            "rss_before_bytes": 10,
            "rss_peak_bytes": 20,
            "allocation_peak_bytes": 30,
            "allocation_total_bytes": 1000,
            "allocation_count": 100,
            "checkpoint_before_bytes": 200,
            "checkpoint_after_bytes": 100,
            "admission_seconds_untimed": 0.1,
            "restore_seconds_untimed": 0.1 if config["restored"] else None,
            "capture_seconds_untimed": 0.01,
            "after_status": {
                "pending_left_rows": 0,
                "matched_rows": config["pending"],
                "retained_right_rows": config["retained"],
            },
            "validated_all_rows": False,
        }
        cases.append(
            {
                "name": name,
                "config": config,
                "oracle": {**sample, "validated_all_rows": True},
                "samples": [sample] * 20,
            }
        )
    return {
        "schema": "calc-flow.asof-finalization.v1",
        "scope": "operator-watermark-settlement",
        "cases": cases,
    }


class AsofEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def test_dominance_retention_requires_one_candidate_per_key(self):
        evidence = copy.deepcopy(report())
        evidence["retention"] = "latest-per-key"
        for case in evidence["cases"]:
            for sample in (case["oracle"], *case["samples"]):
                sample["after_status"]["retained_right_rows"] = 32
                sample["after_status"]["evicted_right_rows"] = (
                    case["config"]["retained"] - 32
                )
        with TemporaryDirectory() as raw:
            path = Path(raw) / "result.json"
            path.write_text(json.dumps(evidence))
            self.assertEqual(len(asof_rows(path)), 8)
            for field, value in (
                ("retained_right_rows", 31),
                ("retained_right_rows", 33),
                ("evicted_right_rows", 0),
            ):
                invalid = copy.deepcopy(evidence)
                invalid["cases"][0]["oracle"]["after_status"][field] = value
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    asof_rows(path)

    def test_current_and_adaptive_output_chunks_cover_all_pending_rows(self):
        evidence = copy.deepcopy(report())
        for case in evidence["cases"]:
            pending = case["config"]["pending"]
            chunks = (
                [[pending, 0.1]]
                if pending < 2048
                else [[pending // 2, 0.05], [pending - pending // 2, 0.1]]
            )
            for sample in (case["oracle"], *case["samples"]):
                sample["chunks"] = chunks
        with TemporaryDirectory() as raw:
            path = Path(raw) / "result.json"
            path.write_text(json.dumps(evidence))
            rows = asof_rows(path)
            for right_fixture in (0.0, 0.1 + 4 * math.ulp(0.1)):
                timed = copy.deepcopy(evidence)
                for case in timed["cases"]:
                    for sample in (case["oracle"], *case["samples"]):
                        sample.update(
                            right_fixture_seconds_untimed=right_fixture,
                            right_admission_seconds_untimed=0.0,
                            left_fixture_seconds_untimed=0.0,
                            left_admission_seconds_untimed=0.0,
                        )
                path.write_text(json.dumps(timed))
                self.assertEqual(len(asof_rows(path)), 8)
        self.assertEqual(len(rows), 8)

    def test_invalid_output_chunk_coverage_is_rejected(self):
        invalid = (
            [[0, 0.1], [512, 0.1]],
            [[True, 0.1], [511, 0.1]],
            [[511, 0.1]],
            [[513, 0.1]],
            [[-1, 0.1], [513, 0.1]],
            [[512, float("nan")]],
            [[512, -0.1]],
            [[10_001, 0.1]],
        )
        with TemporaryDirectory() as raw:
            path = Path(raw) / "result.json"
            for chunks in invalid:
                with self.subTest(chunks=chunks):
                    evidence = copy.deepcopy(report())
                    evidence["cases"][0]["oracle"]["chunks"] = chunks
                    path.write_text(json.dumps(evidence))
                    with self.assertRaises(ValueError):
                        asof_rows(path)

    async def test_maintained_native_target_keeps_eight_cases_and_raw_diagnostics(self):
        self.assertIn("stream_asof_perf", bench_targets(Path.cwd()))
        evidence = report()
        for case in evidence["cases"]:
            for sample in (case["oracle"], *case["samples"]):
                sample.update(
                    right_fixture_seconds_untimed=0.01,
                    right_admission_seconds_untimed=0.04,
                    left_fixture_seconds_untimed=0.01,
                    left_admission_seconds_untimed=0.04,
                )
        with TemporaryDirectory() as raw:
            root = Path(raw)

            async def command(argv, **_kwargs):
                self.assertIn("--output", argv)
                Path(argv[argv.index("--output") + 1]).write_text(json.dumps(evidence))

            with patch("scripts.benchmark_suite.rust.command", side_effect=command):
                rows = await run_binary(
                    "stream_asof_perf", root / "native", root, root, "candidate"
                )
            self.assertEqual(len(rows), 8)
            for case in evidence["cases"]:
                row = rows[f"stream_asof_perf/{case['name']}"]
                self.assertEqual(row["samples"], [0.1] * 20)
                self.assertEqual(row["metadata"]["observations"], case["samples"])
                self.assertEqual(row["metadata"]["oracle"], case["oracle"])

    def test_incomplete_or_corrupt_asof_evidence_is_rejected(self):
        invalid = []
        for field, value in (("schema", "bad"), ("scope", "bad"), ("cases", [])):
            invalid.append({**report(), field: value})
        item = report()
        item["cases"].pop()
        invalid.append(item)
        item = report()
        item["cases"].append(item["cases"][0])
        invalid.append(item)
        for field, value in (
            ("seconds", float("nan")),
            ("seconds", -1),
            ("output_rows", 1),
            ("allocation_total_bytes", None),
            ("allocation_peak_bytes", -1),
            ("rss_available", "yes"),
            ("chunks", []),
            ("after_status", {}),
        ):
            item = copy.deepcopy(report())
            item["cases"][0]["samples"][0][field] = value
            invalid.append(item)
        item = report()
        item["cases"][0]["samples"] = item["cases"][0]["samples"][:19]
        invalid.append(item)
        item = report()
        item["cases"][0]["oracle"]["validated_all_rows"] = False
        invalid.append(item)
        phases = dict(
            right_fixture_seconds_untimed=0.0,
            right_admission_seconds_untimed=0.0,
            left_fixture_seconds_untimed=0.0,
            left_admission_seconds_untimed=0.0,
        )
        invalid_phases = [
            {"right_fixture_seconds_untimed": 0.01},
            dict.fromkeys(phases, 1e308),
            dict.fromkeys(phases, 0.1),
            {**phases, "right_fixture_seconds_untimed": 0.1 + 16 * math.ulp(0.1)},
        ]
        invalid_phases.extend(
            {**phases, "left_admission_seconds_untimed": value}
            for value in (
                -0.01,
                float("nan"),
                float("inf"),
                10**400,
                None,
                True,
                "0.01",
            )
        )
        for breakdown in invalid_phases:
            item = copy.deepcopy(report())
            item["cases"][0]["oracle"].update(breakdown)
            invalid.append(item)
        with TemporaryDirectory() as raw:
            path = Path(raw) / "result.json"
            for index, item in enumerate(invalid):
                with self.subTest(index=index):
                    path.write_text(json.dumps(item))
                    with self.assertRaises(ValueError):
                        asof_rows(path)
        for value in (float("nan"), float("inf")):
            sample = copy.deepcopy(report()["cases"][0]["oracle"])
            sample.update(phases)
            sample["right_fixture_seconds_untimed"] = value
            with self.subTest(nonfinite_phase=value):
                self.assertFalse(
                    _valid_sample(sample, sample["config"], "tolerance-window")
                )
