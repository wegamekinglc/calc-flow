from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from scripts.benchmark_suite.asof_eviction import (
    CASES,
    asof_eviction_rows,
    expected_probe_digest,
    expected_status,
)
from scripts.benchmark_suite.rust import bench_targets, measure_rust, run_binary


def report() -> dict:
    cases = []
    for name, config in CASES.items():
        statuses = [
            expected_status(config, tick)
            | {"state_bytes": 1000 if config["mode"] != "all_expired" else 0}
            for tick in range(1, config["ticks"] + 1)
        ]
        sample = {
            "seconds": 0.01,
            "tick_seconds": [0.01 / config["ticks"]] * config["ticks"],
            "statuses": statuses,
            "before_state_bytes": 2000,
            "output_rows": 0,
            "checkpoint_bytes": 200 if statuses[-1]["state_rows"] else 0,
            "checkpoint_sha256": "0" * 64,
            "validated_status": True,
            "validated_recovery": False,
        }
        if config["mode"] == "none_expired":
            for status in statuses:
                status["state_bytes"] = 2000
        oracle = sample | {
            "validated_recovery": True,
            "restored_state_bytes": statuses[-1]["state_bytes"],
            "restored_state_rows": statuses[-1]["state_rows"],
            "probe_rows": config["keys"],
            "probe_sha256": expected_probe_digest(config),
            "identity_duplicate_rejected": config["mode"] == "sparse_identity_held",
        }
        cases.append(
            {"name": name, "config": config, "oracle": oracle, "samples": [sample] * 20}
        )
    return {
        "schema": "calc-flow.asof-eviction.v1",
        "scope": "operator-watermark-eviction",
        "cases": cases,
    }


class AsofEvictionEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def test_recovery_charge_matches_captured_state_after_preparation(self):
        evidence = copy.deepcopy(report())
        for case in evidence["cases"]:
            oracle = case["oracle"]
            oracle["checkpoint_state_bytes"] = oracle["restored_state_bytes"] + 500
            oracle["restored_state_bytes"] = oracle["checkpoint_state_bytes"]
        with TemporaryDirectory() as raw:
            path = Path(raw) / "eviction.json"
            path.write_text(json.dumps(evidence))
            self.assertEqual(len(asof_eviction_rows(path)), 8)
            for captured in (-1, True, None):
                invalid = copy.deepcopy(evidence)
                invalid["cases"][0]["oracle"]["checkpoint_state_bytes"] = captured
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    asof_eviction_rows(path)
            evidence["cases"][0]["oracle"]["restored_state_bytes"] -= 1
            path.write_text(json.dumps(evidence))
            with self.assertRaises(ValueError):
                asof_eviction_rows(path)

    def test_inventory_and_complete_observations_are_preserved(self):
        self.assertEqual(len(CASES), 8)
        with TemporaryDirectory() as raw:
            path = Path(raw) / "eviction.json"
            path.write_text(json.dumps(report()))
            rows = asof_eviction_rows(path)
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            rows["stream_asof_eviction/sparse_identity_held_65536"]["samples"],
            [0.01] * 20,
        )
        self.assertIn("stream_asof_eviction", bench_targets(Path.cwd()))

    def test_malformed_counts_timings_frontiers_oracles_and_inventory_fail(self):
        invalid = []
        for field, value in (
            ("seconds", float("nan")),
            ("seconds", True),
            ("seconds", 0),
            ("output_rows", 1),
            ("validated_status", False),
            ("checkpoint_sha256", "z" * 64),
        ):
            evidence = copy.deepcopy(report())
            evidence["cases"][0]["samples"][0][field] = value
            invalid.append(evidence)
        for field, value in (
            ("retained_right_rows", 4096),
            ("identity_only_rows", 0),
            ("state_rows", True),
            ("evicted_right_rows", 0),
            ("left_watermark", 0),
            ("right_watermark", 1),
            ("state_bytes", -1),
        ):
            evidence = copy.deepcopy(report())
            evidence["cases"][0]["samples"][0]["statuses"][0][field] = value
            invalid.append(evidence)
        for field, value in (
            ("validated_recovery", False),
            ("probe_rows", 1),
            ("probe_sha256", "0" * 64),
            ("restored_state_bytes", 1),
            ("identity_duplicate_rejected", False),
        ):
            evidence = copy.deepcopy(report())
            evidence["cases"][0]["oracle"][field] = value
            invalid.append(evidence)
        for change in (
            lambda cases: cases.pop(),
            lambda cases: cases.append(cases[0]),
            lambda cases: cases[0]["samples"].pop(),
            lambda cases: cases[0]["samples"][0]["tick_seconds"].pop(),
        ):
            evidence = copy.deepcopy(report())
            change(evidence["cases"])
            invalid.append(evidence)
        with TemporaryDirectory() as raw:
            path = Path(raw) / "eviction.json"
            for index, evidence in enumerate(invalid):
                with self.subTest(index=index):
                    path.write_text(json.dumps(evidence))
                    with self.assertRaises(ValueError):
                        asof_eviction_rows(path)

    async def test_dispatch_uses_strict_loader_and_twenty_samples(self):
        with TemporaryDirectory() as raw:
            source = Path(raw)
            output = source / "result"
            output.mkdir()
            (output / "asof-eviction.json").write_text(json.dumps(report()))
            with patch(
                "scripts.benchmark_suite.rust.command", new_callable=AsyncMock
            ) as command:
                rows = await run_binary(
                    "stream_asof_eviction",
                    source / "bench",
                    source,
                    output,
                    "candidate",
                )
            self.assertEqual(len(rows), 8)
            self.assertEqual(
                command.call_args.args[0],
                [
                    str(source / "bench"),
                    "--samples",
                    "20",
                    "--output",
                    str(output / "asof-eviction.json"),
                ],
            )

    async def test_added_target_is_new_coverage_without_fabricating_baseline(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            binary = root / "binary"
            binary.write_bytes(b"fixture")
            binaries = {
                "baseline": {"core": binary},
                "candidate": {"core": binary, "stream_asof_eviction": binary},
            }
            row = {
                "rows": 4096,
                "scope": "operator-watermark-eviction",
                "metadata": {},
                "samples": [0.1],
            }

            async def block(_binaries, _source, _output, _stamps, side):
                return (
                    {"stream_asof_eviction/sparse_identity_held_4096": row}
                    if side == "candidate"
                    else {},
                    [],
                )

            with (
                patch(
                    "scripts.benchmark_suite.rust.build_binaries",
                    AsyncMock(side_effect=list(binaries.values())),
                ),
                patch("scripts.benchmark_suite.rust._rust_provenance", return_value={}),
                patch(
                    "scripts.benchmark_suite.rust.declared_migrations", return_value={}
                ),
                patch(
                    "scripts.benchmark_suite.rust._stamp_fingerprints",
                    return_value={"baseline": {}, "candidate": {}},
                ),
                patch("scripts.benchmark_suite.rust._rust_block", side_effect=block),
                patch(
                    "scripts.benchmark_suite.rust._allocation_reports",
                    AsyncMock(return_value={}),
                ),
                patch("scripts.benchmark_suite.rust.allocation_rows", return_value=[]),
            ):
                result = await measure_rust(
                    {"id": "rust", "family": "rust"},
                    {},
                    {"baseline": root, "candidate": root},
                    root,
                )
            self.assertEqual(result["errors"], [])
            case = result["cases"][0]
            self.assertEqual(case["comparison"], "new")
            self.assertEqual(case["result"]["verdict"], "new-coverage")
            self.assertEqual(case["baseline"], [])
            self.assertEqual(case["candidate"], [[0.1], [0.1]])


if __name__ == "__main__":
    unittest.main()
