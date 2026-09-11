from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts import measure_performance_plan as controller


class EntityParallelInventoryTests(unittest.TestCase):
    def listed(self, group: str) -> list:
        with (
            patch("sys.argv", ["measure", "--group", group, "--list"]),
            patch("sys.stdout", new_callable=io.StringIO) as output,
            patch.object(controller.Worker, "start") as start,
        ):
            controller.main()
            start.assert_not_called()
            return json.loads(output.getvalue())

    def test_cli_lists_the_predeclared_dual_and_mean_workloads(self):
        listed = self.listed("entity-parallel")
        self.assertEqual([group for group, _ in listed], ["entity-parallel"] * 2)
        common = {
            "family": "native-diagnostic",
            "backend": "calc-flow-stream",
            "rows": 64_000,
            "scope": "warm-enqueue-to-arrow-partial-window",
        }
        config = {
            "history_rows": 64_000,
            "append_rows": 64_000,
            "entities": 64,
            "history_segment_rows": 64_000,
            "window": 20,
            "fast_window": 5,
            "append_entities": None,
        }
        expected = [
            {
                **common,
                "id": "entity-parallel/dual_sma/h64000/a64000/e64/b64000/w5-20",
                "scenario": "dual_sma",
                "config": {**config, "indicator": "dual_sma_spread"},
            },
            {
                **common,
                "id": "entity-parallel/sma20/h64000/a64000/e64/b64000/w20",
                "scenario": "sma20",
                "config": {**config, "indicator": "rolling_mean"},
            },
        ]
        self.assertEqual([case for _, case in listed], expected)

    def test_tail_cli_selects_the_same_predeclared_entity_workloads(self):
        original = self.listed("entity-parallel")
        tail = self.listed("entity-parallel-tail")
        self.assertEqual([group for group, _ in tail], ["entity-parallel-tail"] * 2)
        self.assertEqual([case for _, case in tail], [case for _, case in original])

    def test_original_and_fallback_case_dictionaries_remain_frozen(self):
        # Frozen before introducing this group, at harness commit a1b650fa.
        expected = {
            "core": (
                32,
                "ff367269d7fb78ab77bbb7d9073294ff299a67170359b3e3b304b620d72fb317",
            ),
            "sensitivity": (
                15,
                "46a8bec3f832318b2e113d3c967d230f0719aa48ebb2da09533d7a7d95182ccd",
            ),
            "tail": (
                2,
                "ee417f3f7e3d3b6e7d2a660811a662fa426826e7414029b0857f1bed08e407e4",
            ),
            "fallback-cost": (
                12,
                "ff5f5f95f1c68eb02695543c16b3c8e9df509208b62e2c78ef7e799f7349321e",
            ),
        }
        original = []
        for group, (count, digest) in expected.items():
            with self.subTest(group=group):
                listed = self.listed(group)
                cases = [case for _, case in listed]
                encoded = json.dumps(cases, sort_keys=True, separators=(",", ":"))
                self.assertEqual(len(cases), count)
                self.assertEqual(hashlib.sha256(encoded.encode()).hexdigest(), digest)
                if group != "fallback-cost":
                    original.extend(listed)
        self.assertEqual(self.listed("all"), original)


class EntityParallelTailTests(unittest.IsolatedAsyncioTestCase):
    def assert_round_schedule(self, measured, root):
        expected = controller.entity_parallel_inventory()
        self.assertEqual(measured.await_count, 6)
        self.assertEqual(
            [call.args[0] for call in measured.await_args_list],
            [case for case in expected for _ in range(3)],
        )
        self.assertEqual(
            [call.args[2] for call in measured.await_args_list],
            [
                root / f"case-{case}" / f"round-{round_index}"
                for case in range(2)
                for round_index in range(3)
            ],
        )
        self.assertEqual(
            [call.args[3] for call in measured.await_args_list], [1000] * 6
        )

    def assert_tail_report(self, report):
        for case in report["cases"]:
            self.assertEqual(case["group"], "entity-parallel-tail")
            self.assertEqual(case["sample_count_per_revision"], 3000)
            self.assertEqual(case["result"]["verdict"], "descriptive-tail-evidence")
            self.assertEqual(case["result"]["fresh_worker_pairs"], 3)
            self.assertEqual(case["result"]["pairs_per_round"], 1000)
            for side in ("baseline", "candidate"):
                self.assertEqual(
                    case["result"]["latency_seconds"][side],
                    {"p50": 0.01, "p95": 0.01, "p99": 0.01},
                )
                self.assertEqual(
                    case["result"]["first_sink_seconds"][side],
                    {"p50": 0.006, "p95": 0.006, "p99": 0.006},
                )

    async def test_selected_tail_cases_each_use_three_fresh_thousand_pair_rounds(self):
        shared = {
            "cargo_lock_sha256": "same-lock",
            "rustc_verbose": "same-compiler",
            "features": ["same-feature"],
            "profile": "release",
        }
        sample = {
            "seconds": 0.01,
            "phases_seconds": {
                "enqueue_to_source_data": 0.001,
                "source_data_to_source_watermark": 0.002,
                "source_watermark_to_sink": 0.003,
            },
        }

        async def completed_round(case, pair, root, count, **kwargs):
            root.mkdir(parents=True)
            return {
                "environment": {"same": "environment"},
                "samples": {
                    side: [sample.copy() for _ in range(count)] for side in pair.sites
                },
                "completion": {side: {"state": "completed"} for side in pair.releases},
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            releases = [
                ({**shared, "native_sha256": side}, root / side)
                for side in ("baseline", "candidate")
            ]
            args = SimpleNamespace(
                group="entity-parallel-tail",
                case=None,
                list=False,
                baseline_build=root / "baseline.json",
                candidate_build=root / "candidate.json",
                root=root / "evidence",
            )
            with (
                patch.object(controller, "load_release", side_effect=releases),
                patch.object(controller, "harness_sha256", return_value={}),
                patch.object(controller, "_diagnostic_hashes", return_value={}),
                patch.object(
                    controller, "measure_round", new_callable=AsyncMock
                ) as measured,
                patch.object(controller.Worker, "start") as start,
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                measured.side_effect = completed_round
                await controller.run(args)
                start.assert_not_called()
            self.assert_round_schedule(measured, args.root)
            report = json.loads((args.root / "results.json").read_text())
            self.assert_tail_report(report)


if __name__ == "__main__":
    unittest.main()
