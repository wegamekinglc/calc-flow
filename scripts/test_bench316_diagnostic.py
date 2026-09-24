from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scripts.benchmark_suite.side_bias_diagnostic import (
    ObservedWorker,
    bias_detected,
    build_raw_blocks,
    condition_matrix,
    run,
    swap_site_contents,
)


class SideBiasDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def test_swap_site_contents_preserves_physical_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "A0" / "site"
            right = root / "B1" / "site"
            left.mkdir(parents=True)
            right.mkdir(parents=True)
            (left / "native.so").write_bytes(b"A")
            (right / "native.so").write_bytes(b"B")

            swap_site_contents(left, right, root / "backups")

            self.assertEqual((left / "native.so").read_bytes(), b"B")
            self.assertEqual((right / "native.so").read_bytes(), b"A")
            self.assertEqual(
                (root / "backups" / "left" / "native.so").read_bytes(), b"A"
            )
            self.assertEqual(
                (root / "backups" / "right" / "native.so").read_bytes(), b"B"
            )

    def test_condition_matrix_immediately_crosses_exact_slots_after_trigger(self):
        self.assertEqual(
            condition_matrix(),
            (
                ("aa_exact", "A0", "B1", "baseline", "clone_a"),
                ("aa_exact_reversed", "B1", "A0", "baseline", None),
                ("ba_original", "B1", "A0", "baseline", "restore_b"),
                ("ab_crossed", "B1", "A0", "baseline", "swap"),
                ("ba_crossed", "A0", "B1", "baseline", None),
                ("ab_repeat", "A0", "B1", "candidate", "swap_back"),
                ("ba_repeat", "B1", "A0", "candidate", None),
            ),
        )

    def test_bias_detected_requires_both_rounds_beyond_five_percent(self):
        self.assertTrue(
            bias_detected({"status": "ok", "result": {"round_changes": [13, 14]}})
        )
        self.assertTrue(
            bias_detected({"status": "ok", "result": {"round_changes": [-8, -9]}})
        )
        self.assertFalse(
            bias_detected({"status": "ok", "result": {"round_changes": [13, 0]}})
        )
        self.assertFalse(
            bias_detected({"status": "ok", "result": {"round_changes": [6, -7]}})
        )
        self.assertFalse(bias_detected({"status": "error"}))

    def test_raw_blocks_keep_round_cursor_and_sample_order(self):
        row = {
            "evidence": [
                {
                    "samples": {
                        "baseline": [{"seconds": 1.0, "start_row": 7}],
                        "candidate": [{"seconds": 1.2, "start_row": 7}],
                    }
                },
                {
                    "samples": {
                        "baseline": [{"seconds": 2.0, "start_row": 8}],
                        "candidate": [{"seconds": 2.4, "start_row": 8}],
                    }
                },
            ]
        }
        events = [
            {"event": "sample", "round": 0, "side": "baseline"},
            {"event": "sample", "round": 0, "side": "candidate"},
            {"event": "sample", "round": 1, "side": "candidate"},
            {"event": "sample", "round": 1, "side": "baseline"},
        ]

        self.assertEqual(
            build_raw_blocks(row, events),
            [
                [
                    {
                        "index": 0,
                        "baseline_seconds": 1.0,
                        "candidate_seconds": 1.2,
                        "baseline_start_row": 7,
                        "candidate_start_row": 7,
                        "order": ["baseline", "candidate"],
                    }
                ],
                [
                    {
                        "index": 0,
                        "baseline_seconds": 2.0,
                        "candidate_seconds": 2.4,
                        "baseline_start_row": 8,
                        "candidate_start_row": 8,
                        "order": ["candidate", "baseline"],
                    }
                ],
            ],
        )

    async def test_biased_probe_crosses_contents_on_same_runner(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            async def fake_install(release, install_root):
                site = install_root / "site"
                site.mkdir(parents=True)
                (site / "identity").write_text(release["identity"])
                record = site / "wheel.dist-info" / "RECORD"
                record.parent.mkdir()
                record.write_text(release["identity"])
                return site

            async def fake_measure(_case, _kind, sites, _releases, measurement_root):
                calls.append(
                    (
                        measurement_root.name,
                        tuple(
                            (side, site.parent.name, (site / "identity").read_text())
                            for side, site in sites.items()
                        ),
                    )
                )
                changes = (
                    [13.0, 14.0]
                    if measurement_root.name == "ab_probe_0"
                    else [0.0, 0.0]
                )
                return {
                    "status": "ok",
                    "result": {
                        "round_changes": changes,
                        "verdict": "regression"
                        if changes[0]
                        else "no-confirmed-regression",
                    },
                    "evidence": [],
                }

            releases = iter(
                (
                    {
                        "identity": "A",
                        "native_sha256": "same",
                        "wheel_sha256": "a" * 64,
                        "wheel_path": str(root / "A.whl"),
                    },
                    {
                        "identity": "B",
                        "native_sha256": "same",
                        "wheel_sha256": "b" * 64,
                        "wheel_path": str(root / "B.whl"),
                    },
                )
            )
            (root / "A.whl").write_bytes(b"A")
            (root / "B.whl").write_bytes(b"B")
            with (
                patch(
                    "scripts.benchmark_suite.side_bias_diagnostic.load_release",
                    side_effect=lambda _: next(releases),
                ),
                patch(
                    "scripts.benchmark_suite.side_bias_diagnostic.install",
                    side_effect=fake_install,
                ),
                patch(
                    "scripts.benchmark_suite.side_bias_diagnostic.shard_cases",
                    return_value=[{"id": "engines/100000/calc-flow-stream/group_by"}],
                ),
                patch(
                    "scripts.benchmark_suite.side_bias_diagnostic.get_shard",
                    return_value={},
                ),
                patch(
                    "scripts.benchmark_suite.measure.measure_case",
                    side_effect=fake_measure,
                ),
            ):
                await run(root / "base.json", root / "head.json", root / "results")

            self.assertEqual(
                [label for label, _ in calls],
                [
                    "ab_probe_0",
                    "aa_exact",
                    "aa_exact_reversed",
                    "ba_original",
                    "ab_crossed",
                    "ba_crossed",
                    "ab_repeat",
                    "ba_repeat",
                ],
            )
            self.assertEqual(
                calls[1][1], (("baseline", "A0", "A"), ("candidate", "B1", "A"))
            )
            self.assertEqual(
                calls[4][1], (("baseline", "B1", "A"), ("candidate", "A0", "B"))
            )
            self.assertEqual(
                calls[6][1], (("candidate", "B1", "B"), ("baseline", "A0", "A"))
            )
            self.assertTrue(
                json.loads((root / "results" / "trigger.json").read_text())[
                    "bias_detected"
                ]
            )

    async def test_observed_worker_records_start_warmup_and_sample_order(self):
        class FakeWorker:
            process = SimpleNamespace(pid=123)

            async def request(self, **message):
                if message["operation"] == "prepare":
                    return {"warmup": {"seconds": 0.2}}
                return {"seconds": 0.1, "start_row": 10}

            async def close(self):
                return None

        events = []
        with patch(
            "scripts.benchmark_suite.side_bias_diagnostic.Worker.start",
            return_value=FakeWorker(),
        ):
            worker = await ObservedWorker.start(
                Path("/physical/A0/site"), Path("/trial/round-0/baseline"), events
            )
            await worker.request(operation="prepare", case={"id": "case"})
            await worker.request(operation="sample")
            await worker.close()

        self.assertEqual(
            [event["event"] for event in events],
            ["start", "prepare", "sample", "close"],
        )
        self.assertEqual(events[0]["site"], "/physical/A0/site")
        self.assertEqual(events[0]["side"], "baseline")
        self.assertEqual(events[0]["pid"], 123)
        self.assertEqual(events[1]["warmup_seconds"], 0.2)
        self.assertEqual(events[2]["sample_seconds"], 0.1)


if __name__ == "__main__":
    unittest.main()
