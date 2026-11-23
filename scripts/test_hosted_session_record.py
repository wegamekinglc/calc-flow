from __future__ import annotations

import unittest


class HostedSessionRecord(unittest.TestCase):
    def test_control_keeps_the_same_warmed_single_row(self):
        from scripts.hosted_session_record import control

        sample = {
            "start_row": 4096,
            "seconds": 0.00008,
            "phases_seconds": {"native": 0.00004},
            "correctness": {"passed": True, "float64_digest": "tick"},
        }
        record = {
            "sample": sample,
            "warm_output_digest": "warm",
            "cpu": [1],
            "plan": {"fingerprint": "plan"},
            "phase_probe": {"installed": True},
            "finish": {"output": 0},
        }
        self.assertEqual(control(record, 1), ("warm", "tick", {"fingerprint": "plan"}))
        for changed in (
            {"sample": {**sample, "start_row": 3}},
            {"cpu": [2]},
            {"phase_probe": {"installed": False}},
            {"finish": {"output": 1}},
            {"sample": {**sample, "correctness": {"passed": False}}},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                control({**record, **changed}, 1)

    def test_forty_four_fresh_processes_keep_a_single_worker(self):
        import asyncio
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from scripts import hosted_session_record as bench

        async def child(_argv, *, cwd, env, log):
            self.assertEqual(env["TOKIO_WORKER_THREADS"], "1")
            self.assertIn(str(cwd), env["PYTHONPATH"])
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps({"raw": True}))

        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            with (
                patch.object(bench, "OUT", path),
                patch.object(bench, "command", child),
                patch.object(bench, "control", return_value=("warm", "tick", "plan")),
            ):
                rows = asyncio.run(
                    bench.collect({arm: path / arm for arm in bench.SIDES}, 0)
                )
            self.assertEqual(len(rows), 44)
            self.assertEqual(sum(r["pair"] == -1 for r in rows), 4)
