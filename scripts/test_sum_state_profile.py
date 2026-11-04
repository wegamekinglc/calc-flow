"""Preserve exact profile windows, attribution and executable validation."""

from __future__ import annotations

import gzip
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.benchmark_suite.sum_state_profile import command, decode, frame_names


def small_profile():
    return {
        "libs": [],
        "threads": [
            {
                "pid": 11,
                "tid": 12,
                "name": "worker",
                "samples": {
                    "length": 3,
                    "time": [0.0, 1.0, 2.0],
                    "stack": [0, 1, 1],
                    "weight": None,
                },
                "frameTable": {"length": 2, "func": [0, 1], "address": [None, None]},
                "funcTable": {"resource": [-1, -1], "name": [0, 1]},
                "resourceTable": {"lib": []},
                "stringArray": ["calc_flow::finish_transaction", "scalar"],
                "stackTable": {"frame": [0, 1], "prefix": [None, 0]},
            }
        ],
    }


class ProfileEvidenceTests(unittest.TestCase):
    def test_rejects_nonexecutable_before_starting_a_process(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory, "tool")
            binary.write_text("not executable")
            with (
                patch("subprocess.check_output", return_value="unsafe"),
                self.assertRaises(RuntimeError),
            ):
                command([str(binary)], time.monotonic() + 20, {})

    def test_retains_named_frames_without_native_resources(self):
        profile = small_profile()
        self.assertEqual(
            frame_names(profile, profile["threads"][0], {}),
            ["calc_flow::finish_transaction", "scalar"],
        )

    def test_decode_counts_all_and_only_processing_window_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_root = root / "profiles/candidate"
            profile_root.mkdir(parents=True)
            args = SimpleNamespace(
                output=root,
                perf_library=None,
                samply=Path("/mock/samply"),
                perf=Path("/mock/perf"),
            )
            item = {
                "side": "candidate",
                "path": "profiles/candidate",
                "rows": [{"pid": 11}],
                "perf_data_sha256": "b" * 64,
                "windows": [
                    {"start_ns": 100_000_000_000, "end_ns": 100_000_500_000},
                    {"start_ns": 100_000_600_000, "end_ns": 100_002_500_000},
                ],
            }

            def fake_command(argv, _deadline, _env):
                if argv[1] == "import":
                    (profile_root / "profile.json.gz").write_bytes(
                        gzip.compress(json.dumps(small_profile()).encode())
                    )
                    return ""
                return (
                    " 11/12 100.000000000:\n 11/12 100.001000000:\n"
                    " 11/12 100.002000000:\n"
                )

            with patch(
                "scripts.benchmark_suite.sum_state_profile.command",
                side_effect=fake_command,
            ):
                reports = decode(
                    args,
                    {"profiles": [item]},
                    {"candidate": {"binary_sha256": "a" * 64}},
                    time.monotonic() + 20,
                )
            self.assertEqual(reports[0]["samples"], 3)
            self.assertEqual(reports[0]["samples_per_window"], [1, 2])
            self.assertEqual(reports[0]["clock_alignment"]["matched_sample_count"], 3)
            self.assertEqual(reports[0]["disjoint_phases"][0]["samples"], 3)
            self.assertEqual(reports[0]["unresolved_leaf_samples"], 0)
            stacks = json.loads(
                gzip.decompress((profile_root / "stacks.json.gz").read_bytes())
            )
            self.assertEqual(sum(s["samples"] for s in stacks["stacks"]), 3)
