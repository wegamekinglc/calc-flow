"""Reject incomplete or mismatched focused SUM measurement evidence."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from scripts.benchmark_suite import sum_state
from scripts.benchmark_suite.sum_state import (
    THREAD_ENV,
    compare_builds,
    compare_oracles,
    finalize_acquisition,
    measurement_scope,
    validate_sample,
    validated_argv,
)
from scripts.benchmark_suite.sum_state_profile import clock_alignment
from scripts.benchmark_suite.sum_state_publication import (
    PublicationContext,
    publish,
    validate_inventory,
)


class SumStateEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.case = {"mode": "stream", "partitions": 1, "cpus": [0]}
        self.fixture = {
            "input_sha256": "a" * 64,
            "schema_sha256": "b" * 64,
            "input_rows": 12,
            "batch_rows": [10, 2],
            "snapshot_sha256": ["c" * 64, "d" * 64],
            "snapshot_schema_sha256": "e" * 64,
            "snapshot_rows": [10, 10],
        }
        self.row = {
            "process_observations": {
                "observed_cpu_affinity": [0],
                "observed_thread_environment": {name: "1" for name in THREAD_ENV},
            },
            "schema": "calc-flow.sum-state.v1",
            "mode": "stream",
            "partitions": 1,
            "seconds": 0.01,
            "process_cpu_ns": 9_000_000,
            "start_ns": 100,
            "end_ns": 10_000_100,
            "validated_all_snapshots": True,
            "input_sha256": "a" * 64,
            "input_schema_sha256": "b" * 64,
            "input_rows": 12,
            "batch_rows": [10, 2],
            "snapshot_rows": [10, 10],
            "snapshot_sha256": ["c" * 64, "d" * 64],
            "snapshot_schema_sha256": ["e" * 64, "e" * 64],
            "diagnostics": {
                "state": "completed",
                "terminal_cause": "natural_end",
                "task_count": 0,
                "task_errors": 0,
                "metrics_overflowed": False,
                "checkpoint": {"current_epoch": None, "last_completed_epoch": None},
                "sources": {"s": {"errors": 0, "ended": True}},
                "operators": {"o": {"errors": 0, "ended": True}},
                "sinks": {"s": {"errors": 0, "ended": True}},
            },
        }

    def test_accepts_complete_per_snapshot_evidence(self):
        validate_sample(self.row, self.case, self.fixture)
        compare_oracles(self.row, deepcopy(self.row))

    def test_rejects_corrupt_intermediate_snapshot_with_correct_final(self):
        self.row["snapshot_sha256"][0] = "f" * 64
        with self.assertRaisesRegex(ValueError, "snapshot_sha256"):
            validate_sample(self.row, self.case, self.fixture)

    def test_rejects_different_partition_policy(self):
        self.row["partitions"] = 8
        with self.assertRaisesRegex(ValueError, "partitions"):
            validate_sample(self.row, self.case, self.fixture)

    def test_rejects_unsettled_job(self):
        self.row["diagnostics"]["task_count"] = 1
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            validate_sample(self.row, self.case, self.fixture)

    def test_rejects_changed_intermediate_digest_between_versions(self):
        other = deepcopy(self.row)
        other["snapshot_sha256"][0] = "f" * 64
        with self.assertRaisesRegex(ValueError, "snapshot_sha256"):
            compare_oracles(self.row, other)

    def test_rejects_actual_affinity_mismatch(self):
        self.row["process_observations"]["observed_cpu_affinity"] = [1]
        with self.assertRaisesRegex(ValueError, "affinity"):
            validate_sample(self.row, self.case, self.fixture)

    def test_rejects_nonfinite_timing(self):
        self.row["seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "timing"):
            validate_sample(self.row, self.case, self.fixture)


class InvocationArgvTests(unittest.TestCase):
    def test_rejects_shell_command_text(self):
        with self.assertRaisesRegex(ValueError, "argv"):
            validated_argv("echo unsafe; text")

    def test_rejects_embedded_nul_argument(self):
        with self.assertRaisesRegex(ValueError, "argv"):
            validated_argv(["python", "bad\0argument"])

    def test_resolves_fixed_executable_without_interpreting_arguments(self):
        result = validated_argv([sys.executable, "literal;$(unchanged)"])
        self.assertTrue(Path(result[0]).is_absolute())
        self.assertEqual(result[1], "literal;$(unchanged)")


class InvocationLifecycleTests(unittest.TestCase):
    def test_timeout_terminates_then_kills_and_reaps_process_group(self):
        process = Mock(pid=123, returncode=None)
        process.poll.return_value = None
        process.wait.side_effect = [sum_state.subprocess.TimeoutExpired("worker", 2), 0]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sum_state.subprocess, "Popen", return_value=process) as launch,
            patch.object(
                sum_state,
                "poll_invocation",
                side_effect=TimeoutError("inclusive acquisition deadline"),
            ),
            patch.object(sum_state.os, "killpg", create=True) as kill,
            patch.object(sum_state.signal, "SIGKILL", 9, create=True),
        ):
            env = {name: "1" for name in THREAD_ENV}
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                sum_state.invoke(
                    [sys.executable, "literal;argument"],
                    env,
                    Path(directory) / "invocation",
                    10,
                )
            self.assertEqual(
                kill.call_args_list, [call(123, sum_state.signal.SIGTERM), call(123, 9)]
            )
            self.assertEqual(process.wait.call_count, 2)
            self.assertFalse(launch.call_args.kwargs["shell"])
            self.assertTrue(Path(launch.call_args.args[0][0]).is_absolute())
            self.assertEqual(launch.call_args.args[0][1], "literal;argument")

    def test_spawn_failure_does_not_attempt_to_kill_absent_process(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                sum_state.subprocess, "Popen", side_effect=OSError("spawn failed")
            ),
            patch.object(sum_state.os, "killpg", create=True) as kill,
        ):
            with self.assertRaisesRegex(OSError, "spawn failed"):
                sum_state.invoke(
                    [sys.executable],
                    {name: "1" for name in THREAD_ENV},
                    Path(directory) / "invocation",
                    10,
                )
            kill.assert_not_called()


class MeasurementScopeTests(unittest.TestCase):
    def test_history_borrow_preserves_stream_case_samples_and_residual_budget(self):
        scope = measurement_scope(True, 128.894776064, 471.105223936)
        self.assertEqual(len(scope["cases"]), 3)
        self.assertTrue(all(case["mode"] == "stream" for case in scope["cases"]))
        self.assertEqual(scope["profile_sides"], ("candidate",))
        self.assertEqual(scope["profile_min_remaining_seconds"], 25)

    def test_rejects_resetting_cumulative_measurement_budget(self):
        with self.assertRaisesRegex(ValueError, "cumulative"):
            measurement_scope(True, 600, 471.105223936)

    def test_rejects_history_borrow_without_prior_accounting(self):
        with self.assertRaisesRegex(ValueError, "prior"):
            measurement_scope(True, 600, 0)

    def test_rejects_nonfinite_budget(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            measurement_scope(True, float("nan"), 471.105223936)


class BuildCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.base = {
            "cargo_lock_sha256": "lock",
            "harness_sha256": "harness",
            "rustc": "compiler",
            "features": ["benchmark-internals"],
            "source_sha256": {
                "crates/calc-flow/examples/support/mod.rs": "shared",
                "crates/calc-flow/examples/support/stock_account_data.rs": "fixture",
                "vendor/arrow-ipc/src/lib.rs": "vendor",
                "crates/calc-flow/src/operator/sql/incremental.rs": "old",
            },
        }

    def test_accepts_product_change_with_shared_harness_dependencies(self):
        candidate = deepcopy(self.base)
        candidate["source_sha256"][
            "crates/calc-flow/src/operator/sql/incremental.rs"
        ] = "new"
        compare_builds({"baseline": self.base, "candidate": candidate})

    def test_rejects_different_shared_fixture_module(self):
        candidate = deepcopy(self.base)
        candidate["source_sha256"][
            "crates/calc-flow/examples/support/stock_account_data.rs"
        ] = "other"
        with self.assertRaisesRegex(ValueError, "shared"):
            compare_builds({"baseline": self.base, "candidate": candidate})

    def test_rejects_changed_vendored_dependency(self):
        candidate = deepcopy(self.base)
        candidate["source_sha256"]["vendor/arrow-ipc/src/lib.rs"] = "other"
        with self.assertRaisesRegex(ValueError, "vendor"):
            compare_builds({"baseline": self.base, "candidate": candidate})

    def test_rejects_different_release_optimization_flags(self):
        candidate = deepcopy(self.base)
        candidate["build_environment"] = {"RUSTFLAGS": "-C target-cpu=native"}
        with self.assertRaisesRegex(ValueError, "RUSTFLAGS"):
            compare_builds({"baseline": self.base, "candidate": candidate})


class FinalSealDeadlineTests(unittest.TestCase):
    def test_late_final_seal_demotes_both_durable_source_records(self):
        clock = [100.0]

        def verify(_):
            clock[0] = 130.0

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output=Path(directory), prior_measurement_seconds=471.105223936
            )
            acquisition = {
                "status": "completed",
                "failure": None,
                "cases": [],
                "profiles": [],
            }
            with (
                patch(
                    "scripts.benchmark_suite.sum_state.verify_build", side_effect=verify
                ),
                patch(
                    "scripts.benchmark_suite.sum_state.time.monotonic",
                    side_effect=lambda: clock[0],
                ),
            ):
                finalize_acquisition(args, acquisition, {"candidate": {}}, 0, 128)
            for name in ("acquisition.json", "summary.json"):
                record = json.loads((args.output / name).read_text())
                self.assertEqual(record["status"], "stopped")
                self.assertIn("deadline", record["failure"])
                self.assertGreater(record["cumulative_measurement_seconds"], 600)


class PublicationInventoryTests(unittest.TestCase):
    def test_rejects_completed_status_with_missing_formal_observation(self):
        case = {
            "id": "a",
            "observations": [{}] * 39,
            "baseline": [[1.0] * 10] * 2,
            "candidate": [[1.0] * 10] * 2,
            "statistics": {"samples": 20},
        }
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_inventory({"status": "completed", "cases": [case]}, ["a"])

    def test_retains_partial_inventory_without_claiming_completion(self):
        validate_inventory({"status": "stopped", "cases": []}, ["a"])

    def test_publishes_retained_rows_without_reexecuting_workload(self):
        self._exercise_publication()

    def test_late_publication_persists_stopped_authoritative_records(self):
        self._exercise_publication(late=True)

    def _saved_publication(self):
        previous = Path(__file__).resolve().parents[1] / "benchmarks/sum-state/20261104"
        saved = json.loads(
            gzip.decompress((previous / "complete-evidence.json.gz").read_bytes())
        )
        acquisition = deepcopy(saved["acquisition"])
        acquisition["started_monotonic"] = 0
        acquisition["cases"] = acquisition["cases"][:3]
        acquisition["profiles"] = [
            row for row in acquisition["profiles"] if row["side"] == "candidate"
        ]
        plan = deepcopy(saved["plan"])
        plan["cases"] = plan["cases"][:3]
        plan["unmeasured_final_source"] = ["stock-batch-8cpu"]
        return previous, saved, acquisition, plan

    def _prepare_publication(self, output, previous, saved, acquisition):
        output.mkdir()
        (output / "fixture-manifest.json").write_text(json.dumps(saved["fixtures"]))
        profiles = json.loads((previous / "profile-summary.json").read_text())
        (output / "profile-summary.json").write_text(
            json.dumps([row for row in profiles if row["side"] == "candidate"])
        )
        profile = output / "profiles/candidate"
        profile.mkdir(parents=True)
        (profile / "stacks.json.gz").write_bytes(
            (previous / "candidate-stacks.json.gz").read_bytes()
        )
        (profile / "stderr.log").write_text("")
        self._copy_sample_rows(previous, output)
        (output / "acquisition.json").write_text(json.dumps(acquisition))
        (output / "summary.json").write_text(
            json.dumps({"status": acquisition["status"], "failure": None})
        )

    def _copy_sample_rows(self, previous, output):
        with (previous / "samples.csv").open() as source:
            reader = csv.DictReader(source)
            rows = [row for row in reader if row["case"] != "stock-batch-8cpu"]
            with (output / "samples.csv").open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def _invoke_publication(self, context, late):
        ticks = iter((100, 100, 100, 129))

        def clock():
            return next(ticks, 130) if late else 100

        with (
            patch(
                "scripts.benchmark_suite.sum_state_publication.time.monotonic",
                side_effect=clock,
            ),
            redirect_stdout(io.StringIO()),
        ):
            if late:
                with self.assertRaisesRegex(TimeoutError, "deadline"):
                    publish(context)
            else:
                publish(context)

    def _assert_stopped_publication(self, args, manifest):
        paths = (
            args.output / "acquisition.json",
            args.output / "summary.json",
            args.output / "phase-stopped.json",
            args.publish_directory / "summary.json",
            args.publish_directory / "evidence-manifest.json",
        )
        for path in paths:
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "stopped", str(path))
            self.assertIn("deadline", record["failure"])
        raw = json.loads(
            gzip.decompress(
                (args.publish_directory / "complete-evidence.json.gz").read_bytes()
            )
        )
        self.assertEqual(raw["acquisition"]["status"], "stopped")
        self.assertGreater(manifest["cumulative_measurement_seconds"], 600)
        for name, record in manifest["files"].items():
            contents = (args.publish_directory / name).read_bytes()
            self.assertEqual(hashlib.sha256(contents).hexdigest(), record["sha256"])

    def _assert_publication(self, args, late):
        summary = json.loads((args.publish_directory / "summary.json").read_text())
        self.assertEqual(summary["formal_observations"], 120)
        self.assertEqual(summary["unmeasured_final_source"], ["stock-batch-8cpu"])
        self.assertEqual(len(summary["profiles"]), 1)
        manifest = json.loads(
            (args.publish_directory / "evidence-manifest.json").read_text()
        )
        if late:
            self._assert_stopped_publication(args, manifest)
        else:
            self.assertAlmostEqual(
                manifest["cumulative_measurement_seconds"], 571.105223936
            )

    def _exercise_publication(self, *, late=False):
        previous, saved, acquisition, plan = self._saved_publication()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                output=root / "collection",
                publish_directory=root / "published",
                prior_measurement_seconds=471.105223936,
                budget_seconds=128.894776064,
            )
            self._prepare_publication(args.output, previous, saved, acquisition)
            context = PublicationContext(
                args,
                acquisition,
                plan,
                saved["environment"],
                saved["plan"]["builds"],
                128,
            )
            self._invoke_publication(context, late)
            self._assert_publication(args, late)


class ProfileClockTests(unittest.TestCase):
    def test_matches_every_thread_sample_to_one_clock_offset(self):
        raw = " 11/12 100.001000000: \n 11/12 100.002000000: \n"
        profile = {"threads": [{"pid": 11, "tid": 12, "samples": {"time": [0.0, 1.0]}}]}
        result = clock_alignment(raw, profile, 11)
        self.assertEqual(result["matched_sample_count"], 2)
        self.assertEqual(result["offset_ms"], 100001.0)

    def test_rejects_changed_intermediate_profile_timestamp(self):
        raw = " 11/12 100.001000000: \n 11/12 100.002000000: \n"
        profile = {"threads": [{"pid": 11, "tid": 12, "samples": {"time": [0.0, 1.2]}}]}
        with self.assertRaisesRegex(ValueError, "alignment"):
            clock_alignment(raw, profile, 11)


if __name__ == "__main__":
    unittest.main()
