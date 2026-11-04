"""Keep the new SUM work item's stage identities and budget fail-closed."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.benchmark_suite import sum_state
from scripts.benchmark_suite.sum_followup import phase_budget


class FollowupBudgetTests(unittest.TestCase):
    def test_current_stage_is_capped_at120_seconds(self):
        self.assertEqual(phase_budget("current", None), (120, 0))

    def test_candidate_uses_only_actual_remaining_workitem_budget(self):
        prior = {
            "status": "completed",
            "phase": "current",
            "cumulative_measurement_seconds": 35.5,
        }
        self.assertEqual(phase_budget("candidate", prior), (564.5, 35.5))

    def test_rejects_previous_workitem_or_incomplete_receipt(self):
        for prior in (
            {"status": "stopped", "phase": "current"},
            {"status": "completed", "phase": "history-borrow"},
        ):
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                phase_budget("candidate", prior)


class NativeConflictTests(unittest.TestCase):
    def _worker(self, name, executable, pid=999999):
        return SimpleNamespace(
            pid=pid,
            info={
                "name": name,
                "exe": executable,
                "cmdline": [executable],
                "status": "running",
            },
        )

    def _guard(self, process):
        builds = {
            "baseline": {
                "build_environment": {"CARGO_TARGET_DIR": "/repo/target"},
                "binary": "/repo/sealed",
                "binary_sha256": "same",
            }
        }
        with (
            patch.object(sum_state.psutil, "process_iter", return_value=[process]),
            patch.object(sum_state, "sha", return_value="same"),
        ):
            sum_state.guard(builds)

    def test_rejects_direct_rust_libtest_without_cargo_parent(self):
        with self.assertRaisesRegex(RuntimeError, "competing"):
            self._guard(
                self._worker("calc_flow-abcd", "/repo/target/debug/deps/calc_flow-abcd")
            )

    def test_rejects_direct_sealed_native_benchmark(self):
        with self.assertRaisesRegex(RuntimeError, "competing"):
            self._guard(
                self._worker(
                    "grouped_sum_sta",
                    "/repo/target/build/current/grouped_sum_state_benchmark",
                )
            )

    def test_excludes_collectors_own_pid(self):
        self._guard(
            self._worker(
                "python", "/repo/target/venv/bin/python", sum_state.os.getpid()
            )
        )


class FollowupIdentityTests(unittest.TestCase):
    def _context(self, phase="current", candidate_revision=None, prior=None):
        from scripts.benchmark_suite import sum_followup as followup

        baseline = followup.HISTORICAL if phase == "current" else followup.CURRENT
        revision = candidate_revision or (
            followup.CURRENT if phase == "current" else "new-candidate"
        )
        builds = {
            "baseline": {
                "revision": baseline,
                "harness_sha256": followup.COMMON_HARNESS,
                "binary_sha256": "merged",
            },
            "candidate": {
                "revision": revision,
                "harness_sha256": followup.COMMON_HARNESS,
                "binary_sha256": "candidate",
            },
        }
        collection = sum_state.CollectionContext(None, builds, {}, 0, 120)
        return followup.Followup(collection, {}, "protocol", phase, prior)

    def test_rejects_old841_product_for_current_profile(self):
        from scripts.benchmark_suite.sum_followup import validate_products

        with self.assertRaisesRegex(ValueError, "newly built merged"):
            validate_products(self._context(candidate_revision="84186554"))

    def test_candidate_baseline_must_be_same_binary_as_p0_current(self):
        from scripts.benchmark_suite.sum_followup import validate_products

        with self.assertRaisesRegex(ValueError, "sealed P0 product"):
            validate_products(
                self._context(
                    phase="candidate", prior={"current_binary_sha256": "other"}
                )
            )

    def test_rejects_noop_current_product_as_new_candidate(self):
        from scripts.benchmark_suite.sum_followup import CURRENT, validate_products

        with self.assertRaisesRegex(ValueError, "new candidate"):
            validate_products(
                self._context(phase="candidate", candidate_revision=CURRENT)
            )

    def test_missing_current_profile_analysis_blocks_candidate_stage(self):
        import json
        import tempfile
        from pathlib import Path

        from scripts.benchmark_suite.sum_followup import prior_receipt

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.json"
            path.write_text(
                json.dumps(
                    {
                        "protocol_sha256": "protocol",
                        "current_profile_analysis_complete": False,
                    }
                )
            )
            args = SimpleNamespace(phase="candidate", prior_receipt=path)
            with self.assertRaisesRegex(ValueError, "profile and analysis"):
                prior_receipt(args, "protocol")


class FollowupOrderingTests(unittest.TestCase):
    def _context(self):
        collection = SimpleNamespace(scope={"cases": [{"id": "finite"}]}, deadline=120)
        return SimpleNamespace(collection=collection, phase="current")

    def test_invalid_current_profile_stops_before_finite_case(self):
        from scripts.benchmark_suite import sum_followup as followup

        with (
            patch.object(followup.maintained, "prepare_fixture"),
            patch.object(followup.maintained, "collect_profiles") as profile,
            patch.object(followup, "accepted_profiles", return_value=[]),
            patch.object(followup.maintained, "collect_case") as finite,
        ):
            with self.assertRaisesRegex(ValueError, "profile is missing or invalid"):
                followup.collect(self._context(), {"status": "running"})
            profile.assert_called_once()
            finite.assert_not_called()

    def test_current_profile_and_decode_precede_finite_comparison(self):
        from scripts.benchmark_suite import sum_followup as followup

        order = []
        with (
            patch.object(followup.maintained, "prepare_fixture"),
            patch.object(
                followup.maintained,
                "collect_profiles",
                side_effect=lambda *_: order.append("profile+decode"),
            ),
            patch.object(
                followup, "accepted_profiles", return_value=[{"side": "candidate"}]
            ),
            patch.object(
                followup.maintained,
                "collect_case",
                side_effect=lambda *_: order.append("finite"),
            ),
            patch.object(followup.time, "monotonic", return_value=1),
        ):
            acquisition = {"status": "running"}
            followup.collect(self._context(), acquisition)
            self.assertEqual(order, ["profile+decode", "finite"])
            self.assertEqual(acquisition["status"], "completed")


class FollowupDeadlineTests(unittest.TestCase):
    def test_late_publication_demotes_source_raw_summary_and_receipts(self):
        import gzip
        import json
        import tempfile
        from pathlib import Path

        from scripts.benchmark_suite import sum_followup as followup

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, destination = root / "output", root / "publish"
            output.mkdir()
            destination.mkdir()
            args = SimpleNamespace(
                output=output,
                publish_directory=destination,
                prior_measurement_seconds=0,
            )
            collection = sum_state.CollectionContext(
                args, {"candidate": {"binary_sha256": "current"}}, {}, 0, 10
            )
            context = followup.Followup(collection, {}, "protocol", "current", None)
            acquisition = {
                "status": "completed",
                "failure": None,
                "cases": [],
                "profiles": [],
                "started_monotonic": 0,
            }
            summary = {
                "phase": "current",
                "status": "completed",
                "failure": None,
                "cases": [],
                "profiles": [],
            }
            raw = {"acquisition": acquisition}
            ticks = iter((1, 1, 11))
            with (
                patch.object(followup, "publication_data", return_value=(summary, raw)),
                patch.object(
                    followup.time, "monotonic", side_effect=lambda: next(ticks, 12)
                ),
                self.assertRaisesRegex(TimeoutError, "publication deadline"),
            ):
                followup.publish(context, acquisition, {}, {})
            for path in (
                output / "acquisition.json",
                output / "summary.json",
                output / "phase-stopped.json",
                destination / "summary.json",
                destination / "evidence-manifest.json",
            ):
                record = json.loads(path.read_text())
                self.assertEqual(record["status"], "stopped")
                self.assertIn("deadline", record["failure"])
            raw = json.loads(
                gzip.decompress(
                    (destination / "complete-evidence.json.gz").read_bytes()
                )
            )
            self.assertEqual(raw["acquisition"]["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
