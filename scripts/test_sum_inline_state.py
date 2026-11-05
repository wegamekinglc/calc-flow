"""Preserve the current baseline, fixed 1T/4T contract and complete evidence."""

from __future__ import annotations

import asyncio
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.benchmark_suite import sum_inline_state as inline
from scripts.benchmark_suite import sum_state


def protocol():
    return {
        "slug": inline.SLUG,
        "baseline_revision": inline.BASELINE,
        "candidate_revision": "a" * 40,
        "budget_seconds": 600,
        "rounds": 2,
        "pairs_per_round": 10,
        "warmups_per_arm_per_round": 1,
        "cases": list(inline.CASES),
        "tool_source_sha256": inline.helper_seals(),
    }


def builds():
    return {
        side: {
            "revision": revision,
            "binary_sha256": side,
            "harness_sha256": inline.COMMON_HARNESS,
            "product_recompiled": True,
            "example_recompiled": True,
        }
        for side, revision in (("baseline", inline.BASELINE), ("candidate", "a" * 40))
    }


def profiles():
    return [
        {
            "side": side,
            "binary_sha256": side,
            "windows": [{}] * 5,
            "samples_per_window": [2] * 5,
            "clock_alignment": {"matched_sample_count": 10, "max_error_ms": 0},
        }
        for side in ("baseline", "candidate")
    ]


def completed_cases():
    return [
        {
            **case,
            "observations": [{}] * 40,
            "baseline": [[0.1] * 10] * 2,
            "candidate": [[0.08] * 10] * 2,
            "statistics": {"verdict": "improved", "samples": 20},
        }
        for case in inline.CASES
    ]


def publication_input(directory, records):
    root = Path(directory)
    output, destination = root / "output", root / "publish"
    output.mkdir()
    destination.mkdir()
    args = SimpleNamespace(
        output=output, publish_directory=destination, prior_measurement_seconds=0
    )
    context = inline.followup.Followup(
        sum_state.CollectionContext(args, builds(), {}, 0, 600),
        protocol(),
        "protocol",
        "inline-state",
        None,
    )
    acquisition = {
        "status": "completed",
        "failure": None,
        "cases": completed_cases(),
        "profiles": [],
        "started_monotonic": 0,
    }
    summary = {
        "status": "completed",
        "failure": None,
        "cases": completed_cases(),
        "profiles": records,
    }
    return (
        context,
        acquisition,
        summary,
        {"acquisition": acquisition, "profiles": records},
    )


class InlineContractTests(unittest.TestCase):
    def test_frozen_python_helper_drift_is_rejected_before_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            frozen = protocol()
            frozen["tool_source_sha256"][inline.TOOL_FILES[0]] = "wrong"
            path.write_text(json.dumps(frozen))
            args = SimpleNamespace(protocol=path)
            collection = sum_state.CollectionContext(args, {}, {}, 0, 600)
            with (
                patch.object(inline, "validate_receipt_seals"),
                patch.object(
                    inline.maintained, "collection_context", return_value=collection
                ),
                patch.object(inline, "validate_products"),
                self.assertRaisesRegex(ValueError, "helper"),
            ):
                inline.prepare_context(args, 0)

    def test_current_multithread_control_uses_four_physical_cores(self):
        self.assertEqual(inline.CASES[1]["id"], "stock-stream-4cpu")
        self.assertEqual(inline.CASES[1]["cpus"], [0, 2, 4, 6])
        self.assertTrue(all(case["partitions"] == 1 for case in inline.CASES))

    def test_one_profile_cannot_satisfy_the_required_two_product_inventory(self):
        with self.assertRaisesRegex(ValueError, "profiles"):
            inline.validate_profiles([{"side": "baseline"}], {})

    def test_protocol_cannot_expand_inventory_budget_or_samples(self):
        inline.validate_protocol(protocol())
        for changes in (
            {"budget_seconds": 601},
            {"pairs_per_round": 11},
            {"cases": list(inline.CASES) + [inline.CASES[0]]},
            {"baseline_revision": "af400"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                inline.validate_protocol({**protocol(), **changes})

    def test_old_af400_and_stale_native_receipts_are_rejected(self):
        inline.validate_products(builds(), protocol())
        for side, changes in (
            ("baseline", {"revision": "af400"}),
            ("candidate", {"product_recompiled": False}),
            ("baseline", {"harness_sha256": "other"}),
        ):
            altered = {**builds(), side: {**builds()[side], **changes}}
            with (
                self.subTest(side=side, changes=changes),
                self.assertRaises(ValueError),
            ):
                inline.validate_products(altered, protocol())

    def test_validated_profiles_bind_both_binaries_and_all_five_clock_windows(self):
        inline.validate_profiles(profiles(), builds())
        for change in (
            {"binary_sha256": "wrong"},
            {"samples_per_window": [2, 2, 0, 2, 2]},
            {"clock_alignment": {"matched_sample_count": 10, "max_error_ms": 1}},
        ):
            records = [profiles()[0], {**profiles()[1], **change}]
            with self.subTest(change=change), self.assertRaises(ValueError):
                inline.validate_profiles(records, builds())

    def test_dense_improvement_with_no_regression_is_required(self):
        cases = [
            {**case, "statistics": {"verdict": "inconclusive"}} for case in inline.CASES
        ]
        cases[2]["statistics"]["verdict"] = "improved"
        self.assertFalse(inline.performance_acceptance("completed", cases))
        cases[1]["statistics"]["verdict"] = "improved"
        self.assertTrue(inline.performance_acceptance("completed", cases))
        self.assertFalse(inline.performance_acceptance("stopped", cases))
        cases[0]["statistics"]["verdict"] = "regression"
        self.assertFalse(inline.performance_acceptance("completed", cases))

    def test_build_receipt_drift_fails_before_a_native_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "build.json"
            path.write_text("{}")
            args = SimpleNamespace(baseline_build=path, candidate_build=path)
            with patch.object(inline.maintained, "invoke") as invoke:
                with self.assertRaisesRegex(ValueError, "receipt"):
                    inline.validate_receipt_seals(
                        args,
                        {
                            "build_receipt_sha256": {
                                "baseline": "wrong",
                                "candidate": "wrong",
                            }
                        },
                    )
                invoke.assert_not_called()


class InlineLifecycleTests(unittest.TestCase):
    def test_two_invalid_profile_records_are_retained_but_never_accepted(self):
        invalid = [profiles()[0], {**profiles()[1], "binary_sha256": "wrong"}]
        with tempfile.TemporaryDirectory() as directory:
            context, acquisition, summary, raw = publication_input(directory, invalid)
            with (
                patch.object(
                    inline.followup, "publication_data", return_value=(summary, raw)
                ),
                patch.object(
                    inline.publication, "rendered_rows", return_value=([], [], [])
                ),
                patch.object(inline.time, "monotonic", return_value=1),
                patch("builtins.print"),
            ):
                inline.publish(context, acquisition, {}, {})
            destination = context.collection.args.publish_directory
            published = json.loads((destination / "summary.json").read_text())
            self.assertEqual(published["status"], "stopped")
            self.assertFalse(published["profile_analysis_complete"])
            self.assertNotEqual(published["profile_disposition"], "accepted")
            self.assertEqual(published["accepted_profile_count"], 0)
            manifest = json.loads((destination / "evidence-manifest.json").read_text())
            self.assertEqual(manifest["current_profiles"], 0)
            self.assertEqual(manifest["profile_records"], 2)
            self.assertIn(
                "Accepted profiles: 0", (destination / "README.md").read_text()
            )
            complete = json.loads(
                gzip.decompress(
                    (destination / "complete-evidence.json.gz").read_bytes()
                )
            )
            self.assertEqual(complete["profiles"], invalid)

    def test_postpublication_helper_drift_revokes_every_completed_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            context, acquisition, summary, raw = publication_input(
                directory, profiles()
            )
            with (
                patch.object(
                    inline.followup, "publication_data", return_value=(summary, raw)
                ),
                patch.object(
                    inline.publication, "rendered_rows", return_value=([], [], [])
                ),
                patch.object(inline.time, "monotonic", return_value=1),
                patch.object(
                    inline,
                    "validate_tool_seals",
                    side_effect=[None, ValueError("measurement helper drift")],
                ),
                patch("builtins.print"),
            ):
                inline.publish(context, acquisition, {}, {})
            args = context.collection.args
            for path in (
                args.output / "acquisition.json",
                args.output / "summary.json",
                args.output / "phase-stopped.json",
                args.publish_directory / "summary.json",
                args.publish_directory / "evidence-manifest.json",
            ):
                record = json.loads(path.read_text())
                self.assertEqual(record["status"], "stopped")
                self.assertIn("helper drift", record["failure"])
            manifest = json.loads(
                (args.publish_directory / "evidence-manifest.json").read_text()
            )
            self.assertFalse(manifest["performance_acceptance"])
            for name, seal in manifest["files"].items():
                self.assertEqual(
                    inline.maintained.sha(args.publish_directory / name), seal["sha256"]
                )

    def test_invalid_profiles_stop_before_any_formal_case(self):
        context = SimpleNamespace(
            collection=SimpleNamespace(
                builds=builds(), scope={"cases": inline.CASES}, deadline=600
            )
        )
        with (
            patch.object(inline.maintained, "prepare_fixture"),
            patch.object(inline.maintained, "collect_profiles"),
            patch.object(inline.followup, "accepted_profiles", return_value=[]),
            patch.object(inline.maintained, "collect_case") as case,
        ):
            with self.assertRaisesRegex(ValueError, "profiles"):
                inline.collect(context, {"profiles": []})
            case.assert_not_called()

    def test_profiles_precede_the_three_fixed_cases(self):
        context = SimpleNamespace(
            collection=SimpleNamespace(
                builds=builds(), scope={"cases": inline.CASES}, deadline=600
            )
        )
        order = []
        with (
            patch.object(inline.maintained, "prepare_fixture"),
            patch.object(
                inline.maintained,
                "collect_profiles",
                side_effect=lambda *_: order.append("profiles+decode"),
            ),
            patch.object(inline.followup, "accepted_profiles", return_value=profiles()),
            patch.object(
                inline.maintained,
                "collect_case",
                side_effect=lambda _, case, __: order.append(case["id"]),
            ),
            patch.object(inline.time, "monotonic", return_value=1),
        ):
            acquisition = {"status": "running"}
            inline.collect(context, acquisition)
            self.assertEqual(
                order, ["profiles+decode", *[case["id"] for case in inline.CASES]]
            )
            self.assertEqual(acquisition["status"], "completed")

    def test_cancellation_still_finalizes_and_publishes_partial_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory))
            collection = sum_state.CollectionContext(args, builds(), {}, 0, 600)
            context = SimpleNamespace(collection=collection, protocol=protocol())
            with (
                patch.object(inline, "prepare_context", return_value=context),
                patch.object(inline, "plan_and_machine", return_value=({}, {})),
                patch.object(inline, "collect", side_effect=asyncio.CancelledError()),
                patch.object(inline.maintained, "finalize_acquisition") as finalize,
                patch.object(inline, "publish") as publish,
                self.assertRaises(asyncio.CancelledError),
            ):
                inline.run(args)
            finalize.assert_called_once()
            self.assertEqual(publish.call_args.args[1]["status"], "stopped")
            self.assertIn("CancelledError", publish.call_args.args[1]["failure"])

    def test_late_publication_rewrites_all_statuses_and_keeps_raw_clock_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, destination = root / "output", root / "publish"
            output.mkdir()
            destination.mkdir()
            stamp = output / "profiles/baseline/sample-timestamps.txt"
            stamp.parent.mkdir(parents=True)
            stamp.write_text("11/12 1.000000000:\n")
            args = SimpleNamespace(
                output=output,
                publish_directory=destination,
                prior_measurement_seconds=0,
            )
            context = inline.followup.Followup(
                sum_state.CollectionContext(args, builds(), {}, 0, 10),
                protocol(),
                "protocol",
                "inline-state",
                None,
            )
            acquisition = {
                "status": "completed",
                "failure": None,
                "cases": completed_cases(),
                "profiles": [{"path": "profiles/baseline"}],
                "started_monotonic": 0,
            }
            summary = {
                "status": "completed",
                "failure": None,
                "cases": completed_cases(),
                "profiles": profiles(),
            }
            raw = {"acquisition": acquisition}
            ticks = iter((1, 1, 11))
            with (
                patch.object(
                    inline.followup, "publication_data", return_value=(summary, raw)
                ),
                patch.object(
                    inline.publication, "rendered_rows", return_value=([], [], [])
                ),
                patch.object(
                    inline.time, "monotonic", side_effect=lambda: next(ticks, 12)
                ),
                self.assertRaisesRegex(TimeoutError, "publication deadline"),
            ):
                inline.publish(context, acquisition, {}, {})
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
            complete = json.loads(
                gzip.decompress(
                    (destination / "complete-evidence.json.gz").read_bytes()
                )
            )
            self.assertEqual(complete["acquisition"]["status"], "stopped")
            self.assertEqual(
                complete["profile_clock_evidence"][
                    "profiles/baseline/sample-timestamps.txt"
                ]["text"],
                stamp.read_text(),
            )
            self.assertIn(
                "Performance acceptance: **False**",
                (destination / "README.md").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
