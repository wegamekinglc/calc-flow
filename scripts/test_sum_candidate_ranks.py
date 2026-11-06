"""Preserve the two static SUM contracts while sharing the maintained collector."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

from scripts import test_sum_columnar_keys as controls
from scripts.benchmark_suite import sum_candidate_ranks as ranks
from scripts.benchmark_suite import sum_columnar_keys as shared


def protocol():
    return {
        "slug": ranks.SLUG,
        "baseline_revision": ranks.BASELINE,
        "candidate_revision": "a" * 40,
        "budget_seconds": 600,
        "rounds": 2,
        "pairs_per_round": 10,
        "warmups_per_arm_per_round": 1,
        "cases": list(ranks.CASES),
        "control_fixtures": ranks.CONTROLS,
        "profile_route_anchors": [],
        "tool_source_sha256": shared.helper_seals(ranks.CONTRACT),
    }


class CandidateRankContractTests(unittest.TestCase):
    def test_current_rank_contract_does_not_validate_as_the_old_study(self):
        ranks.validate_protocol(protocol())

    def test_five_cases_keep_low_repetition_without_a_new_route_threshold(self):
        self.assertEqual(len(ranks.CASES), 5)
        self.assertEqual(ranks.CASES[4]["id"], "repeat2-stream-1cpu")
        self.assertEqual(ranks.CASES[4]["fixture"], "edge")
        self.assertEqual(ranks.CONTRACT.phase, "candidate-ranks")
        self.assertEqual(shared.COLUMNAR.phase, "columnar-keys")
        self.assertNotEqual(ranks.CONTRACT.baseline, shared.COLUMNAR.baseline)

    def test_fixed_counts_budget_and_old_baseline_cannot_drift(self):
        for changes in (
            {"budget_seconds": 601},
            {"pairs_per_round": 11},
            {"cases": list(ranks.CASES)[:-1]},
            {"baseline_revision": shared.BASELINE},
            {"candidate_revision": ranks.BASELINE},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ranks.validate_protocol({**protocol(), **changes})

    def test_new_contract_requires_all_thirteen_helpers_and_rejects_legacy_hashes(self):
        self.assertEqual(len(ranks.CONTRACT.tools), 13)
        shared.validate_tool_seals(protocol(), ranks.CONTRACT)
        with self.assertRaisesRegex(ValueError, "inventory"):
            shared.validate_tool_seals(
                {**protocol(), "tool_source_sha256": shared.helper_seals()},
                ranks.CONTRACT,
            )
        legacy = (
            shared.maintained.ROOT
            / "benchmarks/sum-state/sum-columnar-keys-20261105.protocol.json"
        )
        with self.assertRaisesRegex(ValueError, "helper"):
            shared.validate_tool_seals(json.loads(legacy.read_text()))

    def test_rank_acceptance_uses_both_dense_targets_and_every_control(self):
        cases = [
            {**case, "statistics": {"verdict": "inconclusive"}} for case in ranks.CASES
        ]
        self.assertFalse(
            shared.performance_acceptance("completed", cases, ranks.CONTRACT)
        )
        cases[1]["statistics"]["verdict"] = "improved"
        self.assertTrue(
            shared.performance_acceptance("completed", cases, ranks.CONTRACT)
        )
        self.assertFalse(
            shared.performance_acceptance("stopped", cases, ranks.CONTRACT)
        )
        cases[4]["statistics"]["verdict"] = "regression"
        self.assertFalse(
            shared.performance_acceptance("completed", cases, ranks.CONTRACT)
        )

    def test_thin_entry_does_not_mutate_old_contract(self):
        with patch.object(shared, "main") as main:
            ranks.main()
        main.assert_called_once_with(contract=ranks.CONTRACT)
        self.assertEqual(shared.CASES[4]["id"], "edge-stream-1cpu")
        self.assertEqual(shared.COLUMNAR.slug, shared.SLUG)
        self.assertEqual(shared.COLUMNAR.baseline, shared.BASELINE)

    def test_current_product_identity_is_not_replaced_by_an_old_receipt(self):
        builds = controls.builds()
        with self.assertRaisesRegex(ValueError, "fresh public"):
            shared.validate_products(builds, protocol(), ranks.CONTRACT)
        updated = {
            **builds,
            "baseline": {**builds["baseline"], "revision": ranks.BASELINE},
        }
        shared.validate_products(updated, protocol(), ranks.CONTRACT)


def rank_publication(directory):
    context, acquisition, summary, raw = controls.publication_input(
        directory, controls.profiles()
    )
    cases = [
        {**old, **case} for old, case in zip(summary["cases"], ranks.CASES, strict=True)
    ]
    acquisition["cases"] = cases
    summary["cases"] = cases
    context = shared.StudyFollowup(
        context.collection,
        protocol(),
        "protocol",
        "candidate-ranks",
        None,
        ranks.CONTRACT,
    )
    context = replace(context, collection=replace(context.collection, deadline=10))
    return context, acquisition, summary, raw


class CandidateRankPublicationTests(unittest.TestCase):
    def test_late_rank_publication_demotes_all_durable_statuses_without_columnar_claims(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            context, acquisition, summary, raw = rank_publication(directory)
            ticks = iter((1, 1, 11))
            with (
                patch.object(
                    shared.followup, "publication_data", return_value=(summary, raw)
                ),
                patch.object(
                    shared.publication, "rendered_rows", return_value=([], [], [])
                ),
                patch.object(
                    shared.time, "monotonic", side_effect=lambda: next(ticks, 12)
                ),
                self.assertRaisesRegex(TimeoutError, "publication deadline"),
            ):
                shared.publish(context, acquisition, {}, {})
            destination = context.collection.args.publish_directory
            receipt = json.loads((destination / "evidence-manifest.json").read_text())
            self.assertEqual(receipt["status"], "stopped")
            self.assertEqual(receipt["slug"], ranks.SLUG)
            self.assertFalse(receipt["performance_acceptance"])
            self.assertFalse((destination / "profile-route.json").exists())
            published = json.loads((destination / "summary.json").read_text())
            self.assertNotIn("columnar_route_evidence", published)
            self.assertEqual(published["status"], "stopped")
            report = (destination / "README.md").read_text()
            self.assertIn("Performance acceptance: **False**", report)
            self.assertIn("candidate/rank/slot", report)
            self.assertNotIn("first eligible", report)
            self.assertNotIn("selector remains", report)

    def test_two_invalid_profiles_are_never_accepted_for_the_rank_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            context, acquisition, summary, raw = rank_publication(directory)
            summary["profiles"][1]["binary_sha256"] = "wrong"
            with (
                patch.object(
                    shared.followup, "publication_data", return_value=(summary, raw)
                ),
                patch.object(
                    shared.publication, "rendered_rows", return_value=([], [], [])
                ),
                patch.object(shared.time, "monotonic", return_value=1),
                patch("builtins.print"),
            ):
                shared.publish(context, acquisition, {}, {})
            record = json.loads(
                (
                    context.collection.args.publish_directory / "evidence-manifest.json"
                ).read_text()
            )
            self.assertEqual(record["status"], "stopped")
            self.assertEqual(record["current_profiles"], 0)
            self.assertEqual(record["profile_records"], 2)
            self.assertFalse(record["performance_acceptance"])

    def test_late_helper_drift_revokes_the_rank_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            context, acquisition, summary, raw = rank_publication(directory)
            with (
                patch.object(
                    shared.followup, "publication_data", return_value=(summary, raw)
                ),
                patch.object(
                    shared.publication, "rendered_rows", return_value=([], [], [])
                ),
                patch.object(shared.time, "monotonic", return_value=1),
                patch.object(
                    shared,
                    "validate_tool_seals",
                    side_effect=[None, ValueError("helper drift")],
                ),
                patch("builtins.print"),
            ):
                shared.publish(context, acquisition, {}, {})
            destination = context.collection.args.publish_directory
            for name in ("summary.json", "evidence-manifest.json"):
                record = json.loads((destination / name).read_text())
                self.assertEqual(record["status"], "stopped")
                self.assertIn("helper drift", record["failure"])
            self.assertFalse(
                json.loads((destination / "evidence-manifest.json").read_text())[
                    "performance_acceptance"
                ]
            )

    def test_cancellation_keeps_the_frozen_rank_contract_during_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            context, _, _, _ = rank_publication(directory)
            args = context.collection.args
            with (
                patch.object(shared, "prepare_context", return_value=context),
                patch.object(shared, "plan_and_machine", return_value=({}, {})),
                patch.object(shared, "collect", side_effect=asyncio.CancelledError()),
                patch.object(shared.maintained, "finalize_acquisition") as finalize,
                patch.object(shared, "publish") as publish,
                self.assertRaises(asyncio.CancelledError),
            ):
                shared.run(args, ranks.CONTRACT)
            finalize.assert_called_once()
            self.assertIs(publish.call_args.args[0].contract, ranks.CONTRACT)
            self.assertEqual(publish.call_args.args[1]["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
