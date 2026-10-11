from __future__ import annotations

import asyncio
import copy
import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.benchmark_suite.issue406_acceptance import acceptance_inventory
from scripts.benchmark_suite.issue406_fresh import (
    CANDIDATE,
    MAIN,
    PREVIOUS,
    Row5Budget,
    collect_fresh,
    validate_envelope,
    validate_retained_control,
)
from scripts.benchmark_suite.issue406_run import collection_groups


def fresh_plan():
    return {
        "schema": "calc-flow.issue406-execution.v2",
        "budget_seconds": 5400,
        "settlement_seconds": 300,
        "collection_cutoff_seconds": 4800,
        "settlement_deadline_seconds": 5100,
        "startup_allowance_seconds": 5,
        "final_seal_allowance_seconds": 5,
        "row5_arm_seconds": 360,
        "row5_group_seconds": 720,
        "precharged_seconds": 0,
        "cases": [asdict(case) for case in acceptance_inventory()],
        "releases": {
            "baseline": {"product_commit": MAIN},
            "candidate": {"product_commit": CANDIDATE},
        },
        "retained_control": {"product_commit": PREVIOUS},
        "known_main_failure_identity": {
            "product_commit": MAIN,
            "native_sha256": (
                "83c1984005258b702fabbb322d27bb9a0bf7842823d7ddabc4e9660b40997d5f"
            ),
            "wheel_sha256": "b" * 64,
            "build_metadata_sha256": "c" * 64,
            "ordinary_audit_sha256": "d" * 64,
        },
    }


class FreshFakeRunner:
    def __init__(self, *, failure=None, recognized=False):
        self.plan = fresh_plan()
        self.calls = []
        self.finished = []
        self.failure = failure
        self.recognized = recognized
        self.cleanup_tasks = set()

    def observation(self, group, side, round_index, index):
        self.calls.append((group[0].id, side, round_index, index))
        if self.failure == (group[0].row, side):
            raise RuntimeError("controlled workload failure")
        return {
            case.id: {
                "observation_id": f"{group[0].id}/{side}/{round_index}/{index}",
                "raw_observation_sha256": "a" * 64,
                "group_member_ids": [member.id for member in group],
                "product_commit": {
                    "baseline": MAIN,
                    "candidate": CANDIDATE,
                    "previous": PREVIOUS,
                }[side],
            }
            for case in group
        }

    async def prepare(self, group, side, round_index):
        return self.observation(group, side, round_index, None)

    async def sample(self, group, side, round_index, index):
        return self.observation(group, side, round_index, index)

    async def finish_round(self, side=None):
        self.finished.append(side)

    async def known_main_failure(self, group, side, round_index, index, error):
        if (
            self.recognized
            and side == "baseline"
            and round_index == 0
            and index is None
        ):
            return {"category": "controlled-main-only", "settled": True}
        return None


class FreshAcquisitionTests(unittest.IsolatedAsyncioTestCase):
    def test_only_the_reviewed_new_envelope_and_revisions_are_accepted(self):
        validate_envelope(fresh_plan())
        for field, value in (
            ("budget_seconds", 5401),
            ("collection_cutoff_seconds", 4801),
            ("settlement_deadline_seconds", 5101),
            ("row5_arm_seconds", 361),
            ("row5_group_seconds", 721),
            ("precharged_seconds", 4795),
            ("precharged_seconds", float("nan")),
            ("startup_allowance_seconds", 0),
            ("cases", []),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_envelope({**fresh_plan(), field: value})
        for role in ("baseline", "candidate"):
            plan = fresh_plan()
            plan["releases"][role]["product_commit"] = "0" * 40
            with self.subTest(role=role), self.assertRaises(ValueError):
                validate_envelope(plan)

    async def test_fresh_cff_pairs_add_only_44_calls_and_reference_primary_candidate(
        self,
    ):
        with TemporaryDirectory() as raw:
            runner = FreshFakeRunner()
            result = await collect_fresh(runner, Path(raw))
            self.assertEqual(len(runner.calls), 1364)
            self.assertEqual(sum(call[3] is None for call in runner.calls), 124)
            self.assertEqual(len(result["measurements"]), 64)
            control = result["retained_control"]
            self.assertEqual(len(control["measurements"]), 2)
            self.assertEqual(len(control["pairs"]), 44)
            validate_retained_control(result)
            self.assert_retained_calls(runner)
            pairs = [pair for pair in control["pairs"] if pair["index"] is not None]
            self.assertEqual(
                [pair["order"] for pair in pairs[:2]],
                [["previous", "candidate"], ["candidate", "previous"]],
            )
            corrupted = copy.deepcopy(result)
            corrupted["retained_control"]["pairs"][0]["candidate"]["observation_id"] = (
                "wrong"
            )
            with self.assertRaisesRegex(ValueError, "reference"):
                validate_retained_control(corrupted)

    def assert_retained_calls(self, runner):
        selected = [call for call in runner.calls if call[1] == "previous"]
        self.assertEqual(len(selected), 44)
        self.assertTrue(
            all("join_retained" in call[0] and "/off/" in call[0] for call in selected)
        )

    def assert_failed_arms_empty(self, result):
        failed = [
            record
            for record in result["measurements"]
            if record["case"].startswith("row-4/") and record["side"] == "baseline"
        ]
        self.assertEqual(len(failed), 2)
        self.assertTrue(
            all(
                record["rounds"] == [[], []] and not record["warmups"]
                for record in failed
            )
        )

    async def test_exact_main_failure_exhausts_one_arm_without_retry_or_old_samples(
        self,
    ):
        with TemporaryDirectory() as raw:
            runner = FreshFakeRunner(failure=(4, "baseline"), recognized=True)
            result = await collect_fresh(runner, Path(raw))
            self.assertEqual(len(runner.calls), 1364 - 21)
            self.assertEqual(len(result["known_failures"]), 1)
            self.assert_failed_arms_empty(result)
            journal = [
                json.loads(line)
                for line in (Path(raw) / "attempts.jsonl").read_text().splitlines()
            ]
            attempts = [
                item
                for item in journal
                if item.get("cases", [""])[0].startswith("row-4/")
                and item.get("side") == "baseline"
                and item["state"] == "started"
            ]
            self.assertEqual(len(attempts), 1)
            self.assertEqual(result["state"], "collected-with-failures")

    async def test_unrecognized_or_candidate_failure_stops_and_preserves_partial(self):
        for side, recognized in (("baseline", False), ("candidate", True)):
            with self.subTest(side=side), TemporaryDirectory() as raw:
                runner = FreshFakeRunner(failure=(4, side), recognized=recognized)
                with self.assertRaisesRegex(RuntimeError, "controlled"):
                    await collect_fresh(runner, Path(raw))
                self.assertFalse(
                    any(call[0].startswith("row-5/") for call in runner.calls)
                )
                result = json.loads((Path(raw) / "partial.json").read_text())
                self.assertEqual(result["state"], "stopped")
                self.assertTrue(runner.finished)

    async def test_row5_arm_budget_accumulates_across_rounds_and_includes_cleanup(self):
        now = [0.0]

        async def occupied(seconds):
            now[0] += seconds

        with patch(
            "scripts.benchmark_suite.issue406_fresh.time.monotonic",
            side_effect=lambda: now[0],
        ):
            budget = Row5Budget(fresh_plan())
            for _ in range(11):
                await budget.invoke("candidate", lambda: occupied(16.0))
            await budget.invoke("candidate", lambda: occupied(4.0))
            for _ in range(11):
                await budget.invoke("baseline", lambda: occupied(16.0))
            self.assertEqual(budget.spent["candidate"], 180.0)
            for _ in range(10):
                await budget.invoke("candidate", lambda: occupied(16.0))
            with self.assertRaisesRegex(TimeoutError, "row-5"):
                await budget.invoke("candidate", lambda: occupied(21.0))
            self.assertEqual(budget.spent["candidate"], 361.0)

    async def test_cancellation_during_owned_cleanup_stops_before_another_round(self):
        entered, release = asyncio.Event(), asyncio.Event()

        class WaitingRunner(FreshFakeRunner):
            async def finish_round(self, side=None):
                entered.set()
                await release.wait()

        with TemporaryDirectory() as raw:
            runner = WaitingRunner()
            task = asyncio.create_task(collect_fresh(runner, Path(raw)))
            await entered.wait()
            calls = len(runner.calls)
            task.cancel()
            await asyncio.sleep(0)
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.gather(*runner.cleanup_tasks)
            self.assertEqual(len(runner.calls), calls)

    def test_inventory_stays_the_original_36_physical_groups(self):
        self.assertEqual(len(collection_groups()), 36)
