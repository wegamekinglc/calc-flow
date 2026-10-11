from __future__ import annotations

import copy
import unittest

from scripts.benchmark_suite.issue406_fresh import CANDIDATE, PREVIOUS, retained_cases
from scripts.benchmark_suite.release_pairs import pair_order
from scripts.benchmark_suite.report import comparison
from scripts.test_issue406_report import record, report


def retained_report():
    data = report([])
    control = {"product_commit": PREVIOUS, "measurements": [], "pairs": []}
    data["retained_control"] = control
    for case in retained_cases():
        arms = retained_arms(case, control, data)
        for round_index in range(2):
            for index in (None, *range(10)):
                references = pair_references(case, arms, round_index, index)
                order = (
                    ["previous", "candidate"]
                    if index is None
                    else [
                        "previous" if side == "baseline" else side
                        for side in pair_order(index)
                    ]
                )
                control["pairs"].append(
                    {
                        "case": case.id,
                        "round": round_index,
                        "index": index,
                        "order": order,
                        **references,
                    }
                )
    return data


def retained_arms(case, control, data):
    arms = {}
    for side in ("previous", "candidate"):
        arm = record(case, 1.0, side=side)
        arm["warmups"] = [copy.deepcopy(group[0]) for group in arm["rounds"]]
        arms[side] = arm
        destination = (
            control["measurements"] if side == "previous" else data["measurements"]
        )
        destination.append(arm)
    return arms


def pair_references(case, arms, round_index, index):
    references = {}
    for side, revision in (
        ("previous", PREVIOUS),
        ("candidate", CANDIDATE),
    ):
        arm = arms[side]
        sample = (
            arm["warmups"][round_index]
            if index is None
            else arm["rounds"][round_index][index]
        )
        seconds = 1000 if index is None else 1 + index / 100
        factor = 1 if side == "previous" else 0.5
        sample.update(
            product_commit=revision,
            observation_id=f"{case.id}/{side}/{round_index}/{index}",
            raw_observation_sha256="a" * 64,
            processing_seconds=seconds * factor,
            terminal_seconds=2 * seconds * factor,
            full_cycle_seconds=4 * seconds * factor,
        )
        references[side] = {
            key: sample[key] for key in ("observation_id", "raw_observation_sha256")
        }
    return references


class RetainedControlReportTests(unittest.TestCase):
    def test_three_phase_pairs_use_existing_statistics_and_exclude_warmups(self):
        from scripts.benchmark_suite.issue406_fresh_report import (
            summarize_retained_control,
        )

        data = retained_report()
        original = copy.deepcopy(data)
        summary = summarize_retained_control(data)
        self.assertEqual(data, original)
        self.assertEqual(summary["previous"]["product_commit"], PREVIOUS)
        self.assertEqual(summary["candidate"]["product_commit"], CANDIDATE)
        for item, previous, candidate in zip(
            summary["cases"],
            data["retained_control"]["measurements"],
            data["measurements"],
            strict=True,
        ):
            self.assertEqual(item["status"], "complete")
            self.assertEqual(
                item["sample_counts"], {"previous": [10, 10], "candidate": [10, 10]}
            )
            self.assertEqual(
                [phase["metric"] for phase in item["phases"]],
                ["processing_seconds", "terminal_seconds", "full_cycle_seconds"],
            )
            for phase in item["phases"]:
                metric = phase["metric"]
                expected = comparison(
                    {
                        "id": item["case"],
                        "rows": item["rows"],
                        "correctness": True,
                        "comparison": "interleaved",
                        "baseline": [
                            [sample[metric] for sample in group]
                            for group in previous["rounds"]
                        ],
                        "candidate": [
                            [sample[metric] for sample in group]
                            for group in candidate["rounds"]
                        ],
                    }
                )
                self.assertEqual(
                    phase["comparison"],
                    {**expected, "maintained_verdict": expected["verdict"]},
                )
                self.assertEqual(phase["previous"]["samples"], 20)
                self.assertLess(phase["previous"]["p95"], 5)
                self.assertIsNotNone(phase["previous"]["median_interval"])

    def test_one_incomplete_size_preserves_other_complete_control(self):
        from scripts.benchmark_suite.issue406_fresh_report import (
            summarize_retained_control,
        )

        data = retained_report()
        data["retained_control"]["measurements"][1]["rounds"][1].pop()
        summary = summarize_retained_control(data)
        self.assertEqual(summary["cases"][0]["status"], "complete")
        partial = summary["cases"][1]
        self.assertEqual(partial["status"], "incomplete")
        self.assertEqual(partial["sample_counts"]["previous"], [10, 9])
        for phase in partial["phases"]:
            self.assertEqual(phase["comparison"]["verdict"], "unmeasured-or-invalid")
            self.assertEqual(phase["comparison"]["round_intervals"], [])
        self.assertTrue(partial["validation_errors"])

    def test_corrupt_links_runtime_evidence_and_phase_clocks_never_pass(self):
        from scripts.benchmark_suite.issue406_fresh_report import (
            summarize_retained_control,
        )

        mutations = (
            lambda data: data["retained_control"]["pairs"][1]["candidate"].update(
                raw_observation_sha256="b" * 64
            ),
            lambda data: data["measurements"][0]["rounds"][0][0].update(
                oracle_passed=False
            ),
            lambda data: data["retained_control"]["measurements"][0]["rounds"][0][
                0
            ].update(lifecycle="failed"),
            lambda data: data["measurements"][0]["warmups"][0].update(
                cleanup_complete=False
            ),
            lambda data: data["measurements"][0]["rounds"][0][0].update(
                terminal_seconds=float("nan")
            ),
            lambda data: data["measurements"][0]["rounds"][0][0].update(
                full_cycle_seconds=0
            ),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                data = retained_report()
                mutate(data)
                summary = summarize_retained_control(data)
                self.assertEqual(summary["cases"][0]["status"], "invalid")
                self.assertEqual(summary["cases"][1]["status"], "complete")
                self.assertTrue(summary["cases"][0]["validation_errors"])
                self.assertTrue(
                    all(
                        phase["comparison"]["verdict"] == "unmeasured-or-invalid"
                        for phase in summary["cases"][0]["phases"]
                    )
                )

    def test_environment_or_compatibility_failure_retains_only_inconclusive_pairs(self):
        from scripts.benchmark_suite.issue406_fresh_report import (
            summarize_retained_control,
        )

        for reason in ("environment_suitable", "compatibility_verified", "forced_kill"):
            with self.subTest(reason=reason):
                data = retained_report()
                if reason == "forced_kill":
                    data["settlement"]["forced_kill_pids"] = [123]
                else:
                    data["assessment"][reason] = False
                phase = summarize_retained_control(data)["cases"][0]["phases"][0]
                self.assertEqual(phase["comparison"]["verdict"], "inconclusive")
                self.assertEqual(phase["comparison"]["maintained_verdict"], "improved")
                self.assertEqual(len(phase["comparison"]["round_intervals"]), 2)

    def test_render_labels_cff_and_keeps_every_phase_and_missing_size(self):
        from scripts.benchmark_suite.issue406_fresh_report import (
            render_retained_control,
            summarize_retained_control,
        )

        data = retained_report()
        data["retained_control"]["measurements"].pop()
        text = "\n".join(render_retained_control(summarize_retained_control(data)))
        self.assertIn(PREVIOUS, text)
        self.assertIn(CANDIDATE, text)
        self.assertIn("cff", text)
        self.assertIn("1000000", text)
        self.assertIn("4000000", text)
        self.assertIn("terminal_seconds", text)
        self.assertIn("full_cycle_seconds", text)
        self.assertIn("incomplete", text)
        self.assertIn("p95", text)
        self.assertNotIn("Fresh main", text)


if __name__ == "__main__":
    unittest.main()
