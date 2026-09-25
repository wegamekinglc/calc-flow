from __future__ import annotations

import copy
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from scripts.benchmark_suite.catalog import get_shard, shard_cases
from scripts.benchmark_suite.installed_pair_prototype import (
    collect_side,
    planned_schedule,
    run,
)
from scripts.benchmark_suite.installed_pair_validation import (
    gate_exit,
    summarize_report,
    validate_report,
)


def fixture(*, slowdown: float = 0.0, source: str = "none") -> dict:
    slots = {"A0": "/runner/A0/site", "B1": "/runner/B1/site"}
    releases = {
        "baseline": {
            "wheel_sha256": "a" * 64,
            "native_sha256": "c" * 64,
            "wheel_path": "/sealed/a.whl",
        },
        "candidate": {
            "wheel_sha256": ("b" if source != "none" else "a") * 64,
            "native_sha256": ("d" if source == "native" else "c") * 64,
            "wheel_path": "/sealed/b.whl" if source != "none" else "/sealed/a.whl",
        },
    }
    for release in releases.values():
        release["package_files"] = {
            "calc_flow/_native.abi3.so": release["native_sha256"],
            "calc_flow/runtime.py": release["wheel_sha256"],
        }
    blocks = []
    for round_schedule in planned_schedule():
        round_blocks = []
        for spec in round_schedule:
            index, round_index, slot = spec["index"], spec["round"], spec["slot"]
            base_seconds = 1.15 if slot == "B1" else 1.0
            candidate_seconds = base_seconds * (1 + slowdown)
            if round_index == 0 and index == 0:
                candidate_seconds *= 1.13
            sides = {}
            for order_index, side in enumerate(spec["order"]):
                identity = round_index * 100 + index * 2 + order_index + 1
                seconds = base_seconds if side == "baseline" else candidate_seconds
                site = slots[slot]
                files = {
                    "calc_flow/_native.abi3.so": releases[side]["native_sha256"],
                    "calc_flow/runtime.py": releases[side]["wheel_sha256"],
                }
                sides[side] = {
                    "install_id": f"install-{identity}",
                    "install_command": {
                        "argv": [
                            "uv",
                            "pip",
                            "install",
                            "--target",
                            site,
                            "--link-mode",
                            "copy",
                            releases[side]["wheel_path"],
                        ],
                        "exit_code": 0,
                    },
                    "wheel_sha256": releases[side]["wheel_sha256"],
                    "site": site,
                    "tree": {
                        "root": {"dev": 1, "inode": identity, "ctime_ns": identity},
                        "native": {"dev": 1, "inode": identity + 1000},
                        "native_sha256": releases[side]["native_sha256"],
                        "files_sha256": hashlib.sha256(
                            json.dumps(files, sort_keys=True).encode()
                        ).hexdigest(),
                        "files": files,
                    },
                    "loaded_native_inode": identity + 1000,
                    "loaded_native_path": site + "/calc_flow/_native.abi3.so",
                    "native_maps": [
                        "7f00-7f10 r-xp 00000000 08:01 "
                        f"{identity + 1000} {site}/calc_flow/_native.abi3.so"
                    ],
                    "worker_pid": identity + 2000,
                    "environment": {"machine": "same"},
                    "warmup": {"seconds": 0.1, "correctness": {"passed": True}},
                    "replay": [
                        {
                            "seconds": 0.1,
                            "correctness": {"passed": True},
                            "start_row": None,
                        }
                        for _ in range(index)
                    ],
                    "sample": {
                        "seconds": seconds,
                        "correctness": {"passed": True},
                        "start_row": None,
                    },
                    "started_ns": identity * 1_000_000,
                    "finished_ns": identity * 1_000_000 + 100_000,
                }
            round_blocks.append(
                {**spec, "executed_order": list(spec["order"]), "sides": sides}
            )
        blocks.append(round_blocks)
    report = {
        "contract": "installed-pairs-prototype-v1",
        "case_id": "engines/100000/calc-flow-stream/group_by",
        "case": next(
            case
            for case in shard_cases(get_shard("engines-100000"))
            if case["id"] == "engines/100000/calc-flow-stream/group_by"
        ),
        "slots": slots,
        "releases": releases,
        "schedule": planned_schedule(),
        "blocks": blocks,
        "injection_source": source,
    }
    report["summary"] = summarize_report(report)
    return report


class InstalledPairPrototypeTests(unittest.TestCase):
    def test_workflow_gives_installed_pairs_the_full_cost_budget(self):
        workflow = Path(".github/workflows/benchmarks.yml").read_text()
        job = workflow.split("  dal316-side-bias:\n", 1)[1].split(
            "  dal301-profile-build:\n", 1
        )[0]
        self.assertIn(
            "timeout-minutes: ${{ inputs.mode == 'dal316-installed-pairs' "
            "&& 180 || 30 }}",
            job,
        )

    def test_schedule_balances_slot_and_order_in_each_round(self):
        rounds = planned_schedule()
        self.assertEqual([len(round_schedule) for round_schedule in rounds], [10, 10])
        for round_schedule in rounds:
            self.assertEqual(
                Counter(item["slot"] for item in round_schedule), {"A0": 5, "B1": 5}
            )
            self.assertEqual(
                Counter(tuple(item["order"]) for item in round_schedule),
                {("baseline", "candidate"): 5, ("candidate", "baseline"): 5},
            )

    def test_aa_slot_bias_and_one_tree_outlier_do_not_fake_regression(self):
        report = fixture()
        summary = validate_report(report)
        self.assertEqual(summary["verdict"], "no-confirmed-regression")
        self.assertEqual(gate_exit(summary), 0)
        self.assertGreater(max(summary["block_changes"][0]), 12)
        for blocks in report["blocks"]:
            by_slot = {
                slot: [
                    block["sides"]["baseline"]["sample"]["seconds"]
                    for block in blocks
                    if block["slot"] == slot
                ]
                for slot in ("A0", "B1")
            }
            self.assertTrue(all(value == 1.0 for value in by_slot["A0"]))
            self.assertTrue(all(value == 1.15 for value in by_slot["B1"]))

    def test_python_and_native_labeled_eight_percent_timings_still_fail(self):
        for source in ("python", "native"):
            with self.subTest(source=source):
                report = fixture(slowdown=0.08, source=source)
                summary = validate_report(report)
                self.assertEqual(summary["verdict"], "regression")
                self.assertTrue(
                    all(interval["low"] > 5 for interval in summary["round_intervals"])
                )
                self.assertEqual(gate_exit(summary), 1)

    def test_candidate_specific_slot_bias_remains_blocking(self):
        report = fixture()
        for blocks in report["blocks"]:
            for block in blocks:
                if block["slot"] == "B1":
                    block["sides"]["candidate"]["sample"]["seconds"] *= 1.08
        report["summary"] = summarize_report(report)
        summary = validate_report(report)
        self.assertEqual(summary["verdict"], "context-order-dependent")
        self.assertEqual(gate_exit(summary), 1)

    def test_exact_five_percent_is_not_regression_and_uncertain_is_blocking(self):
        exact = fixture(slowdown=0.05, source="python")
        self.assertNotEqual(validate_report(exact)["verdict"], "regression")
        uncertain = fixture()
        for block in uncertain["blocks"][0][1:3]:
            block["sides"]["candidate"]["sample"]["seconds"] *= 1.08
        uncertain["summary"] = summarize_report(uncertain)
        summary = validate_report(uncertain)
        self.assertEqual(summary["verdict"], "unresolved")
        self.assertEqual(gate_exit(summary), 1)

    def test_missing_reused_reordered_or_tampered_evidence_fails_closed(self):
        original = fixture()
        mutations = []
        missing = copy.deepcopy(original)
        missing["blocks"][0].pop()
        mutations.append(missing)
        reused = copy.deepcopy(original)
        reused["blocks"][0][1]["sides"]["baseline"]["install_id"] = reused["blocks"][0][
            0
        ]["sides"]["baseline"]["install_id"]
        mutations.append(reused)
        reordered = copy.deepcopy(original)
        reordered["blocks"][0][0]["executed_order"].reverse()
        mutations.append(reordered)
        tampered = copy.deepcopy(original)
        tampered["blocks"][0][0]["sides"]["candidate"]["sample"]["seconds"] *= 2
        mutations.append(tampered)
        missing_replay = copy.deepcopy(original)
        missing_replay["blocks"][0][5]["sides"]["baseline"]["replay"].pop()
        mutations.append(missing_replay)
        wrong_map = copy.deepcopy(original)
        wrong_map["blocks"][0][0]["sides"]["baseline"]["native_maps"][0] = wrong_map[
            "blocks"
        ][0][0]["sides"]["baseline"]["native_maps"][0].replace(
            "_native.abi3.so", "other.so"
        )
        mutations.append(wrong_map)
        wrong_install = copy.deepcopy(original)
        wrong_install["blocks"][0][0]["sides"]["candidate"]["install_command"]["argv"][
            -1
        ] = "/sealed/other.whl"
        mutations.append(wrong_install)
        changed_case = copy.deepcopy(original)
        changed_case["case"]["rows"] = 10
        mutations.append(changed_case)
        changed_environment = copy.deepcopy(original)
        changed_environment["blocks"][1][0]["sides"]["baseline"]["environment"] = {
            "machine": "other"
        }
        changed_environment["blocks"][1][0]["sides"]["candidate"]["environment"] = {
            "machine": "other"
        }
        mutations.append(changed_environment)
        changed_python = copy.deepcopy(original)
        changed_tree = changed_python["blocks"][0][0]["sides"]["candidate"]["tree"]
        changed_tree["files"]["calc_flow/runtime.py"] = "f" * 64
        changed_tree["files_sha256"] = hashlib.sha256(
            json.dumps(changed_tree["files"], sort_keys=True).encode()
        ).hexdigest()
        mutations.append(changed_python)
        for index, report in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                validate_report(report)


class InstalledPairCollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_collects_forty_fresh_sides_on_balanced_schedule(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            observations = []

            async def fake_side(_case, _release, role, site, index, _root):
                self.assertFalse(site.exists())
                site.mkdir(parents=True)
                observations.append((role, site, index))
                return {"sample": {"seconds": 1.0}}

            releases = iter(
                (
                    {"wheel_sha256": "a" * 64, "wheel_path": "/sealed/a.whl"},
                    {"wheel_sha256": "b" * 64, "wheel_path": "/sealed/b.whl"},
                )
            )
            with (
                patch(
                    "scripts.benchmark_suite.installed_pair_prototype.load_release",
                    side_effect=lambda _: next(releases),
                ),
                patch(
                    "scripts.benchmark_suite.installed_pair_prototype.wheel_package_files",
                    return_value={},
                ),
                patch(
                    "scripts.benchmark_suite.installed_pair_prototype.get_shard",
                    return_value={},
                ),
                patch(
                    "scripts.benchmark_suite.installed_pair_prototype.shard_cases",
                    return_value=[{"id": "engines/100000/calc-flow-stream/group_by"}],
                ),
                patch(
                    "scripts.benchmark_suite.installed_pair_prototype.collect_side",
                    side_effect=fake_side,
                ),
                patch(
                    "scripts.benchmark_suite.installed_pair_prototype.summarize_report",
                    return_value={"verdict": "no-confirmed-regression"},
                ),
                patch(
                    "scripts.benchmark_suite.installed_pair_prototype.validate_report",
                    return_value={"verdict": "no-confirmed-regression"},
                ),
            ):
                report = await run(root / "base.json", root / "head.json", root / "out")

            self.assertEqual(len(observations), 40)
            expected = [
                (role, root / "out" / "slots" / item["slot"] / "site", item["index"])
                for round_schedule in planned_schedule()
                for item in round_schedule
                for role in item["order"]
            ]
            self.assertEqual(observations, expected)
            self.assertEqual([len(blocks) for blocks in report["blocks"]], [10, 10])

    async def test_new_worker_replays_index_before_one_formal_sample(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []
            native_sha = hashlib.sha256(b"native").hexdigest()

            class FakeWorker:
                process = SimpleNamespace(pid=123)

                async def request(self, **message):
                    operation = message["operation"]
                    calls.append(operation)
                    if operation == "hello":
                        return {
                            "native_sha256": native_sha,
                            "polars_threads": 32,
                            "tokio_worker_threads": "32",
                        }
                    if operation == "prepare":
                        return {
                            "case": message["case"],
                            "warmup": {"seconds": 0.1, "correctness": {"passed": True}},
                        }
                    if operation == "sample":
                        return {
                            "seconds": 1.0,
                            "correctness": {"passed": True},
                            "start_row": None,
                        }
                    return {"state": "completed"}

                async def close(self):
                    calls.append("close")

            async def fake_install(_release, site, _log):
                native = site / "calc_flow" / "_native.abi3.so"
                native.parent.mkdir(parents=True)
                native.write_bytes(b"native")

            release = {"wheel_sha256": "a" * 64, "native_sha256": native_sha}
            case = {"id": "engines/100000/calc-flow-stream/group_by"}
            with (
                patch(
                    "scripts.benchmark_suite.installed_pair_prototype.install_fresh",
                    side_effect=fake_install,
                ),
                patch(
                    "scripts.benchmark_suite.installed_pair_prototype.Worker.start",
                    return_value=FakeWorker(),
                ),
                patch(
                    "scripts.benchmark_suite.installed_pair_prototype.native_mapping",
                    side_effect=lambda _pid, site: (
                        (site / "calc_flow" / "_native.abi3.so").stat().st_ino,
                        str((site / "calc_flow" / "_native.abi3.so").resolve()),
                        [
                            "7f00-7f10 r-xp 00000000 08:01 "
                            f"{(site / 'calc_flow' / '_native.abi3.so').stat().st_ino} "
                            f"{(site / 'calc_flow' / '_native.abi3.so').resolve()}"
                        ],
                    ),
                ),
            ):
                evidence = await collect_side(
                    case, release, "baseline", root / "A0" / "site", 3, root / "block"
                )

            self.assertEqual(
                calls,
                [
                    "hello",
                    "prepare",
                    "sample",
                    "sample",
                    "sample",
                    "sample",
                    "finish",
                    "close",
                ],
            )
            self.assertEqual(len(evidence["replay"]), 3)
            self.assertEqual(evidence["sample"]["seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()
