from __future__ import annotations

import asyncio
import copy
import json
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from scripts.benchmark_suite import issue406_milestone as milestone
from scripts.benchmark_suite.issue406_acceptance import acceptance_inventory
from scripts.benchmark_suite.issue406_run import AcceptanceRunner, collection_groups
from scripts.test_issue406_run import FakeRunner
from scripts.toolkit import sha256_file


def plan():
    return {
        **milestone.ENVELOPE,
        "schema": milestone.SCHEMA,
        "baseline_product_commit": "a" * 40,
        "candidate_product_commit": "b" * 40,
        "harness_git_sha": "c" * 40,
        "build_harness_git_sha": "c" * 40,
        "execution_git_sha": "c" * 40,
        "releases": {
            "baseline": {"product_commit": "a" * 40},
            "candidate": {"product_commit": "b" * 40},
        },
        "cases": [asdict(case) for case in milestone.inventory()],
        "cpu_budgets": milestone.cpu_budgets(),
        "preparation_anchor": {"path": "anchor.json", "sha256": "d" * 64},
    }


class MilestonePolicyTests(unittest.TestCase):
    def test_baseline_off_seal_checks_its_binary_product_and_tool(self):
        from scripts.benchmark_suite.issue406_run import _validate_plan_binary

        with TemporaryDirectory() as raw:
            binary = Path(raw) / "baseline-off"
            binary.write_bytes(b"baseline binary fixture")
            target = "stream_join_perf"
            descriptor = {
                "path": str(binary),
                "sha256": sha256_file(binary),
                "build_output": raw,
                "harness_target": target,
            }
            frozen = {
                **plan(),
                "native_binaries": {"baseline": {f"{target}/off": descriptor}},
            }
            files = {
                str(binary): sha256_file(binary),
                str(Path(raw) / f"build-{target}.acceptance-harness.json"): "d" * 64,
            }
            attestation = {
                "targets": [target],
                "harness_git_sha": "c" * 40,
                "product_git_sha": "a" * 40,
            }
            with (
                patch(
                    "scripts.benchmark_suite.issue406_run._validate_source_attestation",
                    return_value=attestation,
                ),
                patch(
                    "scripts.benchmark_suite.issue406_run._validate_plan_native_build",
                    return_value={"artifacts": []},
                ) as compiled,
            ):
                _validate_plan_binary(
                    frozen, "baseline", f"{target}/off", files, "c" * 40
                )
                self.assertEqual(compiled.call_args.args[-1], "off")
                with self.assertRaisesRegex(ValueError, "revisions"):
                    _validate_plan_binary(
                        frozen, "baseline", f"{target}/off", files, "e" * 40
                    )
                attestation["product_git_sha"] = "b" * 40
                with self.assertRaisesRegex(ValueError, "revisions"):
                    _validate_plan_binary(
                        frozen, "baseline", f"{target}/off", files, "c" * 40
                    )
            with self.assertRaisesRegex(ValueError, "sealed"):
                _validate_plan_binary(frozen, "baseline", f"{target}/off", {}, "c" * 40)

    def test_missing_off_binary_or_an_extra_third_arm_is_rejected(self):
        from scripts.benchmark_suite.issue406_run import (
            _plan_binary_inventory,
            _validate_plan_binaries,
        )

        expected = _plan_binary_inventory("baseline", current=True)
        self.assertEqual(len(expected), 4)
        self.assertEqual(expected, _plan_binary_inventory("candidate", current=True))
        self.assertEqual(len(_plan_binary_inventory("baseline")), 2)
        for invalid in (expected - {"stream_join_perf/off"}, expected | {"third/off"}):
            frozen = {**plan(), "native_binaries": {"baseline": dict.fromkeys(invalid)}}
            with self.assertRaisesRegex(ValueError, "inventory"):
                _validate_plan_binaries(frozen, "baseline", {}, "c" * 40, {})

    def test_new_pair_and_both_off_arms_preserve_historical_inventory(self):
        milestone.validate_envelope(plan())
        cases = milestone.inventory()
        self.assertEqual(len(cases), 47)
        self.assertEqual(sum(len(case.sides) for case in cases), 94)
        self.assertTrue(all(case.sides == ("baseline", "candidate") for case in cases))
        groups = collection_groups(cases)
        self.assertEqual(len(groups), 45)
        self.assertEqual(sum(len(group[0].sides) for group in groups), 90)
        self.assertEqual(len(acceptance_inventory()), 38)
        self.assertEqual(sum(len(case.sides) for case in acceptance_inventory()), 64)

    def test_wrong_source_inventory_budget_and_third_product_fail_closed(self):
        for field, value in (
            ("baseline_product_commit", "main"),
            ("candidate_product_commit", "a" * 40),
            ("build_harness_git_sha", "e" * 40),
            ("cases", []),
            ("budget_seconds", 5401),
            ("row5_arm_seconds", 361),
            ("row5_group_seconds", 721),
            ("pairs_per_round", 11),
            ("rounds", 1),
            ("resource_limits", {}),
            ("cpu_budgets", {}),
            ("preparation_anchor", {}),
            ("retained_control", {}),
            ("precharged_seconds", 0),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                milestone.validate_envelope({**plan(), field: value})

    def test_cli_anchor_creation_does_not_run_collection_or_overwrite_an_anchor(self):
        from scripts.benchmark_suite.issue406_run import main

        with TemporaryDirectory() as raw:
            anchor = Path(raw) / "anchor.json"
            with (
                patch("sys.argv", ["issue406_run", "--begin-preparation", str(anchor)]),
                patch("scripts.benchmark_suite.issue406_run._run_cli") as run_cli,
            ):
                self.assertEqual(main(), 0)
                before = anchor.read_bytes()
                with self.assertRaises(FileExistsError):
                    main()
                self.assertEqual(anchor.read_bytes(), before)
                run_cli.assert_not_called()

    def test_elapsed_preparation_is_measured_from_sealed_same_boot_anchor(self):
        with TemporaryDirectory() as raw:
            anchor = Path(raw) / "anchor.json"
            with patch.object(milestone.time, "monotonic", return_value=100.0):
                milestone.begin_preparation(anchor)
            frozen = {
                **plan(),
                "preparation_anchor": {
                    "path": str(anchor),
                    "sha256": sha256_file(anchor),
                },
                "sealed_files": {str(anchor): sha256_file(anchor)},
            }
            with patch.object(milestone.time, "monotonic", return_value=150.0):
                self.assertEqual(milestone.acquisition_start(frozen), 100.0)
            changed = json.loads(anchor.read_text())
            changed["boot_id"] = "different-boot"
            anchor.write_text(json.dumps(changed))
            frozen["preparation_anchor"]["sha256"] = sha256_file(anchor)
            frozen["sealed_files"][str(anchor)] = sha256_file(anchor)
            with self.assertRaisesRegex(ValueError, "boot"):
                milestone.acquisition_start(frozen)

    def test_baseline_off_uses_its_own_release_and_binary(self):
        current = plan()
        current["cpus"] = [0]
        current["row7_releases"] = copy.deepcopy(current["releases"])
        cases = milestone.inventory()
        for workload, target in (
            ("asof_operator", "stream_asof_e2e"),
            ("join_small_batch", "stream_join_perf"),
        ):
            case = next(
                case
                for case in cases
                if case.workload == workload and case.mode == "off"
            )
            current["native_binaries"] = {
                "baseline": {f"{target}/off": {"path": "baseline-off"}}
            }
            runner = AcceptanceRunner(current, Path("output"), time.monotonic())
            self.assertTrue(runner.fresh)
            with patch.object(
                runner, "dispatch_trace_path", return_value=Path("baseline-trace.json")
            ):
                self.assertEqual(
                    runner._native_invocation((case,), "baseline", 0, None)[1]["path"],
                    "baseline-off",
                )
            self.assertEqual(
                runner.release(case, "baseline")["product_commit"], "a" * 40
            )


class MilestoneCollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_driver_routes_current_policy_charges_preparation_and_seals_all_rows(
        self,
    ):
        from scripts.benchmark_suite.issue406_run import run

        with TemporaryDirectory() as raw:
            root = Path(raw)
            anchor = root / "anchor.json"
            with patch.object(
                milestone.time, "monotonic", return_value=time.monotonic() - 3
            ):
                milestone.begin_preparation(anchor)
            frozen = {
                **plan(),
                "cpus": [0],
                "preparation_anchor": {
                    "path": str(anchor),
                    "sha256": sha256_file(anchor),
                },
                "sealed_files": {str(anchor): sha256_file(anchor)},
            }
            collect = AsyncMock(return_value={"state": "collected", "measurements": []})
            with (
                patch("scripts.benchmark_suite.issue406_run.validate_plan"),
                patch.object(milestone, "collect", collect),
            ):
                result = await run(frozen, root / "result")
            collect.assert_awaited_once()
            self.assertGreaterEqual(result["preparation_elapsed_seconds"], 3)
            self.assertNotIn("precharged_seconds", result)
            self.assertEqual(result["state"], "stopped")
            self.assertTrue(result["settlement"]["settled"])
            summary = json.loads((root / "result/summary.json").read_text())
            self.assertEqual(len(summary["targets"]), 8)
            self.assertEqual(len(summary["cases"]), 47)
            self.assertEqual(summary["acceptance"], "inconclusive")

    async def test_all_pairs_include_retained_baseline_off_without_third_product(self):
        with TemporaryDirectory() as raw:
            runner = FakeRunner()
            runner.plan = plan()
            result = await milestone.collect(runner, Path(raw))
        self.assertEqual(len(result["measurements"]), 94)
        self.assertEqual(len(runner.calls), 1980)
        self.assertEqual(sum(call[0] == "sample" for call in runner.calls), 1800)
        self.assertEqual(sum(call[0] == "warmup" for call in runner.calls), 180)
        self.assertEqual({call[2] for call in runner.calls}, {"baseline", "candidate"})
        first = [call[2] for call in runner.calls if call[0] == "sample"][:4]
        self.assertEqual(first, ["baseline", "candidate", "candidate", "baseline"])
        self.assertNotIn("retained_control", result)

    async def test_failure_keeps_partial_evidence_and_settles_without_retry(self):
        with TemporaryDirectory() as raw:
            runner = FakeRunner(fail_at=3)
            runner.plan = plan()
            with self.assertRaises(asyncio.CancelledError):
                await milestone.collect(runner, Path(raw))
            result = json.loads((Path(raw) / "partial.json").read_text())
        self.assertEqual(result["state"], "stopped")
        self.assertEqual(len(runner.calls), 3)
        self.assertEqual(runner.closed, 1)


def _runtime_inputs(case):
    if case.workload == "projection":
        return {"input": case.rows}
    if case.workload in ("asof", "interval_join"):
        return {"reference.input": case.rows, "quotes.input": case.rows}
    return {"left": case.rows, "right": 64}


def _runtime_sample(case, seconds, side, epochs, sample):
    inputs = _runtime_inputs(case)
    return {
        **sample,
        "product_commit": ("a" if side == "baseline" else "b") * 40,
        "cpu_budget": 2.0 if case.threads == 1 else 32.0,
        "cpu_evidence": {
            "schema": "calc-flow.phase-resources.v1",
            "scope": "processing",
            "status": "measured",
            "wall_seconds": seconds,
            "process_cpu_seconds": seconds,
            "threads": [{"tid": 1, "runtime_seconds": seconds}],
            "departed_threads": [],
        },
        "completed_processing_epochs": list(epochs),
        "raw_observation_sha256": "d" * 64,
        "group_member_ids": [case.id],
        "event_layout": {
            "data_events": [
                {
                    "input": name,
                    "rows": rows,
                    "record_batch_rows": [rows],
                    "record_batch_count": 1,
                }
                for name, rows in inputs.items()
            ]
        },
        "runtime_input_data_batches": len(inputs),
    }


def runtime_record(case, seconds, *, side="candidate", epochs=()):
    from scripts.test_issue406_report import record

    result = record(case, seconds, side=side, epochs=list(epochs))
    rounds = [
        [
            _runtime_sample(
                case,
                seconds,
                side,
                epochs,
                {
                    **sample,
                    "observation_id": f"{case.id}/{side}/{round_index}/{index}",
                },
            )
            for index, sample in enumerate(group)
        ]
        for round_index, group in enumerate(result["rounds"])
    ]
    return {
        **result,
        "rounds": rounds,
        "warmups": [copy.deepcopy(group[0]) for group in rounds],
    }


def _interval_fixture(case):
    return {
        "schema": "calc-flow.interval-acceptance-fixture.v1",
        "rows_per_side": case.rows,
        "batch_rows": case.batch_rows,
        "checkpoint_enabled": case.mode != "off",
    }


def _ledger_sample(case, sample, observation_id, members):
    result = {
        **sample,
        "observation_id": observation_id,
        "group_member_ids": members,
        "checkpoint_enabled": case.mode != "off",
        "cpu_evidence": {**sample["cpu_evidence"], "scope": case.cpu_scope},
    }
    if case.row == 7 and case.workload == "asof":
        result.update(
            processing_settled=True,
            match_identity={},
            dispatch_trace={"schema": "calc-flow.asof-dispatch-trace.v1"},
        )
    if case.row == 10:
        result.update(
            oracle={**sample["oracle"], "input_rows": case.rows},
            fixture=_interval_fixture(case),
        )
    return result


def _ledger_record(case, side, members, physical):
    record = runtime_record(case, 0.1, side=side)
    return {
        **record,
        "warmups": [
            _ledger_sample(case, sample, f"{physical}/{side}/{index}/warmup", members)
            for index, sample in enumerate(record["warmups"])
        ],
        "rounds": [
            [
                _ledger_sample(
                    case, sample, f"{physical}/{side}/{round_index}/{slot}", members
                )
                for slot, sample in enumerate(group)
            ]
            for round_index, group in enumerate(record["rounds"])
        ],
    }


def _complete_ledger():
    cases = milestone.inventory()
    records = []
    for case in cases:
        members = (
            sorted(item.id for item in cases if item.row == case.row)
            if case.row in (4, 5)
            else [case.id]
        )
        physical = f"row-{case.row}" if case.row in (4, 5) else case.id
        records.extend(
            _ledger_record(case, side, members, physical) for side in milestone.SIDES
        )
    return {**milestone.report_identity(plan()), "measurements": records}


def _serial_mode_records(row, seconds):
    cases = [
        case for case in milestone.inventory() if case.row == row and case.threads == 1
    ]
    return [
        runtime_record(case, value, epochs=(1,) if case.mode == "on" else ())
        for case, value in zip(cases, seconds, strict=True)
    ]


def _projection_batch_record(case, seconds, batches):
    result = runtime_record(case, seconds)
    for sample in [
        *result["warmups"],
        *(item for group in result["rounds"] for item in group),
    ]:
        sample["event_layout"]["data_events"] = [
            {
                "input": "input",
                "rows": case.rows // batches,
                "record_batch_rows": [case.rows // batches],
                "record_batch_count": 1,
            }
            for _ in range(batches)
        ]
        sample["runtime_input_data_batches"] = batches
    return result


class MilestoneReportTests(unittest.TestCase):
    def test_sealing_expiry_keeps_every_row_and_revokes_all_target_verdicts(self):
        from scripts.benchmark_suite.issue406_run import _expire_fresh_report

        report = {
            **milestone.report_identity(plan()),
            "assessment": {},
            "measurements": [],
        }
        with TemporaryDirectory() as raw:
            output = Path(raw)
            _expire_fresh_report(output, report)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["targets"]), 8)
            self.assertEqual(len(summary["cases"]), 47)
            self.assertEqual(summary["acceptance"], "inconclusive")
            self.assertNotIn(
                "observed-within-target", (output / "performance.md").read_text()
            )
            self.assertFalse(report["budget_satisfied"])

    def test_complete_ledger_requires_every_baseline_off_and_shared_reference(self):
        complete = _complete_ledger()
        milestone.validate_results(complete)
        without_baseline_off = {
            **complete,
            "measurements": [
                record
                for record in complete["measurements"]
                if not ("/off/" in record["case"] and record["side"] == "baseline")
            ],
        }
        with self.assertRaisesRegex(ValueError, "missing measurements"):
            milestone.validate_results(without_baseline_off)
        corrupted = copy.deepcopy(complete)
        corrupted["measurements"][0]["rounds"][0][0]["product_commit"] = "e" * 40
        with self.assertRaisesRegex(ValueError, "source"):
            milestone.validate_results(corrupted)

    def test_new_workloads_require_exact_input_geometry_source_and_cpu(self):
        for case in (case for case in milestone.inventory() if case.row in (9, 10)):
            for side in milestone.SIDES:
                sample = runtime_record(case, 0.1, side=side)["rounds"][0][0]
                if case.row == 10:
                    sample["oracle"]["input_rows"] = case.rows
                    sample["fixture"] = _interval_fixture(case)
                milestone.validate_sample(case, side, sample, plan())
                altered = copy.deepcopy(sample)
                altered["event_layout"]["data_events"][0]["rows"] -= 1
                with (
                    self.subTest(case=case.id, side=side),
                    self.assertRaises(ValueError),
                ):
                    milestone.validate_sample(case, side, altered, plan())

    def test_slow_off_cannot_relax_low_on_formal_or_planning_goals(self):
        from scripts.benchmark_suite.issue406_milestone_report import summarize

        data = {
            **milestone.report_identity(plan()),
            "measurements": _serial_mode_records(1, (0.200, 0.210, 0.250)),
        }
        original = copy.deepcopy(data)
        summary = summarize(data)
        effects = [
            item
            for item in summary["targets"][0]["mode_effects"]
            if item["side"] == "candidate" and item["threads"] == 1
        ]
        self.assertEqual(
            [item["target_status"] for item in effects], ["observed-above-target"] * 3
        )
        self.assertAlmostEqual(effects[1]["limit_seconds"], 0.060 / 0.90)
        self.assertAlmostEqual(effects[2]["limit_seconds"], 0.060 / 0.75)
        self.assertEqual(len(summary["cases"]), 47)
        self.assertEqual(data, original)
        baseline_off = summary["cases"][0]
        self.assertEqual(baseline_off["baseline"]["status"], "unmeasured")
        self.assertNotEqual(
            baseline_off["revision_comparison"]["verdict"], "candidate-only"
        )

    def test_on_without_processing_epochs_and_missing_batch_denominator_cannot_pass(
        self,
    ):
        from scripts.benchmark_suite.issue406_milestone_report import summarize

        case = next(
            case
            for case in milestone.inventory()
            if case.workload == "projection" and case.mode == "on"
        )
        arm = runtime_record(case, 0.00001)
        data = {**milestone.report_identity(plan()), "measurements": [arm]}
        summary = summarize(data)
        target = next(
            item
            for item in summary["runtime_floor"]
            if item["side"] == "candidate" and item["mode"] == "on"
        )
        self.assertEqual(target["target_status"], "unmeasured")
        sample = arm["rounds"][0][0]
        sample.pop("runtime_input_data_batches")
        with self.assertRaisesRegex(ValueError, "denominator"):
            milestone.validate_sample(case, "candidate", sample, plan())

    def test_runtime_floor_uses_actual_batch_counts(self):
        from scripts.benchmark_suite.issue406_milestone_report import summarize

        case = next(
            case
            for case in milestone.inventory()
            if case.workload == "projection" and case.mode == "off"
        )
        arm = _projection_batch_record(case, 0.001, 10)
        summary = summarize(
            {**milestone.report_identity(plan()), "measurements": [arm]}
        )
        target = next(
            item
            for item in summary["runtime_floor"]
            if item["side"] == "candidate" and item["mode"] == "off"
        )
        self.assertAlmostEqual(target["statistics"]["p50"], 0.0001)
        self.assertEqual(target["batch_counts"], [10] * 20)

    def test_wrong_source_cpu_budget_and_false_off_are_rejected(self):
        case = milestone.inventory()[0]
        sample = runtime_record(case, 0.01)["rounds"][0][0]
        milestone.validate_sample(case, "candidate", sample, plan())
        for field, value in (
            ("product_commit", "e" * 40),
            ("cpu_budget", 32.0),
            ("cleanup_complete", False),
            ("lifecycle", "failed"),
            ("processing_checkpoint_free", False),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                milestone.validate_sample(
                    case, "candidate", {**sample, field: value}, plan()
                )
        with self.assertRaises(ValueError):
            milestone.validate_sample(
                case,
                "candidate",
                {key: value for key, value in sample.items() if key != "cpu_evidence"},
                plan(),
            )
