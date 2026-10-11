from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.benchmark_suite import issue406_fresh_v3 as v3
from scripts.benchmark_suite.issue406_fresh import (
    collect_fresh,
    validate_retained_control,
)
from scripts.benchmark_suite.issue406_run import collection_groups
from scripts.test_issue406_fresh import FreshFakeRunner, fresh_plan

PRODUCT = "a" * 40
HARNESS = "b" * 40


def plan_v3():
    plan = fresh_plan()
    plan.update(
        schema=v3.SCHEMA,
        candidate_product_commit=PRODUCT,
        harness_git_sha=HARNESS,
        execution_git_sha=HARNESS,
    )
    plan["releases"]["candidate"]["product_commit"] = PRODUCT
    plan.pop("known_main_failure_identity")
    return plan


class V3Runner(FreshFakeRunner):
    def __init__(self, *, failed_case=None, returned=False, recognized=True):
        super().__init__()
        self.plan = plan_v3()
        self.failed_case = failed_case
        self.returned = returned
        self.recognized = recognized

    def observation(self, group, side, round_index, index):
        result = super().observation(group, side, round_index, index)
        for sample in result.values():
            sample.update(lifecycle="completed")
            if side == "candidate":
                sample["product_commit"] = PRODUCT
        if group[0].id == self.failed_case and side == "baseline":
            if not self.returned:
                raise RuntimeError("new sealed IPC failure")
            for sample in result.values():
                sample.update(lifecycle="failed", lifecycle_error="sentinel")
        return result

    async def known_fresh_failure(self, group, side, round_index, index, error):
        if (
            self.recognized
            and side == "baseline"
            and round_index == 0
            and index is None
        ):
            return {
                "signature": "runtime_eof_shutdown_sentinel",
                "natural_exit": {"returncode": 0, "group_empty": True},
            }
        return None


class V3PolicyTests(unittest.TestCase):
    def test_identity_is_external_and_envelope_is_fixed(self):
        v3.validate_envelope(plan_v3())
        for field, value in (
            ("candidate_product_commit", "main"),
            ("execution_git_sha", "c" * 40),
            ("harness_git_sha", "c" * 40),
            ("budget_seconds", 5401),
            ("precharged_seconds", 4795),
            ("cases", []),
            ("historical_harness_witness", {}),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                v3.validate_envelope({**plan_v3(), field: value})
        altered = plan_v3()
        altered["releases"]["candidate"]["product_commit"] = "c" * 40
        with self.assertRaises(ValueError):
            v3.validate_envelope(altered)

    def test_failed_lifecycle_never_validates_as_complete(self):
        report = {
            "measurements": [
                {
                    "warmups": [],
                    "rounds": [
                        [{"lifecycle": "failed", "lifecycle_error": "sentinel"}],
                        [],
                    ],
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            v3.validate_complete_results(report)


class V3CollectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_product_and_44_control_links_are_preserved(self):
        with TemporaryDirectory() as raw:
            runner = V3Runner()
            report = await collect_fresh(runner, Path(raw))
        self.assertEqual(len(runner.calls), 1364)
        self.assertEqual(len(report["measurements"]), 64)
        self.assertEqual(report["candidate_product_commit"], PRODUCT)
        self.assertEqual(len(report["retained_control"]["pairs"]), 44)
        validate_retained_control(report)
        failed = copy.deepcopy(report)
        failed["measurements"][0]["rounds"][0][0]["lifecycle"] = "failed"
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            v3.validate_complete_results(failed)
        altered = copy.deepcopy(report)
        altered["candidate_product_commit"] = "d" * 40
        with self.assertRaises(ValueError):
            validate_retained_control(altered)

    async def test_raised_and_returned_failures_exhaust_only_one_physical_arm(self):
        group = next(group for group in collection_groups() if group[0].row == 4)
        for returned in (False, True):
            with self.subTest(returned=returned), TemporaryDirectory() as raw:
                runner = V3Runner(failed_case=group[0].id, returned=returned)
                report = await collect_fresh(runner, Path(raw))
            self.assertEqual(len(runner.calls), 1364 - 21)
            self.assertEqual(len({call[0] for call in runner.calls}), 36)
            self.assertEqual(report["state"], "incomplete")
            failure = report["failed_arms"][0]
            self.assertEqual(failure["cases"], [case.id for case in group])
            self.assertEqual(len(failure["unattempted_slots"]), 21)
            self.assertEqual(len(report["retained_control"]["pairs"]), 44)
            self.assertEqual(report["known_failures"], [])
            self.assert_failed_arm_empty(report, failure)

    def assert_failed_arm_empty(self, report, failure):
        for record in report["measurements"]:
            if record["case"] in failure["cases"] and record["side"] == "baseline":
                self.assertEqual(record["rounds"], [[], []])
                self.assertEqual(record["warmups"], [])

    async def test_failed_row7_source_prevents_replay_and_retains_all_slots(self):
        from dataclasses import replace

        from scripts.test_issue406_report import selected

        source = selected(7, workload="asof", mode="low")
        with TemporaryDirectory() as raw:
            runner = V3Runner(failed_case=source.id, returned=True)
            report = await collect_fresh(runner, Path(raw))
        replay = replace(source, workload="asof_operator")
        self.assertEqual(v3.replay_source(replay), source.id)
        self.assertFalse(
            any(call[0] == replay.id and call[1] == "baseline" for call in runner.calls)
        )
        self.assertEqual(len(report["dependent_unattempted"][0]["slots"]), 22)
        self.assertEqual(len(report["failed_arms"]), 1)
        self.assertTrue(any(call[0].startswith("row-8/") for call in runner.calls))

    async def test_unknown_returned_failure_stops_without_timing(self):
        case = next(
            group[0] for group in collection_groups() if "baseline" in group[0].sides
        )
        with TemporaryDirectory() as raw:
            runner = V3Runner(failed_case=case.id, returned=True, recognized=False)
            with self.assertRaisesRegex(ValueError, "lifecycle"):
                await collect_fresh(runner, Path(raw))
            report = json.loads((Path(raw) / "partial.json").read_text())
        self.assertEqual(report["state"], "stopped")
        self.assertEqual(sum(call[0] == case.id for call in runner.calls), 1)

    async def test_unknown_first_warmup_is_failed_in_both_summary_paths(self):
        from unittest.mock import patch

        from scripts.benchmark_suite.issue406_fresh_report import (
            aborted_v3_summary,
            summarize_v3,
        )

        group = next(group for group in collection_groups() if group[0].row == 4)
        for returned in (False, True):
            with self.subTest(returned=returned), TemporaryDirectory() as raw:
                runner = V3Runner(
                    failed_case=group[0].id, returned=returned, recognized=False
                )
                with (
                    patch(
                        "scripts.benchmark_suite.issue406_run.collection_groups",
                        return_value=[group],
                    ),
                    self.assertRaises((RuntimeError, ValueError)),
                ):
                    await collect_fresh(runner, Path(raw))
                report = json.loads((Path(raw) / "partial.json").read_text())
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(report["failed_arms"], [])
            for summarize in (summarize_v3, aborted_v3_summary):
                self.assert_failed_summary(summarize(report), group)

    def assert_failed_summary(self, summary, group):
        failed = {case.id for case in group}
        for case in summary["cases"]:
            for side in ("baseline", "candidate"):
                arm = case[side]
                if arm["status"] == "unsupported":
                    continue
                expected = (
                    "failed"
                    if side == "baseline" and case["case"] in failed
                    else "unattempted"
                )
                self.assertEqual(arm["attempt_status"], expected)
                self.assertIsNone(arm["statistics"])
        self.assertEqual(summary["acceptance_status"], "inconclusive")


def eof_observation(case):
    checkpoint = {
        "current_epoch": None,
        "elapsed_micros": None,
        "phase": None,
        "failure_category": "internal",
        "installed_unknown_epoch": None,
        "last_completed_epoch": 1,
        "runtime_config_changed": False,
        "terminal": False,
        "source_acks": 0,
        "operator_acks": 0,
        "sink_commit_acks": 0,
        "sink_precommit_acks": 0,
        "expected_sources": 2,
        "expected_operators": 2,
        "expected_sink_commits": 1,
        "expected_sink_precommits": 1,
    }
    status = {
        "state": "completed",
        "terminal_cause": "natural_end",
        "task_count": 0,
        "task_errors": 0,
        "metrics_overflowed": False,
        "checkpoint": checkpoint,
        "operators": {
            "join": {"ended": True, "errors": 0},
            "result": {"ended": True, "errors": 0},
        },
        "sources": {
            "left": {"ended": True, "errors": 0},
            "right": {"ended": True, "errors": 0},
        },
        "sinks": {"suite": {"ended": True, "errors": 0, "delivered_rows": case.rows}},
        "stream_joins": {
            "join": {"match_limit_failures": 0, "state_limit_failures": 0}
        },
        "stream_asof_joins": {},
    }
    return {
        "lifecycle": "failed",
        "lifecycle_error": v3.EOF_REASON,
        "terminal_error": v3.EOF_REASON,
        "mode": case.mode,
        "oracle_passed": True,
        "cleanup_complete": True,
        "oracle": {
            "passed": True,
            "rows": case.rows,
            "max_abs_error": 0.0,
            "bitmap_bytes": case.rows,
        },
        "before_eof": {
            "state": "running",
            "task_errors": 0,
            "metrics_overflowed": False,
            "checkpoint": {"failure_category": None},
        },
        "after_eof": copy.deepcopy(status),
        "after_cleanup": copy.deepcopy(status),
    }


class V3ClosedSignatureTests(unittest.TestCase):
    def setUp(self):
        self.case = next(
            group[0]
            for group in collection_groups()
            if group[0].row == 3 and group[0].mode == "low"
        )

    def test_eof_reason_requires_complete_oracle_and_exact_terminal_status(self):
        raw = eof_observation(self.case)
        self.assertEqual(
            v3.runtime_failure_signature(self.case, raw),
            "runtime_eof_shutdown_sentinel",
        )
        changes = (
            ("oracle_passed", False),
            ("cleanup_complete", False),
            ("terminal_error", "unknown error"),
            ("lifecycle", "completed"),
        )
        for key, value in changes:
            with self.subTest(key=key):
                self.assertIsNone(
                    v3.runtime_failure_signature(self.case, {**raw, key: value})
                )
        for phase, field, value in (
            ("after_eof", "state", "failed"),
            ("after_cleanup", "task_errors", 1),
            ("after_eof", "metrics_overflowed", True),
        ):
            altered = copy.deepcopy(raw)
            altered[phase][field] = value
            self.assertIsNone(v3.runtime_failure_signature(self.case, altered))
        altered = copy.deepcopy(raw)
        altered["after_eof"]["checkpoint"]["phase"] = "sources_cut"
        self.assertIsNone(v3.runtime_failure_signature(self.case, altered))
        altered = copy.deepcopy(raw)
        altered["oracle"]["passed"] = False
        self.assertIsNone(v3.runtime_failure_signature(self.case, altered))
        altered = copy.deepcopy(raw)
        altered["after_eof"]["stream_joins"]["join"]["state_limit_failures"] = 1
        self.assertIsNone(v3.runtime_failure_signature(self.case, altered))

    def test_terminal_error_text_cannot_hide_assertions_or_wrong_oracle_types(self):
        raw = eof_observation(self.case)
        for field in ("failure", "cleanup_error", "observer_error", "operation_error"):
            self.assertIsNone(
                v3.runtime_failure_signature(
                    self.case, {**raw, field: "assertion failed"}
                )
            )
        raw["oracle"]["max_abs_error"] = False
        self.assertIsNone(v3.runtime_failure_signature(self.case, raw))

    def test_ipc_reason_is_closed_but_locations_come_from_fresh_sealed_binary(self):
        case = next(group[0] for group in collection_groups() if group[0].row == 4)
        diagnostic = (
            "thread 'main' panicked at crates/calc-flow/benches/support/"
            "native_join_components.rs:999:55:\n"
            "called `Result::unwrap()` on an `Err` value: "
            'Internal { message: "V2 IPC constructor credit admission failed" }\n'
        )
        self.assertEqual(
            v3.native_failure_signature(case, 101, diagnostic),
            "native_checkpoint_credit_admission",
        )
        for code, text in (
            (-9, diagnostic),
            (1, diagnostic),
            (101, diagnostic + "assertion failed: oracle"),
            (101, diagnostic.replace("credit admission", "unknown")),
        ):
            self.assertIsNone(v3.native_failure_signature(case, code, text))


class V3ReportTests(unittest.TestCase):
    def test_incomplete_report_keeps_eight_rows_and_explicit_arm_states(self):
        from scripts.benchmark_suite.issue406_fresh_report import summarize_v3
        from scripts.test_issue406_report import record, report

        case = next(group[0] for group in collection_groups() if group[0].row == 4)
        data = {
            **report([record(case, 0.01)]),
            **v3.report_identity(plan_v3()),
            "state": "incomplete",
            "failed_arms": [{"cases": [case.id], "side": "baseline"}],
            "dependent_unattempted": [],
        }
        summary = summarize_v3(data)
        self.assertEqual(
            {target["row"] for target in summary["targets"]}, set(range(1, 9))
        )
        self.assertFalse(summary["evidence_complete"])
        self.assertEqual(summary["acquisition_status"], "incomplete")
        self.assertEqual(summary["acceptance_status"], "inconclusive")
        selected = next(item for item in summary["cases"] if item["case"] == case.id)
        self.assertEqual(selected["baseline"]["attempt_status"], "failed")
        self.assertEqual(selected["candidate"]["attempt_status"], "attempted")
        off = next(item for item in summary["cases"] if item["mode"] == "off")
        self.assertEqual(off["baseline"]["validity"], "unsupported")

    def test_failed_lifecycle_full_counts_cannot_produce_statistics(self):
        from scripts.benchmark_suite.issue406_fresh_report import summarize_v3
        from scripts.test_issue406_report import record, report

        case = next(group[0] for group in collection_groups() if group[0].row == 4)
        arm = record(case, 0.01, side="baseline")
        arm["rounds"][0][0].update(lifecycle="failed", lifecycle_error=v3.EOF_REASON)
        data = {**report([arm]), **v3.report_identity(plan_v3())}
        summary = summarize_v3(data)
        selected = next(item for item in summary["cases"] if item["case"] == case.id)
        self.assertIsNone(selected["baseline"]["statistics"])
        self.assertFalse(summary["evidence_complete"])
        self.assertEqual(summary["acceptance_status"], "inconclusive")
        self.assertTrue(
            any("lifecycle" in error for error in summary["evidence_validation_errors"])
        )

    def test_final_seal_expiry_keeps_eight_explicit_incomplete_rows(self):
        from scripts.benchmark_suite.issue406_run import _expire_fresh_report

        data = {**v3.report_identity(plan_v3()), "assessment": {}, "measurements": []}
        with TemporaryDirectory() as raw:
            _expire_fresh_report(Path(raw), data)
            summary = json.loads((Path(raw) / "summary.json").read_text())
        self.assertEqual(
            {target["row"] for target in summary["targets"]}, set(range(1, 9))
        )
        self.assertEqual(summary["acquisition_status"], "incomplete")
        self.assertEqual(summary["acceptance_status"], "inconclusive")
        self.assertEqual(len(summary["cases"]), 38)


def native_failure_runner(root):
    import time

    from scripts.benchmark_suite.issue406_harness import FRESH_SCHEMA
    from scripts.benchmark_suite.issue406_run import AcceptanceRunner
    from scripts.toolkit import sha256_file, write_json

    plan = plan_v3()
    plan["cpus"] = [0]
    binary = root / "native"
    binary.write_bytes(b"new sealed native executable identity")
    source = root / "build-stream_join_perf.acceptance-harness.json"
    write_json(
        source,
        {
            "schema": FRESH_SCHEMA,
            "product_git_sha": v3.MAIN,
            "harness_git_sha": HARNESS,
            "compatibility": None,
            "helper_reconstruction": None,
            "product_files_verified": True,
        },
    )
    descriptor = {
        "path": str(binary),
        "sha256": sha256_file(binary),
        "build_output": str(root),
        "harness_target": "stream_join_perf",
    }
    plan["native_binaries"] = {"baseline": {"stream_join_perf/enabled": descriptor}}
    plan["sealed_files"] = {str(path): sha256_file(path) for path in (binary, source)}
    runner = AcceptanceRunner(plan, root, time.monotonic())
    group = next(group for group in collection_groups() if group[0].row == 4)
    destination = runner.destination(group, "baseline", 0, None)
    destination.mkdir(parents=True)
    process = {
        "pid": 12345,
        "returncode": 101,
        "binary_sha256": descriptor["sha256"],
        "argv": [str(binary), "--native-component", "join_capture_restore_1m"],
        "cpu_affinity": [0],
        "elapsed_seconds": 1.0,
    }
    write_json(destination / "process.json", process)
    (destination / "stdout.log").write_bytes(b"")
    (destination / "stderr.log").write_text(
        "thread 'main' panicked at crates/calc-flow/benches/support/"
        "native_join_components.rs:999:55:\n"
        "called `Result::unwrap()` on an `Err` value: "
        'Internal { message: "V2 IPC constructor credit admission failed" }\n'
    )
    runner.retired[12345] = {"returncode": 101, "group_empty": True}
    return runner, group, destination


class V3OwnedFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_binary_failure_requires_exact_seal_slot_invocation_and_cleanup(
        self,
    ):
        from scripts.benchmark_suite.issue406_failures import native_failure_signature

        with TemporaryDirectory() as raw:
            runner, group, destination = native_failure_runner(Path(raw))
            error = RuntimeError(f"native component failed; see {destination}")
            result = await v3.recognize_owned_failure(
                runner, group, ("baseline", 0, None), error
            )
            self.assertEqual(result["phase"], "checkpoint-ipc-construction")
            self.assertEqual(result["candidate_product_commit"], PRODUCT)
            self.assertEqual(result["harness_git_sha"], HARNESS)
            self.assertEqual(len(result["files_sha256"]), 4)
            self.assertTrue((destination / "failure-evidence.json").exists())
            process = json.loads((destination / "process.json").read_text())
            diagnostic = (destination / "stderr.log").read_text()
            self.assertIsNone(native_failure_signature(group[0], process, diagnostic))
            for position in (
                ("candidate", 0, None),
                ("baseline", 1, None),
                ("baseline", 0, 0),
            ):
                self.assertIsNone(
                    await v3.recognize_owned_failure(runner, group, position, error)
                )
            runner.retired[12345]["group_empty"] = False
            self.assertIsNone(
                await v3.recognize_owned_failure(
                    runner, group, ("baseline", 0, None), error
                )
            )

    async def test_new_failure_rejects_missing_seals_provenance_and_stdout(self):
        for corruption in ("hash", "invocation", "stdout", "source", "oracle"):
            with self.subTest(corruption=corruption), TemporaryDirectory() as raw:
                runner, group, destination = native_failure_runner(Path(raw))
                if corruption == "hash":
                    runner.plan["sealed_files"] = {}
                elif corruption == "stdout":
                    (destination / "stdout.log").write_text("malformed evidence")
                elif corruption == "invocation":
                    path = destination / "process.json"
                    value = json.loads(path.read_text())
                    value["argv"][-1] = "unreviewed_workload"
                    path.write_text(json.dumps(value))
                elif corruption == "source":
                    runner.plan["harness_git_sha"] = "c" * 40
                    runner.plan["execution_git_sha"] = "c" * 40
                else:
                    (destination / "stderr.log").write_text("assertion failed: oracle")
                error = RuntimeError(f"native component failed; see {destination}")
                self.assertIsNone(
                    await v3.recognize_owned_failure(
                        runner, group, ("baseline", 0, None), error
                    )
                )

    def test_returned_failure_is_preserved_before_metrics_or_trace(self):
        import time

        from scripts.benchmark_suite.issue406_run import AcceptanceRunner

        case = next(
            group[0]
            for group in collection_groups()
            if group[0].row == 7
            and group[0].workload == "asof"
            and group[0].mode == "low"
        )
        plan = {**plan_v3(), "cpus": [0]}
        plan["row7_releases"] = plan["releases"]
        with TemporaryDirectory() as raw:
            runner = AcceptanceRunner(plan, Path(raw), time.monotonic())
            observation = {
                **eof_observation(case),
                "product_commit": v3.MAIN,
                "cpu_affinity": [0],
            }
            with self.assertRaises(v3.FailedLifecycle):
                runner._persist((case,), "baseline", 0, None, observation)
            destination = runner.destination((case,), "baseline", 0, None)
            self.assertEqual(
                json.loads((destination / "observation.json").read_text()), observation
            )
            self.assertFalse((destination / "metrics.json").exists())
            self.assertFalse((destination / "trace-envelope.json").exists())


class V3PlanRoutingTests(unittest.TestCase):
    def test_fresh_builds_use_exact_common_harness_without_legacy_exception(self):
        from unittest.mock import patch

        from scripts.benchmark_suite.issue406_run import validate_plan

        plan = plan_v3()
        with (
            patch(
                "scripts.benchmark_suite.issue406_run._validate_plan_inventory",
                return_value={},
            ),
            patch(
                "scripts.benchmark_suite.issue406_run.competing_work", return_value=[]
            ),
            patch("scripts.benchmark_suite.issue406_fresh_v3.validate_releases"),
            patch(
                "scripts.benchmark_suite.issue406_run.native_harness_reuse"
            ) as legacy,
            patch(
                "scripts.benchmark_suite.issue406_run._validate_fresh_releases"
            ) as v2,
            patch(
                "scripts.benchmark_suite.issue406_run._validate_plan_wheel",
                return_value={"environment": {}, "validated_artifacts": []},
            ) as wheel,
            patch(
                "scripts.benchmark_suite.issue406_run._validate_plan_binaries"
            ) as native,
        ):
            validate_plan(plan)
        legacy.assert_not_called()
        v2.assert_not_called()
        self.assertEqual(
            [call.args[-1] for call in wheel.call_args_list], [HARNESS, HARNESS]
        )
        self.assertEqual(
            [call.args[3] for call in native.call_args_list], [HARNESS, HARNESS]
        )

    def test_missing_seals_stale_execution_and_main_off_are_rejected(self):
        from unittest.mock import patch

        from scripts.benchmark_suite.issue406_run import (
            _validate_plan_binaries,
            _validate_plan_inventory,
        )
        from scripts.toolkit import sha256_file

        cpus = list(range(32))
        with TemporaryDirectory() as raw:
            source = Path(raw) / "source.py"
            source.write_text("sealed harness")
            plan = {
                **plan_v3(),
                "cpus": cpus,
                "sealed_files": {str(source): sha256_file(source)},
            }
            with (
                patch(
                    "scripts.benchmark_suite.issue406_run.os.sched_getaffinity",
                    return_value=set(cpus),
                ),
                patch(
                    "scripts.benchmark_suite.issue406_run._clean_revision",
                    return_value=HARNESS,
                ),
                patch(
                    "scripts.benchmark_suite.issue406_run._plan_python_sources",
                    return_value={str(source)},
                ),
            ):
                _validate_plan_inventory(plan)
                with self.assertRaisesRegex(ValueError, "missing sealed"):
                    _validate_plan_inventory({**plan, "sealed_files": {}})
                altered = {
                    **plan,
                    "harness_git_sha": "c" * 40,
                    "execution_git_sha": "c" * 40,
                }
                with self.assertRaisesRegex(ValueError, "execution harness"):
                    _validate_plan_inventory(altered)
                source.write_text("changed harness")
                with self.assertRaisesRegex(
                    ValueError, "sealed execution input changed"
                ):
                    _validate_plan_inventory(plan)
            plan["native_binaries"] = {"baseline": {"stream_join_perf/off": {}}}
            with self.assertRaisesRegex(ValueError, "inventory"):
                _validate_plan_binaries(plan, "baseline", {}, HARNESS, {})

    def test_v3_attestation_requires_no_reconstruction(self):
        from unittest.mock import patch

        from scripts.benchmark_suite.issue406_run import _validate_source_attestation

        with patch(
            "scripts.benchmark_suite.issue406_run.validate_fresh_acceptance_harness",
            return_value={},
        ) as fresh:
            self.assertEqual(
                _validate_source_attestation(plan_v3(), Path("build"), "target"), {}
            )
            fresh.assert_called_once_with(
                Path("build"), "target", harness_git_sha=HARNESS
            )
            for field in ("compatibility", "reconstruction"):
                with self.assertRaisesRegex(ValueError, "historical"):
                    _validate_source_attestation(
                        plan_v3(), Path("build"), "target", **{field: {}}
                    )


class V3ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_settlement_never_upgrades_exhausted_acquisition(self):
        from unittest.mock import patch

        from scripts.benchmark_suite.issue406_run import AcceptanceRunner, run

        async def collected(runner, output):
            return {
                "state": "incomplete",
                "measurements": [],
                **v3.report_identity(runner.plan),
                "failed_arms": [{"cases": [], "side": "baseline"}],
            }

        with TemporaryDirectory() as raw:
            plan = {**plan_v3(), "cpus": [0]}
            with (
                patch("scripts.benchmark_suite.issue406_run.validate_plan"),
                patch(
                    "scripts.benchmark_suite.issue406_fresh.collect_fresh", collected
                ),
                patch.object(
                    AcceptanceRunner,
                    "resources_guard",
                    lambda runner: runner.stop.wait(),
                ),
            ):
                result = await run(plan, Path(raw))
            self.assertEqual(result["state"], "incomplete")
            self.assertTrue(result["settlement"]["settled"])
            summary = json.loads((Path(raw) / "summary.json").read_text())
            self.assertEqual(summary["acceptance_status"], "inconclusive")
            self.assertEqual(
                {target["row"] for target in summary["targets"]}, set(range(1, 9))
            )


class V3CliTests(unittest.TestCase):
    def test_incomplete_acquisition_has_nonzero_cli_status(self):
        import contextlib
        import io
        from unittest.mock import AsyncMock, patch

        from scripts.benchmark_suite.issue406_run import main
        from scripts.toolkit import sha256_file, write_json

        with TemporaryDirectory() as raw:
            plan = Path(raw) / "plan.json"
            write_json(plan, plan_v3())
            with (
                patch(
                    "sys.argv",
                    [
                        "issue406",
                        str(plan),
                        str(Path(raw) / "output"),
                        "--plan-sha256",
                        sha256_file(plan),
                    ],
                ),
                patch(
                    "scripts.benchmark_suite.issue406_run._run_cli",
                    AsyncMock(
                        return_value={"state": "incomplete", "budget_satisfied": True}
                    ),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main(), 1)


def runtime_failure_runner(root, *, case=None):
    import sys
    from dataclasses import asdict
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from scripts.toolkit import write_json

    runner, _, _ = native_failure_runner(root)
    descriptor = runner.plan["native_binaries"]["baseline"]["stream_join_perf/enabled"]
    ordinary = root / "ordinary-build.json"
    write_json(ordinary, {"revision": v3.MAIN, "native_sha256": descriptor["sha256"]})
    wheel = root / "ordinary.whl"
    wheel.write_bytes(b"sealed ordinary wheel")
    from scripts.toolkit import sha256_file

    runner.plan["releases"]["baseline"].update(
        native_path=descriptor["path"],
        native_sha256=descriptor["sha256"],
        build_metadata=str(ordinary),
        wheel_path=str(wheel),
        wheel_sha256=sha256_file(wheel),
    )
    runner.plan["sealed_files"].update(
        {str(path): sha256_file(path) for path in (ordinary, wheel)}
    )
    case = case or next(
        group[0]
        for group in collection_groups()
        if group[0].row == 3 and group[0].mode == "low"
    )
    worker_root = root / "workers" / case.id / "round-0" / "baseline"
    worker_root.mkdir(parents=True)
    destination = runner.destination((case,), "baseline", 0, None)
    destination.mkdir(parents=True)
    raw = {**eof_observation(case), "product_commit": v3.MAIN, "cpu_affinity": [0]}
    write_json(destination / "observation.json", raw)
    prepared = {
        **asdict(case),
        "family": "join-asof-acceptance",
        "side": "baseline",
        "product_commit": v3.MAIN,
        "fixture_root": str(runner.fixture_root("baseline", case.mode, 0)),
    }
    runner.prepared_cases[(case.id, "baseline", 0)] = prepared
    runner.workers["baseline"] = SimpleNamespace(process=SimpleNamespace(pid=54321))
    write_json(
        worker_root / "invocation.json",
        {
            "pid": 54321,
            "argv": [
                sys.executable,
                "-m",
                "scripts.benchmark_suite",
                "worker",
                "--root",
                str(worker_root),
            ],
            "cpu_affinity": [0],
            "product_commit": v3.MAIN,
            "native_sha256": descriptor["sha256"],
            "harness_git_sha": HARNESS,
        },
    )
    write_json(
        worker_root / "environment.json",
        {
            "cpu_affinity": [0],
            "native_sha256": descriptor["sha256"],
            "tokio_worker_threads": "1",
        },
    )
    (worker_root / "requests.jsonl").write_text(
        "\n".join(
            json.dumps(request)
            for request in (
                {"operation": "hello", "scope": "core"},
                {"operation": "prepare", "case": prepared},
            )
        )
        + "\n"
    )
    (worker_root / "stdout.log").write_text(
        (worker_root / "environment.json").read_text().replace("\n", "")
        + "\n"
        + json.dumps({"case": prepared, "warmup": raw})
        + "\n"
    )
    (worker_root / "stderr.log").write_bytes(b"")

    async def close(side):
        runner.retired[54321] = {"returncode": 0, "group_empty": True}

    runner.close_worker = AsyncMock(side_effect=close)
    return runner, (case,), destination, worker_root


class V3RuntimeFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_eof_failure_requires_bound_response_and_natural_cleanup(self):
        with TemporaryDirectory() as raw:
            runner, group, destination, _ = runtime_failure_runner(Path(raw))
            result = await v3.recognize_owned_failure(
                runner, group, ("baseline", 0, None), v3.FailedLifecycle("lifecycle")
            )
            self.assertEqual(result["signature"], "runtime_eof_shutdown_sentinel")
            self.assertEqual(
                result["natural_exit"], {"returncode": 0, "group_empty": True}
            )
            self.assertEqual(len(result["files_sha256"]), 9)
            runner.close_worker.assert_awaited_once_with("baseline")
            self.assertNotIn("baseline", runner.workers)
            self.assertFalse((destination / "metrics.json").exists())

    async def test_oracle_status_and_invocation_corruption_stop_before_admission(self):
        for corruption in (
            "oracle",
            "status",
            "request",
            "identity",
            "stdout",
            "extra-response",
            "build",
        ):
            with self.subTest(corruption=corruption), TemporaryDirectory() as raw:
                runner, group, destination, worker = runtime_failure_runner(Path(raw))
                if corruption in ("oracle", "status"):
                    path = destination / "observation.json"
                    value = json.loads(path.read_text())
                    if corruption == "oracle":
                        value["oracle"]["passed"] = False
                    else:
                        value["after_cleanup"]["task_errors"] = 1
                    path.write_text(json.dumps(value))
                elif corruption == "request":
                    (worker / "requests.jsonl").write_text("{}\n")
                elif corruption == "identity":
                    (worker / "environment.json").write_text("{}")
                elif corruption == "extra-response":
                    path = worker / "stdout.log"
                    path.write_text(
                        '{"error": "AssertionError: hidden oracle failure"}\n'
                        + path.read_text()
                    )
                elif corruption == "build":
                    Path(
                        runner.plan["releases"]["baseline"]["build_metadata"]
                    ).write_text("{}")
                else:
                    (worker / "stdout.log").write_text("malformed response")
                self.assertIsNone(
                    await v3.recognize_owned_failure(
                        runner,
                        group,
                        ("baseline", 0, None),
                        v3.FailedLifecycle("lifecycle"),
                    )
                )
                runner.close_worker.assert_not_awaited()


def retained_early_failure(rows=1_000_000):
    from scripts.test_issue406_failures import retained_failure

    raw = retained_failure(rows)
    raw.update(
        checkpoint_observations=[
            {
                "before_ns": 1,
                "after_ns": 2,
                "checkpoint": {**raw["after_eof"]["checkpoint"], "elapsed_micros": 0},
            }
        ],
        cleanup_seconds=0.01,
        job_cpu_seconds=0.1,
        job_seconds=0.2,
        sample_wall_seconds=0.3,
    )
    return raw


class V3RetainedSignatureTests(unittest.TestCase):
    def test_exact_early_retained_failure_keeps_cause_unresolved(self):
        for rows in (1_000_000, 4_000_000):
            case = next(
                group[0]
                for group in collection_groups()
                if group[0].row == 3 and group[0].mode == "on" and group[0].rows == rows
            )
            self.assertEqual(
                v3.retained_failure_signature(case, retained_early_failure(rows)),
                "retained_on_checkpoint_failure_cause_unresolved",
            )

    def test_malformed_checkpoint_observations_are_not_workload_failures(self):
        case = next(
            group[0]
            for group in collection_groups()
            if group[0].row == 3 and group[0].mode == "on"
        )
        for observations in (
            [],
            [{"oracle_error": "bad output"}],
            [{"before_ns": 2, "after_ns": 1, "checkpoint": {}}],
        ):
            raw = retained_early_failure()
            raw["checkpoint_observations"] = observations
            self.assertIsNone(v3.retained_failure_signature(case, raw))

    def test_added_oracle_timing_lifecycle_or_other_failure_fields_stop(self):
        case = next(
            group[0]
            for group in collection_groups()
            if group[0].row == 3 and group[0].mode == "on"
        )
        for field in (
            "oracle",
            "oracle_passed",
            "oracle_error",
            "cleanup_error",
            "observer_error",
            "operation_error",
            "lifecycle",
            "processing_seconds",
            "before_eof",
            "terminal_started_ns",
            "fixture",
            "dispatch_trace",
        ):
            with self.subTest(field=field):
                raw = retained_early_failure()
                raw[field] = False
                self.assertIsNone(v3.retained_failure_signature(case, raw))
        for phase, path in (
            ("after_eof", ("checkpoint", "source_acks")),
            ("after_cleanup", ("checkpoint", "phase")),
            ("after_eof", ("sources", "left", "errors")),
            ("after_cleanup", ("sinks", "suite", "errors")),
        ):
            raw = retained_early_failure()
            item = raw[phase]
            for key in path[:-1]:
                item = item[key]
            item[path[-1]] = "changed"
            self.assertIsNone(v3.retained_failure_signature(case, raw))


def retained_traceback(runner, case):
    import ast
    import asyncio
    import inspect

    from scripts.benchmark_suite.process import ROOT
    from scripts.toolkit import sha256_file

    runtime = ROOT / "benchmarks/join_asof_acceptance_runtime.py"
    lifecycle = ROOT / "benchmarks/engine_lifecycle.py"
    entry = ROOT / "scripts/benchmark_suite/__main__.py"
    worker = ROOT / "scripts/benchmark_suite/worker.py"
    base = Path(inspect.getsourcefile(asyncio.BaseEventLoop.run_until_complete))
    cause = [
        (runtime, "measure_acceptance_job"),
        (lifecycle, "run_with_completion"),
        (lifecycle, "_require_completed"),
    ]
    raised = [
        ("<frozen runpy>", "_run_module_as_main"),
        ("<frozen runpy>", "_run_code"),
        (entry, "<module>"),
        (entry, "main"),
        (worker, "main"),
        (worker, "dispatch"),
        (worker, "prepare_case"),
        (runtime, "sample"),
        (base, "run_until_complete"),
        (runtime, "_sample"),
        (runtime, "measure_acceptance_job"),
        (runtime, "_require_acceptance_outcome"),
    ]

    def frame(path, function):
        line = 1
        if isinstance(path, Path):
            if function != "<module>":
                node = next(
                    node
                    for node in ast.walk(ast.parse(path.read_text()))
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == function
                )
                line = node.lineno
            if path.is_relative_to(ROOT):
                runner.plan["sealed_files"][str(path)] = sha256_file(path)
        return f'  File "{path}", line {line}, in {function}\n'

    error = retained_early_failure(case.rows)["terminal_error"]
    return (
        "Traceback (most recent call last):\n"
        + "".join(frame(*item) for item in cause)
        + f"RuntimeError: {error}\n\n"
        + "The above exception was the direct cause of the following exception:\n\n"
        + "Traceback (most recent call last):\n"
        + "".join(frame(*item) for item in raised)
        + "benchmarks.checkpoint_cycles.CycleValidationError: "
        "invalid acceptance sample\n"
    )


class V3RetainedTracebackTests(unittest.TestCase):
    def test_precise_chain_and_current_harness_frames_are_required(self):
        with TemporaryDirectory() as raw:
            runner, _, _ = native_failure_runner(Path(raw))
            case = next(
                group[0]
                for group in collection_groups()
                if group[0].row == 3 and group[0].mode == "on"
            )
            diagnostic = retained_traceback(runner, case)
            self.assertTrue(v3.retained_traceback_signature(runner, case, diagnostic))
            for altered in (
                diagnostic + "AssertionError: oracle failed\n",
                diagnostic.replace(
                    "RuntimeError: stream failed", "TimeoutError: stream failed"
                ),
                diagnostic.replace("in _require_completed", "in _require_terminal"),
                diagnostic.replace(
                    "The above exception was the direct cause",
                    "During handling of the above exception",
                ),
                "native panic output\n" + diagnostic,
            ):
                self.assertFalse(v3.retained_traceback_signature(runner, case, altered))
            runner.plan["sealed_files"] = {}
            self.assertFalse(v3.retained_traceback_signature(runner, case, diagnostic))


def retained_owned_runner(root):
    from unittest.mock import AsyncMock

    from scripts.toolkit import write_json

    case = next(
        group[0]
        for group in collection_groups()
        if group[0].row == 3 and group[0].mode == "on"
    )
    runner, group, destination, worker = runtime_failure_runner(root, case=case)
    (destination / "observation.json").unlink()
    write_json(worker / "failed-sample-1.json", retained_early_failure(case.rows))
    hello = (worker / "stdout.log").read_text().splitlines()[0]
    (worker / "stdout.log").write_text(
        hello + '\n{"error": "CycleValidationError: invalid acceptance sample"}\n'
    )
    (worker / "stderr.log").write_text(retained_traceback(runner, case))

    async def close(side, *, allow_failure):
        runner.retired[54321] = {"returncode": 1, "group_empty": True}

    runner.close_worker = AsyncMock(side_effect=close)
    return runner, group, destination, worker


class V3RetainedOwnedTests(unittest.IsolatedAsyncioTestCase):
    async def test_raised_retained_failure_has_fresh_evidence_without_timing(self):
        with TemporaryDirectory() as raw:
            runner, group, destination, _ = retained_owned_runner(Path(raw))
            error = RuntimeError(
                "invalid worker result: {'error': 'CycleValidationError: "
                "invalid acceptance sample'}"
            )
            result = await v3.recognize_owned_failure(
                runner, group, ("baseline", 0, None), error
            )
            self.assertEqual(
                result["signature"], "retained_on_checkpoint_failure_cause_unresolved"
            )
            self.assertEqual(
                result["natural_exit"], {"returncode": 1, "group_empty": True}
            )
            self.assertEqual(len(result["files_sha256"]), 9)
            runner.close_worker.assert_awaited_once_with("baseline", allow_failure=True)
            self.assertTrue((destination / "failure-evidence.json").exists())
            self.assertFalse((destination / "metrics.json").exists())

    async def test_retained_oracle_contamination_and_altered_chain_stop(self):
        for corruption in ("oracle", "traceback", "exit"):
            with self.subTest(corruption=corruption), TemporaryDirectory() as raw:
                runner, group, _, worker = retained_owned_runner(Path(raw))
                if corruption == "oracle":
                    path = worker / "failed-sample-1.json"
                    value = json.loads(path.read_text())
                    value["oracle_passed"] = False
                    path.write_text(json.dumps(value))
                elif corruption == "traceback":
                    path = worker / "stderr.log"
                    path.write_text(
                        path.read_text().replace(
                            "RuntimeError: stream failed",
                            "AssertionError: stream failed",
                        )
                    )
                else:

                    async def close(side, *, allow_failure, owned=runner):
                        owned.retired[54321] = {"returncode": -15, "group_empty": True}

                    runner.close_worker.side_effect = close
                error = RuntimeError(
                    "invalid worker result: {'error': 'CycleValidationError: "
                    "invalid acceptance sample'}"
                )
                self.assertIsNone(
                    await v3.recognize_owned_failure(
                        runner, group, ("baseline", 0, None), error
                    )
                )


class V3ReadCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_v3_wire_read_settles_its_owned_process(self):
        import asyncio
        import sys
        import time

        from scripts.benchmark_suite.issue406_run import AcceptanceRunner
        from scripts.benchmark_suite.process import Worker

        with TemporaryDirectory() as raw:
            root = Path(raw)
            runner = AcceptanceRunner(plan_v3(), root, time.monotonic())
            with (root / "stderr.log").open("wb") as log:
                process = await runner.register_process(
                    asyncio.create_task(
                        asyncio.create_subprocess_exec(
                            sys.executable,
                            "-c",
                            "import time; time.sleep(60)",
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=log,
                            start_new_session=True,
                        )
                    )
                )
                worker = Worker(process, log)
                runner.workers["baseline"] = worker
                read = asyncio.create_task(
                    runner._worker_request(worker, operation="hello", scope="core")
                )
                try:
                    await asyncio.sleep(0.01)
                    read.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await read
                finally:
                    await runner.settle()
                self.assertIsNotNone(process.returncode)
                self.assertFalse(runner.group_members())
                self.assertFalse(runner.processes)
