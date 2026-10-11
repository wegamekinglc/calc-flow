from __future__ import annotations

import copy
import unittest

from scripts.benchmark_suite.issue406_failures import (
    MAIN_NATIVE,
    MAIN_WHEEL_NATIVE,
    main_failure_identity,
    native_failure_signature,
    retained_failure_signature,
)
from scripts.test_issue406_report import selected


def retained_failure(rows=1_000_000):
    node = (
        "cf_stream_join_fae6be855b5bef51"
        if rows == 1_000_000
        else "cf_stream_join_70034c097d851384"
    )
    checkpoint = {
        "current_epoch": 1,
        "expected_operators": 2,
        "expected_sink_commits": 1,
        "expected_sink_precommits": 1,
        "expected_sources": 2,
        "failure_category": "internal",
        "installed_unknown_epoch": None,
        "last_completed_epoch": None,
        "operator_acks": 0,
        "phase": "sources_cut",
        "runtime_config_changed": False,
        "sink_commit_acks": 0,
        "sink_precommit_acks": 0,
        "source_acks": 2,
        "terminal": False,
    }
    status = {
        "checkpoint": checkpoint,
        "state": "failed",
        "terminal_cause": "failure",
        "task_count": 0,
        "task_errors": 2,
        "job_id": 1,
        "metrics_overflowed": False,
        "operators": {node: {"errors": 1}, "result": {"errors": 0}},
        "sources": {
            name: {
                "errors": int(name == "left"),
                "ended": False,
                "delivery": "lossless",
                "replay_positioning": "exact_pause_report_and_seek",
                **({"data_rows": 64, "data_batches": 1} if name == "right" else {}),
            }
            for name in ("left", "right")
        },
        "sinks": {"suite": {"errors": 0}},
        "stream_joins": {
            node: {
                "match_limit_failures": 0,
                "state_limit_failures": 0,
                "right": {
                    "retained_rows": 64,
                    "watermark_micros": None,
                    "ended": False,
                    "idle": False,
                },
            }
        },
    }
    terminal = (
        "stream failed: (StreamingError(category='operator', reason_code=None, "
        f"message='operator \"{node}\" execution failed', job_id=1, epoch=None, "
        "checkpoint_phase=None, component_kind='operator', "
        f"component_id='{node}', diagnostic_id=None, position=0), "
        "StreamingError(category='connector', reason_code=None, "
        "message='source \"left\" callback failed', job_id=1, epoch=None, "
        "checkpoint_phase=None, component_kind='source', component_id='left', "
        "diagnostic_id=None, position=1))"
    )
    return {
        "after_eof": status,
        "after_cleanup": copy.deepcopy(status),
        "cleanup_complete": True,
        "failure": "invalid acceptance sample",
        "terminal_error": terminal,
    }


class MainFailureSignatureTests(unittest.TestCase):
    def test_new_main_identity_requires_exact_reviewed_native_and_audit_binding(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from scripts.benchmark_suite.issue406_failures import (
            validate_main_failure_identity,
        )
        from scripts.test_issue406_fresh import fresh_plan
        from scripts.toolkit import sha256_file, write_json

        with TemporaryDirectory() as raw:
            plan = fresh_plan()
            release = plan["releases"]["baseline"]
            release.update(
                native_sha256="1" * 64,
                wheel_sha256="2" * 64,
                build_metadata=str(Path(raw) / "build.json"),
            )
            identity = {
                "product_commit": release["product_commit"],
                "native_sha256": release["native_sha256"],
                "wheel_sha256": release["wheel_sha256"],
                "build_metadata_sha256": "3" * 64,
                "ordinary_audit_sha256": "4" * 64,
            }
            path = Path(raw) / "known-main-failure-identity.json"
            write_json(path, identity)
            plan["known_main_failure_identity"] = identity
            plan["known_main_failure_identity_file"] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            sealed = {
                str(path): sha256_file(path),
                release["build_metadata"]: "3" * 64,
                str(Path(raw) / "wheel-audit.json"): "4" * 64,
            }
            validate_main_failure_identity(plan, sealed)
            case = selected(3, mode="on")
            self.assertTrue(main_failure_identity(plan, case, "baseline", 0, None))
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_main_failure_identity(
                    plan, {**sealed, release["build_metadata"]: "5" * 64}
                )
            plan["known_main_failure_identity"] = {
                **identity,
                "native_sha256": MAIN_WHEEL_NATIVE,
            }
            self.assertFalse(main_failure_identity(plan, case, "baseline", 0, None))
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_main_failure_identity(plan, sealed)

    def test_retained_signature_preserves_unresolved_cause_for_both_known_fixtures(
        self,
    ):
        for rows in (1_000_000, 4_000_000):
            case = selected(3, mode="on", rows=rows)
            observed = retained_failure_signature(case, retained_failure(rows))
            self.assertEqual(
                observed, "retained_on_checkpoint_failure_cause_unresolved"
            )
            varied = retained_failure(rows)
            varied["after_eof"]["checkpoint"]["elapsed_micros"] = 987654
            self.assertEqual(retained_failure_signature(case, varied), observed)

    def test_differing_checkpoint_diagnostic_identity_or_cleanup_is_not_allowlisted(
        self,
    ):
        case = selected(3, mode="on")
        mutations = [
            ("after_eof.checkpoint.phase", "operator_snapshots"),
            ("after_eof.checkpoint.source_acks", 1),
            ("after_eof.checkpoint.operator_acks", 1),
            ("after_eof.checkpoint.sink_precommit_acks", 1),
            ("after_eof.checkpoint.current_epoch", 2),
            ("after_eof.checkpoint.terminal", True),
            ("after_eof.checkpoint.failure_category", "capacity"),
            ("after_cleanup.checkpoint.sink_commit_acks", 1),
            ("after_cleanup.task_count", 1),
            ("after_cleanup.task_errors", 3),
            ("after_eof.sources.right.errors", 1),
            ("after_eof.sources.right.ended", True),
            ("after_eof.sources.right.replay_positioning", "none"),
            ("after_eof.sinks.suite.errors", 1),
            ("after_eof.operators.result.errors", 1),
            ("cleanup_complete", False),
            ("terminal_error", "operator execution failed"),
        ]
        for path, value in mutations:
            observed = retained_failure()
            target = observed
            parts = path.split(".")
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
            with self.subTest(path=path):
                self.assertIsNone(retained_failure_signature(case, observed))
        for other in (
            selected(3, mode="off"),
            selected(3, mode="on", rows=4_000_000),
            selected(1, mode="on"),
        ):
            self.assertIsNone(retained_failure_signature(other, retained_failure()))

    def test_guard_binds_exact_main_release_first_warmup_and_applicable_case(self):
        from scripts.test_issue406_fresh import fresh_plan

        plan = fresh_plan()
        plan["releases"]["baseline"]["native_sha256"] = MAIN_WHEEL_NATIVE
        case = selected(3, mode="on")
        self.assertTrue(main_failure_identity(plan, case, "baseline", 0, None))
        for side, round_index, index in (
            ("candidate", 0, None),
            ("baseline", 1, None),
            ("baseline", 0, 0),
        ):
            self.assertFalse(
                main_failure_identity(plan, case, side, round_index, index)
            )
        plan["releases"]["baseline"]["native_sha256"] = "0" * 64
        self.assertFalse(main_failure_identity(plan, case, "baseline", 0, None))

    def test_native_credit_panic_is_precise_and_bound_to_old_component_callsites(self):
        for row, line, column in ((4, 241, 55), (5, 312, 58)):
            case = selected(row, mode="enabled")
            diagnostic = (
                "\nthread 'main' panicked at "
                f"crates/calc-flow/benches/support/native_join_components.rs:{line}:{column}:\n"
                "called `Result::unwrap()` on an `Err` value: "
                'Internal { message: "V2 IPC constructor credit admission failed" }\n'
                "note: run with `RUST_BACKTRACE=1` environment variable "
                "to display a backtrace\n"
            )
            process = {"returncode": 101, "binary_sha256": MAIN_NATIVE}
            self.assertEqual(
                native_failure_signature(case, process, diagnostic),
                "native_checkpoint_credit_admission",
            )
            for altered in (
                diagnostic.replace(f":{line}:", ":999:"),
                diagnostic.replace("credit admission", "another failure"),
                diagnostic + "unexpected error\n",
            ):
                self.assertIsNone(native_failure_signature(case, process, altered))
            self.assertIsNone(
                native_failure_signature(
                    case, {**process, "returncode": -9}, diagnostic
                )
            )
            self.assertIsNone(
                native_failure_signature(
                    case, {**process, "binary_sha256": "0" * 64}, diagnostic
                )
            )
