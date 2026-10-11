"""Exact observable main failures that exhaust an arm without a timing result."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

from scripts.benchmark_suite.issue406_fresh import MAIN, SCHEMA

MAIN_WHEEL_NATIVE = "83c1984005258b702fabbb322d27bb9a0bf7842823d7ddabc4e9660b40997d5f"
MAIN_NATIVE = "7fc3d88d8a8dd888c5cb4bcaf9b5844f4527452d94e723e09604c53e380ad055"
MAIN_COMPONENT_SOURCE = (
    "e56e9b0082f46d029fc4cede6a41b33f90620f77fdf44feb216a396c7e98e324"
)
JOIN_NODES = {
    1_000_000: "cf_stream_join_fae6be855b5bef51",
    4_000_000: "cf_stream_join_70034c097d851384",
}
CHECKPOINT = {
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


def _contains(observed: dict, expected: dict) -> bool:
    return isinstance(observed, dict) and all(
        key in observed
        and type(observed[key]) is type(value)
        and observed[key] == value
        for key, value in expected.items()
    )


def main_failure_identity(plan: dict, case, side: str, round_index: int, index) -> bool:
    """Require the pinned main product and a first, applicable warmup attempt."""
    release = plan.get("releases", {}).get("baseline", {})
    identity = plan.get("known_main_failure_identity", {})
    return (
        _first_main_warmup(plan, side, round_index, index)
        and _main_release_identity(release, identity)
        and case.threads == 1
        and (_retained_case(case) or _native_case(case))
    )


def _first_main_warmup(plan, side, round_index, index) -> bool:
    return (
        plan.get("schema") == SCHEMA
        and side == "baseline"
        and round_index == 0
        and index is None
    )


def _main_release_identity(release, identity) -> bool:
    return (
        release.get("product_commit") == MAIN
        and identity.get("product_commit") == MAIN
        and release.get("native_sha256") == identity.get("native_sha256")
        and isinstance(identity.get("native_sha256"), str)
    )


def _retained_case(case) -> bool:
    return (
        case.row == 3
        and case.workload == "join_retained"
        and case.mode == "on"
        and case.rows in JOIN_NODES
        and case.batch_rows == 64_000
    )


def _native_case(case) -> bool:
    return (
        case.row in (4, 5)
        and case.mode == "enabled"
        and case.rows == 1_000_000
        and case.batch_rows == (64_000 if case.row == 4 else 500)
    )


def validate_main_failure_identity(plan: dict, sealed: dict) -> None:
    """Bind the guarded main artifact to the exact separately sealed review record."""
    from scripts.benchmark_suite.issue406_reuse import ordinary_audit_paths
    from scripts.toolkit import sha256_file

    reference = plan["known_main_failure_identity_file"]
    path = Path(reference["path"])
    identity = plan["known_main_failure_identity"]
    release = plan["releases"]["baseline"]
    expected = {
        "product_commit": MAIN,
        "native_sha256": release["native_sha256"],
        "wheel_sha256": release["wheel_sha256"],
        "build_metadata_sha256": sealed.get(release["build_metadata"]),
        "ordinary_audit_sha256": sealed.get(
            str(ordinary_audit_paths(release)["audit_path"])
        ),
    }
    if (
        reference["sha256"] != sealed.get(str(path))
        or reference["sha256"] != sha256_file(path)
        or identity != json.loads(path.read_text())
        or identity != expected
        or any(not isinstance(value, str) for value in expected.values())
    ):
        raise ValueError("known main failure artifact identity differs from its review")


def _terminal_error(node: str) -> str:
    return (
        "stream failed: (StreamingError(category='operator', reason_code=None, "
        f"message='operator \"{node}\" execution failed', job_id=1, epoch=None, "
        "checkpoint_phase=None, component_kind='operator', "
        f"component_id='{node}', diagnostic_id=None, position=0), "
        "StreamingError(category='connector', reason_code=None, "
        "message='source \"left\" callback failed', job_id=1, epoch=None, "
        "checkpoint_phase=None, component_kind='source', component_id='left', "
        "diagnostic_id=None, position=1))"
    )


def _retained_status(status: dict, node: str) -> bool:
    if not _contains(
        status,
        {
            "state": "failed",
            "terminal_cause": "failure",
            "task_errors": 2,
            "task_count": 0,
            "job_id": 1,
            "metrics_overflowed": False,
        },
    ) or not _contains(status.get("checkpoint"), CHECKPOINT):
        return False
    if not _retained_entry_errors(status, node):
        return False
    source = {
        "ended": False,
        "delivery": "lossless",
        "replay_positioning": "exact_pause_report_and_seek",
    }
    if not all(
        _contains(status["sources"][name], source) for name in ("left", "right")
    ) or not _contains(
        status["sources"]["right"], {"data_rows": 64, "data_batches": 1}
    ):
        return False
    return _retained_joins(status, node)


def _retained_entry_errors(status: dict, node: str) -> bool:
    for category, errors in (
        ("operators", {node: 1, "result": 0}),
        ("sources", {"left": 1, "right": 0}),
        ("sinks", {"suite": 0}),
    ):
        entries = status.get(category, {})
        if (
            not isinstance(entries, dict)
            or set(entries) != set(errors)
            or any(
                not _contains(entries[name], {"errors": count})
                for name, count in errors.items()
            )
        ):
            return False
    return True


def _retained_joins(status: dict, node: str) -> bool:
    joins = status.get("stream_joins", {})
    return (
        isinstance(joins, dict)
        and set(joins) == {node}
        and _contains(
            joins[node], {"match_limit_failures": 0, "state_limit_failures": 0}
        )
        and _contains(
            joins[node].get("right"),
            {
                "retained_rows": 64,
                "watermark_micros": None,
                "ended": False,
                "idle": False,
            },
        )
        and status.get("stream_asof_joins", {}) == {}
    )


def retained_failure_signature(case, observation: dict) -> str | None:
    """Recognize the recorded signature without inferring its underlying cause."""
    node = JOIN_NODES.get(case.rows)
    if _unsupported_retained_case(case, node):
        return None
    if not _contains(
        observation,
        {
            "cleanup_complete": True,
            "failure": "invalid acceptance sample",
            "terminal_error": _terminal_error(node),
        },
    ):
        return None
    if all(
        _retained_status(observation.get(phase), node)
        for phase in ("after_eof", "after_cleanup")
    ):
        return "retained_on_checkpoint_failure_cause_unresolved"
    return None


def _unsupported_retained_case(case, node) -> bool:
    return (
        node is None
        or case.row != 3
        or case.workload != "join_retained"
        or case.mode != "on"
        or case.threads != 1
        or case.batch_rows != 64_000
    )


def native_failure_signature(case, process: dict, diagnostic: str) -> str | None:
    """Recognize only the two observed old-binary IPC constructor panics."""
    if (
        case.row not in (4, 5)
        or case.mode != "enabled"
        or not _contains(process, {"returncode": 101, "binary_sha256": MAIN_NATIVE})
    ):
        return None
    location = "241:55" if case.row == 4 else "312:58"
    pattern = (
        r"thread 'main' panicked at "
        rf"crates/calc-flow/benches/support/native_join_components\.rs:{location}:\n"
        r"called `Result::unwrap\(\)` on an `Err` value: "
        r'Internal \{ message: "V2 IPC constructor credit admission failed" \}'
        r"(?:\nnote: run with `RUST_BACKTRACE=1` environment variable "
        r"to display a backtrace)?"
    )
    return (
        "native_checkpoint_credit_admission"
        if re.fullmatch(pattern, diagnostic.strip())
        else None
    )


async def recognize_owned_failure(runner, group, position, error):
    """Continue only after exact evidence and a naturally reaped empty group."""
    side, round_index, index = position
    case = group[0]
    if (
        not main_failure_identity(runner.plan, case, side, round_index, index)
        or type(error) is not RuntimeError
    ):
        return None
    if case.row == 3:
        return await _retained_owned_failure(runner, case, error)
    return _native_owned_failure(runner, group, error)


async def _retained_owned_failure(runner, case, error):
    side, round_index = "baseline", 0
    root = runner.output / "workers" / case.id / "round-0" / "baseline"
    expected = {
        **asdict(case),
        "family": "join-asof-acceptance",
        "side": side,
        "product_commit": MAIN,
        "fixture_root": str(runner.fixture_root(side, case.mode, round_index)),
    }
    if (
        runner.prepared_cases.get((case.id, side, round_index)) != expected
        or side not in runner.workers
    ):
        return None
    if str(error) not in (
        "invalid worker result: {'error': 'CycleValidationError: "
        "invalid acceptance sample'}",
        f"benchmark worker exited; see {root / 'stderr.log'}",
    ):
        return None
    evidence = _retained_worker_evidence(runner, case, root)
    if evidence is None:
        return None
    path, signature = evidence
    pid = runner.workers[side].process.pid
    remaining = min(
        10,
        runner.started + runner.plan["collection_cutoff_seconds"] - time.monotonic(),
    )
    async with asyncio.timeout(remaining):
        await runner.close_worker(side, allow_failure=True)
    # The worker writes its traceback before its natural exit.
    diagnostic = (root / "stderr.log").read_text()
    if (
        not diagnostic.rstrip().endswith(
            "benchmarks.checkpoint_cycles.CycleValidationError: "
            "invalid acceptance sample"
        )
        or runner.retired[pid]["returncode"] != 1
    ):
        return None
    runner.workers.pop(side)
    paths = (path, root / "environment.json", root / "stderr.log")
    return _failure_evidence(runner, signature, pid, paths)


def _retained_worker_evidence(runner, case, root):
    path = root / "failed-sample-1.json"
    try:
        signature = retained_failure_signature(case, json.loads(path.read_text()))
        identity = json.loads((root / "environment.json").read_text())
        valid = _contains(
            identity,
            {
                "native_sha256": runner.plan["known_main_failure_identity"][
                    "native_sha256"
                ],
                "cpu_affinity": runner.cpus(case),
                "tokio_worker_threads": "1",
            },
        )
        if not valid or signature is None:
            return None
    except (OSError, ValueError, TypeError):
        return None
    return path, signature


def _native_owned_failure(runner, group, error):
    side, round_index, index = "baseline", 0, None
    case = group[0]
    root = runner.destination(group, side, round_index, index)
    if str(error) != f"native component failed; see {root}":
        return None
    try:
        descriptor = runner.plan["native_binaries"][side]["stream_join_perf/enabled"]
        process = json.loads((root / "process.json").read_text())
        signature = native_failure_signature(
            case, process, (root / "stderr.log").read_text()
        )
        source_path = (
            Path(descriptor["build_output"])
            / f"build-{descriptor['harness_target']}.acceptance-harness.json"
        )
        source = json.loads(source_path.read_text())
        expected_argv = [
            descriptor["path"],
            "--native-component",
            "join_capture_restore_1m" if case.row == 4 else "join_compaction_1m",
        ]
        pid = process["pid"]
        if (
            signature is None
            or not _native_process_identity(
                descriptor, process, expected_argv, runner, case
            )
            or not _native_source_identity(source)
            or not _native_exit_evidence(runner, pid, root)
        ):
            return None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    paths = (
        root / "process.json",
        root / "stderr.log",
        root / "stdout.log",
        source_path,
    )
    return _failure_evidence(runner, signature, pid, paths)


def _native_process_identity(descriptor, process, expected_argv, runner, case) -> bool:
    return (
        descriptor["sha256"] == MAIN_NATIVE
        and process["argv"] == expected_argv
        and process["cpu_affinity"] == runner.cpus(case)
    )


def _native_source_identity(source) -> bool:
    return (
        source["product_git_sha"] == MAIN
        and source["effective_files"].get(
            "crates/calc-flow/benches/support/native_join_components.rs"
        )
        == MAIN_COMPONENT_SOURCE
    )


def _native_exit_evidence(runner, pid, root) -> bool:
    return (
        runner.retired.get(pid) == {"returncode": 101, "group_empty": True}
        and not (root / "stdout.log").read_bytes()
    )


def _failure_evidence(runner, signature, pid, paths):
    from scripts.toolkit import sha256_file

    return {
        "signature": signature,
        "product_commit": MAIN,
        "natural_exit": runner.retired[pid],
        "files_sha256": {str(path): sha256_file(path) for path in paths},
    }
