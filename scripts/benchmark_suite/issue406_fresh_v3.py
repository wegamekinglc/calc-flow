"""Exact identity and incomplete-evidence policy for fresh Issue 406 acquisition."""

from __future__ import annotations

import re
from dataclasses import replace

from scripts.benchmark_suite.issue406_fresh import CANDIDATE, MAIN
from scripts.benchmark_suite.report import ROUNDS, SAMPLES

SCHEMA = "calc-flow.issue406-execution.v3"


def validate_envelope(plan: dict) -> None:
    """Bind the fixed v2 budget/inventory to externally sealed final revisions."""
    from scripts.benchmark_suite.issue406_fresh import SCHEMA as V2_SCHEMA
    from scripts.benchmark_suite.issue406_fresh import validate_envelope as validate_v2

    candidate = plan.get("candidate_product_commit")
    harness = plan.get("harness_git_sha")
    if not all(_revision(value) for value in (candidate, harness)):
        raise ValueError("v3 requires full candidate and harness revisions")
    if plan.get("schema") != SCHEMA or plan.get("execution_git_sha") != harness:
        raise ValueError("v3 execution and harness revisions differ")
    if plan.get("releases", {}).get("candidate", {}).get("product_commit") != candidate:
        raise ValueError("v3 candidate differs from the externally sealed revision")
    _reject_historical_policy(plan)
    validate_v2(
        {
            **plan,
            "schema": V2_SCHEMA,
            "releases": {
                **plan.get("releases", {}),
                "candidate": {"product_commit": CANDIDATE},
            },
        }
    )


def _revision(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None


def _reject_historical_policy(plan):
    historical = {
        "native_harness_reuse",
        "row7_harness_reuse",
        "historical_harness_witness",
        "known_main_failure_identity",
        "known_main_failure_identity_file",
    }
    if historical.intersection(plan):
        raise ValueError("v3 cannot use historical compatibility or failure exceptions")


def report_identity(plan: dict) -> dict:
    """Copy trusted plan identity; observations never select the candidate."""
    return {
        "execution_schema": SCHEMA,
        "candidate_product_commit": plan["candidate_product_commit"],
        "harness_git_sha": plan["harness_git_sha"],
    }


def candidate_commit(report: dict) -> str:
    """Resolve a report's separately sealed candidate, preserving v2 identity."""
    if report.get("execution_schema") != SCHEMA:
        return CANDIDATE
    candidate = report.get("candidate_product_commit")
    if not _revision(candidate):
        raise ValueError("missing plan-bound v3 candidate revision")
    return candidate


def validate_releases(plan: dict, files: dict) -> None:
    """Retain ordinary source, compiler, dependency and installed-member audits."""
    import json
    from pathlib import Path

    from scripts.benchmark_suite.issue406_reuse import validate_ordinary_release

    releases = [*plan["releases"].values(), plan["retained_control"]]
    builds = [validate_ordinary_release(release, files) for release in releases]
    if (
        any(build != builds[0] for build in builds[1:])
        or builds[0]["rustc"] != plan["expected_native_rustc"]
    ):
        raise ValueError(
            "ordinary release dependencies, compiler or build settings differ"
        )
    candidate = json.loads(
        Path(plan["releases"]["candidate"]["build_metadata"]).read_text()
    )
    if candidate.get("harness_git_sha") != plan["harness_git_sha"]:
        raise ValueError("fresh candidate wheel was not built by the sealed harness")


class FailedLifecycle(ValueError):
    """Keep returned failed observations outside the valid timing ledger."""


def require_completed(samples: dict) -> None:
    """Reject failed/missing lifecycle evidence before recording valid samples."""
    if any(sample.get("lifecycle") != "completed" for sample in samples.values()):
        raise FailedLifecycle("v3 observation has invalid lifecycle evidence")


def validate_complete_results(report: dict) -> None:
    """A full count never legitimizes failed lifecycle observations."""
    from scripts.benchmark_suite.issue406_acceptance import (
        validate_complete_results as validate,
    )
    from scripts.benchmark_suite.issue406_fresh import validate_retained_control

    records = [
        *report.get("measurements", []),
        *report.get("retained_control", {}).get("measurements", []),
    ]
    for record in records:
        for sample in [
            *record.get("warmups", []),
            *(sample for group in record.get("rounds", []) for sample in group),
        ]:
            require_completed({"sample": sample})
    if any(
        report.get(field)
        for field in ("failed_attempts", "failed_arms", "dependent_unattempted")
    ):
        raise ValueError("v3 acquisition is incomplete after a failed arm")
    validate(report)
    validate_retained_control(report)


def replay_source(case) -> str | None:
    """Name the exact E2E arm required by a row-7 native dispatch replay."""
    return (
        replace(case, workload="asof").id
        if case.row == 7 and case.workload == "asof_operator"
        else None
    )


def unattempted_dependency(group, side, source):
    return {
        "cases": [case.id for case in group],
        "side": side,
        "source_case": source,
        "reason": "source-observation-failed",
        "slots": [[r, i] for r in range(ROUNDS) for i in (None, *range(SAMPLES))],
    }


EOF_REASON = "terminal status contains a checkpoint failure"
IPC_REASON = "V2 IPC constructor credit admission failed"


def _contains(observed, expected) -> bool:
    return isinstance(observed, dict) and all(
        key in observed
        and type(observed[key]) is type(value)
        and observed[key] == value
        for key, value in expected.items()
    )


def runtime_failure_signature(case, raw: dict) -> str | None:
    """Accept only fully correct output with the terminal shutdown-sentinel status."""
    if not _runtime_failure_case(case):
        return None
    expected = {
        "lifecycle": "failed",
        "lifecycle_error": EOF_REASON,
        "terminal_error": EOF_REASON,
        "cleanup_complete": True,
        "oracle_passed": True,
        "mode": case.mode,
    }
    oracle = {"passed": True, "rows": case.rows, "bitmap_bytes": case.rows}
    if not _contains(raw, expected) or not _contains(raw.get("oracle"), oracle):
        return None
    if not _exact_oracle(raw) or not _before_eof(raw.get("before_eof")):
        return None
    return (
        "runtime_eof_shutdown_sentinel"
        if all(
            _terminal_sentinel(case, raw.get(phase))
            for phase in ("after_eof", "after_cleanup")
        )
        else None
    )


def _exact_oracle(raw):
    error_fields = {"failure", "cleanup_error", "observer_error", "operation_error"}
    difference = raw["oracle"].get("max_abs_error")
    return (
        not error_fields.intersection(raw)
        and type(difference) in (int, float)
        and difference == 0
    )


def _runtime_failure_case(case):
    return (
        case.workload in ("join_sealed", "join_retained", "asof", "projection")
        and case.mode != "off"
    )


def _before_eof(status) -> bool:
    return _contains(
        status, {"state": "running", "task_errors": 0, "metrics_overflowed": False}
    ) and _contains(status.get("checkpoint"), {"failure_category": None})


def _terminal_sentinel(case, status) -> bool:
    if not _contains(
        status,
        {
            "state": "completed",
            "terminal_cause": "natural_end",
            "task_count": 0,
            "task_errors": 0,
            "metrics_overflowed": False,
        },
    ):
        return False
    checkpoint = {
        "current_epoch": None,
        "elapsed_micros": None,
        "phase": None,
        "failure_category": "internal",
        "installed_unknown_epoch": None,
        "runtime_config_changed": False,
        "terminal": False,
        "source_acks": 0,
        "operator_acks": 0,
        "sink_commit_acks": 0,
        "sink_precommit_acks": 0,
        "expected_sink_commits": 1,
        "expected_sink_precommits": 1,
        "expected_sources": 1 if case.workload == "projection" else 2,
        "expected_operators": 1 if case.workload == "projection" else 2,
    }
    if not _contains(status.get("checkpoint"), checkpoint):
        return False
    epoch = status["checkpoint"].get("last_completed_epoch")
    if type(epoch) is not int or epoch < 1:
        return False
    return _terminal_components(case, status) and _no_join_limits(status)


def _terminal_components(case, status) -> bool:
    sources = {
        "projection": {"input"},
        "asof": {"reference.input", "quotes.input"},
    }.get(case.workload, {"left", "right"})
    if set(status.get("sources", {})) != sources or set(status.get("sinks", {})) != {
        "suite"
    }:
        return False
    if len(status.get("operators", {})) != (1 if case.workload == "projection" else 2):
        return False
    return _ended_components(status) and _contains(
        status["sinks"]["suite"], {"delivered_rows": case.rows}
    )


def _ended_components(status):
    return all(
        _contains(value, {"ended": True, "errors": 0})
        for key in ("sources", "operators", "sinks")
        for value in status[key].values()
    )


def _no_join_limits(status) -> bool:
    states = [
        value
        for key in ("stream_joins", "stream_asof_joins")
        for value in status.get(key, {}).values()
    ]
    return all(
        value.get(field, 0) == 0
        for value in states
        for field in ("match_limit_failures", "state_limit_failures")
    )


def native_failure_signature(case, returncode, diagnostic: str) -> str | None:
    """Accept only the IPC-constructor panic during the two checkpoint workloads."""
    if (
        case.row not in (4, 5)
        or case.mode != "enabled"
        or (type(returncode), returncode) != (int, 101)
        or case.rows != 1_000_000
        or case.batch_rows != (64_000 if case.row == 4 else 500)
    ):
        return None
    pattern = (
        r"thread 'main' panicked at crates/calc-flow/benches/support/"
        r"native_join_components\.rs:\d+:\d+:\n"
        r"called `Result::unwrap\(\)` on an `Err` value: Internal \{ message: \""
        + re.escape(IPC_REASON)
        + r"\" \}"
        r"(?:\nnote: run with `RUST_BACKTRACE=1` environment variable "
        r"to display a backtrace)?"
    )
    return (
        "native_checkpoint_credit_admission"
        if re.fullmatch(pattern, diagnostic.strip())
        else None
    )


async def recognize_owned_failure(runner, group, position, error):
    """Classify only first-main warmups with independent, settled v3 evidence."""
    if runner.plan.get("schema") != SCHEMA or position != ("baseline", 0, None):
        return None
    if type(error) not in (RuntimeError, FailedLifecycle):
        return None
    validate_envelope(runner.plan)
    try:
        if group[0].row in (4, 5):
            return _native_owned_failure(runner, group, error)
        if type(error) is FailedLifecycle:
            return await _runtime_owned_failure(runner, group)
        return await _retained_owned_failure(runner, group, error)
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return None


def _sealed_file(plan, path, digest=None):
    from pathlib import Path

    from scripts.toolkit import sha256_file

    expected = plan["sealed_files"].get(str(path))
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("v3 failure input is not sealed")
    if sha256_file(Path(path)) != expected or (
        digest is not None and digest != expected
    ):
        raise ValueError("v3 failure input changed")
    return expected


def _native_owned_failure(runner, group, error):
    import json
    from pathlib import Path

    case = group[0]
    root = runner.destination(group, "baseline", 0, None)
    if str(error) != f"native component failed; see {root}":
        return None
    descriptor = runner.plan["native_binaries"]["baseline"]["stream_join_perf/enabled"]
    _sealed_file(runner.plan, descriptor["path"], descriptor["sha256"])
    source = (
        Path(descriptor["build_output"])
        / f"build-{descriptor['harness_target']}.acceptance-harness.json"
    )
    _sealed_file(runner.plan, source)
    _fresh_source(json.loads(source.read_text()), runner.plan)
    process = json.loads((root / "process.json").read_text())
    expected = {
        "argv": [
            descriptor["path"],
            "--native-component",
            "join_capture_restore_1m" if case.row == 4 else "join_compaction_1m",
        ],
        "binary_sha256": descriptor["sha256"],
        "cpu_affinity": runner.cpus(case),
    }
    if not _contains(process, expected) or (root / "stdout.log").read_bytes():
        return None
    signature = native_failure_signature(
        case, process.get("returncode"), (root / "stderr.log").read_text()
    )
    if signature is None or runner.retired.get(process["pid"]) != {
        "returncode": 101,
        "group_empty": True,
    }:
        return None
    paths = [
        source,
        *(root / name for name in ("process.json", "stdout.log", "stderr.log")),
    ]
    return _failure_evidence(
        runner,
        group,
        signature,
        process,
        paths,
    )


def _fresh_source(record, plan):
    from scripts.benchmark_suite.issue406_harness import FRESH_SCHEMA

    expected = {
        "schema": FRESH_SCHEMA,
        "product_git_sha": MAIN,
        "harness_git_sha": plan["harness_git_sha"],
        "compatibility": None,
        "helper_reconstruction": None,
        "product_files_verified": True,
    }
    if not _contains(record, expected):
        raise ValueError("v3 failure has stale or reconstructed native source")


async def _runtime_owned_failure(runner, group):
    import asyncio
    import json
    import time

    case = group[0]
    root = runner.output / "workers" / case.id / "round-0" / "baseline"
    observation = runner.destination(group, "baseline", 0, None) / "observation.json"
    raw = json.loads(observation.read_text())
    signature = runtime_failure_signature(case, raw)
    if signature is None:
        return None
    process = _runtime_invocation(runner, case, root, raw)
    pid = process["pid"]
    remaining = min(
        10, runner.started + runner.plan["collection_cutoff_seconds"] - time.monotonic()
    )
    async with asyncio.timeout(remaining):
        await runner.close_worker("baseline")
    if runner.retired.get(pid) != {"returncode": 0, "group_empty": True}:
        return None
    runner.workers.pop("baseline")
    if (root / "stderr.log").read_bytes():
        return None
    paths = [
        observation,
        *(
            root / name
            for name in (
                "invocation.json",
                "environment.json",
                "requests.jsonl",
                "stdout.log",
                "stderr.log",
            )
        ),
    ]
    return _failure_evidence(
        runner,
        group,
        signature,
        process,
        [*paths, *_runtime_artifact_paths(runner, case)],
    )


def _runtime_worker_identity(runner, case, root):
    import json
    from dataclasses import asdict

    release = runner.release(case, "baseline")
    _sealed_file(runner.plan, release["native_path"], release["native_sha256"])
    _sealed_file(runner.plan, release["wheel_path"], release["wheel_sha256"])
    _sealed_file(runner.plan, release["build_metadata"])
    expected = {
        **asdict(case),
        "family": "join-asof-acceptance",
        "side": "baseline",
        "product_commit": MAIN,
        "fixture_root": str(runner.fixture_root("baseline", case.mode, 0)),
    }
    if runner.prepared_cases.get((case.id, "baseline", 0)) != expected:
        raise ValueError("v3 failure invocation differs from its warmup")
    requests = [
        json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()
    ]
    if requests != [
        {"operation": "hello", "scope": "core"},
        {"operation": "prepare", "case": expected},
    ]:
        raise ValueError("v3 failure stdout or request differs from its warmup")
    identity = json.loads((root / "environment.json").read_text())
    if not _contains(
        identity,
        {
            "native_sha256": release["native_sha256"],
            "cpu_affinity": runner.cpus(case),
            "tokio_worker_threads": str(case.threads),
        },
    ):
        raise ValueError("v3 failure loaded a different release or environment")
    process = json.loads((root / "invocation.json").read_text())
    _validate_runtime_invocation(process, runner, case, root, release)
    return process, expected, _worker_response(root, identity)


def _worker_response(root, identity):
    import json

    responses = (root / "stdout.log").read_bytes().splitlines()
    if len(responses) != 2 or json.loads(responses[0]) != identity:
        raise ValueError("v3 failure stdout contains unexpected worker responses")
    return json.loads(responses[1])


def _runtime_invocation(runner, case, root, raw):
    process, expected, response = _runtime_worker_identity(runner, case, root)
    if response != {"case": expected, "warmup": raw}:
        raise ValueError("v3 failure stdout differs from its warmup")
    if not _contains(raw, {"product_commit": MAIN, "cpu_affinity": runner.cpus(case)}):
        raise ValueError("v3 failure raw product or affinity differs")
    return process


def _validate_runtime_invocation(process, runner, case, root, release):
    import sys

    expected = {
        "pid": runner.workers["baseline"].process.pid,
        "argv": [
            sys.executable,
            "-m",
            "scripts.benchmark_suite",
            "worker",
            "--root",
            str(root),
        ],
        "cpu_affinity": runner.cpus(case),
        "product_commit": MAIN,
        "native_sha256": release["native_sha256"],
        "harness_git_sha": runner.plan["harness_git_sha"],
    }
    if process != expected:
        raise ValueError("v3 failure worker invocation differs")


def _failure_evidence(runner, group, signature, process, paths):
    from scripts.toolkit import sha256_file, write_json

    phase, reason = {
        "runtime_eof_shutdown_sentinel": ("runtime-eof", EOF_REASON),
        "native_checkpoint_credit_admission": (
            "checkpoint-ipc-construction",
            IPC_REASON,
        ),
        "retained_on_checkpoint_failure_cause_unresolved": (
            "retained-processing-first-checkpoint",
            "structured checkpoint failure; cause unresolved",
        ),
    }[signature]
    evidence = {
        "schema": "calc-flow.issue406-failed-arm.v3",
        **report_identity(runner.plan),
        "product_commit": MAIN,
        "signature": signature,
        "phase": phase,
        "reason": reason,
        "cases": [case.id for case in group],
        "side": "baseline",
        "round": 0,
        "index": None,
        "invocation": process,
        "natural_exit": runner.retired[process["pid"]],
        "files_sha256": {str(path): sha256_file(path) for path in paths},
    }
    destination = runner.destination(group, "baseline", 0, None)
    write_json(destination / "failure-evidence.json", evidence)
    return evidence


# Independent v3 observable signature; no historical binary exception is used.
RETAINED_NODES = {
    1_000_000: "cf_stream_join_fae6be855b5bef51",
    4_000_000: "cf_stream_join_70034c097d851384",
}


RETAINED_CHECKPOINT = {
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


def _retained_terminal_error(node: str) -> str:
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
    ) or not _contains(status.get("checkpoint"), RETAINED_CHECKPOINT):
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


def _unsupported_retained_case(case, node) -> bool:
    return (
        node is None
        or case.row != 3
        or case.workload != "join_retained"
        or case.mode != "on"
        or case.threads != 1
        or case.batch_rows != 64_000
    )


def retained_failure_signature(case, raw):
    """Recognize the early retained checkpoint failure without inferring its cause."""
    node = RETAINED_NODES.get(case.rows)
    if _unsupported_retained_case(case, node) or not _retained_early_shape(raw):
        return None
    expected = {
        "cleanup_complete": True,
        "failure": "invalid acceptance sample",
        "terminal_error": _retained_terminal_error(node),
    }
    if not _contains(raw, expected):
        return None
    return (
        "retained_on_checkpoint_failure_cause_unresolved"
        if all(
            _retained_status(raw[phase], node)
            for phase in ("after_eof", "after_cleanup")
        )
        else None
    )


def _retained_early_shape(raw):
    import math

    clocks = {
        "cleanup_seconds",
        "job_cpu_seconds",
        "job_seconds",
        "sample_wall_seconds",
    }
    fields = {
        "after_eof",
        "after_cleanup",
        "checkpoint_observations",
        "cleanup_complete",
        "failure",
        "terminal_error",
    } | clocks
    if not isinstance(raw, dict) or set(raw) != fields:
        return False
    return _retained_checkpoint_observations(raw["checkpoint_observations"]) and all(
        type(raw[key]) in (int, float) and math.isfinite(raw[key]) and raw[key] >= 0
        for key in clocks
    )


def retained_traceback_signature(runner, case, diagnostic):
    """Require exactly the known early-processing exception chain on sealed H."""
    separator = (
        "\n\nThe above exception was the direct cause of the following exception:\n\n"
    )
    blocks = diagnostic.strip().split(separator)
    if len(blocks) != 2 or re.search(
        r"AssertionError|TimeoutError|CancelledError|panicked|oracle", diagnostic
    ):
        return False
    messages = (
        f"RuntimeError: {_retained_terminal_error(RETAINED_NODES[case.rows])}",
        "benchmarks.checkpoint_cycles.CycleValidationError: invalid acceptance sample",
    )
    try:
        frames = [
            _trace_frames(block, message)
            for block, message in zip(blocks, messages, strict=True)
        ]
        return all(
            _validate_trace_frames(runner, actual, expected)
            for actual, expected in zip(frames, _retained_trace_layout(), strict=True)
        )
    except (OSError, KeyError, TypeError, ValueError, SyntaxError):
        return False


def _retained_trace_layout():
    import asyncio
    import inspect

    from scripts.benchmark_suite.process import ROOT

    runtime = str(ROOT / "benchmarks/join_asof_acceptance_runtime.py")
    lifecycle = str(ROOT / "benchmarks/engine_lifecycle.py")
    entry = str(ROOT / "scripts/benchmark_suite/__main__.py")
    worker = str(ROOT / "scripts/benchmark_suite/worker.py")
    base = inspect.getsourcefile(asyncio.BaseEventLoop.run_until_complete)
    return (
        [
            (runtime, "measure_acceptance_job"),
            (lifecycle, "run_with_completion"),
            (lifecycle, "_require_completed"),
        ],
        [
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
        ],
    )


def _trace_frames(block, message):
    lines = block.splitlines()
    if lines[0] != "Traceback (most recent call last):" or lines[-1] != message:
        raise ValueError("unexpected retained failure exception chain")
    frames = []
    for line in lines[1:-1]:
        match = re.fullmatch(r'  File "([^"]+)", line (\d+), in (.+)', line)
        if match:
            path, number, function = match.groups()
            frames.append((path, int(number), function))
        elif not line.startswith("    "):
            raise ValueError("unexpected retained failure traceback text")
    return frames


def _validate_trace_frames(runner, actual, expected):
    if [(path, function) for path, _, function in actual] != expected:
        return False
    return all(_current_trace_frame(runner.plan, frame) for frame in actual)


def _current_trace_frame(plan, frame):
    import ast
    from pathlib import Path

    from scripts.benchmark_suite.process import ROOT

    path, number, function = frame
    if path == "<frozen runpy>":
        return number > 0
    source = Path(path)
    if source.is_relative_to(ROOT):
        _sealed_file(plan, source)
    text = source.read_text()
    if function == "<module>":
        return 0 < number <= len(text.splitlines())
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function
        and node.lineno <= number <= node.end_lineno
        for node in ast.walk(ast.parse(text))
    )


async def _retained_owned_failure(runner, group, error):
    import asyncio
    import json
    import time

    case = group[0]
    root = runner.output / "workers" / case.id / "round-0" / "baseline"
    if (
        str(error) != "invalid worker result: {'error': 'CycleValidationError: "
        "invalid acceptance sample'}"
    ):
        return None
    path = root / "failed-sample-1.json"
    raw = json.loads(path.read_text())
    signature = retained_failure_signature(case, raw)
    if signature is None:
        return None
    process, _, stdout = _runtime_worker_identity(runner, case, root)
    if stdout != {"error": "CycleValidationError: invalid acceptance sample"}:
        return None
    remaining = min(
        10, runner.started + runner.plan["collection_cutoff_seconds"] - time.monotonic()
    )
    async with asyncio.timeout(remaining):
        await runner.close_worker("baseline", allow_failure=True)
    runner.workers.pop("baseline")
    if runner.retired.get(process["pid"]) != {"returncode": 1, "group_empty": True}:
        return None
    if not retained_traceback_signature(
        runner, case, (root / "stderr.log").read_text()
    ):
        return None
    paths = [
        path,
        *(
            root / name
            for name in (
                "invocation.json",
                "environment.json",
                "requests.jsonl",
                "stdout.log",
                "stderr.log",
            )
        ),
    ]
    destination = runner.destination(group, "baseline", 0, None)
    destination.mkdir(parents=True, exist_ok=True)
    return _failure_evidence(
        runner,
        group,
        signature,
        process,
        [*paths, *_runtime_artifact_paths(runner, case)],
    )


def _runtime_artifact_paths(runner, case):
    from pathlib import Path

    release = runner.release(case, "baseline")
    return [
        Path(release[key]) for key in ("native_path", "wheel_path", "build_metadata")
    ]


def _retained_checkpoint_observations(observations):
    return (
        isinstance(observations, list)
        and bool(observations)
        and all(_retained_checkpoint_observation(value) for value in observations)
    )


def _retained_checkpoint_observation(value):
    if not isinstance(value, dict) or set(value) != {
        "before_ns",
        "after_ns",
        "checkpoint",
    }:
        return False
    before, after = value["before_ns"], value["after_ns"]
    if type(before) is not int or type(after) is not int or not 0 <= before <= after:
        return False
    return _retained_observed_checkpoint(value["checkpoint"])


def _retained_observed_checkpoint(checkpoint):
    dynamic = ("current_epoch", "phase", "failure_category", "source_acks")
    expected = {
        key: value for key, value in RETAINED_CHECKPOINT.items() if key not in dynamic
    }
    if not _contains(checkpoint, expected) or set(checkpoint) != set(
        RETAINED_CHECKPOINT
    ) | {"elapsed_micros"}:
        return False
    states = (
        (None, None, None, 0),
        (1, "requested", None, 0),
        (1, "sources_cut", None, 2),
        (1, "sources_cut", "internal", 2),
    )
    if not any(
        _contains(checkpoint, dict(zip(dynamic, state, strict=True)))
        for state in states
    ):
        return False
    return _retained_checkpoint_elapsed(checkpoint)


def _retained_checkpoint_elapsed(checkpoint):
    elapsed = checkpoint["elapsed_micros"]
    if checkpoint["current_epoch"] is None:
        return elapsed is None
    return type(elapsed) is int and elapsed >= 0
