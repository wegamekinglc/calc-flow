"""Fixed paired SUM-state cases with complete per-snapshot evidence.

Build receipts and reproduction commands are documented in benchmarks/sum-state.
The driver never builds binaries or retries incomplete measurements.
Its owned subprocess boundary uses validated absolute argv without a shell.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import signal
import statistics
import subprocess  # nosec B404
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from scripts.benchmark_suite.report import comparison
from scripts.toolkit import command_output, require_executable, write_json

ROOT = Path(__file__).resolve().parents[2]
HARNESS = "crates/calc-flow/examples/grouped_sum_state_benchmark.rs"
CASES = (
    {
        "id": "stock-stream-1cpu",
        "fixture": "stock",
        "mode": "stream",
        "cpus": [0],
        "partitions": 1,
        "reason": "principal changed integer SUM state path",
    },
    {
        "id": "stock-stream-8cpu",
        "fixture": "stock",
        "mode": "stream",
        "cpus": list(range(8)),
        "partitions": 1,
        "reason": "same changed path with eight workers",
    },
    {
        "id": "sparse-stream-1cpu",
        "fixture": "sparse",
        "mode": "stream",
        "cpus": [0],
        "partitions": 1,
        "reason": "100 updated groups among 10k retained groups",
    },
    {
        "id": "stock-batch-8cpu",
        "fixture": "stock",
        "mode": "batch",
        "cpus": list(range(8)),
        "partitions": 8,
        "reason": "finite execution control unaffected by state implementation",
    },
)
THREAD_ENV = (
    "TOKIO_WORKER_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
ORACLE_FIELDS = (
    "input_sha256",
    "input_schema_sha256",
    "input_rows",
    "batch_rows",
    "snapshot_rows",
    "snapshot_sha256",
    "snapshot_schema_sha256",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_oracles(left, right):
    for field in ORACLE_FIELDS:
        if left[field] != right[field]:
            raise ValueError(f"cross-version {field} mismatch")


def checked_build(path):
    build = json.loads(path.read_text())
    required = (
        "binary",
        "binary_sha256",
        "source_root",
        "revision",
        "command",
        "rustc",
        "cargo_lock_sha256",
        "harness_sha256",
        "features",
        "source_sha256",
        "build_environment",
    )
    if any(not build.get(key) for key in required):
        raise ValueError(f"incomplete build receipt: {path}")
    verify_build(build)
    return build


def verify_build(build):
    source = Path(build["source_root"])
    if (
        sha(build["binary"]) != build["binary_sha256"]
        or sha(source / "Cargo.lock") != build["cargo_lock_sha256"]
        or sha(source / HARNESS) != build["harness_sha256"]
    ):
        raise ValueError("build artifact/lock/harness drift")
    for name, digest in build["source_sha256"].items():
        if sha(source / name) != digest:
            raise ValueError(f"source drift: {name}")


def environment(workers):
    return {
        **os.environ,
        "TOKIO_WORKER_THREADS": str(workers),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "POLARS_MAX_THREADS": "1",
    }


def write_samples(path, acquisition):
    fields = (
        "case",
        "round",
        "pair",
        "side",
        "wall_ms",
        "process_cpu_ms",
        "average_cores",
        "whole_process_peak_rss_bytes",
        "final_snapshot_sha256",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in acquisition["cases"]:
            for observation in case["observations"]:
                row = observation["row"]
                writer.writerow(
                    {
                        "case": case["id"],
                        "round": observation["round"],
                        "pair": observation["pair"],
                        "side": observation["side"],
                        "wall_ms": row["seconds"] * 1000,
                        "process_cpu_ms": row["process_cpu_ns"] / 1e6,
                        "average_cores": row["process_cpu_ns"] / 1e9 / row["seconds"],
                        "whole_process_peak_rss_bytes": observation["resources"][
                            "sampled_process_tree_peak_rss"
                        ],
                        "final_snapshot_sha256": row["snapshot_sha256"][-1],
                    }
                )


def finalize_acquisition(args, acquisition, builds, started, deadline):
    from scripts.benchmark_suite.sum_state_publication import (
        persist_source,
        update_timing,
    )

    try:
        for build in builds.values():
            verify_build(build)
    except BaseException as error:
        acquisition["status"] = "stopped"
        acquisition["failure"] = repr(error)
    write_samples(args.output / "samples.csv", acquisition)
    update_timing(args, acquisition, started, deadline, time.monotonic())
    persist_source(args, acquisition)
    final = time.monotonic()
    if final > deadline:
        update_timing(args, acquisition, started, deadline, final)
        persist_source(args, acquisition)


@dataclass(frozen=True, slots=True)
class CollectionContext:
    args: argparse.Namespace
    builds: dict
    scope: dict
    started: float
    deadline: float


@dataclass(frozen=True, slots=True)
class MeasurementContext:
    collection: CollectionContext
    fixture: dict


@dataclass(frozen=True, slots=True)
class RoundContext:
    measurement: MeasurementContext
    case: dict
    number: int
    record: dict
    path: Path


def validate_timing(row):
    seconds = row.get("seconds")
    if type(seconds) not in (float, int):
        raise ValueError("invalid timing")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid timing")
    if row.get("process_cpu_ns", 0) <= 0:
        raise ValueError("invalid timing clocks")
    if row.get("end_ns", 0) <= row.get("start_ns", 0):
        raise ValueError("invalid timing clocks")


def validate_resources(row, case):
    observed = row.get("process_observations", {})
    if observed.get("observed_cpu_affinity") != case["cpus"]:
        raise ValueError("actual CPU affinity mismatch")
    expected = {name: "1" for name in THREAD_ENV}
    expected["TOKIO_WORKER_THREADS"] = str(len(case["cpus"]))
    if observed.get("observed_thread_environment") != expected:
        raise ValueError("actual thread environment mismatch")


def expected_sample(case, fixture):
    indices = range(len(fixture["batch_rows"])) if case["mode"] == "stream" else [-1]
    return {
        "schema": "calc-flow.sum-state.v1",
        "mode": case["mode"],
        "partitions": case["partitions"],
        "validated_all_snapshots": True,
        "input_sha256": fixture["input_sha256"],
        "input_schema_sha256": fixture["schema_sha256"],
        "input_rows": fixture["input_rows"],
        "batch_rows": fixture["batch_rows"],
        "snapshot_rows": [fixture["snapshot_rows"][i] for i in indices],
        "snapshot_sha256": [fixture["snapshot_sha256"][i] for i in indices],
        "snapshot_schema_sha256": [fixture["snapshot_schema_sha256"] for _ in indices],
    }


def validate_fields(row, expected):
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"invalid evidence {key}: {row.get(key)!r} != {value!r}")


def actor_unsettled(actor):
    return actor["errors"] or not actor["ended"]


def validate_stream_status(status):
    if status["state"] != "completed" or status["terminal_cause"] != "natural_end":
        raise ValueError("incomplete lifecycle")
    if any(status[key] for key in ("task_count", "task_errors", "metrics_overflowed")):
        raise ValueError("incomplete lifecycle")
    epochs = ("current_epoch", "last_completed_epoch")
    if any(status["checkpoint"][key] is not None for key in epochs):
        raise ValueError("unexpected checkpoint lifecycle")
    validate_actor_status(status)


def validate_actor_status(status):
    for group in ("sources", "operators", "sinks"):
        if any(actor_unsettled(value) for value in status[group].values()):
            raise ValueError("unsettled actor lifecycle")


def validate_sample(row, case, fixture):
    validate_timing(row)
    validate_resources(row, case)
    validate_fields(row, expected_sample(case, fixture))
    if case["mode"] == "stream":
        validate_stream_status(row["diagnostics"])


def compare_fields(left, right, keys, label, *, required=False):
    for key in keys:
        left_value = left[key] if required else left.get(key)
        right_value = right[key] if required else right.get(key)
        if left_value != right_value:
            raise ValueError(f"incomparable {label} {key}")


def compare_shared_sources(left, right):
    for path in (
        "crates/calc-flow/examples/support/mod.rs",
        "crates/calc-flow/examples/support/stock_account_data.rs",
    ):
        digest = left.get(path)
        if not digest or digest != right.get(path):
            raise ValueError(f"incomparable shared harness source: {path}")


def vendor_seals(build):
    return {
        name: digest
        for name, digest in build["source_sha256"].items()
        if name.startswith("vendor/")
    }


def compare_builds(builds):
    base, candidate = builds["baseline"], builds["candidate"]
    compare_fields(
        base,
        candidate,
        ("cargo_lock_sha256", "harness_sha256", "rustc", "features"),
        "build",
        required=True,
    )
    compare_fields(
        base.get("build_environment", {}),
        candidate.get("build_environment", {}),
        (
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
            "RUSTUP_TOOLCHAIN",
            "CARGO_PROFILE_RELEASE_LTO",
            "CARGO_PROFILE_RELEASE_OPT_LEVEL",
            "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
            "CARGO_PROFILE_RELEASE_DEBUG",
        ),
        "build environment",
    )
    compare_fields(
        base["source_sha256"],
        candidate["source_sha256"],
        (
            "Cargo.toml",
            "crates/calc-flow/Cargo.toml",
            "rust-toolchain.toml",
            ".cargo/config.toml",
        ),
        "compiled configuration:",
    )
    compare_shared_sources(base["source_sha256"], candidate["source_sha256"])
    vendors = vendor_seals(base)
    if not vendors or vendors != vendor_seals(candidate):
        raise ValueError("incomparable vendored dependencies")


def benchmark_argument(argument):
    return (
        "run_stock_positions_" in argument or argument == "benchmarks.stock_positions"
    )


def competing_process(process):
    if process.pid == os.getpid() or process.info["status"] == psutil.STATUS_ZOMBIE:
        return False
    names = {"cargo", "rustc", "maturin", "pytest", "samply", "perf"}
    arguments = process.info["cmdline"] or []
    return process.info["name"] in names or any(map(benchmark_argument, arguments))


def guard(builds):
    processes = psutil.process_iter(["pid", "name", "cmdline", "status"])
    bad = [process.info for process in processes if competing_process(process)]
    if bad:
        raise RuntimeError(f"competing build/measurement: {bad}")
    for build in builds.values():
        if sha(build["binary"]) != build["binary_sha256"]:
            raise ValueError("binary drift")


def validate_argv_values(arguments):
    if not isinstance(arguments, (list, tuple)) or not arguments:
        raise ValueError("command requires a nonempty argv vector")
    if any(not isinstance(value, str) for value in arguments):
        raise ValueError("argv values must be strings")
    if any("\0" in value for value in arguments):
        raise ValueError("argv values must not contain NUL")


def absolute_executable(value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(require_executable(value))
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("executable must be an executable regular file")
    return str(path)


def validated_argv(arguments):
    validate_argv_values(arguments)
    return [absolute_executable(arguments[0]), *arguments[1:]]


def process_tree_rss(process):
    try:
        parent = psutil.Process(process.pid)
        children = [parent, *parent.children(recursive=True)]
        return sum(child.memory_info().rss for child in children if child.is_running())
    except psutil.Error:
        return 0


def poll_invocation(process, deadline):
    peak = 0
    while process.poll() is None:
        if time.monotonic() > deadline - 5:
            raise TimeoutError("inclusive acquisition deadline")
        peak = max(peak, process_tree_rss(process))
        time.sleep(0.02)
    return peak


def settle_invocation(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def start_invocation(command, env, output, errors):
    command = validated_argv(command)
    # Trusted local receipts/fixed flags form validated, absolute argv. No shell.
    return subprocess.Popen(  # nosec B603  # nosemgrep
        command,
        env=env,
        cwd=ROOT,
        stdout=output,
        stderr=errors,
        start_new_session=True,
        shell=False,
    )


def invoke(command, env, path, deadline):
    command = validated_argv(command)
    path.mkdir(parents=True, exist_ok=False)
    write_json(
        path / "command.json",
        {
            "argv": command,
            "environment": {key: env[key] for key in THREAD_ENV},
        },
    )
    started, process = time.monotonic(), None
    try:
        with (
            (path / "stdout.jsonl").open("wb") as output,
            (path / "stderr.log").open("wb") as errors,
        ):
            process = start_invocation(command, env, output, errors)
            peak = poll_invocation(process, deadline)
            if process.returncode:
                raise RuntimeError(f"command failed ({process.returncode}): {path}")
    finally:
        settle_invocation(process)
    receipt = {
        "whole_process_wall_seconds": time.monotonic() - started,
        "sampled_process_tree_peak_rss": peak,
        "owned_process_reaped": process.returncode is not None,
    }
    write_json(path / "receipt.json", receipt)
    rows = [
        json.loads(line) for line in (path / "stdout.jsonl").read_text().splitlines()
    ]
    return rows, receipt


def validate_budget(budget, prior):
    if not math.isfinite(budget) or not 0 < budget <= 600:
        raise ValueError("invalid measurement budget")
    if not math.isfinite(prior) or prior < 0:
        raise ValueError("invalid measurement budget")
    if budget + prior > 600 + 1e-9:
        raise ValueError("cumulative measurement budget exceeds600 seconds")


def measurement_scope(history_borrow, budget_seconds, prior_seconds):
    validate_budget(budget_seconds, prior_seconds)
    if history_borrow and prior_seconds <= 0:
        raise ValueError("history-borrow requires prior measurement accounting")
    return {
        "cases": CASES[:3] if history_borrow else CASES,
        "profile_sides": ("candidate",)
        if history_borrow
        else ("baseline", "candidate"),
        "profile_min_remaining_seconds": 25 if history_borrow else 0,
        "unmeasured_final_source": ["stock-batch-8cpu"] if history_borrow else [],
    }


def create_output(args):
    if args.output.exists():
        raise FileExistsError(
            "use a new output directory; never overwrite a collection"
        )
    args.output.mkdir(parents=True)
    if args.publish_directory and args.publish_directory.exists():
        raise FileExistsError("use a new publication directory")


def validate_prior_budget(args):
    if args.history_borrow:
        previous = json.loads(
            (ROOT / "benchmarks/sum-state/20261104/evidence-manifest.json").read_text()
        )
        if (
            abs(previous["inclusive_phase_seconds"] - args.prior_measurement_seconds)
            > 1e-6
        ):
            raise ValueError("prior measurement accounting differs from sealed receipt")


def collection_context(args, started):
    scope = measurement_scope(
        args.history_borrow, args.budget_seconds, args.prior_measurement_seconds
    )
    create_output(args)
    builds = {
        side: checked_build(getattr(args, side + "_build"))
        for side in ("baseline", "candidate")
    }
    compare_builds(builds)
    validate_prior_budget(args)
    os.sched_setaffinity(0, {31})
    return CollectionContext(
        args, builds, scope, started, started + args.budget_seconds
    )


def collection_plan(context):
    args, scope, builds = context.args, context.scope, context.builds
    plan = {
        "cases": scope["cases"],
        "rounds": 2,
        "pairs_per_round": 10,
        "warmups_per_arm_per_round": 1,
        "budget_seconds": args.budget_seconds,
        "prior_measurement_seconds": args.prior_measurement_seconds,
        "cumulative_budget_seconds": 600,
        "history_borrow": args.history_borrow,
        "profile_sides": scope["profile_sides"],
        "profile_min_remaining_seconds": scope["profile_min_remaining_seconds"],
        "unmeasured_final_source": scope["unmeasured_final_source"],
        "builds_separate": True,
        "profiles": "selected stream1CPU sides, six jobs each, first excluded"
        if args.perf
        else None,
        "scope": (
            "fresh native graph plan, execution, retained snapshots and cleanup; "
            "excludes IPC preload and subsequent full per-snapshot arithmetic oracle"
        ),
        "statistics": "maintained two-round paired intervals and five-percent rule",
        "stop": (
            "deadline, competing build/benchmark, artifact drift or oracle failure; "
            "no resampling"
        ),
        "builds": builds,
    }
    return plan


def machine_identity(context):
    args = context.args
    machine = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "topology": command_output(
            [require_executable("lscpu"), "-e=CPU,CORE,SOCKET,ONLINE"]
        )
        + "\n",
        "host_limits": (
            "WSL host load and power state unobservable; "
            "eight logical CPUs report four core IDs"
        ),
        "controller_affinity": [31],
        "driver_sha256": sha(__file__),
        "cpu_model": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "memory_bytes": psutil.virtual_memory().total,
        "helper_sha256": {
            str(path.relative_to(ROOT)): sha(path)
            for path in (
                Path(__file__),
                ROOT / "scripts/benchmark_suite/sum_state_profile.py",
                ROOT / "scripts/benchmark_suite/sum_state_publication.py",
                ROOT / "scripts/benchmark_suite/report.py",
                ROOT / "scripts/benchmark_suite/statistics.py",
                ROOT / "scripts/toolkit.py",
            )
        },
        "runtime_defaults": {
            "benchmark_internals": True,
            "dispatch_trace": "Disabled default; harness never starts recorder",
            "join_cpu_trace": "Disabled default; harness never starts recorder",
            "checkpointing": False,
            "sql_compact_state_max_groups": 10000,
            "sql_compact_state_max_bytes": 9_007_199_254_740_991,
            "sql_state_budget": None,
            "sql_snapshot_state_budget": None,
            "edge_budget": {"max_rows": 1_000_000, "max_bytes": 2 * sys.maxsize + 1},
            "retained_epochs": 2,
            "checkpoint_interval_seconds": 60,
            "checkpoint_timeout_seconds": 600,
            "collect_diagnostics": True,
            "datafusion_batch_size": 8192,
            "watermarks": "Disabled",
        },
        "inherited_diagnostic_environment": {
            name: os.environ.get(name) for name in ("RUST_LOG", "RUST_BACKTRACE")
        },
    }
    if args.perf:
        machine["profile_tools"] = profile_tool_identities(context)
    return machine


def profile_tool_identities(context):
    args = context.args
    env = environment(1)
    if args.perf_library:
        env["LD_LIBRARY_PATH"] = str(args.perf_library)
    return {
        name: tool_identity(path, env, context.deadline)
        for name, path in (("perf", args.perf), ("samply", args.samply))
    }


def tool_identity(path, env, deadline):
    executable = absolute_executable(str(path))
    return {
        "path": str(path),
        "sha256": sha(path),
        "version": command_output(
            [executable, "--version"],
            env=env,
            timeout=max(1, deadline - time.monotonic()),
        ),
    }


def acquisition_state(context):
    return {
        "status": "running",
        "started_monotonic": context.started,
        "deadline": context.deadline,
        "cases": [],
        "profiles": [],
        "failure": None,
    }


def save_acquisition(context, acquisition):
    write_json(context.args.output / "acquisition.json", acquisition)


def prepare_fixture(context):
    guard(context.builds)
    prepared, _ = invoke(
        [
            "taskset",
            "-c",
            "0",
            context.builds["baseline"]["binary"],
            "--prepare",
            str(context.args.output / "fixture"),
        ],
        environment(1),
        context.args.output / "prepare",
        context.deadline,
    )
    fixture = {row["fixture"]: row for row in prepared}
    write_json(context.args.output / "fixture-manifest.json", fixture)
    return MeasurementContext(context, fixture)


def collect_pair(context, pair, samples):
    order = ("baseline", "candidate") if pair % 2 == 0 else ("candidate", "baseline")
    rows = {}
    for side in order:
        path = context.path / f"pair-{pair + 1}-{side}"
        row, receipt = measure(context.measurement, context.case, side, path)
        samples[side].append(row["seconds"])
        rows[side] = row
        context.record["observations"].append(
            {
                "round": context.number + 1,
                "pair": pair + 1,
                "side": side,
                "path": str(
                    path.relative_to(context.measurement.collection.args.output)
                ),
                "row": row,
                "resources": receipt,
            }
        )
    compare_oracles(rows["baseline"], rows["candidate"])


def collect_round(context, acquisition):
    collection = context.measurement.collection
    for side in collection.builds:
        measure(
            context.measurement, context.case, side, context.path / ("warmup-" + side)
        )
    samples = {side: [] for side in collection.builds}
    for pair in range(10):
        collect_pair(context, pair, samples)
        save_acquisition(collection, acquisition)
    for side in collection.builds:
        context.record[side].append(samples[side])


def case_record(measurement, case):
    return {
        **case,
        "comparison": "interleaved",
        "correctness": True,
        "rows": measurement.fixture[case["fixture"]]["input_rows"],
        "baseline": [],
        "candidate": [],
        "observations": [],
    }


def side_resources(observations):
    return {
        "processing_cpu_ms_p50": statistics.median(
            row["row"]["process_cpu_ns"] / 1e6 for row in observations
        ),
        "processing_average_cores_p50": statistics.median(
            row["row"]["process_cpu_ns"] / 1e9 / row["row"]["seconds"]
            for row in observations
        ),
        "whole_process_peak_rss_p50": statistics.median(
            row["resources"]["sampled_process_tree_peak_rss"] for row in observations
        ),
    }


def resources_for_side(record, side):
    return side_resources(
        [row for row in record["observations"] if row["side"] == side]
    )


def collect_case(measurement, case, acquisition):
    collection = measurement.collection
    record = case_record(measurement, case)
    acquisition["cases"].append(record)
    for number in range(2):
        path = collection.args.output / "timings" / case["id"] / f"round-{number + 1}"
        collect_round(
            RoundContext(measurement, case, number, record, path), acquisition
        )
    record["statistics"] = comparison(record)
    record["resource_summary"] = {
        side: resources_for_side(record, side) for side in collection.builds
    }
    print(
        json.dumps({"case": case["id"], "statistics": record["statistics"]}), flush=True
    )


def profile_observation(side, path, rows, receipt):
    return {
        "side": side,
        "path": str(path),
        "rows": rows,
        "windows": [
            {"start_ns": row["start_ns"], "end_ns": row["end_ns"]} for row in rows[1:]
        ],
        "resources": receipt,
    }


def collect_profiles(measurement, acquisition):
    context = measurement.collection
    if not context.args.perf:
        return
    if (
        context.deadline - time.monotonic()
        < context.scope["profile_min_remaining_seconds"]
    ):
        acquisition["profile_skip_reason"] = (
            "predeclared remaining-time admission refused; timings prioritized"
        )
        return
    for side in context.scope["profile_sides"]:
        path = context.args.output / "profiles" / side
        rows, receipt = measure(measurement, CASES[0], side, path, profile=True)
        observed = profile_observation(
            side, path.relative_to(context.args.output), rows, receipt
        )
        observed["perf_data_sha256"] = sha(path / "perf.data")
        acquisition["profiles"].append(observed)
        save_acquisition(context, acquisition)
    from scripts.benchmark_suite.sum_state_profile import decode

    decode(context.args, acquisition, context.builds, context.deadline)


def collect(context, acquisition):
    measurement = prepare_fixture(context)
    for case in context.scope["cases"]:
        collect_case(measurement, case, acquisition)
    collect_profiles(measurement, acquisition)
    write_samples(context.args.output / "samples.csv", acquisition)
    if time.monotonic() > context.deadline:
        raise TimeoutError("inclusive acquisition deadline after preflight/cleanup")
    acquisition["status"] = "completed"


def publish_collection(context, acquisition, plan, machine):
    if context.args.publish_directory:
        from scripts.benchmark_suite.sum_state_publication import (
            PublicationContext,
            publish,
        )

        publish(
            PublicationContext(
                context.args,
                acquisition,
                plan,
                machine,
                context.builds,
                context.deadline,
            )
        )


def run(args):
    started = time.monotonic()
    context = collection_context(args, started)
    plan, machine = collection_plan(context), machine_identity(context)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "environment.json", machine)
    acquisition = acquisition_state(context)
    save_acquisition(context, acquisition)
    try:
        collect(context, acquisition)
    except BaseException as error:
        acquisition["status"] = "stopped"
        acquisition["failure"] = repr(error)
        raise
    finally:
        finalize_acquisition(
            args, acquisition, context.builds, started, context.deadline
        )
        publish_collection(context, acquisition, plan, machine)
    if acquisition["status"] != "completed":
        raise RuntimeError(f"measurement stopped: {acquisition['failure']}")


def native_command(context, case, side, path, profile):
    collection = context.collection
    command = [
        "taskset",
        "-c",
        ",".join(map(str, case["cpus"])),
        collection.builds[side]["binary"],
        str(collection.args.output / "fixture" / (case["fixture"] + ".arrow")),
        case["mode"],
        str(case["partitions"]),
        "6" if profile else "1",
    ]
    if profile:
        command = [
            str(collection.args.perf),
            "record",
            "-e",
            "cpu-clock:u",
            "-F",
            "999",
            "--call-graph",
            "dwarf,16384",
            "-k",
            "mono",
            "-o",
            str(path / "perf.data"),
            "--",
            *command,
        ]
    return command


def measure(context, case, side, path, *, profile=False):
    collection = context.collection
    guard(collection.builds)
    env = environment(len(case["cpus"]))
    if collection.args.perf_library:
        env["LD_LIBRARY_PATH"] = str(collection.args.perf_library)
    rows, receipt = invoke(
        native_command(context, case, side, path, profile),
        env,
        path,
        collection.deadline,
    )
    if len(rows) != (6 if profile else 1):
        raise ValueError("wrong invocation count")
    for row in rows:
        validate_sample(row, case, context.fixture[case["fixture"]])
    guard(collection.builds)
    return (rows if profile else rows[0]), receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build", required=True, type=Path)
    parser.add_argument("--candidate-build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--perf", type=lambda x: Path(x).resolve())
    parser.add_argument("--perf-library", type=lambda x: Path(x).resolve())
    parser.add_argument("--samply", type=lambda x: Path(x).resolve())
    parser.add_argument("--history-borrow", action="store_true")
    parser.add_argument("--budget-seconds", type=float, default=600)
    parser.add_argument("--prior-measurement-seconds", type=float, default=0)
    parser.add_argument("--publish-directory", type=lambda x: Path(x).resolve())
    args = parser.parse_args()
    if args.publish_directory and not args.history_borrow:
        parser.error("automatic publication is scoped to --history-borrow")
    if bool(args.perf) != bool(args.samply):
        parser.error("--perf and --samply are required together")
    run(args)


if __name__ == "__main__":
    main()
