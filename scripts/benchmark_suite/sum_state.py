"""Four fixed, paired SUM-state cases with complete per-snapshot evidence.

Build receipts and reproduction commands are documented in benchmarks/sum-state.
The driver never builds binaries or retries incomplete measurements.
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
import subprocess
import sys
import time
from pathlib import Path

import psutil

from scripts.benchmark_suite.report import comparison
from scripts.toolkit import write_json

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


def validate_sample(row, case, fixture):
    seconds = row.get("seconds")
    if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid timing")
    if not (
        row.get("process_cpu_ns", 0) > 0
        and row.get("end_ns", 0) > row.get("start_ns", 0)
    ):
        raise ValueError("invalid timing clocks")
    observed = row.get("process_observations", {})
    if observed.get("observed_cpu_affinity") != case["cpus"]:
        raise ValueError("actual CPU affinity mismatch")
    expected_threads = {name: "1" for name in THREAD_ENV}
    expected_threads["TOKIO_WORKER_THREADS"] = str(len(case["cpus"]))
    if observed.get("observed_thread_environment") != expected_threads:
        raise ValueError("actual thread environment mismatch")
    stream = case["mode"] == "stream"
    indices = range(len(fixture["batch_rows"])) if stream else [-1]
    expected = {
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
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"invalid evidence {key}: {row.get(key)!r} != {value!r}")
    if stream:
        status = row["diagnostics"]
        if (
            status["state"] != "completed"
            or status["terminal_cause"] != "natural_end"
            or status["task_count"]
            or status["task_errors"]
            or status["metrics_overflowed"]
        ):
            raise ValueError("incomplete lifecycle")
        if any(
            status["checkpoint"][key] is not None
            for key in ("current_epoch", "last_completed_epoch")
        ):
            raise ValueError("unexpected checkpoint lifecycle")
        if any(
            v["errors"] or not v["ended"]
            for group in ("sources", "operators", "sinks")
            for v in status[group].values()
        ):
            raise ValueError("unsettled actor lifecycle")


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


def compare_builds(builds):
    base, candidate = builds["baseline"], builds["candidate"]
    for key in ("cargo_lock_sha256", "harness_sha256", "rustc", "features"):
        if base[key] != candidate[key]:
            raise ValueError(f"incomparable build {key}")
    for key in (
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
        "RUSTUP_TOOLCHAIN",
        "CARGO_PROFILE_RELEASE_LTO",
        "CARGO_PROFILE_RELEASE_OPT_LEVEL",
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
        "CARGO_PROFILE_RELEASE_DEBUG",
    ):
        if base.get("build_environment", {}).get(key) != candidate.get(
            "build_environment", {}
        ).get(key):
            raise ValueError(f"incomparable build environment {key}")
    for config in (
        "Cargo.toml",
        "crates/calc-flow/Cargo.toml",
        "rust-toolchain.toml",
        ".cargo/config.toml",
    ):
        if base["source_sha256"].get(config) != candidate["source_sha256"].get(config):
            raise ValueError(f"incomparable compiled configuration: {config}")
    for shared in (
        "crates/calc-flow/examples/support/mod.rs",
        "crates/calc-flow/examples/support/stock_account_data.rs",
    ):
        left = base["source_sha256"].get(shared)
        if not left or left != candidate["source_sha256"].get(shared):
            raise ValueError(f"incomparable shared harness source: {shared}")
    vendors = [
        {
            name: digest
            for name, digest in build["source_sha256"].items()
            if name.startswith("vendor/")
        }
        for build in (base, candidate)
    ]
    if not vendors[0] or vendors[0] != vendors[1]:
        raise ValueError("incomparable vendored dependencies")


def guard(builds):
    bad = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "status"]):
        if process.pid == os.getpid() or process.info["status"] == psutil.STATUS_ZOMBIE:
            continue
        argv = process.info["cmdline"] or []
        if process.info["name"] in {
            "cargo",
            "rustc",
            "maturin",
            "pytest",
            "samply",
            "perf",
        } or any(
            "run_stock_positions_" in arg or arg == "benchmarks.stock_positions"
            for arg in argv
        ):
            bad.append(process.info)
    if bad:
        raise RuntimeError(f"competing build/measurement: {bad}")
    for build in builds.values():
        if sha(build["binary"]) != build["binary_sha256"]:
            raise ValueError("binary drift")


def invoke(command, env, path, deadline):
    path.mkdir(parents=True, exist_ok=False)
    write_json(
        path / "command.json",
        {
            "argv": command,
            "environment": {k: env[k] for k in THREAD_ENV},
        },
    )
    started = time.monotonic()
    peak = 0
    process = None
    try:
        with (
            (path / "stdout.jsonl").open("wb") as output,
            (path / "stderr.log").open("wb") as errors,
        ):
            process = subprocess.Popen(
                command,
                env=env,
                cwd=ROOT,
                stdout=output,
                stderr=errors,
                start_new_session=True,
            )
            while process.poll() is None:
                if time.monotonic() > deadline - 5:
                    raise TimeoutError("inclusive acquisition deadline")
                try:
                    parent = psutil.Process(process.pid)
                    peak = max(
                        peak,
                        sum(
                            p.memory_info().rss
                            for p in [parent, *parent.children(recursive=True)]
                            if p.is_running()
                        ),
                    )
                except psutil.Error:
                    pass
                time.sleep(0.02)
            if process.returncode:
                raise RuntimeError(f"command failed ({process.returncode}): {path}")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
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


def run(args):
    started = time.monotonic()
    deadline = started + 600
    if args.output.exists():
        raise FileExistsError(
            "use a new output directory; never overwrite a collection"
        )
    args.output.mkdir(parents=True)
    builds = {
        side: checked_build(getattr(args, side + "_build"))
        for side in ("baseline", "candidate")
    }
    compare_builds(builds)
    os.sched_setaffinity(0, {31})
    plan = {
        "cases": CASES,
        "rounds": 2,
        "pairs_per_round": 10,
        "warmups_per_arm_per_round": 1,
        "budget_seconds": 600,
        "builds_separate": True,
        "profiles": "baseline/candidate stream1CPU, six jobs each, first excluded"
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
    write_json(args.output / "plan.json", plan)
    machine = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "topology": subprocess.check_output(
            ["lscpu", "-e=CPU,CORE,SOCKET,ONLINE"], text=True
        ),
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
        tool_env = environment(1)
        if args.perf_library:
            tool_env["LD_LIBRARY_PATH"] = str(args.perf_library)
        machine["profile_tools"] = {
            name: {
                "path": str(path),
                "sha256": sha(path),
                "version": subprocess.check_output(
                    [str(path), "--version"],
                    text=True,
                    env=tool_env,
                    timeout=max(1, deadline - time.monotonic()),
                ).strip(),
            }
            for name, path in (("perf", args.perf), ("samply", args.samply))
        }
    write_json(args.output / "environment.json", machine)
    acquisition = {
        "status": "running",
        "started_monotonic": started,
        "deadline": deadline,
        "cases": [],
        "profiles": [],
        "failure": None,
    }
    write_json(args.output / "acquisition.json", acquisition)
    try:
        guard(builds)
        prepared, _ = invoke(
            [
                "taskset",
                "-c",
                "0",
                builds["baseline"]["binary"],
                "--prepare",
                str(args.output / "fixture"),
            ],
            environment(1),
            args.output / "prepare",
            deadline,
        )
        fixture = {row["fixture"]: row for row in prepared}
        write_json(args.output / "fixture-manifest.json", fixture)
        for case in CASES:
            record = {
                **case,
                "comparison": "interleaved",
                "correctness": True,
                "rows": fixture[case["fixture"]]["input_rows"],
                "baseline": [],
                "candidate": [],
                "observations": [],
            }
            acquisition["cases"].append(record)
            for round_index in range(2):
                round_path = (
                    args.output / "timings" / case["id"] / f"round-{round_index + 1}"
                )
                for side in builds:
                    measure(
                        args,
                        builds,
                        case,
                        fixture,
                        side,
                        round_path / ("warmup-" + side),
                        deadline,
                    )
                samples = {side: [] for side in builds}
                for pair in range(10):
                    pair_rows = {}
                    for side in (
                        ("baseline", "candidate")
                        if pair % 2 == 0
                        else ("candidate", "baseline")
                    ):
                        path = round_path / f"pair-{pair + 1}-{side}"
                        row, receipt = measure(
                            args, builds, case, fixture, side, path, deadline
                        )
                        samples[side].append(row["seconds"])
                        pair_rows[side] = row
                        record["observations"].append(
                            {
                                "round": round_index + 1,
                                "pair": pair + 1,
                                "side": side,
                                "path": str(path.relative_to(args.output)),
                                "row": row,
                                "resources": receipt,
                            }
                        )
                    compare_oracles(pair_rows["baseline"], pair_rows["candidate"])
                    write_json(args.output / "acquisition.json", acquisition)
                for side in builds:
                    record[side].append(samples[side])
            record["statistics"] = comparison(record)
            record["resource_summary"] = {
                side: {
                    "processing_cpu_ms_p50": statistics.median(
                        o["row"]["process_cpu_ns"] / 1e6
                        for o in record["observations"]
                        if o["side"] == side
                    ),
                    "processing_average_cores_p50": statistics.median(
                        o["row"]["process_cpu_ns"] / 1e9 / o["row"]["seconds"]
                        for o in record["observations"]
                        if o["side"] == side
                    ),
                    "whole_process_peak_rss_p50": statistics.median(
                        o["resources"]["sampled_process_tree_peak_rss"]
                        for o in record["observations"]
                        if o["side"] == side
                    ),
                }
                for side in builds
            }
            print(
                json.dumps({"case": case["id"], "statistics": record["statistics"]}),
                flush=True,
            )
        if args.perf:
            profile_case = CASES[0]
            for side in builds:
                path = args.output / "profiles" / side
                rows, receipt = measure(
                    args,
                    builds,
                    profile_case,
                    fixture,
                    side,
                    path,
                    deadline,
                    profile=True,
                )
                acquisition["profiles"].append(
                    {
                        "side": side,
                        "path": str(path.relative_to(args.output)),
                        "rows": rows,
                        "windows": [
                            {"start_ns": r["start_ns"], "end_ns": r["end_ns"]}
                            for r in rows[1:]
                        ],
                        "resources": receipt,
                        "perf_data_sha256": sha(path / "perf.data"),
                    }
                )
                write_json(args.output / "acquisition.json", acquisition)
        if args.perf:
            from scripts.benchmark_suite.sum_state_profile import decode

            decode(args, acquisition, builds, deadline)
        for build in builds.values():
            verify_build(build)
        write_samples(args.output / "samples.csv", acquisition)
        if time.monotonic() > deadline:
            raise TimeoutError("inclusive acquisition deadline after preflight/cleanup")
        acquisition["status"] = "completed"
    except BaseException as error:
        acquisition["status"] = "stopped"
        acquisition["failure"] = repr(error)
        raise
    finally:
        write_samples(args.output / "samples.csv", acquisition)
        acquisition["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "acquisition.json", acquisition)
        write_json(
            args.output / "summary.json",
            {
                "status": acquisition["status"],
                "elapsed_seconds": acquisition["elapsed_seconds"],
                "cases": [
                    {k: v for k, v in case.items() if k != "observations"}
                    for case in acquisition["cases"]
                ],
                "profiles": acquisition["profiles"],
                "failure": acquisition["failure"],
            },
        )


def measure(args, builds, case, fixtures, side, path, deadline, *, profile=False):
    guard(builds)
    build = builds[side]
    command = [
        "taskset",
        "-c",
        ",".join(map(str, case["cpus"])),
        build["binary"],
        str(args.output / "fixture" / (case["fixture"] + ".arrow")),
        case["mode"],
        str(case["partitions"]),
        "6" if profile else "1",
    ]
    env = environment(len(case["cpus"]))
    if args.perf_library:
        env["LD_LIBRARY_PATH"] = str(args.perf_library)
    if profile:
        command = [
            str(args.perf),
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
    rows, receipt = invoke(command, env, path, deadline)
    if len(rows) != (6 if profile else 1):
        raise ValueError("wrong invocation count")
    for row in rows:
        validate_sample(row, case, fixtures[case["fixture"]])
    guard(builds)
    return (rows if profile else rows[0]), receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build", required=True, type=Path)
    parser.add_argument("--candidate-build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=lambda x: Path(x).resolve())
    parser.add_argument("--perf", type=lambda x: Path(x).resolve())
    parser.add_argument("--perf-library", type=lambda x: Path(x).resolve())
    parser.add_argument("--samply", type=lambda x: Path(x).resolve())
    args = parser.parse_args()
    if bool(args.perf) != bool(args.samply):
        parser.error("--perf and --samply are required together")
    run(args)


if __name__ == "__main__":
    main()
