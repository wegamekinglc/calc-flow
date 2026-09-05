"""Build provenance-bound wheels and profile paired, persistent streaming workers."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
HISTORY_ROWS = (10_240, 102_400, 1_024_000, 10_240_000)
APPEND_ROWS = (64, 640, 6_400, 64_000)
PHASE_NAMES = (
    "enqueue_to_source_data",
    "source_data_to_source_watermark",
    "source_watermark_to_sink",
    "sink_to_receive",
    "to_pyarrow",
)
SOURCE_PATHS = (
    "crates",
    "python",
    "Cargo.toml",
    "Cargo.lock",
    "pyproject.toml",
    "rust-toolchain.toml",
)
SOURCE_LIST_COMMAND = (
    "ls-files",
    "--cached",
    "--others",
    "--exclude-standard",
    "-z",
    "--",
    *SOURCE_PATHS,
)


def matrix_points(
    history_rows: tuple[int, ...] = HISTORY_ROWS,
    append_rows: tuple[int, ...] = APPEND_ROWS,
) -> list[tuple[int, int]]:
    return sorted(
        {
            *((rows, 64) for rows in history_rows),
            *((1_024_000, rows) for rows in append_rows),
        }
    )


def matrix_cases(args: argparse.Namespace) -> list[tuple[dict[str, Any], str]]:
    points = (
        product(args.history_rows, args.append_rows)
        if args.append_entities is not None
        else matrix_points(tuple(args.history_rows), tuple(args.append_rows))
    )
    cases = []
    for indicator, (history, append), active, mode in product(
        args.indicators, points, args.append_entities or [None], args.gc_modes
    ):
        config = {
            "history_rows": history,
            "append_rows": append,
            "indicator": indicator,
        }
        if active is not None:
            config["append_entities"] = active
        cases.append((config, mode))
    random.Random(args.seed).shuffle(cases)
    return cases


def _summary(values: np.ndarray, rows: int) -> dict[str, Any]:
    median = float(np.median(values))
    return {
        "p50_seconds": median,
        "p95_seconds": float(np.percentile(values, 95)),
        "cv": float(np.std(values, ddof=1) / np.mean(values)),
        "rows_per_second": rows / median,
    }


def paired_summary(
    baseline: list[float],
    candidate: list[float],
    *,
    rows: int,
) -> dict[str, Any]:
    left, right = np.asarray(baseline), np.asarray(candidate)
    if (
        left.ndim != 1
        or right.shape != left.shape
        or len(left) < 2
        or not _positive_finite(left)
        or not _positive_finite(right)
        or rows <= 0
    ):
        raise ValueError(
            "paired samples must have equal lengths and finite positive times"
        )
    ratios = left / right
    rng = np.random.default_rng(771031)
    indices = rng.integers(0, len(ratios), size=(20_000, len(ratios)))
    estimates = np.median(ratios[indices], axis=1)
    return {
        "baseline": _summary(left, rows),
        "candidate": _summary(right, rows),
        "paired_speedup_median": float(np.median(ratios)),
        "paired_speedup_ci95": np.percentile(estimates, [2.5, 97.5]).tolist(),
        "bootstrap_resamples": 20_000,
    }


def _positive_finite(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).all() and np.all(values > 0))


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _wheel_native_sha256(path: Path) -> str:
    with zipfile.ZipFile(path) as wheel:
        names = [
            name
            for name in wheel.namelist()
            if name.startswith("calc_flow/_native") and name.endswith((".so", ".pyd"))
        ]
        if len(names) != 1:
            raise ValueError("wheel must contain exactly one native module")
        with wheel.open(names[0]) as native:
            return hashlib.file_digest(native, "sha256").hexdigest()


async def _command_output(process: asyncio.subprocess.Process) -> str:
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(
            f"benchmark command failed ({process.returncode}): {stderr.decode()}"
        )
    return stdout.decode().strip()


async def _git(source: Path, *args: str) -> str:
    options = {
        "cwd": source,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
    }
    match args:
        case ("rev-parse", "HEAD"):
            process = await asyncio.create_subprocess_exec(
                "git", "rev-parse", "HEAD", **options
            )
        case ("status", "--porcelain"):
            process = await asyncio.create_subprocess_exec(
                "git", "status", "--porcelain", **options
            )
        case _ if args == SOURCE_LIST_COMMAND:
            process = await asyncio.create_subprocess_exec(
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                "crates",
                "python",
                "Cargo.toml",
                "Cargo.lock",
                "pyproject.toml",
                "rust-toolchain.toml",
                **options,
            )
        case _:
            raise ValueError("unsupported git metadata command")
    return await _command_output(process)


async def _rustc_version(source: Path) -> str:
    process = await asyncio.create_subprocess_exec(
        "rustc",
        "-Vv",
        cwd=source,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    return await _command_output(process)


async def source_identity(source: Path) -> dict[str, Any]:
    paths = (await _git(source, *SOURCE_LIST_COMMAND)).split("\0")
    digest = hashlib.sha256()
    for name in sorted(set(paths) - {""}):
        path = source / name
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<deleted>")
        digest.update(b"\0")
    return {
        "git_sha": await _git(source, "rev-parse", "HEAD"),
        "git_clean": not bool(await _git(source, "status", "--porcelain")),
        "source_sha256": digest.hexdigest(),
        "cargo_lock_sha256": _sha256(source / "Cargo.lock"),
    }


def _python_environment(environment: dict[str, str]) -> dict[str, str]:
    """Use a static launcher name only when it resolves to this interpreter."""
    executable = Path(sys.executable)
    search_path = os.pathsep.join((str(executable.parent), environment.get("PATH", "")))
    selected = shutil.which("python", path=search_path)
    if selected is None or not Path(selected).samefile(executable):
        raise ValueError(
            "python launcher must resolve to the current Python interpreter"
        )
    return {**environment, "PATH": search_path}


async def build(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = await source_identity(source)
    env = _python_environment(
        dict(
            os.environ,
            CARGO_TARGET_DIR=str(args.target.resolve()),
            PYO3_PYTHON=sys.executable,
        )
    )
    command = [
        "python",
        "-m",
        "maturin",
        "build",
        "--release",
        "--locked",
        "--out",
        str(output),
    ]
    with (output / "build.log").open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            "python",
            "-m",
            "maturin",
            "build",
            "--release",
            "--locked",
            "--out",
            str(output),
            cwd=source,
            env=env,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
        )
        if await process.wait():
            raise RuntimeError(f"release build failed; see {output / 'build.log'}")
    if before != await source_identity(source):
        raise RuntimeError("source changed during release build")
    wheels = list(output.glob("calc_flow-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("build output must contain exactly one core wheel")
    manifest = {
        "contract": "warm-stream-build-v1",
        "source": str(source),
        **before,
        "wheel": str(wheels[0]),
        "wheel_sha256": _sha256(wheels[0]),
        "native_sha256": _wheel_native_sha256(wheels[0]),
        "build_profile": "release",
        "command": command,
        "rustc": await _rustc_version(source),
        "python": platform.python_version(),
        "python_executable": sys.executable,
    }
    path = output / "build.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(path, flush=True)


async def _load_build(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("contract") != "warm-stream-build-v1"
        or manifest.get("build_profile") != "release"
    ):
        raise ValueError("expected a warm-stream release build manifest")
    if _sha256(Path(manifest["wheel"])) != manifest["wheel_sha256"]:
        raise ValueError("wheel hash changed after build")
    if _wheel_native_sha256(Path(manifest["wheel"])) != manifest["native_sha256"]:
        raise ValueError("native module hash does not match build manifest")
    identity = await source_identity(Path(manifest["source"]))
    if any(identity[key] != manifest[key] for key in ("source_sha256", "git_sha")):
        raise ValueError("source changed after wheel build")
    return manifest


class Worker:
    """Own a benchmark subprocess and keep IPC outside its timed interval."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process

    @classmethod
    async def start(
        cls, manifest: dict[str, Any], root: Path, *, tokio_workers: int | None = None
    ) -> Worker:
        site = root / "site"
        wheel = Path(manifest["wheel"]).resolve(strict=True)
        installer = await asyncio.create_subprocess_exec(
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--target",
            str(site),
            str(wheel),
            cwd=REPOSITORY,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await _command_output(installer)
        env = _python_environment(
            _configured_worker_environment(
                dict(
                    os.environ,
                    PYTHONPATH=os.pathsep.join((str(site), str(REPOSITORY))),
                    PYTHONDONTWRITEBYTECODE="1",
                ),
                tokio_workers,
            )
        )
        process = await asyncio.create_subprocess_exec(
            "python",
            "-m",
            "scripts.profile_warm_stream",
            "_worker",
            "--root",
            str(root),
            cwd=REPOSITORY,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        return cls(process)

    async def request(self, **message: Any) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("warm worker requires stdin and stdout pipes")
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()
        response = await asyncio.wait_for(self.process.stdout.readline(), timeout=180)
        if not response:
            raise RuntimeError(f"warm worker exited: {await self.process.wait()}")
        parsed = json.loads(response)
        if not isinstance(parsed, dict):
            raise ValueError("warm worker response must be an object")
        if "error" in parsed:
            raise RuntimeError(parsed["error"])
        return parsed

    async def close(self) -> None:
        if self.process.returncode is None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=30)
            except TimeoutError:
                self.process.terminate()
                await self.process.wait()


def _configured_worker_environment(
    environment: dict[str, str], workers: int | None
) -> dict[str, str]:
    if workers is None:
        return dict(environment)
    if type(workers) is not int or workers < 1:
        raise ValueError("worker thread override must be a positive integer")
    return {**environment, "TOKIO_WORKER_THREADS": str(workers)}


def _cpu_affinity() -> list[int] | None:
    import psutil

    process = psutil.Process()
    if not hasattr(process, "cpu_affinity"):
        return None
    try:
        return process.cpu_affinity()
    except (psutil.Error, NotImplementedError):
        return None


def _worker_environment() -> dict[str, Any]:
    import pyarrow as pa

    import calc_flow._native as native

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": _cpu_affinity(),
        "native_sha256": _sha256(Path(native.__file__)),
        "tokio_worker_threads": os.environ.get("TOKIO_WORKER_THREADS"),
    }


class WorkerSession:
    """Own and clean up a single worker's active streaming scenario."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.scenario = None
        self.count = 0
        self.profiling = False

    async def start(self, config_values: dict[str, Any]) -> dict[str, Any]:
        from benchmarks.warm_stream import ScenarioConfig, WarmScenario

        if self.scenario is not None:
            raise RuntimeError("previous warm scenario is still active")
        config = ScenarioConfig(**config_values)
        self.count += 1
        self.scenario = await WarmScenario.start(
            config, self.root / f"state-{self.count}"
        )
        return {"config": asdict(config), "warm_seconds": self.scenario.warm_seconds}

    def active_scenario(self):
        if self.scenario is None:
            raise RuntimeError("warm scenario was not started")
        return self.scenario

    async def finish(self) -> dict[str, Any]:
        import psutil

        result = await self.active_scenario().finish()
        if self.profiling:
            result["callback_profile"] = json.loads(
                self.active_scenario().job._inner._take_callback_profile()
            )
        result["rss_bytes"] = psutil.Process().memory_info().rss
        self.scenario = None
        self.profiling = False
        return result

    async def dispatch(self, message: dict[str, Any]) -> dict[str, Any]:
        match message["operation"]:
            case "hello":
                return _worker_environment()
            case "start":
                return await self.start(message["config"])
            case "sample":
                return await self.active_scenario().sample(
                    collect_gc=message["collect_gc"]
                )
            case "finish":
                return await self.finish()
            case "enable_profile":
                self.active_scenario().job._inner._enable_callback_profiling()
                self.profiling = True
                return {"enabled": True}
            case _:
                raise ValueError("unknown worker operation")

    async def close(self) -> None:
        if self.scenario is not None:
            await self.scenario.job.cancel_async()
            self.scenario = None


async def worker_main(root: Path) -> None:
    session = WorkerSession(root)
    try:
        while line := await asyncio.to_thread(sys.stdin.readline):
            message = json.loads(line)
            result = await session.dispatch(message)
            print(json.dumps(result, allow_nan=False), flush=True)
    except Exception as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)
        raise
    finally:
        await session.close()


def _validate_sample(sample: dict[str, Any], expected_start: int) -> None:
    if sample["start_row"] != expected_start or not sample["correctness"]["passed"]:
        raise ValueError("paired workers did not advance equivalent correct states")


def _phase_values(sample: dict[str, Any]) -> np.ndarray:
    try:
        values = np.asarray(
            [sample["phases_seconds"][name] for name in PHASE_NAMES], dtype=float
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("sample must contain all numeric phase intervals") from error
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("phase intervals must be finite and nonnegative")
    if not np.isclose(values[:-1].sum(), sample["seconds"], rtol=1e-9, atol=1e-12):
        raise ValueError("non-overlapping phase intervals must cover the timed sample")
    if values[-1] > values[-2] + 1e-12:
        raise ValueError("to_pyarrow must remain within the sink-to-receive interval")
    return values


def _phase_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("phase summaries require samples")
    values = np.asarray([_phase_values(sample) for sample in samples])
    return {
        name: {
            "p50_seconds": float(np.median(values[:, index])),
            "p95_seconds": float(np.percentile(values[:, index], 95)),
        }
        for index, name in enumerate(PHASE_NAMES)
    }


def _case_summary(measured: list[list[dict[str, Any]]], rows: int) -> dict[str, Any]:
    latency = paired_summary(
        [sample["seconds"] for sample in measured[0]],
        [sample["seconds"] for sample in measured[1]],
        rows=rows,
    )
    return {
        **latency,
        "phase_intervals": {
            "baseline": _phase_summary(measured[0]),
            "candidate": _phase_summary(measured[1]),
        },
    }


async def _measure_case(
    workers: list[Worker], config: dict[str, Any], samples: int, collect_gc: bool
) -> dict[str, Any]:
    for worker in workers:
        await worker.request(operation="start", config=config)
    measured: list[list[dict[str, Any]]] = [[], []]
    for index in range(samples):
        order = (0, 1) if index % 2 == 0 else (1, 0)
        for side in order:
            sample = await workers[side].request(
                operation="sample", collect_gc=collect_gc
            )
            expected = config["history_rows"] + index * config["append_rows"]
            _validate_sample(sample, expected)
            measured[side].append(sample)
    terminal = [await worker.request(operation="finish") for worker in workers]
    return {
        "config": config,
        "collect_gc": collect_gc,
        "samples": measured,
        "terminal": terminal,
        **_case_summary(measured, config["append_rows"]),
    }


def markdown(report: dict[str, Any]) -> str:
    thread_counts = report.get("worker_threads")
    lines = [
        "# Warm-state streaming profile",
        "",
        "Paired release builds; times include the full Python StreamingRunner path.",
        (
            f"Same-wheel configuration experiment: Tokio workers "
            f"{thread_counts[0]} → {thread_counts[1]}."
            if thread_counts
            else "Engine comparison with unchanged scheduler configuration."
        ),
        "",
        "| Indicator | History | Append | Active entities | Forced GC "
        "| Baseline P50 / P95 ms "
        "| Candidate P50 / P95 ms | Paired speedup (95% CI) |",
        "| --------- | ------- | ------ | --------------- | --------- "
        "| --------------------- "
        "| ---------------------- | ----------------------- |",
    ]
    for case in report["cases"]:
        config, left, right = case["config"], case["baseline"], case["candidate"]
        low, high = case["paired_speedup_ci95"]
        lines.append(
            f"| {config['indicator']} | {config['history_rows']:,} "
            f"| {config['append_rows']:,} "
            f"| {config.get('append_entities', 'full tick')} "
            f"| {case['collect_gc']} | "
            f"{left['p50_seconds'] * 1000:.3f} / {left['p95_seconds'] * 1000:.3f} | "
            f"{right['p50_seconds'] * 1000:.3f} / {right['p95_seconds'] * 1000:.3f} | "
            f"{case['paired_speedup_median']:.2f}x ({low:.2f}–{high:.2f}) |"
        )
    lines.extend(_phase_markdown(report["cases"]))
    return "\n".join(lines) + "\n"


def _phase_markdown(cases: list[dict[str, Any]]) -> list[str]:
    lines = [
        "",
        "## Callback interval breakdown",
        "",
        "Representative H=1,024,000 forced-GC cases; independent P50 values in ms.",
        "These wall intervals include scheduling and waiting, not just CPU work.",
        "Conversion is a subset of sink-to-receive and must not be added again.",
        "Independent phase medians need not sum to the total latency median.",
        "",
        "| Indicator | Append | Active entities | Build "
        "| Enqueue to source | Data to watermark "
        "| Watermark to sink | Sink to receive | to_pyarrow subset |",
        "| --------- | ------ | --------------- | ----- "
        "| ----------------- | ----------------- "
        "| ----------------- | --------------- | ----------------- |",
    ]
    for case in cases:
        config = case["config"]
        if config["history_rows"] != 1_024_000 or not case["collect_gc"]:
            continue
        for side in ("baseline", "candidate"):
            phases = case["phase_intervals"][side]
            durations = " | ".join(
                f"{phases[name]['p50_seconds'] * 1000:.3f}" for name in PHASE_NAMES
            )
            lines.append(
                f"| {config['indicator']} | {config['append_rows']:,} "
                f"| {config.get('append_entities', 'full tick')} "
                f"| {side} | {durations} |"
            )
    return lines


async def _compatible_builds(paths: tuple[Path, Path]) -> list[dict[str, Any]]:
    manifests = [await _load_build(path) for path in paths]
    if manifests[0]["cargo_lock_sha256"] != manifests[1]["cargo_lock_sha256"]:
        raise ValueError("baseline and candidate dependencies differ")
    if any(manifests[0][key] != manifests[1][key] for key in ("rustc", "python")):
        raise ValueError("baseline and candidate build toolchains differ")
    return manifests


async def _worker_environments(
    workers: list[Worker],
    manifests: list[dict[str, Any]],
    *,
    thread_counts: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    environments = [await worker.request(operation="hello") for worker in workers]
    for environment, manifest in zip(environments, manifests, strict=True):
        if environment["native_sha256"] != manifest["native_sha256"]:
            raise ValueError("worker loaded a different native module")
    excluded = {"native_sha256"}
    if thread_counts is not None:
        actual = [env.get("tokio_worker_threads") for env in environments]
        if actual != [str(count) for count in thread_counts]:
            raise ValueError("worker did not load the declared thread override")
        excluded.add("tokio_worker_threads")
    comparable = [
        {key: value for key, value in env.items() if key not in excluded}
        for env in environments
    ]
    if comparable[0] != comparable[1]:
        raise ValueError("worker machine or dependency fingerprints differ")
    return environments


def _new_report(
    manifests: list[dict[str, Any]], environments: list[dict[str, Any]], samples: int
) -> dict[str, Any]:
    return {
        "contract": "paired-warm-stream-v1",
        "builds": manifests,
        "environments": environments,
        "sample_pairs": samples,
        "harness_sha256": {
            name: _sha256(REPOSITORY / name)
            for name in (
                "scripts/profile_warm_stream.py",
                "benchmarks/warm_stream.py",
                "benchmarks/rolling_indicator_comparison.py",
            )
        },
        "timed_boundary": (
            "prepared Data enqueue through source/task/channel, rolling, "
            "watermark finalization, projection, sink and to_pyarrow; "
            "excludes IPC, compile, startup, history, validation, "
            "checkpoint, shutdown"
        ),
        "phase_note": (
            "callback wall intervals overlap operator work; to_pyarrow "
            "is contained in sink_to_receive; operator counters span "
            "post-warm through terminal handling, not individual timed samples; "
            "metric accounting depends on the measured source revision"
        ),
        "gc_note": (
            "forced collection happens before timing; normal_gc does "
            "not disable GC or remove IPC gaps between appends"
        ),
    }


@dataclass(frozen=True, slots=True)
class WorkerPairOptions:
    directory: Path
    thread_counts: tuple[int, int] | None = None


async def _fresh_case(
    manifests: list[dict[str, Any]],
    config: dict[str, Any],
    samples: int,
    collect_gc: bool,
    options: WorkerPairOptions,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    thread_counts = options.thread_counts
    workers: list[Worker] = []
    with tempfile.TemporaryDirectory(
        prefix="warm-paired-", dir=options.directory
    ) as temporary:
        try:
            for index, manifest in enumerate(manifests):
                root = Path(temporary) / str(index)
                root.mkdir()
                count = None if thread_counts is None else thread_counts[index]
                workers.append(await Worker.start(manifest, root, tokio_workers=count))
            environments = await _worker_environments(
                workers, manifests, thread_counts=thread_counts
            )
            case = await _measure_case(workers, config, samples, collect_gc)
            return case, environments
        finally:
            for worker in workers:
                await worker.close()


async def compare(args: argparse.Namespace) -> None:
    manifests = await _compatible_builds((args.baseline_build, args.candidate_build))
    counts = None if args.worker_threads is None else tuple(args.worker_threads)
    if counts is not None and any(
        manifests[0][key] != manifests[1][key]
        for key in ("native_sha256", "wheel_sha256")
    ):
        raise ValueError(
            "scheduler comparison requires the same native wheel on both sides"
        )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for config, mode in matrix_cases(args):
        case, environments = await _fresh_case(
            manifests,
            config,
            args.samples,
            mode == "forced",
            WorkerPairOptions(output.parent, counts),
        )
        if not report:
            report = {
                **_new_report(manifests, environments, args.samples),
                "worker_lifetime": "fresh process pair per case",
                "case_order_seed": args.seed,
                "callback_profiling": False,
                "worker_threads": counts,
                "comparison_kind": "engine"
                if counts is None
                else "same-wheel scheduler configuration",
                "cases": [],
            }
        if environments != report["environments"]:
            raise ValueError("worker fingerprints changed between cases")
        report["cases"].append(case)
        print(f"{config} GC={mode}: {case['paired_speedup_median']:.2f}x", flush=True)
    if not report:
        raise ValueError("benchmark matrix must not be empty")
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


async def diagnose(args: argparse.Namespace) -> None:
    manifest = await _load_build(args.build)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "history_rows": args.history_rows,
        "append_rows": args.append_rows,
        "append_entities": args.append_entities,
        "indicator": args.indicator,
    }
    with tempfile.TemporaryDirectory(
        prefix="warm-diagnostic-", dir=output.parent
    ) as temporary:
        worker = await Worker.start(manifest, Path(temporary))
        try:
            environments = await _worker_environments(
                [worker, worker], [manifest, manifest]
            )
            await worker.request(operation="start", config=config)
            await worker.request(operation="enable_profile")
            samples = []
            for index in range(args.samples):
                sample = await worker.request(
                    operation="sample", collect_gc=args.forced_gc
                )
                _validate_sample(sample, args.history_rows + index * args.append_rows)
                samples.append(sample)
            terminal = await worker.request(operation="finish")
        finally:
            await worker.close()
    report = {
        **_new_report([manifest], environments[:1], args.samples),
        "contract": "warm-callback-diagnostics-v1",
        "callback_profiling": True,
        "config": config,
        "collect_gc": args.forced_gc,
        "sample_pairs": 0,
        "samples": samples,
        "terminal": terminal,
        "diagnostic_note": "Instrumented wall times; "
        "not performance evidence or CPU shares. "
        "Requests may span prearmed source waits, IPC gaps, or terminal controls; "
        "records are request-relative offsets, not aligned append-stage costs.",
    }
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--source", type=Path, required=True)
    builder.add_argument("--target", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    paired = commands.add_parser("compare")
    paired.add_argument("--baseline-build", type=Path, required=True)
    paired.add_argument("--candidate-build", type=Path, required=True)
    paired.add_argument("--output", type=Path, required=True)
    paired.add_argument("--samples", type=int, default=30)
    paired.add_argument("--history-rows", type=int, nargs="*", default=HISTORY_ROWS)
    paired.add_argument("--append-rows", type=int, nargs="*", default=APPEND_ROWS)
    paired.add_argument("--append-entities", type=int, nargs="+")
    paired.add_argument("--seed", type=int, default=20260905)
    paired.add_argument(
        "--worker-threads", type=int, nargs=2, metavar=("BASELINE", "CANDIDATE")
    )
    paired.add_argument(
        "--indicators",
        nargs="+",
        choices=("rolling_mean", "dual_sma_spread"),
        default=("rolling_mean", "dual_sma_spread"),
    )
    paired.add_argument(
        "--gc-modes",
        nargs="+",
        choices=("forced", "normal"),
        default=("forced", "normal"),
    )
    worker = commands.add_parser("_worker")
    worker.add_argument("--root", type=Path, required=True)
    diagnostic = commands.add_parser("diagnose")
    diagnostic.add_argument("--build", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    diagnostic.add_argument("--history-rows", type=int, default=1_024_000)
    diagnostic.add_argument("--append-rows", type=int, default=64)
    diagnostic.add_argument("--append-entities", type=int)
    diagnostic.add_argument(
        "--indicator",
        choices=("rolling_mean", "dual_sma_spread"),
        default="rolling_mean",
    )
    diagnostic.add_argument("--samples", type=int, default=30)
    diagnostic.add_argument("--forced-gc", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        asyncio.run(build(args))
    elif args.command == "_worker":
        asyncio.run(worker_main(args.root))
    elif args.command == "diagnose":
        if args.samples < 2:
            parser.error("at least two diagnostic samples are required")
        asyncio.run(diagnose(args))
    else:
        if args.samples < 2:
            parser.error("at least two paired samples are required")
        if args.worker_threads is not None and min(args.worker_threads) < 1:
            parser.error("worker thread overrides must be positive")
        asyncio.run(compare(args))


if __name__ == "__main__":
    main()
