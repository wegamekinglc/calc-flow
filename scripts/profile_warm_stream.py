"""Build provenance-bound wheels and profile paired, persistent streaming workers."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import sys
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
HISTORY_ROWS = (10_240, 102_400, 1_024_000, 10_240_000)
APPEND_ROWS = (64, 640, 6_400, 64_000)


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
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any(left <= 0)
        or np.any(right <= 0)
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


async def _command(argv: list[str], *, cwd: Path, env=None) -> str:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise RuntimeError(
            f"{argv[0]} failed ({process.returncode}): {stderr.decode()}"
        )
    return stdout.decode().strip()


async def source_identity(source: Path) -> dict[str, Any]:
    async def git(*args: str) -> str:
        return await _command(["git", *args], cwd=source)

    paths = (
        await git(
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
        )
    ).split("\0")
    digest = hashlib.sha256()
    for name in sorted(set(paths) - {""}):
        path = source / name
        digest.update(name.encode() + b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<deleted>")
        digest.update(b"\0")
    return {
        "git_sha": await git("rev-parse", "HEAD"),
        "git_clean": not bool(await git("status", "--porcelain")),
        "source_sha256": digest.hexdigest(),
        "cargo_lock_sha256": _sha256(source / "Cargo.lock"),
    }


async def build(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = await source_identity(source)
    env = dict(
        os.environ,
        CARGO_TARGET_DIR=str(args.target.resolve()),
        PYO3_PYTHON=sys.executable,
    )
    command = [
        sys.executable,
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
            *command,
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
        "rustc": await _command(["rustc", "-Vv"], cwd=source),
        "python": platform.python_version(),
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
    async def start(cls, manifest: dict[str, Any], root: Path) -> Worker:
        site = root / "site"
        await _command(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "--target",
                str(site),
                manifest["wheel"],
            ],
            cwd=REPOSITORY,
        )
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join((str(site), str(REPOSITORY))),
            PYTHONDONTWRITEBYTECODE="1",
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).resolve()),
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
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()
        response = await asyncio.wait_for(self.process.stdout.readline(), timeout=180)
        if not response:
            raise RuntimeError(f"warm worker exited: {await self.process.wait()}")
        parsed = json.loads(response)
        if "error" in parsed:
            raise RuntimeError(parsed["error"])
        return parsed

    async def close(self) -> None:
        if self.process.returncode is None:
            assert self.process.stdin is not None
            self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=30)
            except TimeoutError:
                self.process.terminate()
                await self.process.wait()


async def worker_main(root: Path) -> None:
    import psutil
    import pyarrow as pa

    import calc_flow._native as native
    from benchmarks.warm_stream import ScenarioConfig, WarmScenario

    scenario = None
    count = 0
    try:
        while line := await asyncio.to_thread(sys.stdin.readline):
            message = json.loads(line)
            match message["operation"]:
                case "hello":
                    result = {
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "pyarrow": pa.__version__,
                        "platform": platform.platform(),
                        "logical_cpus": os.cpu_count(),
                        "cpu_affinity": psutil.Process().cpu_affinity()
                        if hasattr(psutil.Process, "cpu_affinity")
                        else None,
                        "native_sha256": _sha256(Path(native.__file__)),
                    }
                case "start":
                    if scenario is not None:
                        raise RuntimeError("previous warm scenario is still active")
                    config = ScenarioConfig(**message["config"])
                    count += 1
                    scenario = await WarmScenario.start(config, root / f"state-{count}")
                    result = {
                        "config": asdict(config),
                        "warm_seconds": scenario.warm_seconds,
                    }
                case "sample":
                    if scenario is None:
                        raise RuntimeError("warm scenario was not started")
                    result = await scenario.sample(collect_gc=message["collect_gc"])
                case "finish":
                    if scenario is None:
                        raise RuntimeError("warm scenario was not started")
                    result = await scenario.finish()
                    result["rss_bytes"] = psutil.Process().memory_info().rss
                    scenario = None
                case _:
                    raise ValueError("unknown worker operation")
            print(json.dumps(result, allow_nan=False), flush=True)
    except Exception as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)
        raise
    finally:
        if scenario is not None:
            await scenario.job.cancel_async()


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
            if sample["start_row"] != expected or not sample["correctness"]["passed"]:
                raise ValueError(
                    "paired workers did not advance equivalent correct states"
                )
            measured[side].append(sample)
    terminal = [await worker.request(operation="finish") for worker in workers]
    summary = paired_summary(
        [sample["seconds"] for sample in measured[0]],
        [sample["seconds"] for sample in measured[1]],
        rows=config["append_rows"],
    )
    return {
        "config": config,
        "collect_gc": collect_gc,
        "samples": measured,
        "terminal": terminal,
        **summary,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Warm-state streaming profile",
        "",
        "Paired release builds; times include the full Python StreamingRunner path.",
        "",
        "| Indicator | History | Append | Forced GC | Baseline P50 / P95 ms "
        "| Candidate P50 / P95 ms | Paired speedup (95% CI) |",
        "| --------- | ------- | ------ | --------- | --------------------- "
        "| ---------------------- | ----------------------- |",
    ]
    for case in report["cases"]:
        config, left, right = case["config"], case["baseline"], case["candidate"]
        low, high = case["paired_speedup_ci95"]
        lines.append(
            f"| {config['indicator']} | {config['history_rows']:,} "
            f"| {config['append_rows']:,} | {case['collect_gc']} | "
            f"{left['p50_seconds'] * 1000:.3f} / {left['p95_seconds'] * 1000:.3f} | "
            f"{right['p50_seconds'] * 1000:.3f} / {right['p95_seconds'] * 1000:.3f} | "
            f"{case['paired_speedup_median']:.2f}x ({low:.2f}–{high:.2f}) |"
        )
    return "\n".join(lines) + "\n"


async def compare(args: argparse.Namespace) -> None:
    manifests = [
        await _load_build(path) for path in (args.baseline_build, args.candidate_build)
    ]
    if manifests[0]["cargo_lock_sha256"] != manifests[1]["cargo_lock_sha256"]:
        raise ValueError("baseline and candidate dependencies differ")
    if any(manifests[0][key] != manifests[1][key] for key in ("rustc", "python")):
        raise ValueError("baseline and candidate build toolchains differ")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    workers: list[Worker] = []
    with tempfile.TemporaryDirectory(
        prefix="warm-paired-", dir=output.parent
    ) as temporary:
        try:
            for index, manifest in enumerate(manifests):
                root = Path(temporary) / str(index)
                root.mkdir()
                workers.append(await Worker.start(manifest, root))
            environments = [
                await worker.request(operation="hello") for worker in workers
            ]
            for environment, manifest in zip(environments, manifests, strict=True):
                if environment["native_sha256"] != manifest["native_sha256"]:
                    raise ValueError("worker loaded a different native module")
            comparable = [
                {key: value for key, value in env.items() if key != "native_sha256"}
                for env in environments
            ]
            if comparable[0] != comparable[1]:
                raise ValueError("worker machine or dependency fingerprints differ")
            report = {
                "contract": "paired-warm-stream-v1",
                "builds": manifests,
                "environments": environments,
                "sample_pairs": args.samples,
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
                    "is contained in sink_to_receive; baseline operator "
                    "processing metrics omit watermark handling"
                ),
                "gc_note": (
                    "forced collection happens before timing; normal_gc does "
                    "not disable GC or remove IPC gaps between appends"
                ),
                "cases": [],
            }
            points = matrix_points(tuple(args.history_rows), tuple(args.append_rows))
            for indicator in args.indicators:
                for history, append in points:
                    for mode in args.gc_modes:
                        config = {
                            "history_rows": history,
                            "append_rows": append,
                            "indicator": indicator,
                        }
                        case = await _measure_case(
                            workers, config, args.samples, mode == "forced"
                        )
                        report["cases"].append(case)
                        print(
                            f"{indicator} H={history:,} N={append:,} GC={mode}: "
                            f"{case['paired_speedup_median']:.2f}x",
                            flush=True,
                        )
            output.write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
        finally:
            for worker in workers:
                await worker.close()


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
    args = parser.parse_args()
    if args.command == "build":
        asyncio.run(build(args))
    elif args.command == "_worker":
        asyncio.run(worker_main(args.root))
    else:
        if args.samples < 2:
            parser.error("at least two paired samples are required")
        asyncio.run(compare(args))


if __name__ == "__main__":
    main()
