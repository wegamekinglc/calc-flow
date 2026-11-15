"""Build provenance-bound wheels and profile paired, persistent streaming workers."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.benchmark_suite.process import (
        _complete_owned,
        _raise_command_error,
        stop,
    )
    from scripts.toolkit import (
        sha256_file,
        wheel_native_sha256,
        worker_environment,
        write_json,
    )
except ImportError:  # direct execution puts only scripts/ on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.benchmark_suite.process import (
        _complete_owned,
        _raise_command_error,
        stop,
    )
    from scripts.toolkit import (
        sha256_file,
        wheel_native_sha256,
        worker_environment,
        write_json,
    )

REPOSITORY = Path(__file__).resolve().parents[1]
HISTORY_ROWS = (10_240, 102_400, 1_024_000, 10_240_000)
APPEND_ROWS = (64, 640, 6_400, 64_000)
FLOAT64_DIGEST_FORMAT = "calc-flow-warm-float64-v1"
PHASE_NAMES = (
    "enqueue_to_source_data",
    "source_data_to_source_watermark",
    "source_watermark_to_sink",
    "sink_to_receive",
    "to_pyarrow",
)
SESSION_PHASE_NAMES = ("argument_handling", "native_call", "result_assembly")
DRIVERS = ("runner", "session")
REFERENCES = ("ta_lib",)
DEFAULT_DRIVERS = ("runner", "runner")
TIMED_BOUNDARIES = {
    "runner": (
        "prepared Data enqueue through source/task/channel, rolling, "
        "watermark finalization, projection, sink and to_pyarrow; "
        "excludes IPC, compile, startup, history, validation, "
        "checkpoint, shutdown"
    ),
    "session": (
        "immediately before StreamSession.push with the prepared table Batch "
        "and Watermark to the returned output pa.Table on the session thread; "
        "excludes IPC, executor handoff, input preparation, compile, startup, "
        "history, validation, finish and close"
    ),
}
SESSION_PHASE_NOTE = (
    "session phases come from sys.monitoring CALL/C_RETURN events around the "
    "native _StreamSession.push call, without library instrumentation: "
    "argument_handling is Python work before it, native_call includes PyO3 "
    "conversion and Arrow export, result_assembly is Python work after it; "
    "each timed sample includes the two probe callbacks, estimated per worker "
    "as phase_probe.overhead_seconds_estimate"
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


def _scalar_dimensions(args: argparse.Namespace) -> dict[str, int]:
    names = ("entities", "window", "fast_window", "history_segment_rows")
    return {
        name: value
        for name in names
        if (value := getattr(args, name, None)) is not None
    }


def _positive_scalar(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("scenario dimensions must be positive")
    return parsed


def _add_scalar_dimensions(parser: argparse.ArgumentParser) -> None:
    for name in ("entities", "window", "fast-window", "history-segment-rows"):
        parser.add_argument(f"--{name}", type=_positive_scalar)


def matrix_cases(args: argparse.Namespace) -> list[tuple[dict[str, Any], str]]:
    history_rows = HISTORY_ROWS if args.history_rows is None else args.history_rows
    append_rows = APPEND_ROWS if args.append_rows is None else args.append_rows
    points = (
        product(history_rows, append_rows)
        if args.append_entities is not None
        else matrix_points(tuple(history_rows), tuple(append_rows))
    )
    cases = []
    for indicator, (history, append), active, mode in product(
        args.indicators, points, args.append_entities or [None], args.gc_modes
    ):
        config = {
            "history_rows": history,
            "append_rows": append,
            "indicator": indicator,
            "exact_projection": args.exact_projection,
            **_scalar_dimensions(args),
        }
        if active is not None:
            config["append_entities"] = active
        cases.append((config, mode))
    order = np.random.default_rng(args.seed).permutation(len(cases))
    return [cases[int(index)] for index in order]


CASE_FIELDS = frozenset(("name", "append_rows", "entities", "drivers"))
OPTIONAL_CASE_FIELDS = frozenset(("reference", "p50_limit_us", "regression_below"))
MATRIX_ARMS = {
    "name": None,
    "drivers": DEFAULT_DRIVERS,
    "reference": None,
    "p50_limit_seconds": None,
    "regression_below": None,
}


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("values must be finite and positive")
    return parsed


def parse_case(text: str) -> dict[str, Any]:
    """Parse one explicit ``key=value,...`` case with its per-arm drivers."""
    items = [item.split("=", 1) for item in text.split(",")]
    if any(len(item) != 2 for item in items):
        raise argparse.ArgumentTypeError("a case is comma-separated key=value fields")
    fields = dict(items)
    if (
        len(fields) != len(items)
        or not CASE_FIELDS.issubset(fields)
        or not (CASE_FIELDS | OPTIONAL_CASE_FIELDS).issuperset(fields)
    ):
        raise argparse.ArgumentTypeError(
            "a case has name, append_rows, entities and drivers fields and "
            "optional reference, p50_limit_us and regression_below fields"
        )
    drivers = tuple(fields["drivers"].split(":"))
    if len(drivers) != 2 or any(driver not in DRIVERS for driver in drivers):
        raise argparse.ArgumentTypeError(
            "drivers are BASELINE:CANDIDATE runner/session"
        )
    if fields.get("reference", "ta_lib") not in REFERENCES:
        raise argparse.ArgumentTypeError("the only case reference is ta_lib")
    if re.fullmatch(r"[A-Za-z0-9_-]+", fields["name"]) is None:
        raise argparse.ArgumentTypeError("case names use letters, digits, - and _")
    limit = fields.get("p50_limit_us")
    regression = fields.get("regression_below")
    return {
        "name": fields["name"],
        "append_rows": _positive_scalar(fields["append_rows"]),
        "entities": _positive_scalar(fields["entities"]),
        "drivers": drivers,
        "reference": fields.get("reference"),
        "p50_limit_seconds": None if limit is None else _positive_float(limit) / 1e6,
        "regression_below": None if regression is None else _positive_float(regression),
    }


def case_argument_error(args: argparse.Namespace) -> str | None:
    """Explain why explicit cases conflict with matrix options, if they do."""
    if any(
        value is not None
        for value in (args.append_rows, args.append_entities, args.entities)
    ):
        return "--case sets append rows and entities; omit the matrix options"
    if any(
        values is None or len(values) != 1
        for values in (args.history_rows, args.indicators, args.gc_modes)
    ):
        return "--case requires one history, indicator and GC mode"
    if args.worker_threads is not None:
        return "--case compares engines; omit --worker-threads"
    names = [case["name"] for case in args.cases]
    if len(set(names)) != len(names):
        return "case names must be unique"
    if not args.exact_projection and any(
        "session" in case["drivers"] for case in args.cases
    ):
        return "the session driver requires --exact-projection"
    return None


def selected_cases(
    args: argparse.Namespace,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    """Return seeded cases with their per-arm drivers, references and gates."""
    if args.cases is None:
        return [(config, mode, MATRIX_ARMS) for config, mode in matrix_cases(args)]
    shared = {
        "history_rows": args.history_rows[0],
        "indicator": args.indicators[0],
        "exact_projection": args.exact_projection,
        **_scalar_dimensions(args),
    }
    cases = [
        (
            {
                **shared,
                "append_rows": case["append_rows"],
                "entities": case["entities"],
            },
            args.gc_modes[0],
            case,
        )
        for case in args.cases
    ]
    order = np.random.default_rng(args.seed).permutation(len(cases))
    return [cases[int(index)] for index in order]


def case_verdict(case: dict[str, Any], arms: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered paired-speedup, regression and p50 rules."""
    low, high = case["paired_speedup_ci95"]
    verdict: dict[str, Any] = {
        "paired_speedup": (
            "slower" if high < 1.0 else "faster" if low > 1.0 else "inconclusive"
        )
    }
    if arms["regression_below"] is not None:
        verdict["regression_below"] = arms["regression_below"]
        verdict["confirmed_regression"] = high < arms["regression_below"]
    if arms["p50_limit_seconds"] is not None:
        verdict["p50_limit_seconds"] = arms["p50_limit_seconds"]
        verdict["candidate_p50_within_limit"] = (
            case["candidate"]["p50_seconds"] <= arms["p50_limit_seconds"]
        )
    return verdict


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
        "cargo_lock_sha256": sha256_file(source / "Cargo.lock"),
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


def _core_wheel(source: Path, output: Path) -> Path:
    with (source / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    distribution = re.sub(r"[-_.]+", "_", project["name"]).lower()
    wheels = list(output.glob(f"{distribution}-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("build output must contain exactly one core wheel")
    return wheels[0]


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
        "--features",
        "pyo3/abi3-py313",
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
            "--features",
            "pyo3/abi3-py313",
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
    wheel = _core_wheel(source, output)
    manifest = {
        "contract": "warm-stream-build-v1",
        "source": str(source),
        **before,
        "wheel": str(wheel),
        "wheel_sha256": sha256_file(wheel),
        "native_sha256": wheel_native_sha256(wheel),
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
    if sha256_file(Path(manifest["wheel"])) != manifest["wheel_sha256"]:
        raise ValueError("wheel hash changed after build")
    if wheel_native_sha256(Path(manifest["wheel"])) != manifest["native_sha256"]:
        raise ValueError("native module hash does not match build manifest")
    identity = await source_identity(Path(manifest["source"]))
    if any(identity[key] != manifest[key] for key in ("source_sha256", "git_sha")):
        raise ValueError("source changed after wheel build")
    return manifest


class Worker:
    """Own a benchmark subprocess and keep IPC outside its timed interval."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.close_receipt: dict[str, Any] | None = None

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
        try:
            await _command_output(installer)
        except BaseException as error:
            try:
                await _complete_owned(asyncio.create_task(stop(installer)))
            except BaseException as cleanup_error:
                _raise_command_error(error, cleanup_error)
            raise
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

    async def close(self) -> dict[str, Any]:
        self.close_receipt = {
            "pid": self.process.pid,
            "exit_code": None,
            "reaped": False,
            "close_result": "pending",
        }
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
            code, result = await _wait_for_worker_exit(self.process)
        except BaseException as error:
            try:
                await _complete_owned(asyncio.create_task(stop(self.process)))
            except BaseException as cleanup_error:
                _raise_command_error(error, cleanup_error)
            self.close_receipt = {
                **self.close_receipt,
                "exit_code": await self.process.wait(),
                "reaped": True,
                "close_result": "aborted",
            }
            raise
        self.close_receipt = {
            **self.close_receipt,
            "exit_code": code,
            "reaped": True,
            "close_result": result,
        }
        _require_worker_exit(self.close_receipt)
        return self.close_receipt


async def _wait_for_worker_exit(process) -> tuple[int, str]:
    try:
        return await asyncio.wait_for(process.wait(), timeout=30), "normal"
    except TimeoutError:
        await stop(process)
        return await process.wait(), "terminated"


def _require_worker_exit(receipt) -> None:
    if not isinstance(receipt, dict):
        raise ValueError("missing worker exit receipt")
    pid = receipt.get("pid")
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid worker exit receipt PID")
    valid = (
        type(receipt.get("exit_code")) is int,
        receipt.get("exit_code") == 0,
        receipt.get("reaped") is True,
        receipt.get("close_result") == "normal",
    )
    if not all(valid):
        raise RuntimeError(f"worker did not close cleanly: {receipt}")


def _require_worker_receipts(receipts) -> None:
    if not isinstance(receipts, list) or len(receipts) != 2:
        raise ValueError("two worker exit receipts are required")
    for receipt in receipts:
        _require_worker_exit(receipt)


def _configured_worker_environment(
    environment: dict[str, str], workers: int | None
) -> dict[str, str]:
    if workers is None:
        return {**environment, "TOKIO_WORKER_THREADS": "4"}
    if type(workers) is not int or workers not in (1, 4):
        raise ValueError("worker thread override must be one or four")
    return {**environment, "TOKIO_WORKER_THREADS": str(workers)}


class WorkerSession:
    """Own and clean up a single worker's active streaming scenario."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.scenario = None
        self.count = 0
        self.profiling = False

    async def start(
        self, config_values: dict[str, Any], driver: str = "runner"
    ) -> dict[str, Any]:
        from benchmarks.warm_stream import SCENARIO_DRIVERS, ScenarioConfig

        if self.scenario is not None:
            raise RuntimeError("previous warm scenario is still active")
        if driver not in DRIVERS:
            raise ValueError("unknown warm-stream driver")
        config = ScenarioConfig(**config_values)
        self.count += 1
        self.scenario = await SCENARIO_DRIVERS[driver].start(
            config, self.root / f"state-{self.count}"
        )
        response = {
            "config": asdict(config),
            "driver": driver,
            "warm_seconds": self.scenario.warm_seconds,
            "float64_digest_format": FLOAT64_DIGEST_FORMAT,
            "warm_output_digest": self.scenario.warm_output_digest,
            "plan_identity": self.scenario.plan_identity,
        }
        if driver == "session":
            response["phase_probe"] = self.scenario.phase_probe
        return response

    def ta_lib_reference(self, message: dict[str, Any]) -> dict[str, Any]:
        from benchmarks.warm_stream import ScenarioConfig, ta_lib_reference

        if self.scenario is not None:
            raise RuntimeError("the TA-Lib reference runs between warm scenarios")
        return ta_lib_reference(
            ScenarioConfig(**message["config"]),
            samples=message["samples"],
            collect_gc=message["collect_gc"],
        )

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
                return worker_environment()
            case "start":
                return await self.start(message["config"], message["driver"])
            case "ta_lib_reference":
                return self.ta_lib_reference(message)
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
            await self.scenario.cancel()
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
    _validate_digest(sample["correctness"].get("float64_digest"))


def _validate_digest(value: Any) -> None:
    if not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise ValueError("paired worker requires a valid Float64 digest")


def _validate_plan_identity(identity: Any) -> None:
    if not isinstance(identity, dict) or not isinstance(
        identity.get("preflight"), dict
    ):
        raise ValueError("paired workers require plan preflight identity")
    for key in ("program_fingerprint", "plan_fingerprint", "project_sha256"):
        _validate_digest(identity.get(key))


def _validate_start(response: dict[str, Any], config: dict[str, Any]) -> None:
    fields = {
        "history_rows",
        "append_rows",
        "append_entities",
        "entities",
        "window",
        "fast_window",
        "history_segment_rows",
        "indicator",
        "exact_projection",
    }
    expanded = response.get("config")
    if not isinstance(expanded, dict) or set(expanded) != fields:
        raise ValueError("paired worker configuration is incomplete")
    if any(expanded.get(key) != value for key, value in config.items()):
        raise ValueError("paired worker configuration differs from requested case")
    if response.get("float64_digest_format") != FLOAT64_DIGEST_FORMAT:
        raise ValueError("paired worker Float64 digest format differs")
    _validate_digest(response.get("warm_output_digest"))
    _validate_plan_identity(response.get("plan_identity"))


def _validate_pair_digest(
    left: str, right: str, config: dict[str, Any], phase: str
) -> None:
    # Decay retains the independent value oracle's D13 tolerance.
    if config["indicator"] in ("rolling_mean", "dual_sma_spread") and left != right:
        raise ValueError(f"paired {phase} Float64 digests differ")


def _validate_starts(
    starts: list[dict[str, Any]],
    config: dict[str, Any],
    drivers: tuple[str, str] = DEFAULT_DRIVERS,
) -> None:
    for response, driver in zip(starts, drivers, strict=True):
        _validate_start(response, config)
        if response.get("driver") != driver:
            raise ValueError("paired worker started a different driver")
    if starts[0]["config"] != starts[1]["config"]:
        raise ValueError("paired worker configurations differ")
    _validate_pair_digest(
        starts[0]["warm_output_digest"], starts[1]["warm_output_digest"], config, "warm"
    )
    if starts[0]["plan_identity"] != starts[1]["plan_identity"]:
        raise ValueError("paired plan preflight identities differ")


def _session_phase_values(sample: dict[str, Any]) -> np.ndarray | None:
    attribution = sample.get("phase_attribution")
    phases = sample.get("phases_seconds")
    if attribution != "native_call_probe":
        if not str(attribution).startswith("unavailable: ") or phases:
            raise ValueError("unattributed session samples must name a reason only")
        return None
    try:
        values = np.asarray([phases[name] for name in SESSION_PHASE_NAMES], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("attributed session samples need every phase") from error
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("phase intervals must be finite and nonnegative")
    if not np.isclose(values.sum(), sample["seconds"], rtol=1e-9, atol=1e-12):
        raise ValueError("contiguous session phases must cover the timed sample")
    return values


def _session_phase_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    attributed = [
        values
        for sample in samples
        if (values := _session_phase_values(sample)) is not None
    ]
    summary: dict[str, Any] = {
        "attributed_samples": len(attributed),
        "unattributed_reasons": sorted(
            {
                sample["phase_attribution"]
                for sample in samples
                if sample["phase_attribution"] != "native_call_probe"
            }
        ),
        "phases": {},
    }
    if attributed:
        values = np.asarray(attributed)
        summary["phases"] = {
            name: {
                "p50_seconds": float(np.median(values[:, index])),
                "p95_seconds": float(np.percentile(values[:, index], 95)),
            }
            for index, name in enumerate(SESSION_PHASE_NAMES)
        }
    return summary


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


def _phase_summary(
    samples: list[dict[str, Any]], driver: str = "runner"
) -> dict[str, Any]:
    if not samples:
        raise ValueError("phase summaries require samples")
    if driver == "session":
        return _session_phase_summary(samples)
    values = np.asarray([_phase_values(sample) for sample in samples])
    return {
        name: {
            "p50_seconds": float(np.median(values[:, index])),
            "p95_seconds": float(np.percentile(values[:, index], 95)),
        }
        for index, name in enumerate(PHASE_NAMES)
    }


def _case_summary(
    measured: list[list[dict[str, Any]]],
    rows: int,
    drivers: tuple[str, str] = DEFAULT_DRIVERS,
) -> dict[str, Any]:
    latency = paired_summary(
        [sample["seconds"] for sample in measured[0]],
        [sample["seconds"] for sample in measured[1]],
        rows=rows,
    )
    return {
        **latency,
        "phase_intervals": {
            "baseline": _phase_summary(measured[0], drivers[0]),
            "candidate": _phase_summary(measured[1], drivers[1]),
        },
    }


def _validate_reference(reference: dict[str, Any], samples: int) -> None:
    values = np.asarray(reference.get("samples_seconds", ()), dtype=float)
    if values.shape != (samples,) or not _positive_finite(values):
        raise ValueError("a reference needs one finite positive time per pair")


async def _measure_case(
    workers: list[Worker],
    config: dict[str, Any],
    samples: int,
    collect_gc: bool,
    *,
    drivers: tuple[str, str] = DEFAULT_DRIVERS,
    reference: str | None = None,
) -> dict[str, Any]:
    starts = [
        await worker.request(operation="start", config=config, driver=driver)
        for worker, driver in zip(workers, drivers, strict=True)
    ]
    _validate_starts(starts, config, drivers)
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
        _validate_pair_digest(
            measured[0][-1]["correctness"]["float64_digest"],
            measured[1][-1]["correctness"]["float64_digest"],
            config,
            "sample",
        )
    terminal = [await worker.request(operation="finish") for worker in workers]
    result = {
        "config": config,
        "collect_gc": collect_gc,
        "drivers": list(drivers),
        "start": starts,
        "samples": measured,
        "terminal": terminal,
        **_case_summary(measured, config["append_rows"], drivers),
    }
    if reference is not None:
        # References are unpaired; the idle baseline worker times them last.
        measured_reference = await workers[0].request(
            operation="ta_lib_reference",
            config=config,
            samples=samples,
            collect_gc=collect_gc,
        )
        _validate_reference(measured_reference, samples)
        result["reference"] = {"kind": reference, **measured_reference}
    return result


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(map(len, column)) for column in zip(headers, *rows, strict=True)]

    def line(cells: list[str]) -> str:
        padded = (cell.ljust(width) for cell, width in zip(cells, widths, strict=True))
        return "| " + " | ".join(padded) + " |"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return [line(headers), separator, *(line(row) for row in rows)]


def _micros(seconds: float) -> str:
    return f"{seconds * 1e6:,.1f}"


def _verdict_text(verdict: dict[str, Any]) -> str:
    parts = [verdict["paired_speedup"]]
    if "candidate_p50_within_limit" in verdict:
        state = "met" if verdict["candidate_p50_within_limit"] else "missed"
        parts.append(f"p50 ≤ {_micros(verdict['p50_limit_seconds'])} µs {state}")
    if "confirmed_regression" in verdict:
        state = "regression" if verdict["confirmed_regression"] else "no regression"
        parts.append(f"CI high < {verdict['regression_below']:g}: {state}")
    return "; ".join(parts)


def markdown(report: dict[str, Any]) -> str:
    thread_counts = report.get("worker_threads")
    lines = [
        "# Warm-state streaming profile",
        "",
        "Paired release builds; times include each arm's full Python driver path.",
        (
            f"Same-wheel configuration experiment: Tokio workers "
            f"{thread_counts[0]} → {thread_counts[1]}."
            if thread_counts
            else "Engine comparison with unchanged scheduler configuration."
        ),
        f"Status: {report.get('status', 'completed')}; "
        f"{report.get('elapsed_seconds', 0.0):.1f} s elapsed"
        + (
            f" of a {report['budget_seconds']:g} s budget."
            if report.get("budget_seconds") is not None
            else "."
        ),
        "",
    ]
    rows = []
    for case in report["cases"]:
        config, left, right = case["config"], case["baseline"], case["candidate"]
        low, high = case["paired_speedup_ci95"]
        drivers = case.get("drivers", DEFAULT_DRIVERS)
        rows.append(
            [
                case.get("name") or "-",
                config["indicator"],
                f"{config['history_rows']:,}",
                f"{config['append_rows']:,}",
                f"{config.get('entities', 64):,}",
                str(config.get("append_entities") or "full tick"),
                str(case["collect_gc"]),
                f"{drivers[0]} → {drivers[1]}",
                f"{_micros(left['p50_seconds'])} / {_micros(left['p95_seconds'])}",
                f"{_micros(right['p50_seconds'])} / {_micros(right['p95_seconds'])}",
                f"{case['paired_speedup_median']:.2f}x ({low:.2f}–{high:.2f})",
                _verdict_text(case["verdict"]) if "verdict" in case else "-",
            ]
        )
    headers = [
        "Case",
        "Indicator",
        "History",
        "Append",
        "Entities",
        "Active entities",
        "Forced GC",
        "Drivers",
        "Baseline P50 / P95 µs",
        "Candidate P50 / P95 µs",
        "Paired speedup (95% CI)",
        "Verdict",
    ]
    lines.extend(_markdown_table(headers, rows))
    lines.extend(_boundary_markdown(report))
    lines.extend(_session_phase_markdown(report["cases"]))
    lines.extend(_reference_markdown(report["cases"]))
    lines.extend(_phase_markdown(report["cases"]))
    lines.extend(_unmeasured_markdown(report.get("unmeasured_cases", [])))
    return "\n".join(lines) + "\n"


def _boundary_markdown(report: dict[str, Any]) -> list[str]:
    drivers = sorted(
        {
            driver
            for case in report["cases"]
            for driver in case.get("drivers", DEFAULT_DRIVERS)
        }
    )
    boundaries = report.get("timed_boundaries", {"runner": report["timed_boundary"]})
    return [
        "",
        "Timed boundaries:",
        "",
        *(f"- {driver}: {boundaries[driver]}." for driver in drivers),
    ]


def _session_phase_markdown(cases: list[dict[str, Any]]) -> list[str]:
    rows = []
    for case in cases:
        drivers = case.get("drivers", DEFAULT_DRIVERS)
        for index, side in enumerate(("baseline", "candidate")):
            if drivers[index] != "session":
                continue
            summary = case["phase_intervals"][side]
            probe = case["start"][index].get("phase_probe", {})
            overhead = probe.get("overhead_seconds_estimate")
            phases = summary["phases"]
            rows.append(
                [
                    case.get("name") or "-",
                    side,
                    f"{case['config']['append_rows']:,}",
                    f"{summary['attributed_samples']} / {len(case['samples'][index])}",
                    *(
                        _micros(phases[name]["p50_seconds"]) if phases else "-"
                        for name in SESSION_PHASE_NAMES
                    ),
                    "-" if overhead is None else f"{overhead * 1e6:.3f}",
                ]
            )
    if not rows:
        return []
    return [
        "",
        "## Session phase attribution",
        "",
        "Independent P50 values in µs from the native-call probe; they need not",
        "sum to the total median. Probe callbacks are inside every session sample.",
        "",
        *_markdown_table(
            [
                "Case",
                "Arm",
                "Append",
                "Attributed samples",
                "Argument handling",
                "Native call",
                "Result assembly",
                "Probe overhead estimate µs",
            ],
            rows,
        ),
    ]


def _reference_markdown(cases: list[dict[str, Any]]) -> list[str]:
    rows = [
        [
            case.get("name") or "-",
            case["reference"]["kind"],
            f"{case['config']['append_rows']:,}",
            f"{case['config'].get('entities', 64):,}",
            str(len(case["reference"]["samples_seconds"])),
            f"{_micros(case['reference']['median_seconds'])} / "
            f"{_micros(case['reference']['p95_seconds'])}",
            f"{case['reference']['throughput_rows_per_second']:,.0f}",
        ]
        for case in cases
        if "reference" in case
    ]
    if not rows:
        return []
    return [
        "",
        "## Unpaired references",
        "",
        "TA-Lib recalculates `window - 1` rows per entity before each paired",
        "append plus the append in the idle baseline worker; every value matches",
        "the warm oracle. References are not paired arms.",
        "",
        *_markdown_table(
            [
                "Case",
                "Reference",
                "Append",
                "Entities",
                "Samples",
                "P50 / P95 µs",
                "Rows per second",
            ],
            rows,
        ),
    ]


def _unmeasured_markdown(unmeasured: list[dict[str, Any]]) -> list[str]:
    if not unmeasured:
        return []
    return [
        "",
        "## Unmeasured after the budget stop",
        "",
        *(
            f"- {case.get('name') or '-'}: {case['config']} "
            f"({case['drivers'][0]} → {case['drivers'][1]})"
            for case in unmeasured
        ),
    ]


def _phase_markdown(cases: list[dict[str, Any]]) -> list[str]:
    rows = []
    for case in cases:
        config = case["config"]
        drivers = case.get("drivers", DEFAULT_DRIVERS)
        if config["history_rows"] != 1_024_000 or not case["collect_gc"]:
            continue
        for index, side in enumerate(("baseline", "candidate")):
            if drivers[index] != "runner":
                continue
            phases = case["phase_intervals"][side]
            rows.append(
                [
                    config["indicator"],
                    f"{config['append_rows']:,}",
                    str(config.get("append_entities") or "full tick"),
                    side,
                    *(_micros(phases[name]["p50_seconds"]) for name in PHASE_NAMES),
                ]
            )
    if not rows:
        return []
    return [
        "",
        "## Callback interval breakdown",
        "",
        "Representative H=1,024,000 forced-GC runner arms; independent P50 µs.",
        "These wall intervals include scheduling and waiting, not just CPU work.",
        "Conversion is a subset of sink-to-receive and must not be added again.",
        "Independent phase medians need not sum to the total latency median.",
        "",
        *_markdown_table(
            [
                "Indicator",
                "Append",
                "Active entities",
                "Build",
                "Enqueue to source",
                "Data to watermark",
                "Watermark to sink",
                "Sink to receive",
                "to_pyarrow subset",
            ],
            rows,
        ),
    ]


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
    excluded = _allowed_fingerprint_differences(environments, thread_counts)
    comparable = [
        {key: value for key, value in env.items() if key not in excluded}
        for env in environments
    ]
    if comparable[0] != comparable[1]:
        raise ValueError("worker machine or dependency fingerprints differ")
    return environments


def _allowed_fingerprint_differences(
    environments: list[dict[str, Any]], thread_counts: tuple[int, int] | None
) -> set[str]:
    actual = [env.get("tokio_worker_threads") for env in environments]
    if thread_counts is None:
        if actual != ["4"] * len(environments):
            raise ValueError("worker did not load the default four Tokio workers")
        return {"native_sha256"}
    if actual != [str(count) for count in thread_counts]:
        raise ValueError("worker did not load the declared thread override")
    return {"native_sha256", "tokio_worker_threads"}


def _new_report(
    manifests: list[dict[str, Any]], environments: list[dict[str, Any]], samples: int
) -> dict[str, Any]:
    return {
        "contract": "paired-warm-stream-v1",
        "builds": manifests,
        "environments": environments,
        "sample_pairs": samples,
        "harness_sha256": {
            name: sha256_file(REPOSITORY / name)
            for name in (
                "scripts/profile_warm_stream.py",
                "scripts/toolkit.py",
                "benchmarks/warm_stream.py",
                "benchmarks/rolling_indicator_comparison.py",
            )
        },
        "timed_boundary": TIMED_BOUNDARIES["runner"],
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
    drivers: tuple[str, str] = DEFAULT_DRIVERS
    reference: str | None = None


async def _close_workers(workers, path):
    results = await asyncio.gather(
        *(worker.close() for worker in workers), return_exceptions=True
    )
    receipts = [getattr(worker, "close_receipt", None) for worker in workers]
    write_json(
        path,
        {
            "workers": receipts,
            "close_errors": [
                repr(result) if isinstance(result, BaseException) else None
                for result in results
            ],
        },
    )
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return receipts


async def _owned_close_workers(workers, path):
    task = asyncio.create_task(_close_workers(workers, path))
    try:
        return await asyncio.shield(task)
    except BaseException as error:
        try:
            await _complete_owned(task)
        except BaseException as cleanup_error:
            _raise_command_error(error, cleanup_error)
        raise


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
        receipt_path = options.directory / f"{Path(temporary).name}-workers.json"
        primary_error = None
        try:
            for index, manifest in enumerate(manifests):
                root = Path(temporary) / str(index)
                root.mkdir()
                count = None if thread_counts is None else thread_counts[index]
                workers.append(await Worker.start(manifest, root, tokio_workers=count))
            environments = await _worker_environments(
                workers, manifests, thread_counts=thread_counts
            )
            case = await _measure_case(
                workers,
                config,
                samples,
                collect_gc,
                drivers=options.drivers,
                reference=options.reference,
            )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                receipts = await _owned_close_workers(workers, receipt_path)
            except BaseException as cleanup_error:
                if primary_error is not None:
                    _raise_command_error(primary_error, cleanup_error)
                raise
        _require_worker_receipts(receipts)
        return {
            **case,
            "worker_receipts": receipts,
            "worker_receipt_path": str(receipt_path.resolve()),
        }, environments


def _validate_scheduler_builds(
    manifests: list[dict[str, Any]], counts: tuple[int, int] | None
) -> None:
    if counts is None:
        return
    if any(
        manifests[0][key] != manifests[1][key]
        for key in ("native_sha256", "wheel_sha256")
    ):
        raise ValueError(
            "scheduler comparison requires the same native wheel on both sides"
        )


def _compare_report(
    manifests: list[dict[str, Any]],
    environments: list[dict[str, Any]],
    args: argparse.Namespace,
    counts: tuple[int, int] | None,
) -> dict[str, Any]:
    return {
        **_new_report(manifests, environments, args.samples),
        "worker_lifetime": "fresh process pair per case",
        "case_order_seed": args.seed,
        "case_selection": "matrix" if args.cases is None else "explicit",
        "callback_profiling": False,
        "worker_threads": counts,
        "comparison_kind": "engine"
        if counts is None
        else "same-wheel scheduler configuration",
        "timed_boundaries": TIMED_BOUNDARIES,
        "session_phase_note": SESSION_PHASE_NOTE,
        "budget_seconds": args.budget_seconds,
        "cases": [],
    }


def _unmeasured(
    planned: list[tuple[dict[str, Any], str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": arms["name"],
            "config": config,
            "gc_mode": mode,
            "drivers": list(arms["drivers"]),
        }
        for config, mode, arms in planned
    ]


async def compare(args: argparse.Namespace) -> None:
    started = time.monotonic()
    manifests = await _compatible_builds((args.baseline_build, args.candidate_build))
    counts = None if args.worker_threads is None else tuple(args.worker_threads)
    _validate_scheduler_builds(manifests, counts)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent if args.work_dir is None else args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    planned = selected_cases(args)
    if not planned:
        raise ValueError("benchmark matrix must not be empty")
    report: dict[str, Any] = {}
    unmeasured: list[dict[str, Any]] = []
    for position, (config, mode, arms) in enumerate(planned):
        remaining = (
            None
            if args.budget_seconds is None
            else args.budget_seconds - (time.monotonic() - started)
        )
        if remaining is not None and remaining <= 0:
            unmeasured = _unmeasured(planned[position:])
            break
        try:
            async with asyncio.timeout(remaining) as deadline:
                case, environments = await _fresh_case(
                    manifests,
                    config,
                    args.samples,
                    mode == "forced",
                    WorkerPairOptions(work, counts, arms["drivers"], arms["reference"]),
                )
        except TimeoutError:
            if not deadline.expired():
                raise
            # Cancellation already settled this case's owned workers.
            unmeasured = _unmeasured(planned[position:])
            break
        if not report:
            report = _compare_report(manifests, environments, args, counts)
        if environments != report["environments"]:
            raise ValueError("worker fingerprints changed between cases")
        case = {"name": arms["name"], **case, "verdict": case_verdict(case, arms)}
        report["cases"].append(case)
        print(
            f"{arms['name'] or config} GC={mode} {'/'.join(arms['drivers'])}: "
            f"{case['paired_speedup_median']:.2f}x",
            flush=True,
        )
    if not report:
        report = _compare_report(manifests, [], args, counts)
    if unmeasured:
        print(f"budget exhausted; {len(unmeasured)} cases unmeasured", flush=True)
    report["status"] = "stopped" if unmeasured else "completed"
    report["unmeasured_cases"] = unmeasured
    report["elapsed_seconds"] = time.monotonic() - started
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
        "exact_projection": args.exact_projection,
        **_scalar_dimensions(args),
    }
    with tempfile.TemporaryDirectory(
        prefix="warm-diagnostic-", dir=output.parent
    ) as temporary:
        worker = await Worker.start(manifest, Path(temporary))
        try:
            environments = await _worker_environments(
                [worker, worker], [manifest, manifest]
            )
            await worker.request(operation="start", config=config, driver="runner")
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
    paired.add_argument("--history-rows", type=int, nargs="*")
    paired.add_argument("--append-rows", type=int, nargs="*")
    paired.add_argument("--append-entities", type=int, nargs="+")
    paired.add_argument("--seed", type=int, default=20260905)
    _add_scalar_dimensions(paired)
    paired.add_argument("--exact-projection", action="store_true")
    paired.add_argument(
        "--case",
        dest="cases",
        action="append",
        type=parse_case,
        metavar="FIELDS",
        help=(
            "explicit case instead of the matrix: name=N,append_rows=N,entities=N,"
            "drivers=BASELINE:CANDIDATE[,reference=ta_lib][,p50_limit_us=F]"
            "[,regression_below=F]"
        ),
    )
    paired.add_argument("--budget-seconds", type=_positive_float)
    paired.add_argument(
        "--work-dir", type=Path, help="worker temporaries and exit receipts"
    )
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
    diagnostic.add_argument("--exact-projection", action="store_true")
    _add_scalar_dimensions(diagnostic)
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
        if args.worker_threads is not None and any(
            count not in (1, 4) for count in args.worker_threads
        ):
            parser.error("worker thread overrides must be one or four")
        if args.cases is not None and (error := case_argument_error(args)):
            parser.error(error)
        asyncio.run(compare(args))


if __name__ == "__main__":
    main()
