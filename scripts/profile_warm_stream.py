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
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from functools import partial
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
NATIVE_CONTRACT = "calc-flow.rolling-kernel-v2-evidence/1"
NATIVE_EVIDENCE_TEST = (
    "runtime::streaming::operator_task::performance_evidence::"
    "rolling_kernel_v2_evidence"
)
NATIVE_CASES = (
    "1x1-dual",
    "99x1-dual",
    "6400x64-dual",
    "64x64-dual-tick",
    "6400x64-dual-lag1",
)
NATIVE_ARMS = ("absent", "present")
NATIVE_SIDES = ("baseline", "candidate")
NATIVE_TIMED_SAMPLES = 30
NATIVE_HISTORY_ROWS = 4_096
NATIVE_WARM_DISPATCHES = 3
NATIVE_MAX_CASES = 5
NATIVE_REGRESSION_BELOW = 0.95
NATIVE_CASE_ROWS = {
    "1x1-dual": 1,
    "99x1-dual": 99,
    "6400x64-dual": 6_400,
    "64x64-dual-tick": 64,
    "6400x64-dual-lag1": 6_400,
}
# Rolling-kernel-v2 spec §10.2: (gate, case, arm, limit in seconds).
NATIVE_LATENCY_GATES = (
    ("G1", "1x1-dual", "absent", 1e-6),
    ("G3", "6400x64-dual", "absent", 100e-6),
)
NATIVE_G2_LIMIT_NS = 4.0
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
ALL_CASE_FIELDS = CASE_FIELDS | OPTIONAL_CASE_FIELDS
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


def _case_fields(text: str) -> dict[str, str]:
    items = [item.split("=", 1) for item in text.split(",")]
    if any(len(item) != 2 for item in items):
        raise argparse.ArgumentTypeError("a case is comma-separated key=value fields")
    fields = dict(items)
    keys = set(fields)
    if len(keys) != len(items) or CASE_FIELDS - keys or keys - ALL_CASE_FIELDS:
        raise argparse.ArgumentTypeError(
            "a case has name, append_rows, entities and drivers fields and "
            "optional reference, p50_limit_us and regression_below fields"
        )
    return fields


def _case_drivers(value: str) -> tuple[str, str]:
    drivers = tuple(value.split(":"))
    if len(drivers) != 2 or not set(drivers).issubset(DRIVERS):
        raise argparse.ArgumentTypeError(
            "drivers are BASELINE:CANDIDATE runner/session"
        )
    return drivers


def _case_reference(value: str | None) -> str | None:
    if value is not None and value not in REFERENCES:
        raise argparse.ArgumentTypeError("the only case reference is ta_lib")
    return value


def _case_name(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise argparse.ArgumentTypeError("case names use letters, digits, - and _")
    return value


def _optional_positive_float(value: str | None) -> float | None:
    return None if value is None else _positive_float(value)


def parse_case(text: str) -> dict[str, Any]:
    """Parse one explicit ``key=value,...`` case with its per-arm drivers."""
    fields = _case_fields(text)
    drivers = _case_drivers(fields["drivers"])
    reference = _case_reference(fields.get("reference"))
    name = _case_name(fields["name"])
    append_rows = _positive_scalar(fields["append_rows"])
    entities = _positive_scalar(fields["entities"])
    limit = _optional_positive_float(fields.get("p50_limit_us"))
    return {
        "name": name,
        "append_rows": append_rows,
        "entities": entities,
        "drivers": drivers,
        "reference": reference,
        "p50_limit_seconds": None if limit is None else limit / 1e6,
        "regression_below": _optional_positive_float(fields.get("regression_below")),
    }


def _case_option_error(args: argparse.Namespace) -> str | None:
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
    return None


def _case_set_error(cases: list[dict[str, Any]], exact_projection: bool) -> str | None:
    names = [case["name"] for case in cases]
    if len(set(names)) != len(names):
        return "case names must be unique"
    if not exact_projection and any("session" in case["drivers"] for case in cases):
        return "the session driver requires --exact-projection"
    return None


def case_argument_error(args: argparse.Namespace) -> str | None:
    """Explain why explicit cases conflict with matrix options, if they do."""
    return _case_option_error(args) or _case_set_error(
        args.cases, args.exact_projection
    )


def compare_argument_error(args: argparse.Namespace) -> str | None:
    """Explain why compare arguments cannot produce paired evidence, if so."""
    if args.samples < 2:
        return "at least two paired samples are required"
    if args.worker_threads is not None and any(
        count not in (1, 4) for count in args.worker_threads
    ):
        return "worker thread overrides must be one or four"
    return None if args.cases is None else case_argument_error(args)


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
    left: str,
    right: str,
    config: dict[str, Any],
    phase: str,
    numeric_contract_change: bool = False,
) -> None:
    # Decay retains the independent value oracle's D13 tolerance.
    if (
        not numeric_contract_change
        and config["indicator"] in ("rolling_mean", "dual_sma_spread")
        and left != right
    ):
        raise ValueError(f"paired {phase} Float64 digests differ")


def _validate_starts(
    starts: list[dict[str, Any]],
    config: dict[str, Any],
    drivers: tuple[str, str] = DEFAULT_DRIVERS,
    numeric_contract_change: bool = False,
) -> None:
    for response, driver in zip(starts, drivers, strict=True):
        _validate_start(response, config)
        if response.get("driver") != driver:
            raise ValueError("paired worker started a different driver")
    if starts[0]["config"] != starts[1]["config"]:
        raise ValueError("paired worker configurations differ")
    _validate_pair_digest(
        starts[0]["warm_output_digest"],
        starts[1]["warm_output_digest"],
        config,
        "warm",
        numeric_contract_change,
    )
    if starts[0]["plan_identity"] != starts[1]["plan_identity"]:
        raise ValueError("paired plan preflight identities differ")


def _require_phase_range(values: np.ndarray) -> None:
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("phase intervals must be finite and nonnegative")


def _require_unattributed_reason(sample: dict[str, Any]) -> None:
    reason = str(sample.get("phase_attribution"))
    if not reason.startswith("unavailable: ") or sample.get("phases_seconds"):
        raise ValueError("unattributed session samples must name a reason only")


def _attributed_session_phases(sample: dict[str, Any]) -> np.ndarray:
    phases = sample.get("phases_seconds")
    try:
        values = np.asarray([phases[name] for name in SESSION_PHASE_NAMES], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("attributed session samples need every phase") from error
    _require_phase_range(values)
    if not np.isclose(values.sum(), sample["seconds"], rtol=1e-9, atol=1e-12):
        raise ValueError("contiguous session phases must cover the timed sample")
    return values


def _session_phase_values(sample: dict[str, Any]) -> np.ndarray | None:
    if sample.get("phase_attribution") == "native_call_probe":
        return _attributed_session_phases(sample)
    _require_unattributed_reason(sample)
    return None


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
    _require_phase_range(values)
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
    numeric_contract_change: bool = False,
) -> dict[str, Any]:
    starts = [
        await worker.request(operation="start", config=config, driver=driver)
        for worker, driver in zip(workers, drivers, strict=True)
    ]
    _validate_starts(starts, config, drivers, numeric_contract_change)
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
            numeric_contract_change,
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


def _markdown_section(
    title: str, notes: list[str], headers: list[str], rows: list[list[str]]
) -> list[str]:
    if not rows:
        return []
    return ["", f"## {title}", "", *notes, "", *_markdown_table(headers, rows)]


def _driver_arms(case: dict[str, Any], driver: str) -> list[tuple[int, str]]:
    drivers = case.get("drivers", DEFAULT_DRIVERS)
    return [
        (index, side)
        for index, side in enumerate(("baseline", "candidate"))
        if drivers[index] == driver
    ]


def _session_phase_cells(phases: dict[str, Any]) -> list[str]:
    if not phases:
        return ["-"] * len(SESSION_PHASE_NAMES)
    return [_micros(phases[name]["p50_seconds"]) for name in SESSION_PHASE_NAMES]


def _session_phase_row(case: dict[str, Any], index: int, side: str) -> list[str]:
    summary = case["phase_intervals"][side]
    probe = case["start"][index].get("phase_probe", {})
    overhead = probe.get("overhead_seconds_estimate")
    return [
        case.get("name") or "-",
        side,
        f"{case['config']['append_rows']:,}",
        f"{summary['attributed_samples']} / {len(case['samples'][index])}",
        *_session_phase_cells(summary["phases"]),
        "-" if overhead is None else f"{overhead * 1e6:.3f}",
    ]


def _session_phase_markdown(cases: list[dict[str, Any]]) -> list[str]:
    rows = [
        _session_phase_row(case, index, side)
        for case in cases
        for index, side in _driver_arms(case, "session")
    ]
    return _markdown_section(
        "Session phase attribution",
        [
            "Independent P50 values in µs from the native-call probe; they need not",
            "sum to the total median. Probe callbacks are inside every session sample.",
        ],
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
    )


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
    return _markdown_section(
        "Unpaired references",
        [
            "TA-Lib recalculates `window - 1` rows per entity before each paired",
            "append plus the append in the idle baseline worker; every value matches",
            "the warm oracle. References are not paired arms.",
        ],
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
    )


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


def _callback_breakdown_case(case: dict[str, Any]) -> bool:
    return case["config"]["history_rows"] == 1_024_000 and case["collect_gc"]


def _runner_phase_row(case: dict[str, Any], side: str) -> list[str]:
    config = case["config"]
    phases = case["phase_intervals"][side]
    return [
        config["indicator"],
        f"{config['append_rows']:,}",
        str(config.get("append_entities") or "full tick"),
        side,
        *(_micros(phases[name]["p50_seconds"]) for name in PHASE_NAMES),
    ]


def _phase_markdown(cases: list[dict[str, Any]]) -> list[str]:
    rows = [
        _runner_phase_row(case, side)
        for case in cases
        if _callback_breakdown_case(case)
        for _, side in _driver_arms(case, "runner")
    ]
    return _markdown_section(
        "Callback interval breakdown",
        [
            "Representative H=1,024,000 forced-GC runner arms; independent P50 µs.",
            "These wall intervals include scheduling and waiting, not just CPU work.",
            "Conversion is a subset of sink-to-receive and must not be added again.",
            "Independent phase medians need not sum to the total latency median.",
        ],
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
    )


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
    numeric_contract_change: bool = False


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
                numeric_contract_change=options.numeric_contract_change,
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
        "numeric_contract_change": args.numeric_contract_change,
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


def _remaining_budget(budget: float | None, started: float) -> float | None:
    return None if budget is None else budget - (time.monotonic() - started)


async def _budgeted(
    remaining: float | None, run: Callable[[], Awaitable[Any]]
) -> Any | None:
    """Return ``run()``'s result, or ``None`` when the budget expires first."""
    if remaining is not None and remaining <= 0:
        return None
    deadline = asyncio.timeout(remaining)
    try:
        async with deadline:
            return await run()
    except TimeoutError:
        if not deadline.expired():
            raise
        # Cancellation already settled this case's owned workers.
        return None


def _measured_case(
    case: dict[str, Any],
    environments: list[dict[str, Any]],
    report: dict[str, Any],
    arms: dict[str, Any],
) -> dict[str, Any]:
    if environments != report["environments"]:
        raise ValueError("worker fingerprints changed between cases")
    return {"name": arms["name"], **case, "verdict": case_verdict(case, arms)}


def _case_line(
    case: dict[str, Any], config: dict[str, Any], mode: str, arms: dict[str, Any]
) -> str:
    return (
        f"{arms['name'] or config} GC={mode} {'/'.join(arms['drivers'])}: "
        f"{case['paired_speedup_median']:.2f}x"
    )


def _write_compare_report(output: Path, report: dict[str, Any]) -> None:
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")


def _compare_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output = args.output.resolve()
    work = output.parent if args.work_dir is None else args.work_dir.resolve()
    for directory in (output.parent, work):
        directory.mkdir(parents=True, exist_ok=True)
    return output, work


async def compare(args: argparse.Namespace) -> None:
    started = time.monotonic()
    manifests = await _compatible_builds((args.baseline_build, args.candidate_build))
    counts = None if args.worker_threads is None else tuple(args.worker_threads)
    _validate_scheduler_builds(manifests, counts)
    output, work = _compare_paths(args)
    planned = selected_cases(args)
    if not planned:
        raise ValueError("benchmark matrix must not be empty")
    report: dict[str, Any] = {}
    unmeasured: list[dict[str, Any]] = []
    for position, (config, mode, arms) in enumerate(planned):
        options = WorkerPairOptions(
            work,
            counts,
            arms["drivers"],
            arms["reference"],
            args.numeric_contract_change,
        )
        measured = await _budgeted(
            _remaining_budget(args.budget_seconds, started),
            partial(
                _fresh_case, manifests, config, args.samples, mode == "forced", options
            ),
        )
        if measured is None:
            unmeasured = _unmeasured(planned[position:])
            print(f"budget exhausted; {len(unmeasured)} cases unmeasured", flush=True)
            break
        case, environments = measured
        report = report or _compare_report(manifests, environments, args, counts)
        case = _measured_case(case, environments, report, arms)
        report["cases"].append(case)
        print(_case_line(case, config, mode, arms), flush=True)
    report = report or _compare_report(manifests, [], args, counts)
    _write_compare_report(
        output,
        {
            **report,
            "status": "stopped" if unmeasured else "completed",
            "unmeasured_cases": unmeasured,
            "elapsed_seconds": time.monotonic() - started,
        },
    )


def _validate_native_case_names(names: list[str]) -> None:
    if any(not name for name in names):
        raise argparse.ArgumentTypeError("native evidence case names must not be empty")
    unknown = [name for name in names if name not in NATIVE_CASES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown native evidence case {unknown[0]!r}; "
            f"known cases: {', '.join(NATIVE_CASES)}"
        )


def _validate_native_case_counts(names: list[str]) -> None:
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise argparse.ArgumentTypeError(
            f"native evidence case {duplicates[0]!r} is selected more than once"
        )
    if len(names) > NATIVE_MAX_CASES:
        raise argparse.ArgumentTypeError(
            f"native evidence accepts at most {NATIVE_MAX_CASES} cases"
        )


def parse_native_cases(text: str) -> tuple[str, ...]:
    """Parse a bounded, ordered selection of rolling-dispatch evidence cases."""
    names = text.split(",")
    _validate_native_case_names(names)
    _validate_native_case_counts(names)
    return tuple(names)


def _native_round_count(value: str) -> int:
    rounds = _positive_scalar(value)
    if rounds < 2:
        raise argparse.ArgumentTypeError("native evidence requires at least two rounds")
    return rounds


def _native_selection(cases: tuple[str, ...]) -> tuple[str, ...]:
    selected = tuple(cases)
    valid = (
        1 <= len(selected) <= NATIVE_MAX_CASES
        and len(set(selected)) == len(selected)
        and all(name in NATIVE_CASES for name in selected)
    )
    if not valid:
        raise ValueError("invalid native evidence case selection")
    return selected


def _native_number(value: Any, *, positive: bool) -> float:
    if type(value) not in (int, float):
        raise ValueError("native evidence times must be JSON numbers")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError("native evidence times must be finite")
    if positive and parsed <= 0:
        raise ValueError("native dispatch times must be positive")
    if not positive and parsed < 0:
        raise ValueError("native stage times must be nonnegative")
    return parsed


def _validate_native_sample(sample: Any, index: int) -> None:
    if not isinstance(sample, dict):
        raise ValueError("native evidence samples must be objects")
    if type(sample.get("index")) is not int or sample["index"] != index:
        raise ValueError("native evidence sample indices must be contiguous")
    expected_order = "AB" if index % 2 == 0 else "BA"
    if sample.get("order") != expected_order:
        raise ValueError("native evidence samples must alternate AB/BA order")
    for arm in NATIVE_ARMS:
        _native_number(sample.get(f"{arm}_seconds"), positive=True)
    _native_number(sample.get("present_numeric_update_seconds"), positive=False)


def _validate_native_case_metadata(case: dict[str, Any], name: str) -> None:
    if type(case.get("rows")) is not int or case["rows"] != NATIVE_CASE_ROWS[name]:
        raise ValueError(f"native evidence case {name!r} has the wrong row count")
    expected = {
        "name": name,
        "kind": "dispatch",
        "window_outputs": 2,
        "history_rows": NATIVE_HISTORY_ROWS,
        "warm_dispatches": NATIVE_WARM_DISPATCHES,
    }
    if any(case.get(key) != value for key, value in expected.items()):
        raise ValueError(f"native evidence case {name!r} does not match its workload")


def _validate_native_case(case: Any, name: str) -> None:
    if not isinstance(case, dict):
        raise ValueError("native evidence cases must be objects")
    _validate_native_case_metadata(case, name)
    samples = case.get("samples")
    if not isinstance(samples, list) or len(samples) != NATIVE_TIMED_SAMPLES:
        raise ValueError("native evidence cases require exactly 30 timed samples")
    for index, sample in enumerate(samples):
        _validate_native_sample(sample, index)


def _validate_native_envelope(
    evidence: dict[str, Any], selected: tuple[str, ...]
) -> None:
    if evidence.get("contract") != NATIVE_CONTRACT:
        raise ValueError("unexpected native evidence contract")
    if type(evidence.get("debug_assertions")) is not bool:
        raise ValueError("native evidence must declare its build mode")
    if evidence["debug_assertions"]:
        raise ValueError("native evidence requires a release test binary")
    if evidence.get("selection") != list(selected):
        raise ValueError("native evidence selection differs from the request")


def _validate_native_run_metadata(evidence: dict[str, Any]) -> None:
    if (
        type(evidence.get("timed_samples")) is not int
        or evidence["timed_samples"] != NATIVE_TIMED_SAMPLES
    ):
        raise ValueError("native evidence declares the wrong timed-sample count")
    if not isinstance(evidence.get("executable"), str) or not evidence["executable"]:
        raise ValueError("native evidence must identify its executable")


def _validate_native_evidence(evidence: Any, selected: tuple[str, ...]) -> None:
    """Reject stale, debug, mis-selected, malformed or non-finite evidence."""
    if not isinstance(evidence, dict):
        raise ValueError("native evidence must be a JSON object")
    _validate_native_envelope(evidence, selected)
    _validate_native_run_metadata(evidence)
    cases = evidence.get("cases")
    if not isinstance(cases, list) or len(cases) != len(selected):
        raise ValueError("native evidence cases differ from the request")
    for case, name in zip(cases, selected, strict=True):
        _validate_native_case(case, name)


async def _settle_native_process(creation, process, error: BaseException) -> None:
    try:
        if process is None:
            process = await _complete_owned(creation)
        await _complete_owned(asyncio.create_task(stop(process)))
    except BaseException as cleanup_error:
        _raise_command_error(error, cleanup_error)


async def _run_native_evidence(
    binary: Path, selected: tuple[str, ...], output: Path
) -> dict[str, Any]:
    """Run one owned release-test process and load its exclusive JSON output."""
    binary = binary.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite native evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        CALC_FLOW_NATIVE_EVIDENCE_CASES=",".join(selected),
        CALC_FLOW_NATIVE_EVIDENCE_OUTPUT=str(output),
    )
    with output.with_suffix(".log").open("wb") as log:
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                str(binary),
                "--exact",
                NATIVE_EVIDENCE_TEST,
                "--ignored",
                "--nocapture",
                "--test-threads=1",
                cwd=REPOSITORY,
                env=environment,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
            )
        )
        process = None
        try:
            process = await asyncio.shield(creation)
            code = await process.wait()
        except BaseException as error:
            await _settle_native_process(creation, process, error)
            raise
    if code:
        raise RuntimeError(
            f"native evidence binary exited {code}; see {output.with_suffix('.log')}"
        )
    try:
        evidence = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"native evidence output is missing or invalid: {output}"
        ) from error
    _validate_native_evidence(evidence, selected)
    return evidence


def _native_binary_identity(binary: Path, side: str) -> dict[str, Any]:
    path = binary.resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{side} native evidence binary must be an executable file")
    return {"side": side, "path": str(path), "sha256": sha256_file(path)}


def _verify_native_binaries(binaries: list[dict[str, Any]]) -> None:
    for binary in binaries:
        if sha256_file(Path(binary["path"])) != binary["sha256"]:
            raise ValueError(
                f"{binary['side']} native evidence binary changed during the run"
            )


async def _native_round(
    binaries: list[dict[str, Any]],
    selected: tuple[str, ...],
    work: Path,
    number: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_side = {binary["side"]: binary for binary in binaries}
    order = NATIVE_SIDES if number % 2 else tuple(reversed(NATIVE_SIDES))
    paths = {side: work / f"round-{number}-{side}.json" for side in NATIVE_SIDES}
    evidence = {}
    for side in order:
        evidence[side] = await _run_native_evidence(
            Path(by_side[side]["path"]), selected, paths[side]
        )
        _validate_native_evidence(evidence[side], selected)
    outputs = {
        side: {
            "path": str(paths[side].resolve()),
            "sha256": sha256_file(paths[side]),
            "executable": evidence[side]["executable"],
        }
        for side in NATIVE_SIDES
    }
    return evidence, {
        "round": number,
        "order": "AB" if order == NATIVE_SIDES else "BA",
        "outputs": outputs,
    }


def _native_case(evidence: dict[str, Any], name: str) -> dict[str, Any]:
    return next(case for case in evidence["cases"] if case["name"] == name)


def _native_round_samples(
    rounds: list[dict[str, Any]], name: str, arm: str, side: str
) -> list[list[float]]:
    return [
        [
            _native_number(sample[f"{arm}_seconds"], positive=True)
            for sample in _native_case(paired[side], name)["samples"]
        ]
        for paired in rounds
    ]


def _native_paired_statistics(
    baseline: list[float], candidate: list[float], rows: int
) -> tuple[float, list[float] | None, int]:
    if len(baseline) == 1:
        ratio = baseline[0] / candidate[0]
        return ratio, None, 0
    paired = paired_summary(baseline, candidate, rows=rows)
    return (
        paired["paired_speedup_median"],
        paired["paired_speedup_ci95"],
        paired["bootstrap_resamples"],
    )


def _native_verdict(interval: list[float] | None) -> dict[str, Any]:
    if interval is None:
        return {"paired_speedup": "inconclusive", "confirmed_regression": None}
    low, high = interval
    return {
        "paired_speedup": (
            "slower" if high < 1.0 else "faster" if low > 1.0 else "inconclusive"
        ),
        "confirmed_regression": high < NATIVE_REGRESSION_BELOW,
    }


def _native_numeric_update(
    rounds: list[dict[str, Any]], name: str, rows: int, window_outputs: int
) -> dict[str, float]:
    divisor = rows * window_outputs
    return {
        side: float(
            np.median(
                [
                    _native_number(
                        sample["present_numeric_update_seconds"], positive=False
                    )
                    * 1e9
                    / divisor
                    for paired in rounds
                    for sample in _native_case(paired[side], name)["samples"]
                ]
            )
        )
        for side in NATIVE_SIDES
    }


def _native_arm_summary(
    rounds: list[dict[str, Any]], name: str, arm: str
) -> dict[str, Any]:
    case = _native_case(rounds[0]["baseline"], name)
    rows = case["rows"]
    samples = {
        side: _native_round_samples(rounds, name, arm, side) for side in NATIVE_SIDES
    }
    medians = {
        side: [float(np.median(values)) for values in samples[side]]
        for side in NATIVE_SIDES
    }
    speedup, interval, resamples = _native_paired_statistics(
        medians["baseline"], medians["candidate"], rows
    )
    numeric = (
        _native_numeric_update(rounds, name, rows, case["window_outputs"])
        if arm == "present"
        else None
    )
    return {
        "case": name,
        "kind": "dispatch",
        "arm": arm,
        "rows": rows,
        "rounds": len(rounds),
        "baseline": _summary(np.asarray(samples["baseline"]).ravel(), rows),
        "candidate": _summary(np.asarray(samples["candidate"]).ravel(), rows),
        "round_medians_seconds": medians,
        "paired_round_speedups": [
            baseline / candidate
            for baseline, candidate in zip(
                medians["baseline"], medians["candidate"], strict=True
            )
        ],
        "paired_round_speedup_median": speedup,
        "paired_round_speedup_ci95": interval,
        "bootstrap_resamples": resamples,
        "verdict": _native_verdict(interval),
        **({} if numeric is None else {"numeric_update_ns_per_window_row": numeric}),
    }


def _native_arm_summaries(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rounds:
        return []
    selected = rounds[0]["baseline"]["selection"]
    return [
        _native_arm_summary(rounds, name, arm)
        for name in selected
        for arm in NATIVE_ARMS
    ]


def _native_find_arm(arms: list[dict[str, Any]], case: str, arm: str):
    return next(
        (item for item in arms if (item["case"], item["arm"]) == (case, arm)),
        None,
    )


def _native_latency_gate(
    gate: str,
    case: str,
    arm: str,
    limit: float,
    arms: list[dict[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    measured = _native_find_arm(arms, case, arm)
    value = None if measured is None else measured["candidate"]["p50_seconds"]
    return {
        "case": case,
        "arm": arm,
        "value_seconds": value,
        "limit_seconds": limit,
        "met": value <= limit if complete and value is not None else None,
    }


def _native_g2_gate(arms: list[dict[str, Any]], complete: bool) -> dict[str, Any]:
    measured = _native_find_arm(arms, "6400x64-dual", "present")
    value = (
        None
        if measured is None
        else measured["numeric_update_ns_per_window_row"]["candidate"]
    )
    return {
        "case": "6400x64-dual",
        "arm": "present",
        "value_ns_per_window_row": value,
        "limit_ns_per_window_row": NATIVE_G2_LIMIT_NS,
        "met": (
            value <= NATIVE_G2_LIMIT_NS if complete and value is not None else None
        ),
    }


def _native_gate_values(
    arms: list[dict[str, Any]], complete: bool
) -> dict[str, dict[str, Any]]:
    latency = {
        gate: _native_latency_gate(gate, case, arm, limit, arms, complete)
        for gate, case, arm, limit in NATIVE_LATENCY_GATES
    }
    return {
        "G1": latency["G1"],
        "G2": _native_g2_gate(arms, complete),
        "G3": latency["G3"],
    }


def _native_report(
    args: argparse.Namespace,
    binaries: list[dict[str, Any]],
    rounds: list[dict[str, Any]],
    round_records: list[dict[str, Any]],
    status: str,
    started: float,
) -> dict[str, Any]:
    arms = _native_arm_summaries(rounds)
    return {
        "contract": "paired-native-evidence-v1",
        "status": status,
        "cases": list(args.cases),
        "requested_rounds": args.rounds,
        "completed_rounds": len(rounds),
        "timed_samples_per_round": NATIVE_TIMED_SAMPLES,
        "sample_pairs_per_case": len(rounds) * NATIVE_TIMED_SAMPLES,
        "budget_seconds": args.budget_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "numeric_contract_change": args.numeric_contract_change,
        "binaries": binaries,
        "round_evidence": round_records,
        "arms": arms,
        "gate_values": _native_gate_values(arms, status == "completed"),
        "timed_boundary": (
            "current-thread dispatch_data plus dispatch_watermark_handler on a "
            "RollingOperator; no projection; output drain outside the timer"
        ),
        "comparison_note": (
            "P2.1 uses the async dispatch boundary; later cross-boundary "
            "speedups are informational, while G1-G3 remain absolute"
        ),
        "harness_sha256": sha256_file(Path(__file__).resolve()),
    }


def _native_speedup(arm: dict[str, Any]) -> str:
    interval = arm["paired_round_speedup_ci95"]
    if interval is None:
        return f"{arm['paired_round_speedup_median']:.2f}x (one round; no CI)"
    low, high = interval
    return f"{arm['paired_round_speedup_median']:.2f}x ({low:.2f}–{high:.2f})"


def _native_arm_markdown(arms: list[dict[str, Any]]) -> list[str]:
    rows = [
        [
            arm["case"],
            arm["arm"],
            str(arm["rounds"]),
            f"{_micros(arm['baseline']['p50_seconds'])} / "
            f"{_micros(arm['baseline']['p95_seconds'])}",
            f"{_micros(arm['candidate']['p50_seconds'])} / "
            f"{_micros(arm['candidate']['p95_seconds'])}",
            _native_speedup(arm),
            arm["verdict"]["paired_speedup"],
        ]
        for arm in arms
    ]
    return _markdown_table(
        [
            "Case",
            "Arm",
            "Rounds",
            "Baseline P50 / P95 µs",
            "Candidate P50 / P95 µs",
            "Paired-round speedup (95% CI)",
            "Verdict",
        ],
        rows,
    )


def _native_gate_measurement(
    gate: dict[str, Any],
) -> tuple[float | None, float, str]:
    if "value_seconds" in gate:
        value = gate["value_seconds"]
        return (
            None if value is None else value * 1e6,
            gate["limit_seconds"] * 1e6,
            "µs",
        )
    return (
        gate["value_ns_per_window_row"],
        gate["limit_ns_per_window_row"],
        "ns/window-row",
    )


def _native_gate_cell(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value:.3f}"


def _native_gate_markdown(gates: dict[str, dict[str, Any]]) -> list[str]:
    rows = []
    for name, gate in gates.items():
        value, limit, unit = _native_gate_measurement(gate)
        rows.append(
            [
                name,
                gate["case"],
                gate["arm"],
                _native_gate_cell(value),
                _native_gate_cell(limit),
                unit,
                "incomplete" if gate["met"] is None else str(gate["met"]),
            ]
        )
    return _markdown_table(
        ["Gate", "Case", "Arm", "Candidate value", "Limit", "Unit", "Met"], rows
    )


def native_markdown(report: dict[str, Any]) -> str:
    budget = report["budget_seconds"]
    lines = [
        "# Paired native rolling-dispatch evidence",
        "",
        (
            "Release test binaries alternate AB/BA across rounds; each round warms "
            "both recorder arms and collects 30 timed dispatches per case."
        ),
        f"Status: {report['status']}; {report['completed_rounds']} / "
        f"{report['requested_rounds']} rounds completed; "
        f"{report['elapsed_seconds']:.1f} s elapsed"
        + (f" of a {budget:g} s budget." if budget is not None else "."),
        "",
        *_native_arm_markdown(report["arms"]),
        "",
        "## Absolute gate values",
        "",
        (
            "Gate values use candidate P50 over all retained samples. Incomplete "
            "budget-stopped reports do not assert gate outcomes."
        ),
        "",
        *_native_gate_markdown(report["gate_values"]),
        "",
        (
            "G2 uses recorder-present numeric_update ns divided by 6,400 × 2 "
            "window rows. P2.1 cross-boundary speedups are informational."
        ),
        "",
    ]
    return "\n".join(lines)


def _native_remaining_budget(budget: float | None, started: float) -> float | None:
    return None if budget is None else budget - (time.monotonic() - started)


async def _native_rounds(
    args: argparse.Namespace,
    binaries: list[dict[str, Any]],
    selected: tuple[str, ...],
    work: Path,
    started: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    rounds: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    stopped = False
    for number in range(1, args.rounds + 1):
        remaining = _native_remaining_budget(args.budget_seconds, started)
        if remaining is not None and remaining <= 0:
            stopped = True
            break
        try:
            async with asyncio.timeout(remaining) as deadline:
                paired, record = await _native_round(binaries, selected, work, number)
        except TimeoutError:
            if not deadline.expired():
                raise
            stopped = True
            break
        rounds.append(paired)
        records.append(record)
        print(f"native evidence round {number}: {record['order']}", flush=True)
    return rounds, records, stopped


async def native(args: argparse.Namespace) -> None:
    """Pair prebuilt release binaries without compiling or expanding the matrix."""
    started = time.monotonic()
    selected = _native_selection(args.cases)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.parent if args.work_dir is None else args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    binaries = [
        _native_binary_identity(binary, side)
        for binary, side in zip(
            (args.baseline_binary, args.candidate_binary), NATIVE_SIDES, strict=True
        )
    ]
    rounds, records, stopped = await _native_rounds(
        args, binaries, selected, work, started
    )
    _verify_native_binaries(binaries)
    if stopped:
        print(f"budget exhausted; retained {len(rounds)} completed rounds", flush=True)
    report = _native_report(
        args, binaries, rounds, records, "stopped" if stopped else "completed", started
    )
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    output.with_suffix(".md").write_text(native_markdown(report), encoding="utf-8")


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
        "--numeric-contract-change",
        action="store_true",
        help="allow unequal cross-arm rolling digests; retain all other validation",
    )
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
    native_parser = commands.add_parser("native")
    native_parser.add_argument("--baseline-binary", type=Path, required=True)
    native_parser.add_argument("--candidate-binary", type=Path, required=True)
    native_parser.add_argument("--cases", type=parse_native_cases, required=True)
    native_parser.add_argument("--rounds", type=_native_round_count, default=10)
    native_parser.add_argument("--output", type=Path, required=True)
    native_parser.add_argument("--work-dir", type=Path)
    native_parser.add_argument("--budget-seconds", type=_positive_float)
    native_parser.add_argument("--numeric-contract-change", action="store_true")
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
    elif args.command == "native":
        asyncio.run(native(args))
    elif args.command == "diagnose":
        if args.samples < 2:
            parser.error("at least two diagnostic samples are required")
        asyncio.run(diagnose(args))
    else:
        if error := compare_argument_error(args):
            parser.error(error)
        asyncio.run(compare(args))


if __name__ == "__main__":
    main()
