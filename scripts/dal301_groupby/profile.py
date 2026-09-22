from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from scripts.benchmark_suite.measure import _prepare, validate_sample
from scripts.benchmark_suite.process import ROOT, command, install
from scripts.dal301_groupby.contract import SEALS, cases
from scripts.dal301_groupby.runtime import AuditWorker
from scripts.toolkit import sha256_file, write_json

PROFILE_ENV = {
    "CARGO_PROFILE_RELEASE_DEBUG": "1",
    "CARGO_PROFILE_RELEASE_STRIP": "false",
    "RUSTFLAGS": "-C force-frame-pointers=yes",
    "CARGO_INCREMENTAL": "0",
    "CARGO_BUILD_JOBS": "2",
}


async def logged(
    argv: list[str], root: Path, name: str, *, cwd: Path = ROOT, env=None
) -> str:
    log = root / f"{name}.log"
    record = {"argv": argv, "cwd": str(cwd), "time_ns": time.time_ns()}
    started = time.monotonic_ns()
    try:
        await command(argv, cwd=cwd, log=log, env=env, timeout=3600)
        record["exit_code"] = 0
    except Exception as error:
        record["error"] = repr(error)
        match = re.search(r"command exited (-?\d+)", str(error))
        record["exit_code"] = int(match[1]) if match else None
        raise
    finally:
        record["elapsed_ns"] = time.monotonic_ns() - started
        write_json(root / f"{name}.command.json", record)
    return log.read_text(encoding="utf-8", errors="replace")


def unpack_native(wheel: Path, output: Path) -> Path:
    with zipfile.ZipFile(wheel) as archive:
        names = [
            n
            for n in archive.namelist()
            if n.startswith("calc_flow/_native") and n.endswith(".so")
        ]
        if len(names) != 1:
            raise ValueError("expected exactly one native ELF")
        output.write_bytes(archive.read(names[0]))
    return output


async def elf_identity(native: Path, root: Path) -> dict:
    notes = await logged(["readelf", "-n", str(native)], root, "elf-notes")
    sections = await logged(["readelf", "-S", str(native)], root, "elf-sections")
    symbols = await logged(["nm", "-C", "--defined-only", str(native)], root, "symbols")
    match = re.search(r"Build ID: ([0-9a-f]+)", notes)
    return {
        "native_sha256": sha256_file(native),
        "build_id": match[1] if match else "",
        "symbols": ".symtab" in sections
        and ".debug_info" in sections
        and "calc_flow::operator::window" in symbols,
    }


def validate_profile(side: str, value: dict) -> None:
    if (
        value.get("contract") != "dal301-profile-only-v1"
        or value.get("source_ref") != SEALS[side]["git_sha"]
        or value.get("native_sha256") == SEALS[side]["native_sha256"]
        or not value.get("build_id")
        or value.get("symbols") is not True
    ):
        raise ValueError(
            "profile requires separately identified, symbolized fixed-source build"
        )


def validate_build_settings(value: dict) -> None:
    expected = {
        "env": PROFILE_ENV,
        "rustc_release": "1.88.0",
        "dependency_lock": sha256_file(ROOT / "benchmarks/requirements.lock"),
    }
    if {key: value.get(key) for key in expected} != expected:
        raise ValueError("profiling build settings or dependency lock differ")


def rustc_release(log: str) -> str:
    releases = re.findall(r"^[ \t]*release[ \t]*:[ \t]*([^\r\n]*)", log, re.MULTILINE)
    if len(releases) != 1:
        raise ValueError("expected exactly one rustc release field")
    return releases[0].strip()


async def build(side: str, source: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    try:
        from scripts.profile_warm_stream import source_identity

        original = await source_identity(source)
        if original["git_sha"] != SEALS[side]["git_sha"] or not original["git_clean"]:
            raise ValueError("profile source is not the clean fixed A/B ref")
        write_json(output / "build-plan.json", {"source": original, "env": PROFILE_ENV})
        config = source / "pyproject.toml"
        before = config.read_text(encoding="utf-8")
        if before.count("strip = true") != 1:
            raise ValueError("expected one explicit maturin strip setting")
        config.write_text(
            before.replace("strip = true", "strip = false"), encoding="utf-8"
        )
        (output / "overlay.patch").write_text(
            "-strip = true\n+strip = false\n", encoding="utf-8"
        )
        env = {**os.environ, **PROFILE_ENV, "CARGO_TARGET_DIR": str(output / "cargo")}
        rustc = await logged(["rustc", "-Vv"], output, "rustc", cwd=source)
        settings = {
            "env": PROFILE_ENV,
            "rustc_release": rustc_release(rustc),
            "dependency_lock": sha256_file(ROOT / "benchmarks/requirements.lock"),
        }
        validate_build_settings(settings)
        await logged(
            [
                sys.executable,
                "-m",
                "maturin",
                "build",
                "--release",
                "--locked",
                "--out",
                str(output),
            ],
            output,
            "build",
            cwd=source,
            env=env,
        )
        wheels = list(output.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("expected one profiling wheel")
        with tempfile.TemporaryDirectory(dir=output) as temporary:
            native = unpack_native(wheels[0], Path(temporary) / "native.so")
            identity = await elf_identity(native, output)
        manifest = {
            "contract": "dal301-profile-only-v1",
            "source_ref": original["git_sha"],
            "original_source": original,
            "overlay_source": await source_identity(source),
            **settings,
            "wheel": wheels[0].name,
            "wheel_sha256": sha256_file(wheels[0]),
            **identity,
            "rustc": rustc,
        }
        validate_profile(side, manifest)
        write_json(output / "profile.json", manifest)
        write_json(output / "outcome.json", {"status": "completed", "exit_code": 0})
        return 0
    except Exception as error:
        write_json(
            output / "outcome.json",
            {"status": "failed", "error": repr(error), "exit_code": 1},
        )
        return 1


async def profile_release(side: str, root: Path, output: Path) -> dict:
    value = json.loads((root / "profile.json").read_text(encoding="utf-8"))
    validate_profile(side, value)
    validate_build_settings(value)
    wheel = root / value["wheel"]
    if wheel.name != value["wheel"] or sha256_file(wheel) != value["wheel_sha256"]:
        raise ValueError("profile wheel hash/path mismatch")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        actual = await elf_identity(
            unpack_native(wheel, Path(temporary) / "native.so"), output
        )
    if any(actual[key] != value[key] for key in actual):
        raise ValueError("profile ELF no longer matches its symbol manifest")
    return {**value, "wheel_path": str(wheel.resolve())}


async def perf_preflight(native: Path, root: Path) -> None:
    await logged(["perf", "--version"], root, "perf-version")
    functions = await logged(
        ["perf", "probe", "-x", str(native), "--funcs", "*window*"],
        root,
        "perf-functions",
    )
    if "window" not in functions.lower():
        raise ValueError("perf cannot resolve target window functions")
    write_json(
        root / "permissions.json",
        {
            "perf_event_paranoid": Path(
                "/proc/sys/kernel/perf_event_paranoid"
            ).read_text(),
            "uid": os.getuid(),
            "frequency": 49,
            "unwind": "fp",
        },
    )
    await logged(
        [
            "perf",
            "record",
            "-e",
            "cpu-clock",
            "-F",
            "49",
            "--strict-freq",
            "--call-graph",
            "fp",
            "-o",
            str(root / "permission-probe.data"),
            "--",
            "/bin/true",
        ],
        root,
        "permission-probe",
    )


async def record_profile(worker: AuditWorker, root: Path) -> None:
    maps = Path(f"/proc/{worker.process.pid}/maps")
    (root / "maps-before.txt").write_bytes(maps.read_bytes())
    profiler = asyncio.create_task(
        logged(
            [
                "perf",
                "record",
                "-e",
                "cpu-clock",
                "-F",
                "49",
                "--strict-freq",
                "--call-graph",
                "fp",
                "-o",
                str(root / "perf.data"),
                "-p",
                str(worker.process.pid),
                "--",
                "sleep",
                "10",
            ],
            root,
            "perf-record",
        )
    )
    try:
        count = await profile_workload(worker, profiler)
    finally:
        await profiler
        (root / "maps-after.txt").write_bytes(maps.read_bytes())
    script = await logged(
        ["perf", "script", "--show-lost-events", "-i", str(root / "perf.data")],
        root,
        "perf-script",
    )
    raw = await logged(
        ["perf", "script", "-D", "-i", str(root / "perf.data")], root, "perf-raw"
    )
    validate_trace(raw, script)
    write_json(
        root / "profile-result.json",
        {
            "frequency": 49,
            "workloads": count,
            "window_seconds": 10,
            "lost_events": 0,
            "throttle_events": 0,
            "lag": "not reported by perf; retain raw timestamps and resource trace",
            "purpose": "profile-only, includes preparation outside paired timer",
        },
    )


def validate_trace(raw: str, decoded: str) -> None:
    if "PERF_RECORD_LOST" in raw or "PERF_RECORD_THROTTLE" in raw:
        raise ValueError("profile reports lost samples or frequency throttling")
    if "calc_flow::operator::window" not in decoded:
        raise ValueError("profile lacks resolved target Rust frames")


async def profile_workload(worker: AuditWorker, profiler) -> int:
    count = 0
    for _ in range(200):
        if profiler.done():
            break
        validate_sample(await worker.request(operation="sample"))
        count += 1
    if count == 0:
        raise ValueError("profile ended before any workload")
    return count


async def profile_case(case: dict, release: dict, site: Path, root: Path) -> None:
    worker = await AuditWorker.start(site, root)
    try:
        await _prepare({"candidate": worker}, {"candidate": release}, case)
        natives = list((site / "calc_flow").glob("_native*.so"))
        if len(natives) != 1:
            raise ValueError("missing installed profile ELF")
        await perf_preflight(natives[0], root)
        await record_profile(worker, root)
        await worker.request(operation="finish")
    finally:
        await worker.close()


async def collect_profiles(profiles: Path, output: Path) -> None:
    for side in ("A", "B"):
        release = await profile_release(side, profiles / side, output / side)
        site = await install(release, output / side / "installed")
        for case in cases():
            await profile_case(case, release, site, output / side / str(case["rows"]))
