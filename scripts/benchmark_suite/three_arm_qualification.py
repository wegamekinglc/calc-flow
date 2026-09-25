"""Pre-registered three-arm qualification of fresh installed benchmark wheels."""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

from scripts.benchmark_suite.catalog import get_shard, shard_cases
from scripts.benchmark_suite.installed_pair_prototype import (
    _write_report,
    collect_side,
    planned_schedule,
)
from scripts.benchmark_suite.installed_pair_validation import (
    CASE_ID,
    CONTRACT,
    summarize_report,
    validate_report,
    wheel_package_files,
)
from scripts.benchmark_suite.release import load_release
from scripts.toolkit import sha256_file, wheel_native_sha256

PYTHON_DELAY_NS = 1_000_000
NATIVE_DELAY_NS = 1_000_000
BASE_GIT_SHA = "508fd9fe702a809e22ce37bbfe3b134fb28db626"
ARMS = ("aa", "python", "native")


class DiskMonitor:
    def __init__(self, root: Path, *, interval: float = 0.01) -> None:
        self.root = root.resolve()
        self.interval = interval
        self.samples: list[dict[str, int]] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    def _sample(self) -> None:
        usage = shutil.disk_usage(self.root)
        self.samples.append(
            {
                "monotonic_ns": time.monotonic_ns(),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }
        )

    async def _observe(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(self.interval)
            self._sample()

    async def __aenter__(self) -> DiskMonitor:
        self._sample()
        self._task = asyncio.create_task(self._observe())
        return self

    async def __aexit__(self, *_error: object) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task

    def summary(self) -> dict:
        if not self.samples:
            raise ValueError("runner disk was not sampled")
        return {
            "filesystem_root": str(self.root),
            "filesystem_dev": self.root.stat().st_dev,
            "interval_seconds": self.interval,
            "sample_count": len(self.samples),
            "initial_free_bytes": self.samples[0]["free_bytes"],
            "min_free_bytes": min(item["free_bytes"] for item in self.samples),
            "peak_used_bytes": max(item["used_bytes"] for item in self.samples),
            "total_bytes": self.samples[0]["total_bytes"],
        }


def qualification_plan(base_wheel_sha256: str) -> dict:
    return {
        "contract": "bench316-three-arm-v1",
        "case_id": CASE_ID,
        "base_git_sha": BASE_GIT_SHA,
        "base_wheel_sha256": base_wheel_sha256,
        "python_delay_ns": PYTHON_DELAY_NS,
        "native_delay_ns": NATIVE_DELAY_NS,
        "threshold_percent": 5.0,
        "max_abs_lag1_correlation": 0.3,
        "max_abs_sequence_correlation": 0.5,
        "max_stratum_median_gap_percent": 5.0,
        "round_arm_order": [
            ["aa", "python", "native"],
            ["native", "aa", "python"],
        ],
        "schedule": planned_schedule(),
    }


def python_injection_source(original: str) -> str:
    anchor = "        value = await self.source.next()\n        if value is None:"
    if original.count(anchor) != 1 or original.count("import threading\n") != 1:
        raise ValueError("Python source injection anchor changed")
    return (
        original.replace("import threading\n", "import threading\nimport time\n", 1)
        .replace(
            anchor,
            "        value = await self.source.next()\n"
            "        if isinstance(value, Data):\n"
            "            started = time.perf_counter_ns()\n"
            "            while time.perf_counter_ns() - started < "
            "_BENCH316_PYTHON_DATA_DELAY_NS:\n"
            "                pass\n"
            "        if value is None:",
            1,
        )
        .replace(
            "import time\n",
            f"import time\n\n_BENCH316_PYTHON_DATA_DELAY_NS = {PYTHON_DELAY_NS}\n",
            1,
        )
    )


def native_injection_source(original: str) -> str:
    anchor = "        decode_source_event(&value)\n    }\n\n    async fn close"
    if original.count(anchor) != 1:
        raise ValueError("native source injection anchor changed")
    return original.replace(
        anchor,
        "        const BENCH316_NATIVE_DATA_DELAY_NS: u64 = "
        f"{NATIVE_DELAY_NS};\n"
        "        let event = decode_source_event(&value)?;\n"
        "        if matches!(&event, Some(calc_flow::SourceEvent::Data { .. })) {\n"
        "            let started = std::time::Instant::now();\n"
        "            while started.elapsed()\n"
        "                < std::time::Duration::from_nanos("
        "BENCH316_NATIVE_DATA_DELAY_NS)\n"
        "            {\n"
        "                std::hint::spin_loop();\n"
        "            }\n"
        "        }\n"
        "        Ok(event)\n"
        "    }\n\n    async fn close",
        1,
    )


def patch_python_wheel(base: Path, variant: Path) -> None:
    """Repack one sealed wheel with a modified installed Python source file."""
    variant.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(base) as original, ZipFile(variant, "w") as output:
        records = []
        record_names = [
            name for name in original.namelist() if name.endswith(".dist-info/RECORD")
        ]
        if len(record_names) != 1:
            raise ValueError("expected one wheel RECORD")
        record_name = record_names[0]
        if original.namelist().count("calc_flow/runtime.py") != 1:
            raise ValueError("wheel lacks one Python hot-path module")
        for info in original.infolist():
            if info.filename == record_name:
                continue
            data = original.read(info.filename)
            if info.filename == "calc_flow/runtime.py":
                data = python_injection_source(data.decode()).encode()
            output.writestr(info, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(
                b"="
            )
            records.append((info.filename, "sha256=" + digest.decode(), str(len(data))))
        records.append((record_name, "", ""))
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerows(records)
        output.writestr(original.getinfo(record_name), buffer.getvalue().encode())


def _variant_manifest(
    base: dict, wheel: Path, kind: str, delay_ns: int, output: Path
) -> dict:
    package_files = wheel_package_files(str(wheel))
    base_files = wheel_package_files(base["wheel_path"])
    changed = {
        name
        for name in set(package_files) | set(base_files)
        if package_files.get(name) != base_files.get(name)
    }
    expected = {
        "python": {"calc_flow/runtime.py"},
        "native": {"calc_flow/_native.abi3.so"},
    }[kind]
    if changed != expected:
        raise ValueError(f"{kind} wheel changed unexpected package files: {changed}")
    native_sha256 = wheel_native_sha256(wheel)
    if (native_sha256 != base["native_sha256"]) != (kind == "native"):
        raise ValueError("native extension identity does not match injection arm")
    manifest = {
        "contract": "bench316-variant-v1",
        "base_git_sha": BASE_GIT_SHA,
        "base_wheel_sha256": base["wheel_sha256"],
        "kind": kind,
        "delay_ns": delay_ns,
        "wheel": wheel.name,
        "wheel_sha256": sha256_file(wheel),
        "native_sha256": native_sha256,
        "package_files": package_files,
    }
    (output / "variant.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build_python_variant(base_path: Path, output: Path) -> dict:
    base = load_release(base_path)
    if base["git_sha"] != BASE_GIT_SHA:
        raise ValueError("Python injection base SHA differs from pre-registration")
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / Path(base["wheel_path"]).name
    patch_python_wheel(Path(base["wheel_path"]), wheel)
    return _variant_manifest(base, wheel, "python", PYTHON_DELAY_NS, output)


def build_native_variant(base_path: Path, source: Path, output: Path) -> dict:
    base = load_release(base_path)
    if base["git_sha"] != BASE_GIT_SHA:
        raise ValueError("native injection base SHA differs from pre-registration")
    source = source.resolve()
    revision = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source)
        .decode()
        .strip()
    )
    if revision != BASE_GIT_SHA:
        raise ValueError("native build source differs from baseline wheel revision")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source).strip():
        raise ValueError("native build source was not clean before injection")
    path = source / "crates/calc-flow-python/src/continuous.rs"
    path.write_text(native_injection_source(path.read_text()))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PYO3_PYTHON": sys.executable,
        "CARGO_TARGET_DIR": str(output.parents[2] / "bench316-native-cargo"),
        "CARGO_BUILD_JOBS": "2",
        "CARGO_INCREMENTAL": "0",
    }
    argv = [
        sys.executable,
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
    started = time.monotonic()
    with (output / "build.log").open("wb") as log:
        result = subprocess.run(
            argv,
            cwd=source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    (output / "build-command.json").write_text(
        json.dumps(
            {
                "argv": argv,
                "source": str(source),
                "source_sha": revision,
                "exit_code": result.returncode,
                "elapsed_seconds": time.monotonic() - started,
            },
            indent=2,
        )
        + "\n"
    )
    if result.returncode:
        raise RuntimeError(f"native diagnostic build failed: {result.returncode}")
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("native build did not produce exactly one wheel")
    patch = subprocess.check_output(["git", "diff", "--", str(path)], cwd=source)
    (output / "native-injection.patch").write_bytes(patch)
    return _variant_manifest(base, wheels[0], "native", NATIVE_DELAY_NS, output)


def load_variant(path: Path, base: dict, kind: str) -> dict:
    manifest = json.loads(path.read_text())
    if Path(manifest["wheel"]).name != manifest["wheel"]:
        raise ValueError("variant wheel must be inside its manifest directory")
    wheel = path.parent / manifest["wheel"]
    expected = {
        "contract": "bench316-variant-v1",
        "base_git_sha": BASE_GIT_SHA,
        "base_wheel_sha256": base["wheel_sha256"],
        "kind": kind,
        "delay_ns": PYTHON_DELAY_NS if kind == "python" else NATIVE_DELAY_NS,
        "wheel_sha256": sha256_file(wheel),
        "native_sha256": wheel_native_sha256(wheel),
        "package_files": wheel_package_files(str(wheel)),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("injected wheel no longer matches its sealed variant manifest")
    return {
        **manifest,
        "wheel_path": str(wheel.resolve()),
        "git_sha": BASE_GIT_SHA,
    }


def _release_evidence(release: dict) -> dict:
    return {
        key: release[key]
        for key in (
            "wheel_sha256",
            "native_sha256",
            "wheel_path",
            "git_sha",
            "package_files",
        )
    }


async def collect_qualification(plan_path: Path, base_path: Path, output: Path) -> dict:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = load_release(base_path)
    base = {**base, "package_files": wheel_package_files(base["wheel_path"])}
    plan = json.loads(plan_path.read_text())
    if plan != qualification_plan(base["wheel_sha256"]):
        raise ValueError("three-arm schedule differs from the pre-registered plan")
    variants = {
        kind: load_variant(output / "variants" / kind / "variant.json", base, kind)
        for kind in ("python", "native")
    }
    candidates = {"aa": base, **variants}
    case = next(
        case
        for case in shard_cases(get_shard("engines-100000"))
        if case["id"] == CASE_ID
    )
    slots = {slot: str(output / "slots" / slot / "site") for slot in ("A0", "B1")}
    reports = {
        arm: {
            "contract": CONTRACT,
            "case_id": CASE_ID,
            "case": case,
            "slots": slots,
            "releases": {
                "baseline": _release_evidence(base),
                "candidate": _release_evidence(candidates[arm]),
            },
            "schedule": plan["schedule"],
            "blocks": [[], []],
            "summary": None,
        }
        for arm in ARMS
    }
    started = time.monotonic_ns()
    async with DiskMonitor(output) as monitor:
        try:
            for round_index, arms in enumerate(plan["round_arm_order"]):
                for arm in arms:
                    report = reports[arm]
                    arm_root = output / "arms" / arm
                    for spec in plan["schedule"][round_index]:
                        block_root = (
                            arm_root / f"round-{round_index}" / f"block-{spec['index']}"
                        )
                        block_root.mkdir(parents=True)
                        block = {**spec, "executed_order": [], "sides": {}}
                        report["blocks"][round_index].append(block)
                        site = Path(slots[spec["slot"]])
                        for role in spec["order"]:
                            release = base if role == "baseline" else candidates[arm]
                            evidence = await collect_side(
                                case, release, role, site, spec["index"], block_root
                            )
                            block["executed_order"].append(role)
                            block["sides"][role] = evidence
                            _write_report(arm_root, report)
                            shutil.rmtree(site)
                        _write_report(arm_root, report)
                        print(
                            json.dumps(
                                {
                                    "arm": arm,
                                    "round": round_index,
                                    "index": spec["index"],
                                    "slot": spec["slot"],
                                    "order": spec["order"],
                                    "baseline_seconds": block["sides"]["baseline"][
                                        "sample"
                                    ]["seconds"],
                                    "candidate_seconds": block["sides"]["candidate"][
                                        "sample"
                                    ]["seconds"],
                                }
                            ),
                            flush=True,
                        )
        finally:
            disk = {**monitor.summary(), "samples": monitor.samples}
            (output / "disk.json").write_text(json.dumps(disk, indent=2) + "\n")
    duration_seconds = (time.monotonic_ns() - started) / 1e9
    for arm, report in reports.items():
        report["summary"] = summarize_report(report)
        _write_report(output / "arms" / arm, report)
        validate_report(report)
    return {
        "elapsed_seconds": duration_seconds,
        "reports": reports,
        "disk": disk,
    }


def validate_qualification(plan_path: Path, base_path: Path, output: Path) -> dict:
    base = load_release(base_path)
    base = {**base, "package_files": wheel_package_files(base["wheel_path"])}
    plan = json.loads(plan_path.read_text())
    if plan != qualification_plan(base["wheel_sha256"]):
        raise ValueError("pre-registered plan changed after measurement")
    candidates = {
        "aa": base,
        **{
            kind: load_variant(output / "variants" / kind / "variant.json", base, kind)
            for kind in ("python", "native")
        },
    }
    reports = {
        arm: json.loads((output / "arms" / arm / "report.json").read_text())
        for arm in ARMS
    }
    summaries = {}
    effects = {}
    seen = set()
    previous_end = None
    install_intervals = []
    for round_index, arm_order in enumerate(plan["round_arm_order"]):
        for arm in arm_order:
            report = reports[arm]
            if report["schedule"] != plan["schedule"]:
                raise ValueError("arm schedule was changed")
            if report["releases"] != {
                "baseline": _release_evidence(base),
                "candidate": _release_evidence(candidates[arm]),
            }:
                raise ValueError("arm wheel or package provenance changed")
            summaries[arm] = validate_report(report)
            effects[arm] = effect_diagnostics(summaries[arm], plan)
            for block, spec in zip(
                report["blocks"][round_index],
                plan["schedule"][round_index],
                strict=True,
            ):
                if block["index"] != spec["index"]:
                    raise ValueError("arm block order changed")
                for role in spec["order"]:
                    side = block["sides"][role]
                    identity = (
                        side["tree"]["root"]["dev"],
                        side["tree"]["root"]["inode"],
                        side["tree"]["root"]["ctime_ns"],
                    )
                    if identity in seen or side["install_id"] in seen:
                        raise ValueError("installation tree was reused between arms")
                    seen.add(identity)
                    seen.add(side["install_id"])
                    if previous_end is not None and side["started_ns"] <= previous_end:
                        raise ValueError("arm collection order or timestamps changed")
                    previous_end = side["finished_ns"]
                    phase = side["phase_ns"]["install"]
                    install_intervals.append(
                        (phase["started_ns"], phase["finished_ns"])
                    )
    disk = json.loads((output / "disk.json").read_text())
    samples = disk["samples"]
    if len(samples) < 3 or disk["sample_count"] != len(samples):
        raise ValueError("runner disk sampling is incomplete")
    if any(
        next_sample["monotonic_ns"] <= sample["monotonic_ns"]
        for sample, next_sample in zip(samples[:-1], samples[1:], strict=True)
    ):
        raise ValueError("runner disk sample order changed")
    if (
        disk["peak_used_bytes"] != max(sample["used_bytes"] for sample in samples)
        or disk["min_free_bytes"] != min(sample["free_bytes"] for sample in samples)
        or disk["filesystem_dev"] != output.stat().st_dev
    ):
        raise ValueError("runner filesystem evidence disagrees with raw samples")
    install_samples = [
        sample
        for sample in samples
        if any(
            start <= sample["monotonic_ns"] <= end for start, end in install_intervals
        )
    ]
    if any(
        not any(start <= sample["monotonic_ns"] <= end for sample in samples)
        for start, end in install_intervals
    ):
        raise ValueError("disk monitor missed one or more fresh installations")
    phase_totals = {}
    for report in reports.values():
        for blocks in report["blocks"]:
            for block in blocks:
                for side in block["sides"].values():
                    for phase, bounds in side["phase_ns"].items():
                        phase_totals[phase] = phase_totals.get(phase, 0) + (
                            bounds["finished_ns"] - bounds["started_ns"]
                        )
    verdicts = {arm: summary["verdict"] for arm, summary in summaries.items()}
    qualified = qualification_verdict(verdicts) == "qualified" and not any(
        effect["material"] for effect in effects.values()
    )
    result = {
        "contract": "bench316-three-arm-result-v1",
        "status": "controls-pass-cost-unverified" if qualified else "blocked",
        "verdicts": verdicts,
        "round_intervals": {
            arm: summary["round_intervals"] for arm, summary in summaries.items()
        },
        "effects": effects,
        "install_peak_used_bytes": max(
            sample["used_bytes"] for sample in install_samples
        ),
        "install_min_free_bytes": min(
            sample["free_bytes"] for sample in install_samples
        ),
        "phase_totals_seconds": {
            phase: value / 1e9 for phase, value in phase_totals.items()
        },
        "tree_count": len(install_intervals),
    }
    (output / "qualification.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("plan", "python", "native", "collect", "validate")
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    match arguments.action:
        case "plan":
            base = load_release(arguments.baseline)
            if base["git_sha"] != BASE_GIT_SHA:
                raise ValueError(
                    "baseline wheel differs from pre-registered source SHA"
                )
            path = output / "plan.json"
            if path.exists():
                raise ValueError("pre-registered plan already exists")
            path.write_text(
                json.dumps(qualification_plan(base["wheel_sha256"]), indent=2) + "\n"
            )
        case "python":
            build_python_variant(arguments.baseline, output / "variants" / "python")
        case "native":
            if arguments.source is None:
                raise ValueError("native wheel build requires --source")
            build_native_variant(
                arguments.baseline, arguments.source, output / "variants" / "native"
            )
        case "collect":
            asyncio.run(
                collect_qualification(output / "plan.json", arguments.baseline, output)
            )
        case "validate":
            result = validate_qualification(
                output / "plan.json", arguments.baseline, output
            )
            print(json.dumps(result, sort_keys=True))
            raise SystemExit(1)


def qualification_verdict(verdicts: dict[str, str]) -> str:
    if set(verdicts) != set(ARMS):
        raise ValueError("three-arm results are incomplete")
    return (
        "qualified"
        if verdicts
        == {
            "aa": "no-confirmed-regression",
            "python": "regression",
            "native": "regression",
        }
        else "blocked"
    )


def _correlation(left: list[float], right: list[float]) -> float:
    if statistics.pstdev(left) == 0 or statistics.pstdev(right) == 0:
        return 0.0
    return statistics.correlation(left, right)


def effect_diagnostics(summary: dict, plan: dict) -> dict:
    rounds = []
    for round_index, changes in enumerate(summary["block_changes"]):
        lag1 = _correlation(changes[:-1], changes[1:])
        sequence = _correlation(list(range(len(changes))), changes)
        strata = summary["strata_medians"]
        slot_gap = abs(strata["slot:A0"][round_index] - strata["slot:B1"][round_index])
        order_gap = abs(
            strata["order:AB"][round_index] - strata["order:BA"][round_index]
        )
        material = (
            abs(lag1) >= plan["max_abs_lag1_correlation"]
            or abs(sequence) >= plan["max_abs_sequence_correlation"]
            or slot_gap >= plan["max_stratum_median_gap_percent"]
            or order_gap >= plan["max_stratum_median_gap_percent"]
        )
        rounds.append(
            {
                "lag1_correlation": lag1,
                "sequence_correlation": sequence,
                "slot_median_gap": slot_gap,
                "order_median_gap": order_gap,
                "material": material,
            }
        )
    return {"rounds": rounds, "material": any(item["material"] for item in rounds)}


if __name__ == "__main__":
    main()
