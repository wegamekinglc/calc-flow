"""Bounded installed-wheel pairing prototype for one group-by benchmark case."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import shutil
import sys
import time
import uuid
from pathlib import Path

from scripts.benchmark_suite.catalog import get_shard, shard_cases
from scripts.benchmark_suite.installed_pair_validation import (
    CASE_ID,
    CONTRACT,
    summarize_report,
    validate_report,
    wheel_package_files,
)
from scripts.benchmark_suite.measure import validate_environment, validate_sample
from scripts.benchmark_suite.process import ROOT, Worker, child_environment, command
from scripts.benchmark_suite.release import load_release
from scripts.benchmark_suite.tree_identity_diagnostic import (
    _native_path,
    tree_fingerprint,
)


def planned_schedule() -> list[list[dict]]:
    cells = (
        ("A0", ["baseline", "candidate"]),
        ("A0", ["candidate", "baseline"]),
        ("B1", ["baseline", "candidate"]),
        ("B1", ["candidate", "baseline"]),
    )
    rounds = []
    for round_index in range(2):
        counts = (3, 2, 2, 3) if round_index == 0 else (2, 3, 3, 2)
        choices = [
            (slot, order)
            for (slot, order), count in zip(cells, counts, strict=True)
            for _ in range(count)
        ]
        random.Random(31620260925 + round_index).shuffle(choices)
        rounds.append(
            [
                {"round": round_index, "index": index, "slot": slot, "order": order}
                for index, (slot, order) in enumerate(choices)
            ]
        )
    return rounds


async def install_fresh(release: dict, site: Path, log: Path) -> dict:
    if site.exists():
        raise ValueError(f"install slot is not empty: {site}")
    site.parent.mkdir(parents=True, exist_ok=True)
    await command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--target",
            str(site),
            "--link-mode",
            "copy",
            release["wheel_path"],
        ],
        cwd=ROOT,
        log=log,
        env=child_environment(),
    )
    return json.loads(log.with_suffix(".command.json").read_text())


def native_mapping(pid: int, site: Path) -> tuple[int, str, list[str]]:
    expected = str(_native_path(site).resolve())
    inode = _native_path(site).stat().st_ino
    matches = []
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) >= 6 and fields[4] == str(inode) and fields[5] == expected:
            matches.append(line)
    if not matches:
        raise ValueError("worker did not map the newly installed native file")
    return inode, expected, matches


def _tree_evidence(site: Path) -> dict:
    fingerprint = tree_fingerprint(site)
    return {
        "root": {
            **fingerprint["root"],
            "ctime_ns": site.stat().st_ctime_ns,
        },
        "native": fingerprint["native"],
        "native_sha256": hashlib.sha256(_native_path(site).read_bytes()).hexdigest(),
        "files_sha256": hashlib.sha256(
            json.dumps(fingerprint["files"], sort_keys=True).encode()
        ).hexdigest(),
        "files": fingerprint["files"],
        "bytes": sum(path.stat().st_size for path in site.rglob("*") if path.is_file()),
    }


async def collect_side(
    case: dict,
    release: dict,
    role: str,
    site: Path,
    index: int,
    root: Path,
) -> dict:
    """Install one fresh wheel and replay to the matching logical sample."""
    started_ns = time.monotonic_ns()
    install_id = uuid.uuid4().hex
    install_log = root / f"install-{role}.log"
    install_record = await install_fresh(release, site, install_log)
    tree = _tree_evidence(site)
    if tree["native_sha256"] != release["native_sha256"]:
        raise ValueError("installed native file differs from sealed wheel")
    worker = await Worker.start(site, root / f"worker-{role}")
    try:
        environment = validate_environment(
            await worker.request(operation="hello"), release
        )
        prepared = await worker.request(operation="prepare", case=case)
        if prepared["case"] != case:
            raise ValueError("worker prepared a different workload")
        warmup = prepared["warmup"]
        validate_sample(warmup)
        replay = []
        for _ in range(index):
            sample = await worker.request(operation="sample")
            validate_sample(sample)
            replay.append(sample)
        formal = await worker.request(operation="sample")
        validate_sample(formal)
        loaded_inode, loaded_path, native_maps = native_mapping(
            worker.process.pid, site
        )
        completion = await worker.request(operation="finish")
        if completion["state"] != "completed":
            raise ValueError("worker did not finish the paired sample")
    finally:
        await worker.close()
    return {
        "install_id": install_id,
        "install_command": install_record,
        "wheel_sha256": release["wheel_sha256"],
        "site": str(site.resolve()),
        "tree": tree,
        "loaded_native_inode": loaded_inode,
        "loaded_native_path": loaded_path,
        "native_maps": native_maps,
        "worker_pid": worker.process.pid,
        "environment": environment,
        "warmup": warmup,
        "replay": replay,
        "sample": formal,
        "started_ns": started_ns,
        "finished_ns": time.monotonic_ns(),
    }


def _output_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _write_report(root: Path, report: dict) -> None:
    (root / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


async def run(baseline_path: Path, candidate_path: Path, output: Path) -> dict:
    """Collect exactly two rounds of ten independently installed tree pairs."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    releases = {
        role: {
            **release,
            "package_files": wheel_package_files(release["wheel_path"]),
        }
        for role, release in (
            ("baseline", load_release(baseline_path)),
            ("candidate", load_release(candidate_path)),
        )
    }
    case = next(
        case
        for case in shard_cases(get_shard("engines-100000"))
        if case["id"] == CASE_ID
    )
    schedule = planned_schedule()
    slots = {slot: str(output / "slots" / slot / "site") for slot in ("A0", "B1")}
    report = {
        "contract": CONTRACT,
        "case_id": CASE_ID,
        "case": case,
        "slots": slots,
        "releases": {
            side: {
                key: release[key]
                for key in (
                    "wheel_sha256",
                    "native_sha256",
                    "wheel_path",
                    "git_sha",
                    "package_files",
                )
                if key in release
            }
            for side, release in releases.items()
        },
        "schedule": schedule,
        "blocks": [[], []],
        "summary": None,
    }
    started_ns = time.monotonic_ns()
    peak_output_bytes = 0
    min_free_bytes = shutil.disk_usage(output).free
    for round_schedule in schedule:
        for spec in round_schedule:
            block_root = output / f"round-{spec['round']}" / f"block-{spec['index']}"
            block_root.mkdir(parents=True)
            block = {**spec, "executed_order": [], "sides": {}}
            report["blocks"][spec["round"]].append(block)
            site = Path(slots[spec["slot"]])
            for role in spec["order"]:
                evidence = await collect_side(
                    case, releases[role], role, site, spec["index"], block_root
                )
                block["executed_order"].append(role)
                block["sides"][role] = evidence
                peak_output_bytes = max(peak_output_bytes, _output_bytes(output))
                min_free_bytes = min(min_free_bytes, shutil.disk_usage(output).free)
                _write_report(output, report)
                shutil.rmtree(site)
            _write_report(output, report)
            print(
                json.dumps(
                    {
                        "round": spec["round"],
                        "index": spec["index"],
                        "slot": spec["slot"],
                        "order": spec["order"],
                        "change_percent": 100
                        * (
                            block["sides"]["candidate"]["sample"]["seconds"]
                            / block["sides"]["baseline"]["sample"]["seconds"]
                            - 1
                        ),
                    }
                ),
                flush=True,
            )
    report["elapsed_seconds"] = (time.monotonic_ns() - started_ns) / 1e9
    report["peak_output_bytes"] = peak_output_bytes
    report["min_free_bytes"] = min_free_bytes
    report["summary"] = summarize_report(report)
    _write_report(output, report)
    validate_report(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(run(arguments.baseline, arguments.candidate, arguments.output))


if __name__ == "__main__":
    main()
