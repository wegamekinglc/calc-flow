"""Independent fail-closed checks for the #316 installed-pair prototype artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from sys import exit as sys_exit
from zipfile import ZipFile

from scripts.benchmark_suite.catalog import get_shard, shard_cases
from scripts.benchmark_suite.release import load_release
from scripts.benchmark_suite.statistics import paired_round

CONTRACT = "installed-pairs-prototype-v1"
CASE_ID = "engines/100000/calc-flow-stream/group_by"
THRESHOLD = 5.0 + 1e-12


def wheel_package_files(path: str) -> dict[str, str]:
    with ZipFile(path) as wheel:
        files = {
            name: hashlib.sha256(wheel.read(name)).hexdigest()
            for name in wheel.namelist()
            if name.startswith("calc_flow/") and not name.endswith("/")
        }
    if not files or not any(name.endswith("/_native.abi3.so") for name in files):
        raise ValueError("sealed wheel lacks the expected calc_flow package")
    return files


def _schedule() -> list[list[dict]]:
    import random

    rounds = []
    cells = (
        ("A0", ["baseline", "candidate"]),
        ("A0", ["candidate", "baseline"]),
        ("B1", ["baseline", "candidate"]),
        ("B1", ["candidate", "baseline"]),
    )
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


def _sample(sample: dict) -> None:
    seconds = sample["seconds"]
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or seconds <= 0
        or sample["correctness"]["passed"] is not True
    ):
        raise ValueError("invalid timing or correctness")


def _validate_side(side: dict, role: str, slot: str, report: dict, index: int) -> None:
    if side["site"] != report["slots"][slot]:
        raise ValueError("paired sides must use the exact same physical slot")
    if side["wheel_sha256"] != report["releases"][role]["wheel_sha256"]:
        raise ValueError("installed wheel differs from sealed release")
    tree = side["tree"]
    package_files = {
        name: digest
        for name, digest in tree["files"].items()
        if name.startswith("calc_flow/")
    }
    expected_native_path = side["site"] + "/calc_flow/_native.abi3.so"
    file_digest = hashlib.sha256(
        json.dumps(tree["files"], sort_keys=True).encode()
    ).hexdigest()
    command = side["install_command"]
    argv = command["argv"]
    if (
        tree["native_sha256"] != report["releases"][role]["native_sha256"]
        or package_files != report["releases"][role]["package_files"]
        or tree["files"]["calc_flow/_native.abi3.so"]
        != report["releases"][role]["native_sha256"]
        or tree["files_sha256"] != file_digest
        or side["loaded_native_inode"] != tree["native"]["inode"]
        or side["loaded_native_path"] != expected_native_path
    ):
        raise ValueError("worker loaded a different native tree")
    if not any(
        len(fields) >= 6
        and fields[4] == str(tree["native"]["inode"])
        and fields[5] == expected_native_path
        for fields in (line.split(maxsplit=5) for line in side["native_maps"])
    ):
        raise ValueError("native map evidence disagrees with the installed tree")
    if (
        command["exit_code"] != 0
        or argv[:3] != ["uv", "pip", "install"]
        or argv[argv.index("--target") + 1] != side["site"]
        or argv[argv.index("--link-mode") + 1] != "copy"
        or argv[-1] != report["releases"][role]["wheel_path"]
    ):
        raise ValueError("installation command differs from the sealed wheel")
    if len(side["replay"]) != index:
        raise ValueError("missing or extra cursor replay")
    modules = {
        "calc_flow": "calc_flow/__init__.py",
        "calc_flow.runtime": "calc_flow/runtime.py",
        "calc_flow.pipeline": "calc_flow/pipeline.py",
        "calc_flow._native": "calc_flow/_native.abi3.so",
    }
    if set(side["loaded_modules"]) != set(modules):
        raise ValueError("loaded Python/native module inventory changed")
    for module, relative_path in modules.items():
        loaded = side["loaded_modules"][module]
        if (
            loaded["path"] != side["site"] + "/" + relative_path
            or loaded["sha256"] != tree["files"][relative_path]
        ):
            raise ValueError("worker imported code outside the installed tree")
    phases = (
        "install",
        "fingerprint",
        "worker_start",
        "hello",
        "prepare_warmup",
        "replay",
        "formal",
    )
    if set(side["phase_ns"]) != set(phases):
        raise ValueError("missing measurement phase")
    prior_end = side["started_ns"]
    for phase in phases:
        bounds = side["phase_ns"][phase]
        if not (
            prior_end
            <= bounds["started_ns"]
            < bounds["finished_ns"]
            <= side["finished_ns"]
        ):
            raise ValueError("measurement phase timestamps are invalid")
        prior_end = bounds["finished_ns"]
    for item in (side["warmup"], *side["replay"], side["sample"]):
        _sample(item)
    if side["finished_ns"] <= side["started_ns"]:
        raise ValueError("non-monotonic worker timing")


def _validate_block(block: dict, spec: dict, report: dict, seen: set) -> None:
    for key in ("round", "index", "slot", "order"):
        if block[key] != spec[key]:
            raise ValueError("actual block differs from the sealed schedule")
    if block["executed_order"] != spec["order"]:
        raise ValueError("block execution order changed")
    if set(block["sides"]) != {"baseline", "candidate"}:
        raise ValueError("block is missing a wheel side")
    first, second = (block["sides"][side] for side in spec["order"])
    if first["finished_ns"] >= second["started_ns"]:
        raise ValueError("paired installations overlapped or ran in the wrong order")
    if first["environment"] != second["environment"]:
        raise ValueError("paired worker environments differ")
    for role, side in block["sides"].items():
        _validate_side(side, role, spec["slot"], report, spec["index"])
        tree = side["tree"]
        identity = (
            tree["root"]["dev"],
            tree["root"]["inode"],
            tree["root"]["ctime_ns"],
        )
        if side["install_id"] in seen or identity in seen:
            raise ValueError("installation tree was reused")
        seen.add(side["install_id"])
        seen.add(identity)
    for item_index in range(spec["index"]):
        if first["replay"][item_index].get("start_row") != second["replay"][
            item_index
        ].get("start_row"):
            raise ValueError("paired replay cursors differ")
    if first["sample"].get("start_row") != second["sample"].get("start_row"):
        raise ValueError("paired formal cursors differ")


def summarize_report(report: dict) -> dict:
    baseline = [
        [block["sides"]["baseline"]["sample"]["seconds"] for block in blocks]
        for blocks in report["blocks"]
    ]
    candidate = [
        [block["sides"]["candidate"]["sample"]["seconds"] for block in blocks]
        for blocks in report["blocks"]
    ]
    intervals = [
        paired_round(left, right)
        for left, right in zip(baseline, candidate, strict=True)
    ]
    changes = [
        [100 * (right / left - 1) for left, right in zip(base, head, strict=True)]
        for base, head in zip(baseline, candidate, strict=True)
    ]
    strata = {}
    for dimension in ("slot", "order"):
        for value in ("A0", "B1") if dimension == "slot" else ("AB", "BA"):
            strata[f"{dimension}:{value}"] = [
                statistics.median(
                    change
                    for block, change in zip(blocks, round_changes, strict=True)
                    if (
                        block["slot"]
                        if dimension == "slot"
                        else ("AB" if block["order"][0] == "baseline" else "BA")
                    )
                    == value
                )
                for blocks, round_changes in zip(report["blocks"], changes, strict=True)
            ]
    if all(interval["low"] > THRESHOLD for interval in intervals):
        verdict = "regression"
    elif any(
        all(value > THRESHOLD for value in medians) for medians in strata.values()
    ):
        verdict = "context-order-dependent"
    elif any(interval["high"] > THRESHOLD for interval in intervals):
        verdict = "unresolved"
    elif all(interval["high"] < -THRESHOLD for interval in intervals):
        verdict = "improved"
    else:
        verdict = "no-confirmed-regression"
    return {
        "baseline_seconds": baseline,
        "candidate_seconds": candidate,
        "block_changes": changes,
        "round_intervals": intervals,
        "strata_medians": strata,
        "verdict": verdict,
    }


def validate_report(report: dict) -> dict:
    try:
        if report["contract"] != CONTRACT or report["case_id"] != CASE_ID:
            raise ValueError("unexpected prototype contract or case")
        expected_case = next(
            case
            for case in shard_cases(get_shard("engines-100000"))
            if case["id"] == CASE_ID
        )
        if report["case"] != expected_case:
            raise ValueError("benchmark workload differs from the catalog")
        expected = _schedule()
        if report["schedule"] != expected or len(report["blocks"]) != 2:
            raise ValueError("missing or modified installed-pair schedule")
        if set(report["slots"]) != {"A0", "B1"}:
            raise ValueError("physical slot inventory changed")
        seen: set = set()
        machine_environment = None
        for blocks, schedule in zip(report["blocks"], expected, strict=True):
            if len(blocks) != 10:
                raise ValueError("round has fewer than ten independent blocks")
            for block, spec in zip(blocks, schedule, strict=True):
                _validate_block(block, spec, report, seen)
                environment = block["sides"]["baseline"]["environment"]
                if machine_environment is None:
                    machine_environment = environment
                elif environment != machine_environment:
                    raise ValueError("worker environment changed across blocks")
        summary = summarize_report(report)
        if summary != report["summary"]:
            raise ValueError("stored statistics differ from raw block evidence")
        return summary
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"invalid installed-pair evidence: {error}") from error


def gate_exit(summary: dict) -> int:
    return 0 if summary["verdict"] in {"no-confirmed-regression", "improved"} else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    arguments = parser.parse_args()
    report = json.loads(arguments.report.read_text())
    for role, path in (
        ("baseline", arguments.baseline),
        ("candidate", arguments.candidate),
    ):
        release = load_release(path)
        for key in ("wheel_sha256", "native_sha256", "wheel_path", "git_sha"):
            if report["releases"][role][key] != release[key]:
                raise ValueError(f"{role} release seal differs from raw artifact")
        if report["releases"][role]["package_files"] != wheel_package_files(
            release["wheel_path"]
        ):
            raise ValueError(f"{role} package files differ from the sealed wheel")
    summary = validate_report(report)
    print(json.dumps(summary, sort_keys=True))
    sys_exit(gate_exit(summary))


if __name__ == "__main__":
    main()
