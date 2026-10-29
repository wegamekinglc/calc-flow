"""Offline audit of only the newly collected adjacent Arrow engine pairs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import pyarrow as pa
from fixtures import CASES, NATIVE, ROWS, digest, encode, fixture, oracle, save


def read(path):
    return json.loads(path.read_text())


def check_rows(case, table):
    expected = oracle(case)
    if table.column_names != ["id", "y"] or any(
        not table[name].equals(expected[name]) for name in expected.column_names
    ):
        raise ValueError("raw Arrow values, columns or sequence differ")


def cf_boundary(directory, sample):
    metadata = read(directory / "worker.json")
    if (
        metadata["native_sha256"] != NATIVE
        or metadata["output_ipc_ns"][1] > metadata["source_eof_ns"]
    ):
        raise ValueError("CF native seal/EOF boundary mismatch")
    if sample["observer"]["observer_ns"][1] >= metadata["source_eof_ns"]:
        raise ValueError("CF input closed before IPC consumption")
    return (metadata["core_ns"][1] - metadata["core_ns"][0]) / 1e6


def native_boundary(directory, sample, batches):
    metadata = read(directory / "worker.json")
    if (
        metadata["clock_contract"] != "ready-native-Batch-v2"
        or metadata["source_eof_before_output"]
        or metadata["source_eof_at_consume"]
    ):
        raise ValueError("Arroyo did not meet the open native boundary")
    if metadata["input"]["input_rows"] != ROWS or metadata["input"][
        "input_batches"
    ] != len(batches):
        raise ValueError("Arroyo input binding dimension mismatch")
    if (
        digest(directory / "input.arrow") != sample["input_sha256"]
        or not (directory / "output.arrow.closed").exists()
    ):
        raise ValueError("Arroyo input/terminal proof missing")
    return metadata["core_seconds"] * 1000


def check_one(root, case, sample):
    arm = sample["arm"]
    directory = root / f"{case.name}-{sample['pair']}-{arm}"
    if read(directory / "sample.json") != sample:
        raise ValueError("per-run/master observations differ")
    output = directory / "output.arrow"
    check_rows(case, pa.ipc.open_stream(output).read_all())
    if digest(output) != sample["output_sha256"]:
        raise ValueError("output hash mismatch")
    _, batches = fixture(case)
    if encode(batches) != (root / f"{case.name}-input.arrow").read_bytes():
        raise ValueError("input bytes/batch boundaries differ")
    if digest(root / f"{case.name}-input.arrow") != sample["input_sha256"]:
        raise ValueError("input seal mismatch")
    core_ms = (
        cf_boundary(directory, sample)
        if arm == "cf"
        else native_boundary(directory, sample, batches)
    )
    if (
        not math.isclose(core_ms, sample["core_ms"], rel_tol=1e-12)
        or sample["output_rows"] != ROWS // 2
    ):
        raise ValueError("core statistic or complete output count differs")
    if read(directory / "command.json")["argv"][:3] != ["taskset", "-c", "4-7"]:
        raise ValueError("worker resource mismatch")
    allowed_exit = {0} if arm == "cf" else {0, 128}
    if read(directory / "exit.json")["code"] not in allowed_exit:
        raise ValueError("unexpected process exit")
    if sample["rss_peak_bytes"] != max(
        row["rss_bytes"] for row in read(directory / "memory.json")
    ):
        raise ValueError("RSS differs from raw samples")


def pairs(samples):
    result = []
    for case in CASES[:2]:
        ratios = []
        details = []
        for pair in range(5):
            group = [s for s in samples if s["case"] == case.name and s["pair"] == pair]
            arms = {row["arm"]: row for row in group}
            ratios.append(arms["cf"]["core_ms"] / arms["arroyo"]["core_ms"])
            details.append(
                {
                    "pair": pair,
                    "order": [s["arm"] for s in group],
                    "gap_ms": (
                        group[1]["invocation_ns"][0] - group[0]["invocation_ns"][1]
                    )
                    / 1e6,
                }
            )
        result.append(
            {
                "case": case.name,
                "ratio": "Arroyo/CF core throughput",
                "ratios": ratios,
                "median": statistics.median(ratios),
                "min": min(ratios),
                "max": max(ratios),
                "adjacency": details,
            }
        )
    return result


def audit(root):
    identity = read(root / "identity.json")
    for name, checksum in identity["harnesses"].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError(f"measured harness changed: {name}")
    if (
        digest(Path(__file__).with_name("arroyo-ipc-observed.patch"))
        != identity["arroyo_patch_sha256"]
    ):
        raise ValueError("observed adapter patch changed")
    expected = [
        (case.name, pair, arm)
        for case in CASES[:2]
        for pair in range(5)
        for arm in (("cf", "arroyo") if pair % 2 == 0 else ("arroyo", "cf"))
    ]
    samples = read(root / "samples.json")
    if [(s["case"], s["pair"], s["arm"]) for s in samples] != expected:
        raise ValueError("adjacent AB/BA inventory is incomplete")
    for sample in samples:
        case = next(case for case in CASES if case.name == sample["case"])
        check_one(root, case, sample)
    save(root / "pairs.json", pairs(samples))
    result = {
        "audit": "passed",
        "samples": len(samples),
        "pairs": 10,
        "output_rows": sum(s["output_rows"] for s in samples),
    }
    save(root / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root)))
