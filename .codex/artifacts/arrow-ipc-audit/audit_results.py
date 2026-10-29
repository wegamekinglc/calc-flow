"""Revalidate raw IPC rows, clocks, resources and derive descriptive summaries."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import pyarrow as pa
from fixtures import CASES, NATIVE, ROWS, digest, encode, fixture, save, validate


def read(path):
    return json.loads(path.read_text())


def distribution(values):
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
    }


def check_sample(root, case, sample, native_sha=NATIVE):
    directory = root / f"{case.name}-{sample['pair']}-{sample['arm']}"
    if read(directory / "sample.json") != sample:
        raise ValueError("master sample differs from per-run evidence")
    output = directory / "output.arrow"
    with pa.ipc.open_stream(output) as reader:
        table = reader.read_all()
    validate(case, table)
    _, batches = fixture(case)
    if (
        digest(root / f"{case.name}-input.arrow") != sample["input_sha256"]
        or digest(output) != sample["output_sha256"]
    ):
        raise ValueError("IPC payload hash mismatch")
    if encode(batches) != (root / f"{case.name}-input.arrow").read_bytes():
        raise ValueError("IPC input differs from the declared fixture/batch boundaries")
    worker = read(directory / "worker.json")
    start, end = worker["core_ns"]
    if worker["native_sha256"] != native_sha or sample["native_sha256"] != native_sha:
        raise ValueError("native seal mismatch")
    if (
        not worker["input_ipc_ns"][1]
        <= worker["input_binding_ns"][0]
        <= worker["input_binding_ns"][1]
        <= start
        < end
        <= worker["output_ipc_ns"][0]
    ):
        raise ValueError("worker stage clocks overlap or finish before a result")
    if not sample["round_trip_ns"][0] < start < end <= sample["round_trip_ns"][1]:
        raise ValueError("parent round-trip clock excludes core completion")
    if (
        sample["core_ms"] != (end - start) / 1e6
        or sample["output_rows"] != table.num_rows
    ):
        raise ValueError("core statistic/output count mismatch")
    if sample["rss_peak_bytes"] != max(
        point["rss_bytes"] for point in read(directory / "memory.json")
    ):
        raise ValueError("RSS differs from process-tree observations")
    if (
        read(directory / "command.json")[:3] != ["taskset", "-c", "4-7"]
        or read(directory / "exit.json")["code"] != 0
    ):
        raise ValueError("worker resource/exit contract differs")
    return worker


def stage_values(workers, field):
    return [(worker[field][1] - worker[field][0]) / 1e6 for worker in workers]


def summarize(samples, workers):
    groups = dict.fromkeys((sample["case"], sample["arm"]) for sample in samples)
    result = []
    for case, arm in groups:
        rows = [
            sample
            for sample in samples
            if (sample["case"], sample["arm"]) == (case, arm)
        ]
        active = [workers[(case, sample["pair"], arm)] for sample in rows]
        result.append(
            {
                "case": case,
                "arm": arm,
                "samples": len(rows),
                **{
                    name: distribution([row[name] for row in rows])
                    for name in ("core_ms", "round_trip_ms")
                },
                **{
                    name.replace("_ns", "_ms"): distribution(stage_values(active, name))
                    for name in ("input_ipc_ns", "input_binding_ns", "output_ipc_ns")
                },
                "rss_peak_mib": distribution(
                    [row["rss_peak_bytes"] / 2**20 for row in rows]
                ),
                "core_rows_per_second": distribution(
                    [ROWS / (row["core_ms"] / 1000) for row in rows]
                ),
                "round_trip_rows_per_second": distribution(
                    [ROWS / (row["round_trip_ms"] / 1000) for row in rows]
                ),
            }
        )
    return result


def paired(samples):
    result = []
    for case in CASES[:2]:
        for field in ("core_ms", "round_trip_ms"):
            ratios = []
            for pair in range(5):
                group = {
                    row["arm"]: row
                    for row in samples
                    if row["case"] == case.name and row["pair"] == pair
                }
                ratios.append(group["sql"][field] / group["sql_plan"][field])
            result.append(
                {
                    "case": case.name,
                    "metric": field,
                    "comparison": "SQL time / precompiled-plan time",
                    "ratios": ratios,
                    **distribution(ratios),
                }
            )
    return result


def audit(root):
    identity = read(root / "identity.json")
    for name, expected in identity["scripts"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"measured script changed: {name}")
    samples, expected_order = read(root / "samples.json"), []
    workers = {}
    for case in CASES:
        for pair in range(5):
            for arm in case.arms if pair % 2 == 0 else case.arms[::-1]:
                expected_order.append((case.name, pair, arm))
    actual = [(row["case"], row["pair"], row["arm"]) for row in samples]
    if actual != expected_order:
        raise ValueError("fixed sample inventory/order differs")
    for sample in samples:
        case = next(case for case in CASES if case.name == sample["case"])
        workers[(case.name, sample["pair"], sample["arm"])] = check_sample(
            root, case, sample, identity["native_sha256"]
        )
    save(root / "summary.json", summarize(samples, workers))
    save(root / "paired.json", paired(samples))
    return {
        "audit": "passed",
        "samples": len(samples),
        "output_rows": sum(row["output_rows"] for row in samples),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    save(args.root / "audit.json", result)
    print(json.dumps(result))
