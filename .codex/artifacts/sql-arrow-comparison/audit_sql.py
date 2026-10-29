"""Offline correctness, timer, inventory and paired SQL/Arroyo summaries."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import pyarrow as pa
from common import (
    ARMS,
    ARROYO,
    CASES,
    NATIVE,
    PRIOR,
    ROWS,
    check_rows,
    digest,
    encode,
    fixture,
    inventory,
    milliseconds,
    read,
    save,
)


def summed(spans):
    return sum(milliseconds(span) for span in spans)


def cf_clocks(worker, sample):
    observer = sample["observer"]
    api, ipc = worker["api_ready_ns"], worker["output_ipc_ns"]
    if not (
        worker["native_sha256"] == NATIVE
        and worker["input_ipc_ns"][1] <= observer["ready_ns"]
        and observer["observer_ns"][0] <= api[0] < api[1] <= ipc[0]
        and ipc[1] <= observer["observer_ns"][1]
    ):
        raise ValueError("CF input/API/output completion boundary differs")
    expected = 1 if sample["case"] == "transform-full" else 64
    if worker["input_rows"] != ROWS or worker["input_batches"] != expected:
        raise ValueError("CF input dimension differs")
    nested = worker["normal_collect_nested_ns"]
    phases = worker["plan_nested_ns"]
    observed = nested.values() if sample["arm"] == "sql" else phases.values()
    if any(len(spans) != expected for spans in observed):
        raise ValueError("normal API calls were omitted or moved out of timer")
    if any(
        not api[0] <= span[0] <= span[1] <= api[1]
        for spans in observed
        for span in spans
    ):
        raise ValueError("normal API subphase escaped total timer")
    return milliseconds(api)


def native_clocks(worker, directory):
    if (
        worker["clock_contract"] != "ready-native-Batch-v2"
        or worker["source_eof_before_output"]
        or worker["source_eof_at_consume"]
        or not (directory / "output.arrow.closed").exists()
    ):
        raise ValueError("real Arroyo completion/open-source evidence missing")
    return (worker["core_seconds"] + worker["output_binding_seconds"]) * 1000


def verify_one(root, sample):
    directory = root / f"{sample['case']}-{sample['group']}-{sample['arm']}"
    if read(directory / "sample.json") != sample:
        raise ValueError("master/per-run observations differ")
    case = next(case for case in CASES[:2] if case.name == sample["case"])
    check_rows(case, pa.ipc.open_stream(directory / "output.arrow").read_all())
    if (directory / "input.arrow").read_bytes() != encode(fixture(case)[1]):
        raise ValueError("actual IPC input bytes/batch boundaries differ")
    for name in ("input", "output"):
        if digest(directory / f"{name}.arrow") != sample[name + "_sha256"]:
            raise ValueError("raw IPC hash differs")
    if sample["output_rows"] != ROWS // 2:
        raise ValueError("incomplete result")
    worker = read(directory / "worker.json")
    engine_ms = (
        native_clocks(worker, directory)
        if sample["arm"] == "arroyo"
        else cf_clocks(worker, sample)
    )
    for actual, expected in (
        (sample["engine_ready_ms"], engine_ms),
        (sample["consumable_ms"], milliseconds(sample["observer"]["observer_ns"])),
        (sample["consumable_input_rows_s"], ROWS * 1000 / sample["consumable_ms"]),
        (sample["engine_ready_input_rows_s"], ROWS * 1000 / engine_ms),
    ):
        if not math.isclose(actual, expected, rel_tol=1e-12):
            raise ValueError("reported statistic differs from raw clocks")
    verify_resources(directory, sample)


def verify_resources(directory, sample):
    allowed = {0, 128} if sample["arm"] == "arroyo" else {0}
    if read(directory / "exit.json")["code"] not in allowed:
        raise ValueError("unexpected process exit")
    if read(directory / "command.json")["argv"][:3] != ["taskset", "-c", "4-7"]:
        raise ValueError("worker resources differ")
    if sample["rss_peak_bytes"] != max(
        row["rss_bytes"] for row in read(directory / "memory.json")
    ):
        raise ValueError("RSS differs from recorded samples")


def phases(root, sample):
    worker = read(
        root / f"{sample['case']}-{sample['group']}-{sample['arm']}" / "worker.json"
    )
    observer = sample["observer"]
    result = {
        "engine_ready_ms": sample["engine_ready_ms"],
        "consumable_ms": sample["consumable_ms"],
        "consumable_wan_rows_s": sample["consumable_input_rows_s"] / 10000,
        "engine_ready_wan_rows_s": sample["engine_ready_input_rows_s"] / 10000,
        "launch_to_ready_ms": (observer["ready_ns"] - observer["launch_ns"]) / 1e6,
        "rss_peak_mib": sample["rss_peak_bytes"] / 2**20,
    }
    if sample["arm"] == "arroyo":
        result.update(
            {
                "native_core_ms": worker["core_seconds"] * 1000,
                "result_packaging_ms": worker["output_binding_seconds"] * 1000,
                "output_ipc_ms": worker["output_ipc_seconds"] * 1000,
                "input_ipc_ms": worker["input"]["input_ipc_seconds"] * 1000,
                "pre_ready_input_binding_ms": worker["input"]["input_binding_seconds"]
                * 1000,
            }
        )
    else:
        for name in (
            "input_ipc",
            "caller_table_views",
            "declaration",
            "precompile",
            "output_ipc",
        ):
            result[name + "_ms"] = milliseconds(worker[name + "_ns"])
        if sample["arm"] == "sql":
            nested = worker["normal_collect_nested_ns"]
            result["normal_input_binding_ms"] = summed(nested["_collect_batches"])
            result["normal_prepare_compile_ms"] = summed(
                nested["_prepare_collect_batches"]
            )
            result["result_packaging_ms"] = summed(nested["_tables"])
        else:
            nested = worker["plan_nested_ns"]
            for name, spans in nested.items():
                result[name.removesuffix("_ns") + "_ms"] = summed(spans)
        result["final_concat_ms"] = milliseconds(worker["final_concat_ns"])
    return result


def describe(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def summaries(root, samples):
    groups = []
    ratios = []
    for case in CASES[:2]:
        for arm in ARMS:
            rows = [
                phases(root, s)
                for s in samples
                if (s["case"], s["arm"]) == (case.name, arm)
            ]
            groups.append(
                {
                    "case": case.name,
                    "arm": arm,
                    "samples": len(rows),
                    "phases": {
                        key: describe([row[key] for row in rows]) for key in rows[0]
                    },
                }
            )
        for arm in ("sql", "sql_plan"):
            entries = []
            for group in sorted({s["group"] for s in samples}):
                rows = [
                    s for s in samples if s["case"] == case.name and s["group"] == group
                ]
                lookup = {s["arm"]: s for s in rows}
                entries.append(
                    {
                        "group": group,
                        "order": [s["arm"] for s in rows],
                        "consumable_ratio": lookup[arm]["consumable_ms"]
                        / lookup["arroyo"]["consumable_ms"],
                        "engine_stage_ratio": lookup[arm]["engine_ready_ms"]
                        / lookup["arroyo"]["engine_ready_ms"],
                        "adjacent_gap_ms": min(
                            abs(
                                lookup[arm]["invocation_ns"][0]
                                - lookup["arroyo"]["invocation_ns"][1]
                            ),
                            abs(
                                lookup["arroyo"]["invocation_ns"][0]
                                - lookup[arm]["invocation_ns"][1]
                            ),
                        )
                        / 1e6,
                    }
                )
            ratios.append(
                {
                    "case": case.name,
                    "cf_arm": arm,
                    "direction": "Arroyo/CF input throughput",
                    "consumable_ratio": describe(
                        [e["consumable_ratio"] for e in entries]
                    ),
                    "engine_stage_ratio": describe(
                        [e["engine_stage_ratio"] for e in entries]
                    ),
                    "groups": entries,
                }
            )
    save(root / "summary.json", groups)
    save(root / "paired.json", ratios)


def audit(root):
    provenance = read(root / "identity.json")
    if (
        provenance["arroyo_binary_sha256"] != ARROYO
        or provenance["cf_native"] != NATIVE
    ):
        raise ValueError("binary provenance mismatch")
    for name, checksum in provenance["harnesses"].items():
        if digest(PRIOR / name) != checksum:
            raise ValueError("reviewed shared harness changed")
    for name, checksum in provenance["sql_harnesses"].items():
        if digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError("measured SQL harness changed")
    preflight = read(root / "completion.json")["preflight"]
    samples = read(root / "samples.json")
    if [(s["case"], s["group"], s["arm"]) for s in samples] != inventory(preflight):
        raise ValueError("fixed interleaved sample inventory differs")
    for sample in samples:
        verify_one(root, sample)
    summaries(root, samples)
    result = {
        "audit": "passed",
        "preflight": preflight,
        "samples": len(samples),
        "output_rows": sum(s["output_rows"] for s in samples),
    }
    save(root / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(audit(parser.parse_args().root))
