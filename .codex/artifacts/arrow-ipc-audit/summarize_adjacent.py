"""Post-collection phase summaries; never used inside acquisition timers."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from audit_adjacent import audit, read
from fixtures import CASES, save


def milliseconds(span):
    return (span[1] - span[0]) / 1e6


def phases(root, sample):
    directory = root / f"{sample['case']}-{sample['pair']}-{sample['arm']}"
    worker = read(directory / "worker.json")
    observer = sample["observer"]
    result = {
        "core_ms": sample["core_ms"],
        "million_input_rows_s": sample["rows_per_second"] / 1e6,
        "launch_to_ready_ms": (observer["ready_ns"] - observer["launch_ns"]) / 1e6,
        "observer_ms": milliseconds(observer["observer_ns"]),
        "rss_peak_mib": sample["rss_peak_bytes"] / 2**20,
    }
    if sample["arm"] == "cf":
        for name in ("input_ipc", "input_binding", "output_binding", "output_ipc"):
            result[name + "_ms"] = milliseconds(worker[name + "_ns"])
        result["declaration_compile_job_ready_ms"] = milliseconds(
            worker["compile_job_ready_ns"]
        )
        metrics = worker["status_before_eof"]
        result["operator_processing_ms"] = (
            sum(
                row["processing_duration_micros"]
                for row in metrics["operators"].values()
            )
            / 1000
        )
        result["sink_write_ms"] = (
            sum(row["write_duration_micros"] for row in metrics["sinks"].values())
            / 1000
        )
    else:
        for name in ("input_ipc", "input_binding"):
            result[name + "_ms"] = worker["input"][name + "_seconds"] * 1000
        for name in ("output_binding", "output_ipc"):
            result[name + "_ms"] = worker[name + "_seconds"] * 1000
    return result


def summary(root):
    audit(root)
    samples = read(root / "samples.json")
    result = []
    for case in CASES[:2]:
        for arm in ("cf", "arroyo"):
            rows = [
                phases(root, sample)
                for sample in samples
                if (sample["case"], sample["arm"]) == (case.name, arm)
            ]
            result.append(
                {
                    "case": case.name,
                    "arm": arm,
                    "samples": len(rows),
                    "phases": {
                        key: {
                            "median": statistics.median(row[key] for row in rows),
                            "min": min(row[key] for row in rows),
                            "max": max(row[key] for row in rows),
                        }
                        for key in rows[0]
                    },
                }
            )
    save(root / "phase-summary.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summary(parser.parse_args().root)
