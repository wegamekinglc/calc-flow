"""Recompute descriptive summaries and diagnostic phase deltas offline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from common import save


def distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def summary(samples, root):
    groups = dict.fromkeys((row["case"], row["arm"]) for row in samples)
    result = []
    for case, arm in groups:
        rows = [row for row in samples if (row["case"], row["arm"]) == (case, arm)]
        record = {
            "case": case,
            "arm": arm,
            "observations": len(rows),
            "elapsed_ms": distribution([row["elapsed_s"] * 1000 for row in rows]),
            "rows_per_second": distribution([row["rows_per_second"] for row in rows]),
            "rss_peak_mib": distribution(
                [row["rss_peak_bytes"] / 2**20 for row in rows]
            ),
            "rss_peak_through_shutdown_mib": distribution(
                [
                    max(
                        point["rss_bytes"]
                        for point in json.loads(
                            (
                                root / f"{case}-{row['pair']}-{arm}" / "memory.json"
                            ).read_text()
                        )
                    )
                    / 2**20
                    for row in rows
                ]
            ),
        }
        if "latency_ms" in rows[0]:
            record["latency_ms"] = {
                key: distribution([row["latency_ms"][key] for row in rows])
                for key in ("p50", "p95", "p99")
            }
        result.append(record)
    return result


def paired_ratios(samples):
    cases = dict.fromkeys(row["case"] for row in samples)
    result = []
    for case in cases:
        names = tuple(
            dict.fromkeys(row["arm"] for row in samples if row["case"] == case)
        )
        comparisons = [
            (names[i], names[j]) for i in range(len(names)) for j in range(i)
        ]
        for numerator, denominator in comparisons:
            ratios = []
            for pair in range(5):
                group = {
                    row["arm"]: row
                    for row in samples
                    if row["case"] == case and row["pair"] == pair
                }
                ratios.append(
                    group[numerator]["rows_per_second"]
                    / group[denominator]["rows_per_second"]
                )
            result.append(
                {
                    "case": case,
                    "numerator": numerator,
                    "denominator": denominator,
                    "ratios": ratios,
                    **distribution(ratios),
                }
            )
    return result


def callback_summary(records):
    result = []
    for name in dict.fromkeys(row["callback"] for row in records):
        group = [
            row
            for row in records
            if row["callback"] == name and row["outcome"] == "completed"
        ]
        phases = (
            "attached_ns",
            "queued_ns",
            "dispatched_ns",
            "completed_ns",
            "elapsed_ns",
        )
        deltas = {}
        for index, phase in enumerate(phases):
            previous = phases[index - 1] if index else None
            deltas[phase] = sum(
                row[phase] - (row[previous] if previous else 0) for row in group
            )
        result.append(
            {
                "callback": name,
                "completed_records": len(group),
                "exclusive_lifecycle_phase_ms": {k: v / 1e6 for k, v in deltas.items()},
                "elapsed_sum_ms": sum(row["elapsed_ns"] for row in group) / 1e6,
            }
        )
    return result


def diagnostics(root):
    result = []
    for path in sorted(root.glob("profile-*/profile.json")):
        doc = json.loads(path.read_text())
        sample = json.loads((path.parent / "sample.json").read_text())
        details = doc["details"]
        record = {
            "name": path.parent.name,
            "instrumented_elapsed_ms": sample["elapsed_s"] * 1000,
            "stages": {
                name: {"calls": len(values), "sum_ms": sum(values) / 1e6}
                for name, values in doc["stage_ns"].items()
            },
        }
        metrics = details.get("datafusion_metrics", [])
        if metrics:
            keys = [key for key in metrics[0] if key.endswith("_ns")]
            record["datafusion_sum_ms"] = {
                k: sum(m[k] for m in metrics) / 1e6 for k in keys
            }
            record["datafusion_queries"] = len(metrics)
            record["physical_planning_count"] = sum(
                m["physical_planning_count"] for m in metrics
            )
        if "stream_status" in details:
            status = details["stream_status"]
            record["operators"] = status["operators"]
            record["sinks"] = status["sinks"]
            record["edge_blocked_sends"] = sum(
                edge["blocked_sends"] for edge in status["edges"].values()
            )
        if "native_callbacks" in details:
            callbacks = details["native_callbacks"]
            record["callback_dropped"] = callbacks["dropped"]
            record["callbacks"] = callback_summary(callbacks["records"])
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    samples = json.loads((args.root / "results/samples.json").read_text())
    save(args.root / "summary.json", summary(samples, args.root / "results"))
    save(args.root / "paired-ratios.json", paired_ratios(samples))
    save(args.root / "diagnostic-summary.json", diagnostics(args.root / "results"))
    print(f"Recomputed {len(samples)} observations and eight diagnostic summaries")


if __name__ == "__main__":
    main()
