"""Run and summarize the P3 SQL/DataFusion tuning matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

try:
    from scripts.verify_sql_datafusion_performance import verify_report
except ModuleNotFoundError:  # Direct `python scripts/...` invocation.
    from verify_sql_datafusion_performance import verify_report

ROWS = (100_000, 1_000_000, 2_100_000)
ENTITIES = (1, 4, 16, 64)
PARTITIONS = (1, 2, 4, 8, 16, 32)
BATCH_SIZES = (4_096, 8_192, 16_384, 32_768)


def matrix_points(
    *, rows: Iterable[int] = ROWS, entities: Iterable[int] = ENTITIES
) -> list[tuple[int, int, int, int]]:
    return sorted(
        (row_count, entity_count, partitions, batch_size)
        for row_count in rows
        for entity_count in entities
        for partitions in PARTITIONS
        for batch_size in BATCH_SIZES
    )


def pareto_frontier(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_latency = float(left["median_ms"])
        right_latency = float(right["median_ms"])
        left_memory = int(left["peak_rss_bytes"])
        right_memory = int(right["peak_rss_bytes"])
        return (
            left_latency <= right_latency
            and left_memory <= right_memory
            and (left_latency < right_latency or left_memory < right_memory)
        )

    return [
        entry
        for entry in entries
        if not any(other is not entry and dominates(other, entry) for other in entries)
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_one(
    binary: Path,
    output: Path,
    *,
    rows: int,
    entities: int,
    partitions: int,
    batch_size: int,
    samples: int,
) -> None:
    subprocess.run(
        [
            str(binary),
            "--profile",
            "matrix",
            "--rows",
            str(rows),
            "--entities",
            str(entities),
            "--partitions",
            str(partitions),
            "--batch-size",
            str(batch_size),
            "--samples",
            str(samples),
            "--warmups",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )


def run_screening(args: argparse.Namespace) -> None:
    points = matrix_points(rows=args.rows or ROWS, entities=args.entities or ENTITIES)
    for rows, entities, partitions, batch_size in points:
        output = args.output_dir / (
            f"r{rows}-e{entities}-p{partitions}-b{batch_size}.json"
        )
        _run_one(
            args.bench_binary,
            output,
            rows=rows,
            entities=entities,
            partitions=partitions,
            batch_size=batch_size,
            samples=5,
        )


def summarize(reports: list[Path], output: Path) -> dict[str, Any]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    fingerprints: set[tuple[str, str]] = set()
    git_shas: set[str] = set()
    for path in sorted(reports):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("profile") != "matrix":
            raise ValueError(f"{path} is not a matrix report")
        verify_report(report, minimum_samples=5, require_stable=False)
        git_shas.add(report["git_sha"])
        environment = report["environment"]
        fingerprints.add(
            (environment["machine_fingerprint"], environment["dependency_fingerprint"])
        )
        for case in report["cases"]:
            calc = case["calc_flow"]
            grouped[(case["rows"], case["active_entities"], case["name"])].append(
                {
                    "report": str(path),
                    "rows": case["rows"],
                    "active_entities": case["active_entities"],
                    "workload": case["name"],
                    "partitions": calc["configured_partitions"],
                    "effective_partitions": calc["effective_partitions"],
                    "batch_size": calc["batch_size"],
                    "median_ms": calc["median_ms"],
                    "cv": calc["cv"],
                    "peak_rss_bytes": calc["peak_rss_bytes"],
                    "spill_bytes": calc["spill_bytes"],
                    "empty_partitions": calc["empty_partitions"],
                    "partition_skew": calc["partition_skew"],
                }
            )
    if len(git_shas) != 1 or len(fingerprints) != 1:
        raise ValueError(
            "matrix reports must share one SHA and machine/dependency fingerprint"
        )
    candidates = [
        candidate
        for key in sorted(grouped)
        for candidate in pareto_frontier(grouped[key])
        if candidate["spill_bytes"] == 0 and candidate["empty_partitions"] == 0
    ]
    manifest = {
        "schema_version": 1,
        "git_sha": next(iter(git_shas)),
        "screening_reports": len(reports),
        "screening_cases": sum(len(entries) for entries in grouped.values()),
        "pareto_candidates": candidates,
    }
    _write_json(output, manifest)
    return manifest


def rerun_candidates(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    configurations = sorted(
        {
            (
                entry["rows"],
                entry["active_entities"],
                entry["partitions"],
                entry["batch_size"],
            )
            for entry in manifest["pareto_candidates"]
        }
    )
    for rows, entities, partitions, batch_size in configurations:
        stem = f"r{rows}-e{entities}-p{partitions}-b{batch_size}"
        for repeat in ("first", "second"):
            _run_one(
                args.bench_binary,
                args.output_dir / f"{stem}-{repeat}.json",
                rows=rows,
                entities=entities,
                partitions=partitions,
                batch_size=batch_size,
                samples=20,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    screen = subparsers.add_parser("screen")
    screen.add_argument("--bench-binary", required=True, type=Path)
    screen.add_argument("--output-dir", required=True, type=Path)
    screen.add_argument("--rows", action="append", type=int)
    screen.add_argument("--entities", action="append", type=int)
    select = subparsers.add_parser("summarize")
    select.add_argument("reports", nargs="+", type=Path)
    select.add_argument("--output", required=True, type=Path)
    rerun = subparsers.add_parser("rerun")
    rerun.add_argument("--bench-binary", required=True, type=Path)
    rerun.add_argument("--manifest", required=True, type=Path)
    rerun.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "screen":
            run_screening(args)
        elif args.command == "summarize":
            summarize(args.reports, args.output)
        else:
            rerun_candidates(args)
    except (OSError, KeyError, ValueError, subprocess.CalledProcessError) as error:
        print(f"SQL/DataFusion matrix failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
