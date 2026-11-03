"""Publish the fixed history-borrow measurement before its residual deadline."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path

from scripts.toolkit import linear_percentile, write_json

TARGETS = {
    "candidate": "IncrementalSql::candidate [",
    "candidate_accumulators": "IncrementalSql::candidate_accumulators",
    "scalar_drop": "drop_in_place<datafusion_common::scalar::ScalarValue>",
    "group_drop": "drop_in_place<calc_flow::operator::sql::incremental::Group>",
    "row_converter": "arrow_row::RowConverter::",
    "row_encode": "arrow_row::",
    "hashing": "core::hash::BuildHasher::hash_one",
    "hash_map_any": "hashbrown::",
    "memory_reservation": "MemoryReservation",
    "new_empty": "::new_empty",
    "output_chunk": "IncrementalSql::output_chunk [",
    "merge_integer_history": "PartialGroups::merge_integer_history",
    "merge_rank": "PartialGroups::merge_rank",
    "finalize_candidate": "IncrementalSql::finalize_candidate [",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_timing(args, acquisition, started, deadline, now):
    acquisition["elapsed_seconds"] = now - started
    acquisition["prior_measurement_seconds"] = args.prior_measurement_seconds
    acquisition["cumulative_measurement_seconds"] = (
        args.prior_measurement_seconds + acquisition["elapsed_seconds"]
    )
    if now > deadline:
        acquisition["status"] = "stopped"
        reason = "TimeoutError('measurement deadline exhausted')"
        previous = acquisition.get("failure")
        acquisition["failure"] = (
            f"{previous}; {reason}"
            if previous and "deadline" not in previous
            else previous or reason
        )


def persist_source(args, acquisition):
    write_json(args.output / "acquisition.json", acquisition)
    write_json(
        args.output / "summary.json",
        {
            "status": acquisition["status"],
            "failure": acquisition["failure"],
            "elapsed_seconds": acquisition["elapsed_seconds"],
            "prior_measurement_seconds": acquisition["prior_measurement_seconds"],
            "cumulative_measurement_seconds": acquisition[
                "cumulative_measurement_seconds"
            ],
            "cases": [
                {k: v for k, v in case.items() if k != "observations"}
                for case in acquisition["cases"]
            ],
            "profiles": acquisition["profiles"],
        },
    )


def profile_targets(path):
    data = json.loads(gzip.decompress(path.read_bytes()))
    frames, stacks = data["frames"], data["stacks"]
    total = sum(row["samples"] for row in stacks)
    targets = {}
    for name, needle in TARGETS.items():
        matching = {i for i, frame in enumerate(frames) if needle in frame}
        inclusive = sum(
            row["samples"]
            for row in stacks
            if any(index in matching for index in row["frames"])
        )
        own = sum(row["samples"] for row in stacks if row["frames"][0] in matching)
        targets[name] = {
            "inclusive_samples": inclusive,
            "inclusive_percent": 100 * inclusive / total,
            "self_samples": own,
            "self_percent": 100 * own / total,
            "observed_symbols": [frames[i] for i in sorted(matching)],
        }
    return {"total_samples": total, "targets": targets}


def table(headers, rows):
    text = [[str(v) for v in row] for row in [headers, *rows]]
    widths = [max(len(row[i]) + 2 for row in text) for i in range(len(headers))]
    lines = []
    for index, row in enumerate(text):
        if index == 1:
            lines.append("|" + "|".join("-" * width for width in widths) + "|")
        lines.append(
            "|"
            + "|".join(
                " " + value.ljust(width - 2) + " "
                for value, width in zip(row, widths, strict=True)
            )
            + "|"
        )
    return "\n".join(lines)


def render(summary):
    complete = summary["status"] == "completed"
    lines = [
        "# History-borrow candidate: focused verification",
        "",
        f"Collection status: **{summary['status']}**; "
        f"{summary['formal_observations']} formal observations.",
        "The artifact manifest records the first-phase 471.105224 seconds plus this",
        "phase's collection, analysis, publication and cleanup within 600 seconds.",
        "",
        "The final source changes integer SUM history seeding to borrow the prior",
        "immutable groups by stable slot, avoiding the per-candidate history copy.",
        "The Rust harness, fixtures, affinity, compact state budget and two rounds of",
        "ten alternating pairs are unchanged. These three stream cases directly",
        "exercise the changed state path. **Finite control was not measured on this",
        "final source.** Its earlier inconclusive result does not establish final",
        "finite equivalence or absence of regression.",
        "",
        "## Timings and exact maintained verdicts",
        "",
    ]
    rows, intervals, resources = [], [], []
    for case in summary["cases"]:
        statistics = case.get("statistics")
        if statistics is None:
            rows.append([case["id"], "incomplete", "incomplete", "unknown", "partial"])
            continue
        rows.append(
            [
                case["id"],
                f"{1000 * statistics['base_p50']:.3f}",
                f"{1000 * statistics['head_p50']:.3f}",
                f"{statistics['change_percent']:+.2f}%",
                statistics["verdict"],
            ]
        )
        for index, interval in enumerate(statistics["round_intervals"], 1):
            intervals.append(
                [
                    case["id"],
                    index,
                    f"{interval['median']:+.2f}%",
                    f"[{interval['low']:+.2f}%, {interval['high']:+.2f}%]",
                    f"{100 * interval['coverage']:.5f}%",
                ]
            )
        for side in ("baseline", "candidate"):
            resource = case["resource_summary"][side]
            distribution = case["wall_distribution"][side]
            resources.append(
                [
                    case["id"],
                    side,
                    f"{distribution['p95_ms']:.3f}",
                    f"{resource['processing_cpu_ms_p50']:.3f}",
                    f"{resource['processing_average_cores_p50']:.3f}",
                    f"{resource['whole_process_peak_rss_p50'] / 2**20:.2f}",
                ]
            )
    lines += [
        table(
            ["Case", "Baseline P50 ms", "Candidate P50 ms", "P50 change", "Verdict"],
            rows,
        ),
        "",
    ]
    if intervals:
        lines += [
            table(
                ["Case", "Round", "Paired median", "Median interval", "Coverage"],
                intervals,
            ),
            "",
        ]
    if resources:
        lines += [
            "## CPU and whole-process memory",
            "",
            table(
                [
                    "Case",
                    "Side",
                    "P95 ms",
                    "CPU P50 ms",
                    "Cores P50",
                    "Peak RSS P50 MiB",
                ],
                resources,
            ),
            "",
        ]
    lines += [
        "Verdicts use the existing paired-median confidence rule and 5% threshold.",
        "P50 ratios and paired medians are distinct statistics. Inconclusive timing",
        "does not establish improvement, regression or equivalence. No cases, pairs",
        "or retries were added to obtain a favorable verdict. Peak RSS includes",
        "IPC preload, retained outputs and validation; it is not operator-only memory.",
        "",
        "## Profiles and remaining work",
        "",
        f"Accepted current candidate profiles: {len(summary['profiles'])}.",
        f"Profile budget disposition: {summary['profile_disposition']}.",
        "The predeclared candidate-only capture requires at least 25 seconds remaining",
        "after formal timings; otherwise it is skipped to preserve complete timings",
        "and publication. A capture has six jobs, excludes the first, and verifies",
        "every per-thread timestamp before selecting five exact processing windows.",
        "",
        "The baseline profile is retained from the **earlier window cohort**, with the",
        "same baseline binary, common Rust harness, fixture and CPU resources. It is",
        "qualitative context rather than a newly paired elapsed-speedup comparison.",
        "All accepted stacks and exact self/inclusive target counts are preserved.",
        "Inclusive paths overlap; an absent or inlined symbol does not mean zero cost.",
        "Percentages describe user-mode on-CPU samples, not wall-time attribution.",
        "Off-CPU waits, hardware counters and memory bandwidth were not measured.",
        "Composite-key rewrites, group sharding and finite filter/projection changes",
        "remain unimplemented candidates; no performance benefit for them is claimed.",
        "",
        "## Scope, budget and complete evidence",
        "",
        "Dense input: 2M business rows (1.6M trades), plus 10k zero seed trades.",
        "Sparse input initializes 10k groups and sends 20 batches of 100 rows:",
        "80 trades and 20 non-trades per batch, touching 80 trade groups each batch",
        "and 100 distinct groups across the fixture. Each stream emits 21 complete",
        "10k-group snapshots. All retained snapshots are validated after the primary",
        "timer; absolute times differ from the older final-only sink diagnostic.",
        "Native affinity, thread settings and clean lifecycle are checked each time.",
        "Eight logical CPUs correspond to four reported core IDs on this WSL2 host;",
        "host load and power state are unobservable. Builds are timed separately.",
        "",
    ]
    for side, build in summary["builds"].items():
        lines += [
            f"- {side}: `{build['revision']}`; "
            f"binary SHA256 `{build['binary_sha256']}`; "
            f"Release build {build['build_elapsed_seconds']:.3f}s."
        ]
    lines += [
        "",
        "[summary.json](summary.json) retains every case, resource statistic, build",
        "identity, fixture/oracle digest and environment/tool version/hash. Read all",
        "formal rows directly in [samples.csv](samples.csv).",
        "[complete-evidence.json.gz](complete-evidence.json.gz) includes all raw",
        "warmup/formal/profile rows, commands, resource receipts, lifecycle metrics,",
        "source seals and intermediate snapshot digests.",
        "[Profile summary](profile-summary.json) and [targets](profile-targets.json)",
        "retain",
        "accepted current-profile and historical-context attribution. Artifact hashes",
        "and cumulative timing are in the [manifest](evidence-manifest.json).",
        "The [first complete four-case collection](../20261104/README.md) remains",
        "unchanged, including its non-confirmed streaming results and imprecise finite",
        "control. See [reproduction](../README.md#history-borrow-followup).",
        "",
    ]
    if not complete:
        lines += [
            f"Stopped reason: `{summary['failure']}`. "
            "Incomplete paths remain unverified.",
            "",
        ]
    return "\n".join(lines)


def validate_inventory(acquisition, expected_ids):
    if acquisition["status"] != "completed":
        return
    cases = acquisition["cases"]
    complete = [case["id"] for case in cases] == list(expected_ids)
    for case in cases:
        complete = complete and len(case["observations"]) == 40
        complete = complete and case.get("statistics", {}).get("samples") == 20
        for side in ("baseline", "candidate"):
            complete = complete and len(case[side]) == 2
            complete = complete and all(len(values) == 10 for values in case[side])
    if not complete:
        raise ValueError("incomplete completed evidence inventory")


def publish(args, acquisition, plan, machine, builds, deadline):
    update_timing(
        args, acquisition, acquisition["started_monotonic"], deadline, time.monotonic()
    )
    validate_inventory(acquisition, [case["id"] for case in plan["cases"]])
    destination = args.publish_directory
    destination.mkdir(parents=True, exist_ok=False)
    root = args.output
    previous = Path(__file__).resolve().parents[2] / "benchmarks/sum-state/20261104"
    fixtures_path = root / "fixture-manifest.json"
    fixtures = read(fixtures_path) if fixtures_path.exists() else {}
    first = read(previous / "summary.json")
    if (
        first["builds"]["baseline"]["binary_sha256"]
        != builds["baseline"]["binary_sha256"]
    ):
        raise ValueError("historical profile baseline binary differs")
    if (
        fixtures
        and first["fixtures"]["stock"]["input_sha256"]
        != fixtures["stock"]["input_sha256"]
    ):
        raise ValueError("historical profile fixture differs")
    profiles_path = root / "profile-summary.json"
    profiles = read(profiles_path) if profiles_path.exists() else []
    targets = {
        "historical_baseline": read(previous / "profile-targets.json")["baseline"]
    }
    for profile in acquisition["profiles"]:
        path = root / profile["path"]
        stderr = (path / "stderr.log").read_text()
        if any(
            word in stderr.lower()
            for word in ("lost", "unwind", "warning", "corrupt", "truncat")
        ):
            profiles = []
            acquisition["profile_skip_reason"] = (
                "captured profile excluded: lost/unwind/warning diagnostic"
            )
        if profiles:
            targets[profile["side"]] = profile_targets(path / "stacks.json.gz")
            shutil.copy2(
                path / "stacks.json.gz",
                destination / f"{profile['side']}-stacks.json.gz",
            )
    shutil.copy2(
        previous / "baseline-stacks.json.gz",
        destination / "historical-baseline-stacks.json.gz",
    )
    historical_profile = next(
        row for row in first["profiles"] if row["side"] == "baseline"
    )
    cases = [
        {k: v for k, v in case.items() if k != "observations"}
        for case in acquisition["cases"]
    ]
    for case, original in zip(cases, acquisition["cases"], strict=True):
        case["wall_distribution"] = {}
        for side in ("baseline", "candidate"):
            values = [
                row["row"]["seconds"] * 1000
                for row in original["observations"]
                if row["side"] == side
            ]
            if values:
                case["wall_distribution"][side] = {
                    "p50_ms": linear_percentile(values, 0.5),
                    "p95_ms": linear_percentile(values, 0.95),
                    "min_ms": min(values),
                    "max_ms": max(values),
                }
    summary = {
        "status": acquisition["status"],
        "failure": acquisition["failure"],
        "formal_observations": sum(
            len(case["observations"]) for case in acquisition["cases"]
        ),
        "cases": cases,
        "environment": machine,
        "fixtures": fixtures,
        "builds": {
            side: {k: v for k, v in build.items() if k != "source_sha256"}
            for side, build in builds.items()
        },
        "profiles": profiles,
        "historical_baseline_profile": historical_profile,
        "profile_disposition": "accepted"
        if profiles
        else acquisition.get("profile_skip_reason", "capture/analysis incomplete"),
        "unmeasured_final_source": plan["unmeasured_final_source"],
        "prior_measurement_seconds": args.prior_measurement_seconds,
        "phase_budget_seconds": args.budget_seconds,
    }
    invocations = {}
    for directory in (root / "timings", root / "prepare", root / "profiles"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name in {
                "command.json",
                "receipt.json",
                "stdout.jsonl",
                "stderr.log",
                "clock-alignment.json",
            }:
                invocations[str(path.relative_to(root))] = path.read_text()
    raw = {
        "acquisition": acquisition,
        "plan": plan,
        "environment": machine,
        "fixtures": fixtures,
        "builds": builds,
        "invocations": invocations,
        "profiles": profiles,
        "targets": targets,
        "historical_profile_context": {
            "artifact": "../20261104",
            "baseline": historical_profile,
            "source": first["builds"]["baseline"],
        },
    }

    def metadata(now):
        update_timing(
            args, acquisition, acquisition["started_monotonic"], deadline, now
        )
        summary.update(
            {
                key: acquisition[key]
                for key in (
                    "status",
                    "failure",
                    "elapsed_seconds",
                    "cumulative_measurement_seconds",
                )
            }
        )
        persist_source(args, acquisition)
        (destination / "complete-evidence.json.gz").write_bytes(
            gzip.compress(json.dumps(raw, separators=(",", ":")).encode(), mtime=0)
        )
        write_json(destination / "summary.json", summary)
        (destination / "README.md").write_text(render(summary))

    def stopped_receipt(now):
        phase_seconds = now - acquisition["started_monotonic"]
        return {
            "prior_measurement_seconds": args.prior_measurement_seconds,
            "phase_seconds": phase_seconds,
            "cumulative_measurement_seconds": args.prior_measurement_seconds
            + phase_seconds,
            "cumulative_budget_seconds": 600,
            "remaining_seconds": 600 - args.prior_measurement_seconds - phase_seconds,
            "status": acquisition["status"],
            "failure": acquisition["failure"],
            "phase_stopped_monotonic": now,
            "formal_observations": summary["formal_observations"],
            "current_profiles": len(profiles),
            "retries": 0,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
                for path in sorted(destination.iterdir())
                if path.is_file() and path.name != "evidence-manifest.json"
            },
        }

    write_json(
        destination / "profile-summary.json",
        {
            "current_candidate": profiles,
            "historical_baseline_context": historical_profile,
        },
    )
    write_json(destination / "profile-targets.json", targets)
    shutil.copy2(root / "samples.csv", destination / "samples.csv")
    metadata(time.monotonic())
    receipt = stopped_receipt(time.monotonic())
    write_json(destination / "evidence-manifest.json", receipt)
    write_json(root / "phase-stopped.json", receipt)
    final = time.monotonic()
    if final > deadline:
        # Publication can cross the deadline after writing optimistic metadata.
        # Rewrite every authoritative status and dependent hash before returning.
        metadata(final)
        receipt = stopped_receipt(time.monotonic())
        write_json(destination / "evidence-manifest.json", receipt)
        write_json(root / "phase-stopped.json", receipt)
        raise TimeoutError("residual deadline exhausted during publication")
    print(json.dumps({"publication": str(destination), "receipt": receipt}), flush=True)
