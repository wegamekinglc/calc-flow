"""One bounded profile-selected SUM followup using the maintained native harness."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path

from scripts.benchmark_suite import sum_state as maintained
from scripts.benchmark_suite import sum_state_publication as publication
from scripts.toolkit import write_json

SLUG = "sum-followup-columnar-20261104"
HISTORICAL = "199d70b4156591153eb6b15cc2c8ffc238b3348f"
CURRENT = "4f2a6190d20ce18d10750cd87d86da0bec6f778f"
COMMON_HARNESS = "657de3e697907e473691c2fd0d1b99967d46577333135c50176801f6faa4e04e"


@dataclass(frozen=True, slots=True)
class Followup:
    collection: maintained.CollectionContext
    protocol: dict
    protocol_sha256: str
    phase: str
    prior: dict | None


@dataclass(frozen=True, slots=True)
class PublishedFollowup:
    context: Followup
    acquisition: dict
    summary: dict
    raw: dict


def phase_budget(phase, prior):
    if phase == "current":
        return 120, 0
    if prior is None or prior.get("status") != "completed":
        raise ValueError("candidate requires a completed current-stage receipt")
    if prior.get("phase") != "current":
        raise ValueError("prior receipt belongs to a different work item stage")
    elapsed = prior_elapsed(prior)
    return 600 - elapsed, elapsed


def prior_elapsed(prior):
    elapsed = prior["cumulative_measurement_seconds"]
    if not math.isfinite(elapsed) or not 0 <= elapsed <= 120:
        raise ValueError("invalid prior measurement budget")
    return elapsed


def prior_receipt(args, protocol_hash):
    if args.phase == "current":
        return None
    if args.prior_receipt is None:
        raise ValueError("candidate requires --prior-receipt")
    prior = publication.read(args.prior_receipt)
    if prior.get("protocol_sha256") != protocol_hash:
        raise ValueError("prior receipt is not from this frozen protocol")
    if not prior.get("current_profile_analysis_complete"):
        raise ValueError("P0 current profile and analysis are mandatory")
    return prior


def prepare_context(args, started):
    protocol = publication.read(args.protocol)
    digest = maintained.sha(args.protocol)
    prior = prior_receipt(args, digest)
    budget, spent = phase_budget(args.phase, prior)
    options = {
        **vars(args),
        "budget_seconds": budget,
        "prior_measurement_seconds": spent,
        "history_borrow": False,
    }
    collection = maintained.collection_context(argparse.Namespace(**options), started)
    scope = dict(collection.scope)
    scope["cases"] = (
        (maintained.CASES[3],) if args.phase == "current" else maintained.CASES
    )
    scope["profile_sides"] = ("candidate",)
    scope["profile_min_remaining_seconds"] = 0 if args.phase == "current" else 25
    collection = replace(collection, scope=scope)
    context = Followup(collection, protocol, digest, args.phase, prior)
    validate_products(context)
    return context


def validate_revisions(context):
    builds = context.collection.builds
    expected = HISTORICAL if context.phase == "current" else CURRENT
    if builds["baseline"]["revision"] != expected:
        raise ValueError("wrong baseline product revision")
    if context.phase == "current" and builds["candidate"]["revision"] != CURRENT:
        raise ValueError("current profile must use newly built merged product")
    if context.phase == "candidate" and builds["candidate"]["revision"] == CURRENT:
        raise ValueError("new candidate must differ from the merged product")


def validate_products(context):
    validate_revisions(context)
    for build in context.collection.builds.values():
        if build["harness_sha256"] != COMMON_HARNESS:
            raise ValueError("all arms require the exact common diagnostic harness")
    if (
        context.prior
        and context.collection.builds["baseline"]["binary_sha256"]
        != context.prior["current_binary_sha256"]
    ):
        raise ValueError("candidate baseline differs from the sealed P0 product")


def accepted_profiles(context, acquisition):
    output = context.collection.args.output
    path = output / "profile-summary.json"
    profiles = publication.read(path) if path.exists() else []
    for item in acquisition["profiles"]:
        if publication.profile_has_warning(output / item["path"]):
            acquisition["profile_skip_reason"] = (
                "captured profile excluded for diagnostic warnings"
            )
            return []
    return profiles


def collect(context, acquisition):
    collection = context.collection
    measurement = maintained.prepare_fixture(collection)
    if context.phase == "current":
        maintained.collect_profiles(measurement, acquisition)
        if not accepted_profiles(context, acquisition):
            raise ValueError("current-stage profile is missing or invalid")
    for case in collection.scope["cases"]:
        maintained.collect_case(measurement, case, acquisition)
    if context.phase == "candidate":
        maintained.collect_profiles(measurement, acquisition)
    if time.monotonic() > collection.deadline:
        raise TimeoutError("followup measurement deadline")
    acquisition["status"] = "completed"


def plan_and_machine(context):
    plan = maintained.collection_plan(context.collection)
    plan["followup"] = {
        "slug": SLUG,
        "phase": context.phase,
        "protocol": context.protocol,
        "protocol_sha256": context.protocol_sha256,
        "prior": context.prior,
    }
    machine = maintained.machine_identity(context.collection)
    machine["helper_sha256"]["scripts/benchmark_suite/sum_followup.py"] = (
        maintained.sha(__file__)
    )
    return plan, machine


def report(summary):
    rows = []
    for case in summary["cases"]:
        stats = case.get("statistics")
        if stats:
            rows.append(
                [
                    case["id"],
                    f"{stats['base_p50'] * 1000:.3f}",
                    f"{stats['head_p50'] * 1000:.3f}",
                    f"{stats['change_percent']:+.2f}%",
                    stats["verdict"],
                ]
            )
        else:
            rows.append([case["id"], "partial", "partial", "unknown", "partial"])
    comparison = (
        "original199d versus merged4f2 finite control"
        if summary["phase"] == "current"
        else "merged4f2 versus new profile-selected candidate"
    )
    return "\n".join(
        [
            f"# {SLUG}: {summary['phase']}",
            "",
            f"Status: **{summary['status']}**.",
            f"Comparison: {comparison}.",
            "",
            publication.table(
                ["Case", "Baseline P50 ms", "Candidate P50 ms", "Change", "Verdict"],
                rows,
            ),
            "",
            "Two rounds of ten adjacent alternating pairs use the maintained "
            "paired-median",
            "confidence intervals and 5% rule. All raw observations and full "
            "per-snapshot",
            "oracles, schemas, hashes, native affinity/thread environment, "
            "lifecycle and",
            "source/binary seals are retained. An inconclusive result proves no "
            "equivalence.",
            "",
            "The common native diagnostic harness is 657de3e6 from public "
            "84186554, overlaid",
            "only in permanently frozen build trees. Product revisions are "
            "distinct from",
            "instrumentation provenance; a historical 841 product is never used "
            "as 4f2.",
            "",
            f"Accepted current-stage profiles: {len(summary['profiles'])}.",
            "Profiles use user-mode CPU-clock 999Hz and exact monotonic "
            "processing windows,",
            "excluding the first of six jobs. Every imported per-thread timestamp is",
            "verified against perf before attribution. Inclusive shares overlap; "
            "unresolved",
            "or inlined symbols do not imply zero cost. No hardware "
            "bandwidth/caches or",
            "off-CPU wait attribution is claimed. Profiles are never mixed into "
            "timings.",
            "",
            "[summary.json](summary.json) contains all per-case intervals, CPU/RSS and",
            "build/environment identities. [samples.csv](samples.csv) contains "
            "every formal",
            "row. [complete-evidence.json.gz](complete-evidence.json.gz) retains "
            "every raw",
            "warmup/formal/profile row, command, oracle digest and full source seal.",
            "[Profile summary](profile-summary.json) and "
            "[targets](profile-targets.json)",
            "preserve accepted stacks/counts and provenance.",
            "[evidence-manifest.json](evidence-manifest.json) records the exact "
            "shared 600s",
            "budget debit, hashes and stop status. Native builds are separately timed.",
            "",
            f"Stopped reason: {summary['failure']}",
            "",
        ]
    )


def publication_data(context, acquisition, plan, machine):
    collection = context.collection
    output, destination = collection.args.output, collection.args.publish_directory
    destination.mkdir(parents=True, exist_ok=False)
    profiles = accepted_profiles(context, acquisition)
    targets = {}
    for item in acquisition["profiles"]:
        if profiles:
            path = output / item["path"] / "stacks.json.gz"
            targets[item["side"]] = publication.profile_targets(path)
            shutil.copy2(path, destination / f"{item['side']}-stacks.json.gz")
    fixtures = output / "fixture-manifest.json"
    fixtures = publication.read(fixtures) if fixtures.exists() else {}
    cases = publication.publication_cases(acquisition)
    summary = {
        "phase": context.phase,
        "status": acquisition["status"],
        "failure": acquisition["failure"],
        "cases": cases,
        "profiles": profiles,
        "profile_disposition": acquisition.get("profile_skip_reason", "accepted"),
        "builds": {
            side: publication.build_view(build)
            for side, build in collection.builds.items()
        },
        "environment": machine,
        "fixtures": fixtures,
    }
    raw = {
        "acquisition": acquisition,
        "plan": plan,
        "machine": machine,
        "builds": collection.builds,
        "fixtures": fixtures,
        "profiles": profiles,
        "invocations": publication.invocation_files(output),
        "targets": targets,
    }
    write_json(destination / "profile-summary.json", profiles)
    write_json(destination / "profile-targets.json", targets)
    shutil.copy2(output / "samples.csv", destination / "samples.csv")
    return summary, raw


def write_metadata(data, now):
    context, acquisition, summary, raw = (
        data.context,
        data.acquisition,
        data.summary,
        data.raw,
    )
    collection = context.collection
    publication.update_timing(
        collection.args, acquisition, collection.started, collection.deadline, now
    )
    publication.persist_source(collection.args, acquisition)
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
    destination = collection.args.publish_directory
    (destination / "complete-evidence.json.gz").write_bytes(
        gzip.compress(json.dumps(raw, separators=(",", ":")).encode(), mtime=0)
    )
    write_json(destination / "summary.json", summary)
    (destination / "README.md").write_text(report(summary))


def receipt(data, now):
    context, acquisition, summary = data.context, data.acquisition, data.summary
    collection = context.collection
    elapsed = now - collection.started
    return {
        "slug": SLUG,
        "phase": context.phase,
        "status": acquisition["status"],
        "failure": acquisition["failure"],
        "protocol_sha256": context.protocol_sha256,
        "phase_seconds": elapsed,
        "cumulative_measurement_seconds": collection.args.prior_measurement_seconds
        + elapsed,
        "prior_measurement_seconds": collection.args.prior_measurement_seconds,
        "current_binary_sha256": collection.builds["candidate"]["binary_sha256"],
        "current_profiles": len(summary["profiles"]),
        "current_profile_analysis_complete": bool(summary["profiles"]),
        "formal_observations": sum(
            len(case["observations"]) for case in acquisition["cases"]
        ),
        "phase_stopped_monotonic": now,
        "budget_seconds": 600,
        "retries": 0,
        "files": publication.artifact_hashes(collection.args.publish_directory),
    }


def write_receipt(context, value):
    args = context.collection.args
    write_json(args.publish_directory / "evidence-manifest.json", value)
    write_json(args.output / "phase-stopped.json", value)


def publish(context, acquisition, plan, machine):
    summary, raw = publication_data(context, acquisition, plan, machine)
    data = PublishedFollowup(context, acquisition, summary, raw)
    write_metadata(data, time.monotonic())
    value = receipt(data, time.monotonic())
    write_receipt(context, value)
    final = time.monotonic()
    if final > context.collection.deadline:
        write_metadata(data, final)
        write_receipt(context, receipt(data, time.monotonic()))
        raise TimeoutError("followup publication deadline")
    print(json.dumps(value), flush=True)


def run(args):
    started = time.monotonic()
    context = prepare_context(args, started)
    plan, machine = plan_and_machine(context)
    collection = context.collection
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "environment.json", machine)
    acquisition = maintained.acquisition_state(collection)
    maintained.save_acquisition(collection, acquisition)
    try:
        collect(context, acquisition)
    except BaseException as error:
        acquisition["status"], acquisition["failure"] = "stopped", repr(error)
        raise
    finally:
        maintained.finalize_acquisition(
            collection.args,
            acquisition,
            collection.builds,
            started,
            collection.deadline,
        )
        publish(context, acquisition, plan, machine)
    if acquisition["status"] != "completed":
        raise RuntimeError(acquisition["failure"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("current", "candidate"))
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--prior-receipt", type=Path)
    parser.add_argument("--baseline-build", required=True, type=Path)
    parser.add_argument("--candidate-build", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=lambda value: Path(value).resolve()
    )
    parser.add_argument(
        "--publish-directory", required=True, type=lambda value: Path(value).resolve()
    )
    parser.add_argument(
        "--perf", required=True, type=lambda value: Path(value).resolve()
    )
    parser.add_argument(
        "--samply", required=True, type=lambda value: Path(value).resolve()
    )
    parser.add_argument("--perf-library", type=lambda value: Path(value).resolve())
    run(parser.parse_args())


if __name__ == "__main__":
    main()
