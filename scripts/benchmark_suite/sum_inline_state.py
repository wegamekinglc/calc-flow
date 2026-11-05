"""Bounded paired evidence for the inline grouped integer SUM representation."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from dataclasses import replace
from pathlib import Path

from scripts.benchmark_suite import sum_batch_parallel as evidence
from scripts.benchmark_suite import sum_followup as followup
from scripts.benchmark_suite import sum_state as maintained
from scripts.benchmark_suite import sum_state_publication as publication
from scripts.toolkit import write_json

SLUG = "sum-inline-state-20261105"
BASELINE = "222f4e2d88f56b5b4ce635abab24eceed732da6d"
COMMON_HARNESS = followup.COMMON_HARNESS
CASES = (
    {**maintained.CASES[0], "reason": "direct single-SUM allocation and drop target"},
    {
        **maintained.CASES[0],
        "id": "stock-stream-4cpu",
        "cpus": [0, 2, 4, 6],
        "reason": "same representation with current four-thread diagnostic policy",
    },
    {
        **maintained.CASES[2],
        "reason": "sparse updates and retained complete-snapshot control",
    },
)
DENSE_TARGETS = ("stock-stream-1cpu", "stock-stream-4cpu")
TOOL_FILES = (
    "scripts/benchmark_suite/sum_inline_state.py",
    "scripts/benchmark_suite/sum_state.py",
    "scripts/benchmark_suite/sum_state_profile.py",
    "scripts/benchmark_suite/sum_state_publication.py",
    "scripts/benchmark_suite/sum_followup.py",
    "scripts/benchmark_suite/sum_batch_parallel.py",
    "scripts/benchmark_suite/report.py",
    "scripts/benchmark_suite/statistics.py",
    "scripts/toolkit.py",
    "scripts/test_sum_inline_state.py",
)


def helper_seal(path):
    try:
        return maintained.sha(path)
    except OSError:
        return None


def helper_seals():
    return {name: helper_seal(maintained.ROOT / name) for name in TOOL_FILES}


def validate_tool_seals(protocol):
    expected = protocol["tool_source_sha256"]
    if set(expected) != set(TOOL_FILES):
        raise ValueError("frozen measurement helper inventory differs")
    maintained.compare_fields(
        helper_seals(), expected, TOOL_FILES, "measurement helper", required=True
    )


def verify_helper_seals(context, acquisition):
    try:
        validate_tool_seals(context.protocol)
    except ValueError as error:
        previous = acquisition.get("failure")
        reason = f"{previous}; {error!r}" if previous else repr(error)
        acquisition["status"], acquisition["failure"] = "stopped", reason
        return False
    return True


def validate_protocol(protocol):
    expected = {
        "slug": SLUG,
        "baseline_revision": BASELINE,
        "budget_seconds": 600,
        "rounds": 2,
        "pairs_per_round": 10,
        "warmups_per_arm_per_round": 1,
        "cases": list(CASES),
    }
    maintained.compare_fields(
        protocol, expected, expected, "inline protocol", required=True
    )
    revision = protocol["candidate_revision"]
    if (
        not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or revision == BASELINE
    ):
        raise ValueError("a new public candidate revision is required")


def validate_fresh_build(build):
    if build["harness_sha256"] != COMMON_HARNESS:
        raise ValueError("both products require the common657 instrumentation")
    if not build.get("product_recompiled") or not build.get("example_recompiled"):
        raise ValueError("fresh core and example compilation are required")


def validate_products(builds, protocol):
    if builds["baseline"]["revision"] != BASELINE:
        raise ValueError("fresh current222 baseline is required; af400 is incompatible")
    if builds["candidate"]["revision"] != protocol["candidate_revision"]:
        raise ValueError("candidate differs from frozen public source")
    for build in builds.values():
        validate_fresh_build(build)


def validate_receipt_seals(args, protocol):
    for side in ("baseline", "candidate"):
        if (
            maintained.sha(getattr(args, side + "_build"))
            != protocol["build_receipt_sha256"][side]
        ):
            raise ValueError(f"frozen {side} build receipt drift")


def prepare_context(args, started):
    protocol = publication.read(args.protocol)
    validate_protocol(protocol)
    validate_tool_seals(protocol)
    validate_receipt_seals(args, protocol)
    options = argparse.Namespace(
        **{
            **vars(args),
            "budget_seconds": 600,
            "prior_measurement_seconds": 0,
            "history_borrow": False,
        }
    )
    collection = maintained.collection_context(options, started)
    scope = {
        **collection.scope,
        "cases": CASES,
        "profile_sides": ("baseline", "candidate"),
        "profile_min_remaining_seconds": 0,
    }
    collection = replace(collection, scope=scope)
    validate_products(collection.builds, protocol)
    for build in collection.builds.values():
        evidence.source_equivalence(build, build["revision"], collection.deadline)
    return followup.Followup(
        collection, protocol, maintained.sha(args.protocol), "inline-state", None
    )


def validate_profile_clock(record):
    alignment = record.get("clock_alignment", {})
    if (
        alignment.get("matched_sample_count", 0) <= 0
        or not 0 <= alignment.get("max_error_ms", float("inf")) < 0.00001
    ):
        raise ValueError("profile clock alignment is missing or invalid")


def validate_profile_record(record, builds):
    if record.get("binary_sha256") != builds[record["side"]]["binary_sha256"]:
        raise ValueError("profile binary differs from sealed product")
    counts = record.get("samples_per_window", [])
    if (
        len(record.get("windows", [])) != 5
        or len(counts) != 5
        or min(counts, default=0) <= 0
    ):
        raise ValueError("profile requires five sampled processing windows")
    validate_profile_clock(record)


def validate_profiles(profiles, builds):
    if len(profiles) != 2 or {record["side"] for record in profiles} != {
        "baseline",
        "candidate",
    }:
        raise ValueError("both product profiles are required")
    for record in profiles:
        validate_profile_record(record, builds)


def profiles_valid(profiles, builds):
    try:
        validate_profiles(profiles, builds)
    except ValueError:
        return False
    return True


def collect(context, acquisition):
    collection = context.collection
    measurement = maintained.prepare_fixture(collection)
    maintained.collect_profiles(measurement, acquisition)
    validate_profiles(
        followup.accepted_profiles(context, acquisition), collection.builds
    )
    for case in collection.scope["cases"]:
        maintained.collect_case(measurement, case, acquisition)
    if time.monotonic() > collection.deadline:
        raise TimeoutError("inline SUM measurement deadline")
    acquisition["status"] = "completed"


def accepted_verdicts(verdicts):
    return (
        all(verdicts.values())
        and "regression" not in verdicts.values()
        and any(verdicts.get(target) == "improved" for target in DENSE_TARGETS)
    )


def performance_acceptance(status, cases):
    if status != "completed" or [case["id"] for case in cases] != [
        case["id"] for case in CASES
    ]:
        return False
    return accepted_verdicts(
        {case["id"]: case.get("statistics", {}).get("verdict") for case in cases}
    )


def report(summary):
    rows, intervals, resources = publication.rendered_rows(summary)
    accepted = performance_acceptance(summary["status"], summary["cases"])
    return "\n".join(
        [
            f"# {SLUG}",
            "",
            f"Status: **{summary['status']}**.",
            f"Performance acceptance: **{accepted}**.",
            "",
            publication.table(
                ["Case", "Baseline P50 ms", "Candidate P50 ms", "Change", "Verdict"],
                rows,
            ),
            "",
            publication.table(
                ["Case", "Round", "Paired median", "Median interval", "Coverage"],
                intervals,
            ),
            "",
            publication.table(
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
            "Three fixed cases use two rounds of ten alternating pairs and one",
            "600-second budget including preflight, fixtures, warmups, independent",
            "full-snapshot oracles, capture/decoding, analysis, publication and",
            "cleanup. Builds are separate; no cases, samples or retries are added.",
            "",
            "A dense1/dense4 target must have the exact maintained improved verdict,",
            "with no regression on any case, after complete evidence validation.",
            "Inconclusive is uncertainty, not equivalence or a confirmed speedup.",
            "",
            "The baseline is freshly built public222f4e2d. The older af400 binary",
            "is incompatible with current DataFusion/input-registration source.",
            "Both products use common657 native instrumentation from public841 in",
            "permanently frozen build trees; development instrumentation is unchanged.",
            "Input partitions remain at their default one. All streams use one",
            "DataFusion target partition, min_rows_per_partition1 and no checkpoints.",
            "The 4T case uses CPUs0/2/4/6, four reported physical cores, Tokio4;",
            "single-thread cases use CPU0/Tokio1. The controller uses CPU31 on WSL2.",
            "External helper pools stay at one; host load/power is unobservable.",
            "",
            "Dense input is 10k zero seed trades plus20x100k source rows, filtering",
            "to80k trades per business batch. Sparse input retains10k groups and",
            "adds20x100 source rows (80trades/batch,100updatedgroups overall).",
            "Every stream retains21 complete snapshots, independently checked after",
            "the native timer. Whole-process peak RSS includes inputs and retained",
            "outputs; it is not operator-only allocation or an engine memory cap.",
            "Common657 requires non-null inputs. UInt64, nullable/generic aggregates",
            "and checkpoint latency are unmeasured; focused allocation/size/budget",
            "and correctness tests separately cover their changed representation.",
            "",
            f"Accepted profiles: {summary.get('accepted_profile_count', 0)}.",
            "Both serial profiles require five verified processing windows after",
            "excluding the first of six jobs; all per-thread timestamps must align.",
            "Self/inclusive counts and full stacks cover the allocation/drop path.",
            "Inclusive CPU shares overlap and are not phase wall-time measurements.",
            "Missing/inlined frames do not imply zero cost; unresolved leaves remain.",
            "",
            "[Summary](summary.json) and [all formal rows](samples.csv) retain exact",
            "verdicts, every interval, CPU/RSS and source/build/fixture identities.",
            "[Complete raw evidence](complete-evidence.json.gz) includes every warmup,",
            "profile and formal observation, snapshot/schema digest, command, resource",
            "receipt and clock timestamp text. Large perf files stay at recorded local",
            "paths with hashes. [Profiles](profile-summary.json),",
            "[targets](profile-targets.json) and full stacks preserve attribution.",
            "[Manifest](evidence-manifest.json) seals actual budget debit and status.",
            "",
            f"Stopped reason: {summary['failure']}",
            "",
        ]
    )


def validate_publication(context, acquisition, profiles):
    if acquisition["status"] != "completed":
        return
    try:
        publication.validate_inventory(acquisition, [case["id"] for case in CASES])
        validate_profiles(profiles, context.collection.builds)
    except ValueError as error:
        acquisition["status"], acquisition["failure"] = "stopped", repr(error)


def write_metadata(data, now):
    context, acquisition, summary, raw = (
        data.context,
        data.acquisition,
        data.summary,
        data.raw,
    )
    collection = context.collection
    raw["helper_source_sha256_after"] = helper_seals()
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


def write_receipt(data, now):
    value = {
        **followup.receipt(data, now),
        "slug": SLUG,
        "performance_acceptance": performance_acceptance(
            data.acquisition["status"], data.summary["cases"]
        ),
        "current_profile_analysis_complete": data.summary["profile_analysis_complete"],
        "current_profiles": data.summary["accepted_profile_count"],
        "profile_records": len(data.summary["profiles"]),
    }
    followup.write_receipt(data.context, value)
    return value


def publish(context, acquisition, plan, machine):
    summary, raw = followup.publication_data(context, acquisition, plan, machine)
    validate_publication(context, acquisition, summary["profiles"])
    complete_profiles = profiles_valid(summary["profiles"], context.collection.builds)
    summary = {
        **summary,
        "profile_analysis_complete": complete_profiles,
        "accepted_profile_count": len(summary["profiles"]) if complete_profiles else 0,
        "profile_disposition": "accepted"
        if complete_profiles
        else acquisition.get(
            "profile_skip_reason", "incomplete or invalid required profiles"
        ),
        "helper_seals_verified": verify_helper_seals(context, acquisition),
    }
    raw = {
        **raw,
        "profile_clock_evidence": evidence.clock_evidence(context, acquisition),
    }
    data = followup.PublishedFollowup(context, acquisition, summary, raw)
    write_metadata(data, time.monotonic())
    value = write_receipt(data, time.monotonic())
    if not verify_helper_seals(context, acquisition):
        summary["helper_seals_verified"] = False
        write_metadata(data, time.monotonic())
        value = write_receipt(data, time.monotonic())
    final = time.monotonic()
    if final > context.collection.deadline:
        write_metadata(data, final)
        write_receipt(data, time.monotonic())
        raise TimeoutError("inline SUM publication deadline")
    print(json.dumps(value), flush=True)


def plan_and_machine(context):
    plan = maintained.collection_plan(context.collection)
    plan["study"] = {
        "slug": SLUG,
        "protocol": context.protocol,
        "protocol_sha256": context.protocol_sha256,
    }
    machine = maintained.machine_identity(context.collection)
    machine["verified_tool_source_sha256"] = helper_seals()
    for path in (Path(__file__), Path(followup.__file__), Path(evidence.__file__)):
        machine["helper_sha256"][str(path.relative_to(maintained.ROOT))] = (
            maintained.sha(path)
        )
    return plan, machine


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
        verify_helper_seals(context, acquisition)
        publish(context, acquisition, plan, machine)
    if acquisition["status"] != "completed":
        raise RuntimeError(acquisition["failure"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "protocol",
        "baseline-build",
        "candidate-build",
        "output",
        "publish-directory",
        "perf",
        "samply",
    ):
        parser.add_argument(
            "--" + name, required=True, type=lambda value: Path(value).resolve()
        )
    parser.add_argument("--perf-library", type=lambda value: Path(value).resolve())
    run(parser.parse_args())


if __name__ == "__main__":
    main()
