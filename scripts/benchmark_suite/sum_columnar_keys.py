"""Bounded paired evidence for the columnar composite-key SUM path."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pyarrow as pa

from scripts.benchmark_suite import sum_batch_parallel as evidence
from scripts.benchmark_suite import sum_followup as followup
from scripts.benchmark_suite import sum_inline_state as inline
from scripts.benchmark_suite import sum_state as maintained
from scripts.benchmark_suite import sum_state_publication as publication
from scripts.toolkit import write_json

SLUG = "sum-columnar-keys-20261105"
BASELINE = "22f4b9a562bdd087a75d810aeeb2161f29a3d6b7"
COMMON_HARNESS = followup.COMMON_HARNESS
helper_seal = inline.helper_seal
validate_fresh_build = inline.validate_fresh_build
validate_receipt_seals = inline.validate_receipt_seals
validate_profiles = inline.validate_profiles
profiles_valid = inline.profiles_valid
CASES = (
    {**maintained.CASES[0], "reason": "transaction-wide composite-key encoding target"},
    {
        **maintained.CASES[0],
        "id": "stock-stream-4cpu",
        "cpus": [0, 2, 4, 6],
        "reason": "same key path with four physical cores",
    },
    {
        **maintained.CASES[2],
        "reason": "legacy selector sparse retained-snapshot control",
    },
    {
        **maintained.CASES[0],
        "id": "distinct-stream-1cpu",
        "fixture": "distinct",
        "reason": "all-distinct N=G selects legacy whole-pipeline control",
    },
    {
        **maintained.CASES[0],
        "id": "edge-stream-1cpu",
        "fixture": "edge",
        "reason": "first eligible N=2G edge and least canonical-row saving",
    },
)
CONTROL_FIXTURES = {
    "distinct": {
        "copies": 1,
        "groups": 10_000,
        "business_batches": 20,
        "business_sql_rows": 10_000,
        "input_rows": 210_000,
        "selector": "legacy N=G",
    },
    "edge": {
        "copies": 2,
        "groups": 10_000,
        "business_batches": 20,
        "business_sql_rows": 20_000,
        "input_rows": 410_000,
        "selector": "first eligible N=2G",
    },
}

DENSE_TARGETS = ("stock-stream-1cpu", "stock-stream-4cpu")
TOOL_FILES = (
    "scripts/benchmark_suite/sum_columnar_keys.py",
    "scripts/benchmark_suite/sum_inline_state.py",
    "scripts/benchmark_suite/sum_state.py",
    "scripts/benchmark_suite/sum_state_profile.py",
    "scripts/benchmark_suite/sum_state_publication.py",
    "scripts/benchmark_suite/sum_followup.py",
    "scripts/benchmark_suite/sum_batch_parallel.py",
    "scripts/benchmark_suite/report.py",
    "scripts/benchmark_suite/statistics.py",
    "scripts/toolkit.py",
    "scripts/test_sum_columnar_keys.py",
)
PROFILE_ANCHORS = (
    "calc_flow::operator::sql::incremental::columnar_keys::Dictionary::intern",
    "calc_flow::operator::sql::incremental::IncrementalSql::update_columnar_chunk",
    "calc_flow::operator::sql::incremental::IncrementalSql::admit_columnar_keys",
)

WORK_ANCHORS = (PROFILE_ANCHORS[0], PROFILE_ANCHORS[2])


def canonical_symbol(name):
    return re.sub(r"::h[0-9a-f]{16}$", "", name.rsplit(" [", 1)[0])


def anchor_samples(data, anchors):
    matching = {
        index
        for index, name in enumerate(data["frames"])
        if canonical_symbol(name) in anchors
    }
    return publication.inclusive_count(data["stacks"], matching)


def validate_symbol_anchors(builds):
    for side, build in builds.items():
        anchors = build.get("compiled_columnar_anchors")
        if not isinstance(anchors, list) or not set(anchors).issubset(PROFILE_ANCHORS):
            raise ValueError(
                "build requires an exact compiled columnar anchor inventory"
            )
        if side == "baseline" and anchors:
            raise ValueError(
                "current baseline cannot contain candidate columnar anchors"
            )


def route_status(anchors, samples):
    if not anchors:
        return "nonseparable_or_inlined"
    return "observed" if samples else "unresolved_by_sampling"


def profile_route(path, anchors):
    if not anchors:
        return {
            "compiled_anchors": [],
            "status": "nonseparable_or_inlined",
            "inclusive_samples": None,
        }
    data = json.loads(gzip.decompress(path.read_bytes()))
    work = [anchor for anchor in anchors if anchor in WORK_ANCHORS]
    samples = anchor_samples(data, work)
    return {
        "compiled_anchors": anchors,
        "status": route_status(work, samples),
        "inclusive_samples": anchor_samples(data, anchors),
        "active_work_samples": samples if work else None,
        "total_samples": sum(row["samples"] for row in data["stacks"]),
    }


def route_evidence(destination, builds, complete_profiles):
    if not complete_profiles:
        return {"status": "required profiles are not accepted"}
    return {
        side: profile_route(
            destination / (side + "-stacks.json.gz"), build["compiled_columnar_anchors"]
        )
        for side, build in builds.items()
    }


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
        "control_fixtures": CONTROL_FIXTURES,
        "profile_route_anchors": list(PROFILE_ANCHORS),
    }
    maintained.compare_fields(
        protocol, expected, expected, "columnar-key protocol", required=True
    )
    revision = protocol["candidate_revision"]
    if (
        not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or revision == BASELINE
    ):
        raise ValueError("a new public candidate revision is required")


def validate_products(builds, protocol):
    if builds["baseline"]["revision"] != BASELINE:
        raise ValueError("fresh latest22f baseline is required; P74 is incompatible")
    if builds["candidate"]["revision"] != protocol["candidate_revision"]:
        raise ValueError("candidate differs from frozen public source")
    for build in builds.values():
        validate_fresh_build(build)


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
    validate_symbol_anchors(collection.builds)
    for build in collection.builds.values():
        evidence.source_equivalence(build, build["revision"], collection.deadline)
    return followup.Followup(
        collection, protocol, maintained.sha(args.protocol), "columnar-keys", None
    )


def validate_control_source(record, groups, copies):
    if copies not in (1, 2) or groups <= 0 or record.num_rows != groups * 10:
        raise ValueError("control source requires ten complete positive key cycles")
    if any(column.null_count for column in record.columns):
        raise ValueError("common657 fixtures require non-null input columns")


def validate_control_keys(record, groups, copies):
    counts = Counter(
        zip(
            record.column("account").to_pylist(),
            record.column("stock").to_pylist(),
            strict=True,
        )
    )
    if len(counts) != groups or set(counts.values()) != {copies}:
        raise ValueError("control must contain each retained key exactly copies times")


def control_record(record, index, groups, copies):
    validate_control_source(record, groups, copies)
    selected = [
        record.slice(((index + offset) % 10) * groups, groups)
        for offset in range(copies)
    ]
    arrays = [
        pa.concat_arrays([part.column(i) for part in selected])
        for i in range(record.num_columns)
    ]
    result = pa.RecordBatch.from_arrays(arrays, schema=record.schema)
    column = result.schema.get_field_index("flow_type")
    result = result.set_column(
        column, result.schema.field(column), pa.array(["trade"] * result.num_rows)
    )
    validate_control_keys(result, groups, copies)
    return result


def validate_control_manifest(manifest, groups, copies):
    if manifest["batch_rows"] != [groups] + [groups * copies] * 20:
        raise ValueError("control requires seed and twenty complete business batches")
    if manifest["snapshot_rows"] != [groups] * 21:
        raise ValueError("control keys must remain in the retained seed set")


def control_fixture(source, destination, copies, deadline=None):
    if source.resolve() == destination.resolve():
        raise FileExistsError("never overwrite the maintained fixture")
    manifest = {
        "fixture": "distinct" if copies == 1 else "edge",
        "batch_rows": [],
        "snapshot_rows": [],
        "snapshot_sha256": [],
    }
    output = destination.open("xb")
    try:
        with (
            output,
            pa.ipc.open_stream(source) as reader,
            pa.ipc.new_stream(output, reader.schema) as writer,
        ):
            manifest["schema_sha256"] = evidence.schema_hash(reader.schema)
            positions, groups = {}, None
            for index, record in enumerate(reader):
                evidence.ensure_remaining(deadline, "control fixture")
                groups = record.num_rows if index == 0 else groups
                transformed = (
                    record
                    if index == 0
                    else control_record(record, index - 1, groups, copies)
                )
                positions = evidence.append_reference(manifest, positions, transformed)
                writer.write_batch(transformed)
            validate_control_manifest(manifest, groups, copies)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    output_schema = pa.schema(
        [
            pa.field("account", pa.string()),
            pa.field("stock", pa.string()),
            pa.field("position", pa.int64()),
        ]
    )
    return {
        **manifest,
        "input_rows": sum(manifest["batch_rows"]),
        "input_sha256": maintained.sha(destination),
        "snapshot_schema_sha256": evidence.schema_hash(output_schema),
        "source_stock_sha256": maintained.sha(source),
        "mapping": f"seed unchanged; cycles index%10, {copies} cycle(s), all trade",
    }


def prepare_fixtures(collection):
    measurement = maintained.prepare_fixture(collection)
    fixtures = dict(measurement.fixture)
    root = collection.args.output / "fixture"
    for name, spec in CONTROL_FIXTURES.items():
        fixtures[name] = control_fixture(
            root / "stock.arrow",
            root / (name + ".arrow"),
            spec["copies"],
            collection.deadline,
        )
    write_json(collection.args.output / "fixture-manifest.json", fixtures)
    return replace(measurement, fixture=fixtures)


def collect(context, acquisition):
    collection = context.collection
    measurement = prepare_fixtures(collection)
    maintained.collect_profiles(measurement, acquisition)
    validate_profiles(
        followup.accepted_profiles(context, acquisition), collection.builds
    )
    for case in collection.scope["cases"]:
        maintained.collect_case(measurement, case, acquisition)
    if time.monotonic() > collection.deadline:
        raise TimeoutError("columnar-key SUM measurement deadline")
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
            "Five fixed cases use two rounds of ten alternating pairs (200 formal",
            "observations and20warmups) and one",
            "600-second budget including preflight, fixtures, warmups, independent",
            "full-snapshot oracles, capture/decoding, analysis, publication and",
            "cleanup. Builds are separate; no cases, samples or retries are added.",
            "",
            "A dense1/dense4 target must have the exact maintained improved verdict,",
            "with no regression on any case, after complete evidence validation.",
            "Inconclusive is uncertainty, not equivalence or a confirmed speedup.",
            "",
            "The baseline is freshly built public22f4b9a. The older P74 binary",
            "is incompatible with current Batch/DataFusion trusted metadata source.",
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
            "Distinct control retains10k seed keys, adding20x10k all-trade rows,",
            "one occurrence per key per business batch (N=G); the selector remains",
            "legacy. Edge control adds20x20k rows with two occurrences per key",
            "(N=2G), the first eligible edge. Both cross the8192-row chunk boundary",
            "and keep at most10k groups; cycles follow business_index%10, with",
            "the edge also using the next cycle modulo10. These are independent",
            "new fixtures, not old workload digests. Sparse is also a legacy control.",
            "Dense80k-row SQL batches contain10k keys; within-chunk dedup could",
            "save only1.89% canonical encodings versus87.5% across the whole batch.",
            "These counts motivate the hypothesis and are not measured speedups.",
            "Every stream retains21 complete snapshots, independently checked after",
            "the native timer. Whole-process peak RSS includes inputs and retained",
            "outputs; it is not operator-only allocation or an engine memory cap.",
            "Common657 requires non-null inputs. UInt64, nullable/generic aggregates,",
            "checkpoint/finite/below-edge and tight-byte latency are unmeasured.",
            "Focused allocation/size/budget and correctness tests separately cover",
            "semantics and budgets; they do not establish end-to-end latency.",
            "",
            f"Accepted profiles: {summary.get('accepted_profile_count', 0)}.",
            "Both serial profiles require five verified processing windows after",
            "excluding the first of six jobs; all per-thread timestamps must align.",
            "Self/inclusive counts and full stacks cover the row-encoding/hash path.",
            "Inclusive CPU shares overlap and are not phase wall-time measurements.",
            "[Exact route evidence](profile-route.json) intersects sealed nm anchors",
            "with complete canonical symbols in the existing profiles. It is serial",
            "path attribution, not a worker gate. Dispatcher-only samples do not prove",
            "interner work. Missing/inlined/unsampled active-work anchors",
            "remain unresolved, not zero cost or proof that the route did not run.",
            "Same-shape focused path-hit tests remain separate evidence. Unresolved",
            "profile leaves and attribution limits are retained.",
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
    routes = route_evidence(
        context.collection.args.publish_directory,
        context.collection.builds,
        complete_profiles,
    )
    summary["columnar_route_evidence"] = routes
    write_json(context.collection.args.publish_directory / "profile-route.json", routes)
    raw = {
        **raw,
        "profile_route_evidence": routes,
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
        raise TimeoutError("columnar-key SUM publication deadline")
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
