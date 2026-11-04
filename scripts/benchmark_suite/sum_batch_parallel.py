"""Five fixed paired SUM cases, with required SQL-specific multicore evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from scripts.benchmark_suite import sum_followup as followup
from scripts.benchmark_suite import sum_state as maintained
from scripts.benchmark_suite import sum_state_profile as profile
from scripts.benchmark_suite import sum_state_publication as publication
from scripts.toolkit import command_output, require_executable, write_json

SLUG = "sum-batch-parallel-20261104"
MERGED_BASE = "c9afa829ff025280ab346667582c62dc26a34785"
BASELINE = "af400b38511bc7f6255fd3549518095b51b20e0c"
BASELINE_BINARY = "be89d670e89c7142fdf89849ae86240d3df3b86b2bfa508b52979a525f4c96bd"
COMMON_HARNESS = followup.COMMON_HARNESS
SQL_PHASE_FRAMES = (
    (
        "sql_local_update",
        "calc_flow::operator::sql::incremental::grouped_parallel::Work::aggregate_range",
    ),
    (
        "sql_local_update",
        "calc_flow::operator::sql::incremental::grouped_parallel::Work::aggregate_chunk",
    ),
    (
        "sql_local_update",
        "<calc_flow::operator::sql::incremental::grouped_parallel::Work as "
        "calc_flow::runtime::streaming::gather_work::parallel::ParallelCpuWork>::run",
    ),
    (
        "sql_coordinator_merge",
        "calc_flow::operator::sql::incremental::grouped_parallel::merge_fragment",
    ),
    (
        "sql_coordinator_merge",
        "calc_flow::operator::sql::incremental::grouped_parallel::merge_fragment"
        "::{{closure}}",
    ),
    (
        "sql_coordinator_merge",
        "calc_flow::operator::sql::incremental::grouped_parallel::resolve_ranks"
        "::{{closure}}",
    ),
    (
        "sql_coordinator_merge",
        "calc_flow::operator::sql::incremental::grouped_parallel::merge_states"
        "::{{closure}}",
    ),
)
CASES = (
    {**maintained.CASES[0], "reason": "one-CPU serial admission and overhead control"},
    {
        **maintained.CASES[0],
        "id": "stock-stream-4cpu",
        "cpus": [0, 2, 4, 6],
        "reason": "four physical cores and bounded partial reduction",
    },
    {
        **maintained.CASES[1],
        "reason": "eight logical CPUs, four cores, at most five local units",
    },
    {
        **maintained.CASES[2],
        "reason": "preserve sparse single-CPU retained-state performance",
    },
    {
        **maintained.CASES[1],
        "id": "hotkey-stream-8cpu",
        "fixture": "hotkey",
        "reason": "90-percent hot-key skew with retained seed tail",
    },
)


def json_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def schema_hash(schema):
    names = {pa.string(): "Utf8", pa.int64(): "Int64", pa.float64(): "Float64"}
    return json_hash(
        [(field.name, names[field.type], field.nullable) for field in schema]
    )


def reference_batch(previous, record):
    if any(column.null_count for column in record.columns):
        raise ValueError("common657 fixtures require non-null input columns")
    columns = [
        record.column(name).to_pylist()
        for name in ("account", "stock", "quantity", "side", "flow_type")
    ]
    updated = dict(previous)
    for account, stock, quantity, side, flow in zip(*columns, strict=True):
        if flow != "trade":
            continue
        delta = quantity if side == "buy" else -quantity
        value = updated.get((account, stock), 0) + delta
        updated[(account, stock)] = (value + (1 << 63)) % (1 << 64) - (1 << 63)
    return updated


def skew_record(record):
    mask = pa.array([row % 100 < 90 for row in range(record.num_rows)])
    updated = record
    for index, value in ((0, "account-000"), (1, "stock-000")):
        column = pc.if_else(mask, pa.scalar(value), record.column(index))
        updated = updated.set_column(index, record.schema.field(index), column)
    return updated


def append_reference(manifest, positions, record):
    updated = reference_batch(positions, record)
    rows = [
        [account, stock, value] for (account, stock), value in sorted(updated.items())
    ]
    manifest["batch_rows"].append(record.num_rows)
    manifest["snapshot_rows"].append(len(updated))
    manifest["snapshot_sha256"].append(json_hash(rows))
    return updated


def ensure_remaining(deadline, phase):
    if deadline is not None and time.monotonic() > deadline - 5:
        raise TimeoutError(f"{phase} deadline")


def hotkey_fixture(source, destination, deadline=None):
    if source.resolve() == destination.resolve():
        raise FileExistsError("never overwrite the maintained fixture")
    manifest = {
        "fixture": "hotkey",
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
            manifest["schema_sha256"] = schema_hash(reader.schema)
            positions = {}
            for index, record in enumerate(reader):
                ensure_remaining(deadline, "fixture")
                transformed = skew_record(record) if index else record
                positions = append_reference(manifest, positions, transformed)
                writer.write_batch(transformed)
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
        "snapshot_schema_sha256": schema_hash(output_schema),
        "source_stock_sha256": maintained.sha(source),
        "mapping": "business row%100<90 maps to account-000/stock-000; seed unchanged",
    }


def validate_protocol(protocol):
    expected = {
        "slug": SLUG,
        "budget_seconds": 600,
        "rounds": 2,
        "pairs_per_round": 10,
        "warmups_per_arm_per_round": 1,
        "cases": list(CASES),
    }
    maintained.compare_fields(
        protocol, expected, expected, "frozen protocol", required=True
    )
    validate_revision(protocol["candidate_revision"])
    validate_worker_rules(protocol["sql_phase_frames"])


def validate_revision(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or revision in (
        BASELINE,
        MERGED_BASE,
    ):
        raise ValueError("a new public candidate revision is required")


def validate_worker_rules(rules):
    if rules != [list(rule) for rule in SQL_PHASE_FRAMES]:
        raise ValueError("SQL worker rules must match the frozen complete symbols")


def validate_products(builds, protocol):
    base, candidate = builds["baseline"], builds["candidate"]
    if (base["revision"], base["binary_sha256"]) != (BASELINE, BASELINE_BINARY):
        raise ValueError("baseline must retain its actual sealed af400 identity")
    if candidate["revision"] != protocol["candidate_revision"]:
        raise ValueError("candidate differs from frozen public revision")
    if not candidate.get("product_recompiled") or not candidate.get(
        "example_recompiled"
    ):
        raise ValueError("candidate requires fresh product and example compilation")
    if any(build["harness_sha256"] != COMMON_HARNESS for build in builds.values()):
        raise ValueError("all arms require the exact common657 harness")


def source_equivalence(build, revision, deadline):
    ensure_remaining(deadline, "source preflight")
    raw = command_output(
        [
            require_executable("git"),
            "ls-tree",
            "-r",
            "--format=%(objectname) %(path)",
            revision,
        ],
        cwd=maintained.ROOT,
        timeout=max(1, deadline - time.monotonic()),
    )
    objects = dict(line.split(maxsplit=1)[::-1] for line in raw.splitlines())
    for name in build["source_sha256"]:
        if name == maintained.HARNESS:
            continue
        data = (Path(build["source_root"]) / name).read_bytes()
        identity = hashlib.sha1(
            f"blob {len(data)}\0".encode() + data, usedforsecurity=False
        ).hexdigest()
        if objects.get(name) != identity:
            raise ValueError(
                f"compiled source differs from public revision {revision}: {name}"
            )


def baseline_equivalence(build, deadline):
    source_equivalence(build, MERGED_BASE, deadline)


def admitted_symbols(names):
    observed = {
        label: [
            needle
            for tag, needle in SQL_PHASE_FRAMES
            if tag == label and needle in names
        ]
        for label in ("sql_local_update", "sql_coordinator_merge")
    }
    if not observed["sql_local_update"]:
        raise ValueError("candidate has no compiled exact SQL worker symbol")
    return observed


def compiled_symbols(build, args, deadline):
    executable = require_executable("nm")
    command = [executable, "-n", "-S", "-C", "--defined-only", build["binary"]]
    raw = profile.command(command, deadline, maintained.environment(1))
    path = args.output / "candidate-symbols.txt"
    path.write_text(raw)
    names = {
        re.sub(r"::h[0-9a-f]{16}$", "", name) for _, _, name in profile._symbols(raw)
    }
    return {
        "observed": admitted_symbols(names),
        "command": command,
        "path": str(path),
        "sha256": maintained.sha(path),
        "tool": maintained.tool_identity(
            Path(executable), maintained.environment(1), deadline
        ),
    }


def prepare_context(args, started):
    protocol = publication.read(args.protocol)
    validate_protocol(protocol)
    options = argparse.Namespace(
        **{
            **vars(args),
            "budget_seconds": 600,
            "prior_measurement_seconds": 0,
            "history_borrow": False,
            "sql_phase_frames": protocol["sql_phase_frames"],
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
    baseline_equivalence(collection.builds["baseline"], collection.deadline)
    source_equivalence(
        collection.builds["candidate"],
        protocol["candidate_revision"],
        collection.deadline,
    )
    symbols = compiled_symbols(
        collection.builds["candidate"], options, collection.deadline
    )
    collection = replace(
        collection, scope={**scope, "candidate_symbol_admission": symbols}
    )
    return followup.Followup(
        collection, protocol, maintained.sha(args.protocol), "batch-parallel", None
    )


def prepare_fixture(context):
    measurement = maintained.prepare_fixture(context.collection)
    fixture = dict(measurement.fixture)
    if fixture["stock"]["batch_rows"] != [10000] + [100000] * 20:
        raise ValueError("dense fixture schedule changed")
    directory = context.collection.args.output / "fixture"
    fixture["hotkey"] = hotkey_fixture(
        directory / "stock.arrow",
        directory / "hotkey.arrow",
        context.collection.deadline,
    )
    if fixture["hotkey"]["snapshot_rows"] != [10000] * 21:
        raise ValueError("hotkey fixture lost its retained group tail")
    write_json(context.collection.args.output / "fixture-manifest.json", fixture)
    return maintained.MeasurementContext(context.collection, fixture)


def collect_profiles(measurement, acquisition):
    collection = measurement.collection
    for side in collection.builds:
        path = collection.args.output / "profiles" / side
        rows, receipt = maintained.measure(
            measurement, CASES[2], side, path, profile=True
        )
        item = maintained.profile_observation(
            side, path.relative_to(collection.args.output), rows, receipt
        )
        acquisition["profiles"].append(
            {**item, "perf_data_sha256": maintained.sha(path / "perf.data")}
        )
        maintained.save_acquisition(collection, acquisition)
    profile.decode(collection.args, acquisition, collection.builds, collection.deadline)


def candidate_profile(profiles):
    if len(profiles) != 2 or {row["side"] for row in profiles} != {
        "baseline",
        "candidate",
    }:
        raise ValueError("required SQL worker profiles are missing or invalid")
    return next(row for row in profiles if row["side"] == "candidate")


def window_workers(distribution, index):
    return {
        str(row["tid"]).split(".")[0]
        for row in distribution
        if row["window"] == index and row["samples"] > 0
    }


def validate_worker_profiles(profiles):
    candidate = candidate_profile(profiles)
    windows = candidate.get("windows", [])
    distribution = candidate.get("sql_worker_samples_by_window", [])
    workers = [window_workers(distribution, index) for index in range(1, 6)]
    if len(windows) != 5 or any(len(tids) < 2 for tids in workers):
        raise ValueError(
            "SQL worker distribution must cover multiple threads in every window"
        )


def worker_evidence_accepted(profiles):
    try:
        validate_worker_profiles(profiles)
    except ValueError:
        return False
    return True


def collect(context, acquisition):
    measurement = prepare_fixture(context)
    collect_profiles(measurement, acquisition)
    validate_worker_profiles(followup.accepted_profiles(context, acquisition))
    for case in context.collection.scope["cases"]:
        maintained.collect_case(measurement, case, acquisition)
    if time.monotonic() > context.collection.deadline:
        raise TimeoutError("parallel SUM measurement deadline")
    acquisition["status"] = "completed"


def performance_acceptance(cases):
    if [case["id"] for case in cases] != [case["id"] for case in CASES]:
        return False
    verdicts = {case["id"]: case.get("statistics", {}).get("verdict") for case in cases}
    targets = ("stock-stream-4cpu", "stock-stream-8cpu")
    return (
        all(verdicts.values())
        and "regression" not in verdicts.values()
        and any(verdicts[target] == "improved" for target in targets)
    )


def accepted_performance(status, cases):
    return status == "completed" and performance_acceptance(cases)


def report(summary):
    rows, intervals, resources = publication.rendered_rows(summary)
    acceptance = accepted_performance(summary["status"], summary["cases"])
    worker_verified = summary.get("sql_worker_evidence_verified", False)
    return "\n".join(
        [
            f"# {SLUG}",
            "",
            f"Collection status: **{summary['status']}**.",
            f"Performance acceptance: **{acceptance}**.",
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
            "Five fixed cases, two rounds of ten alternating pairs, and one 600-second",
            "budget include fixture, preflight, warmup, full 21-snapshot oracles,",
            "profiles,",
            "decode, analysis, publication and cleanup. Builds are recorded",
            "separately.",
            "The actual reused baseline product is af400, byte-compatible with",
            "merged c9afa;",
            "it is never relabeled as a fresh c9afa build. All arms use audited",
            "common657",
            "instrumentation from public841 only in permanently frozen native",
            "source trees.",
            "",
            "All five cases must finish, no selected case may regress, and dense4",
            "or dense8",
            "must be improved under the maintained paired-median/5% rule for",
            "acceptance.",
            "Inconclusive does not establish equivalence or speedup. No retries",
            "are allowed.",
            "All streams retain one DataFusion partition; the CPU work allowance",
            "is separate.",
            "Eight logical CPUs represent four reported cores on WSL2; host load",
            "is unknown.",
            "100k source rows filter to 80k SQL rows: ten legacy SQL chunks,",
            "admitting at most five local units with eight CPUs.",
            "The seed and sparse batches remain below parallel admission. Hot-key",
            "input maps",
            "90/100 business rows to one existing key while preserving the 10k",
            "seed tail.",
            "Common657 requires non-null input; nullable semantics are covered by",
            "Rust tests.",
            "",
            f"Accepted profiles: {len(summary.get('profiles', []))}.",
            f"SQL worker evidence verified: {worker_verified}.",
            "The protocol requires both 8CPU profiles to contain five verified",
            "processing windows, excluding",
            "the first of six jobs. SQL worker samples are attributed by exact",
            "frozen frame",
            "rules and per-window TID. These samples describe CPU shares, not",
            "phase wall",
            "time or proof of elapsed speedup; unknown/inlined frames remain",
            "disclosed.",
            "",
            "[summary.json](summary.json) and [samples.csv](samples.csv) retain",
            "all timing",
            "intervals, CPU/RSS, fixture hashes and accepted profiles. Every",
            "warmup/formal/",
            "profile observation, source/build seal, command, snapshot/schema",
            "digest and",
            "Resource receipts: [complete evidence](complete-evidence.json.gz).",
            "[Profile summary](profile-summary.json) and",
            "[targets](profile-targets.json) retain",
            "CPU/thread attribution; [manifest](evidence-manifest.json) seals",
            "budget/status.",
            "",
            f"Stopped reason: {summary['failure']}",
            "",
        ]
    )


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


def write_receipt(data, now):
    value = {
        **followup.receipt(data, now),
        "slug": SLUG,
        "current_profile_analysis_complete": data.summary.get(
            "sql_worker_evidence_verified", False
        ),
        "performance_acceptance": accepted_performance(
            data.acquisition["status"], data.summary["cases"]
        ),
    }
    followup.write_receipt(data.context, value)
    return value


def clock_evidence(context, acquisition):
    output = context.collection.args.output
    records = {}
    for item in acquisition["profiles"]:
        for name in ("sample-timestamps.txt", "clock-alignment.json", "perf.data"):
            path = output / item["path"] / name
            if not path.is_file():
                continue
            record = {
                "path": str(path.resolve()),
                "sha256": maintained.sha(path),
                "bytes": path.stat().st_size,
            }
            if name != "perf.data":
                record["text"] = path.read_text()
            records[str(path.relative_to(output))] = record
    return records


def publish(context, acquisition, plan, machine):
    publication.validate_inventory(acquisition, [case["id"] for case in CASES])
    summary, raw = followup.publication_data(context, acquisition, plan, machine)
    disposition = acquisition.get(
        "profile_skip_reason", "incomplete or invalid required profiles"
    )
    summary = {
        **summary,
        "profile_disposition": "accepted"
        if len(summary["profiles"]) == 2
        else disposition,
        "sql_worker_evidence_verified": worker_evidence_accepted(summary["profiles"]),
    }
    if (
        acquisition["status"] == "completed"
        and not summary["sql_worker_evidence_verified"]
    ):
        acquisition["status"] = "stopped"
        acquisition["failure"] = "required SQL worker evidence invalid at publication"
    raw = {**raw, "profile_clock_evidence": clock_evidence(context, acquisition)}
    data = followup.PublishedFollowup(context, acquisition, summary, raw)
    write_metadata(data, time.monotonic())
    value = write_receipt(data, time.monotonic())
    final = time.monotonic()
    if final > context.collection.deadline:
        write_metadata(data, final)
        write_receipt(data, time.monotonic())
        raise TimeoutError("parallel SUM publication deadline")
    print(json.dumps(value), flush=True)


def run(args):
    started = time.monotonic()
    context = prepare_context(args, started)
    plan, machine = followup.plan_and_machine(context)
    plan["followup"] = {**plan["followup"], "slug": SLUG}
    plan["candidate_symbol_admission"] = context.collection.scope.get(
        "candidate_symbol_admission"
    )
    plan["profiles"] = (
        "required baseline/candidate stock-stream-8cpu, six jobs each, first excluded"
    )
    machine["helper_sha256"]["scripts/benchmark_suite/sum_batch_parallel.py"] = (
        maintained.sha(__file__)
    )
    machine["fixture_pyarrow_version"] = pa.__version__
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
