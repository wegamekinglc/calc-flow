"""The two preregistered P1b stock-position fusion cases, using SUM collectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import pyarrow as pa

from examples.stock_account_data import TRADE_SCHEMA, demo_batches
from scripts.benchmark_suite import sum_state
from scripts.toolkit import write_json

BUSINESS_ROWS = 256_000
ACCOUNTS = 100
STOCKS = 100
CASES = (
    {
        "id": "stock-stream-4cpu-batch1000",
        "fixture": "stock-batch1000",
        "mode": "stream",
        "cpus": [0, 2, 4, 6],
        "partitions": 4,
        "batch_size": 1_000,
        "reason": "Expression CASE/filter to compact SUM with frequent full snapshots",
    },
    {
        "id": "stock-stream-4cpu-batch64000",
        "fixture": "stock-batch64000",
        "mode": "stream",
        "cpus": [0, 2, 4, 6],
        "partitions": 4,
        "batch_size": 64_000,
        "reason": "larger batches expose lost pipeline overlap on the same graph",
    },
)


def digest(value):
    data = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def schema_digest(schema):
    names = {pa.string(): "Utf8", pa.int64(): "Int64", pa.float64(): "Float64"}
    return digest([(field.name, names[field.type], field.nullable) for field in schema])


def write_fixture(path, *, business_rows, accounts, stocks, batch_size):
    """Write real example batches and independently hash each cumulative table."""
    positions = {}
    snapshot_rows, snapshots, batch_rows = [], [], []
    source = pa.concat_tables(
        list(
            demo_batches(
                row_count=business_rows,
                account_count=accounts,
                stock_count=stocks,
                batch_size=batch_size,
            )
        )
    ).combine_chunks()
    seeds = accounts * stocks
    # The example's template period must not create extra IPC batch boundaries.
    parts = (source.slice(0, seeds), source.slice(seeds))
    with pa.ipc.new_stream(path, TRADE_SCHEMA) as writer:
        for record in (
            batch
            for part in parts
            for batch in part.to_batches(max_chunksize=batch_size)
        ):
            batch = pa.Table.from_batches([record])
            writer.write_table(batch)
            batch_rows.append(batch.num_rows)
            for row in batch.to_pylist():
                if row["flow_type"] == "trade":
                    key = (row["account"], row["stock"])
                    quantity = row["quantity"]
                    delta = quantity if row["side"] == "buy" else -quantity
                    positions[key] = positions.get(key, 0) + delta
            snapshots.append(
                digest([(a, s, value) for (a, s), value in sorted(positions.items())])
            )
            snapshot_rows.append(len(positions))
    output_schema = pa.schema(
        [("account", pa.string()), ("stock", pa.string()), ("position", pa.int64())]
    )
    return {
        "fixture": path.stem,
        "input_sha256": sum_state.sha(path),
        "schema_sha256": schema_digest(TRADE_SCHEMA),
        "input_rows": sum(batch_rows),
        "batch_rows": batch_rows,
        "snapshot_rows": snapshot_rows,
        "snapshot_sha256": snapshots,
        "snapshot_schema_sha256": schema_digest(output_schema),
    }


def prepare_fixture(context):
    sum_state.guard(context.builds)
    directory = context.args.output / "fixture"
    directory.mkdir()
    fixtures = {
        case["fixture"]: write_fixture(
            directory / (case["fixture"] + ".arrow"),
            business_rows=BUSINESS_ROWS,
            accounts=ACCOUNTS,
            stocks=STOCKS,
            batch_size=case["batch_size"],
        )
        for case in CASES
    }
    write_json(context.args.output / "fixture-manifest.json", fixtures)
    return sum_state.MeasurementContext(context, fixtures)


def context(args, started):
    sum_state.validate_budget(args.budget_seconds, 0)
    sum_state.create_output(args)
    builds = {
        side: sum_state.checked_build(getattr(args, side + "_build"))
        for side in ("baseline", "candidate")
    }
    sum_state.compare_builds(builds)
    os.sched_setaffinity(0, {31})
    scope = {
        "cases": CASES,
        "profile_sides": (),
        "profile_min_remaining_seconds": 0,
        "unmeasured_final_source": [],
    }
    return sum_state.CollectionContext(
        args, builds, scope, started, started + args.budget_seconds
    )


def run(args):
    started = time.monotonic()
    collection = context(args, started)
    plan = sum_state.collection_plan(collection)
    machine = sum_state.machine_identity(collection)
    machine["focused_adapter_sha256"] = sum_state.sha(__file__)
    machine["fixture_source_sha256"] = sum_state.sha(
        sum_state.ROOT / "examples/stock_account_data.py"
    )
    machine["helper_sha256"].update(
        {
            "scripts/benchmark_suite/fusion_sum.py": machine["focused_adapter_sha256"],
            "examples/stock_account_data.py": machine["fixture_source_sha256"],
        }
    )
    machine["pyarrow_version"] = pa.__version__
    machine["host_limits"] = "WSL host load and power remain unobservable"
    plan["fixture_selection"] = {
        "business_rows": BUSINESS_ROWS,
        "accounts": ACCOUNTS,
        "stocks": STOCKS,
        "seed_rows": ACCOUNTS * STOCKS,
        "batches": "separate seed/business sections, exact case batch-size cap",
    }
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "environment.json", machine)
    acquisition = sum_state.acquisition_state(collection)
    sum_state.save_acquisition(collection, acquisition)
    try:
        measurement = prepare_fixture(collection)
        for case in CASES:
            sum_state.collect_case(measurement, case, acquisition)
        if time.monotonic() > collection.deadline:
            raise TimeoutError("inclusive acquisition deadline after cleanup")
        acquisition["status"] = "completed"
    except BaseException as error:
        acquisition["status"] = "stopped"
        acquisition["failure"] = repr(error)
        raise
    finally:
        if any(
            sum_state.sha(sum_state.ROOT / name) != digest
            for name, digest in machine["helper_sha256"].items()
        ):
            acquisition["status"] = "stopped"
            acquisition["failure"] = "measurement helper changed during acquisition"
        sum_state.finalize_acquisition(
            args, acquisition, collection.builds, started, collection.deadline
        )
    if acquisition["status"] != "completed":
        raise RuntimeError(f"measurement stopped: {acquisition['failure']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-build", required=True, type=Path)
    parser.add_argument("--candidate-build", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=lambda path: Path(path).resolve()
    )
    parser.add_argument("--budget-seconds", type=float, default=600)
    parser.set_defaults(
        history_borrow=False,
        prior_measurement_seconds=0,
        publish_directory=None,
        perf=None,
        perf_library=None,
        samply=None,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
