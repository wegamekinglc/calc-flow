"""Measure the public async changelog stream on the grouped SUM sparse fixture."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from time import perf_counter_ns

import pyarrow as pa

import calc_flow as cf
from benchmarks.native_release_identity import file_hash, release_identity
from scripts.toolkit import worker_environment, write_json

FIXTURE_SHA256 = "641effd7042bd3d0cd1be8f3fef6e6b5d7d726b47c71eaf761a6ec76f10ff3e0"
OUTPUT = pa.schema(
    [
        ("account", pa.string()),
        ("stock", pa.string()),
        ("position", pa.int64()),
        pa.field("_cf_op", pa.string(), nullable=False),
    ]
)


def load_fixture(path: Path) -> tuple[pa.Table, ...]:
    if file_hash(path) != FIXTURE_SHA256:
        raise ValueError("expected grouped_sum_state_benchmark sparse.arrow")
    with pa.memory_map(str(path), "r") as source:
        batches = tuple(
            pa.Table.from_batches([batch]) for batch in pa.ipc.open_stream(source)
        )
    if [batch.num_rows for batch in batches] != [10_000, *([100] * 20)]:
        raise ValueError("unexpected sparse fixture batch boundaries")
    return batches


async def observe(batches: tuple[pa.Table, ...]) -> tuple[int, list[pa.Table]]:
    async def source():
        for batch in batches:
            yield batch

    started = perf_counter_ns()
    flows = cf.table_input("trades", schema=batches[0].schema)
    trades = flows.filter(flows["flow_type"] == "trade")
    deltas = trades.select(
        "account",
        "stock",
        delta=cf.row.where(
            trades["side"] == "buy", trades["quantity"], -trades["quantity"]
        ),
    )
    positions = deltas.changelog_sql(
        "SELECT account, stock, SUM(delta) AS position "
        "FROM input GROUP BY account, stock"
    )
    config = cf.StreamRuntimeConfig(
        checkpointing=False, sql_compact_state_budget=cf.StateBudget(10_000, 2**53 - 1)
    )
    async with positions.stream(source(), config=config) as stream:
        outputs = [table async for table in stream]
    return perf_counter_ns() - started, outputs


def _positions_after(
    previous: dict[tuple[str, str], int], batch: pa.Table
) -> dict[tuple[str, str], int]:
    positions = previous.copy()
    columns = ["account", "stock", "side", "quantity", "flow_type"]
    for account, stock, side, quantity, kind in zip(
        *batch.select(columns).to_pydict().values()
    ):
        if kind == "trade":
            key = (account, stock)
            positions[key] = positions.get(key, 0) + (
                quantity if side == "buy" else -quantity
            )
    return positions


def _ordered_changes(previous, positions):
    for key, value in positions.items():
        if key not in previous or value != previous[key]:
            if key in previous:
                yield (*key, previous[key], "retract")
            yield (*key, value, "upsert")


def _checked_changes(table: pa.Table, expected, index: int):
    if not table.schema.equals(OUTPUT, check_metadata=True):
        raise ValueError(f"unexpected output schema at batch {index}: {table.schema}")
    changes = list(zip(*table.to_pydict().values()))
    if changes != list(expected):
        raise ValueError(f"incorrect ordered changelog at batch {index}")
    return changes


def validate(batches: tuple[pa.Table, ...], outputs: list[pa.Table]) -> list[int]:
    if len(batches) != len(outputs):
        raise ValueError(f"expected {len(batches)} outputs, got {len(outputs)}")
    positions = {}
    reconstructed = {}
    for index, (batch, table) in enumerate(zip(batches, outputs)):
        previous = positions
        positions = _positions_after(previous, batch)
        changes = _checked_changes(table, _ordered_changes(previous, positions), index)
        for account, stock, value, op in changes:
            key = (account, stock)
            if op == "retract":
                del reconstructed[key]
            else:
                reconstructed[key] = value
        if reconstructed != positions:
            raise ValueError(f"incorrect reconstructed positions at batch {index}")
    return [table.num_rows for table in outputs]


async def collect(batches: tuple[pa.Table, ...], metadata: dict, output: Path) -> None:
    report = {**metadata, "observations": []}
    write_json(output, report, exclusive=True)
    for round_number in (1, 2):
        for sample in range(11):
            elapsed, outputs = await observe(batches)
            rows = validate(batches, outputs)
            report["observations"].append(
                {
                    "round": round_number,
                    "sample": sample,
                    "warmup": sample == 0,
                    "elapsed_ns": elapsed,
                    "verified_output_rows": rows,
                }
            )
            write_json(output, report)
            del outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "contract": "calc-flow.public-sparse-changelog.v1",
        "build": release_identity(args.build),
        "environment": worker_environment(),
        "fixture_sha256": FIXTURE_SHA256,
        "harness_sha256": file_hash(Path(__file__)),
    }
    batches = load_fixture(args.fixture)
    asyncio.run(collect(batches, report, args.output))


if __name__ == "__main__":
    main()
