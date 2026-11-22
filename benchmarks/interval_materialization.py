"""Time the three interval output columns given already matched row positions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from benchmarks.interval_acceptance import validate_interval_output
from scripts.toolkit import linear_percentile, worker_environment, write_json


def prepare_gathers(
    source: pa.Table, pairs: pa.Table, *, ingress_rows: int = 4096
) -> tuple[tuple[pa.Array, pa.Array], list[tuple[pa.Array, pa.Array]]]:
    """Group known pairs by right-side ingress morsel outside the measured clock."""
    sequence = source["sequence"].combine_chunks()
    price = source["price"].combine_chunks()
    left = pc.index_in(pairs["sequence"], value_set=sequence).combine_chunks()
    right = pc.index_in(pairs["right_sequence"], value_set=sequence).combine_chunks()
    if left.null_count or right.null_count:
        raise ValueError("oracle pair references an unknown source identity")
    order = pa.array(np.lexsort((left.to_numpy(), right.to_numpy())))
    left, right = left.take(order), right.take(order)
    boundaries = np.searchsorted(
        right.to_numpy(), np.arange(0, source.num_rows, ingress_rows)
    ).tolist()
    groups = [
        (left.slice(start, end - start), right.slice(start, end - start))
        for start, end in zip(boundaries, boundaries[1:] + [len(right)])
        if end > start
    ]
    return (sequence, price), groups


def gather_projection(
    columns: tuple[pa.Array, pa.Array], groups: list[tuple[pa.Array, pa.Array]]
) -> pa.Table:
    """Gather both sequence columns and multiply left/right prices in Arrow."""
    sequence, price = columns
    return pa.Table.from_batches(
        [
            pa.record_batch(
                [
                    sequence.take(left),
                    sequence.take(right),
                    pc.multiply(price.take(left), price.take(right)),
                ],
                names=["sequence", "right_sequence", "value"],
            )
            for left, right in groups
        ]
    )


def measure(rows: int) -> dict:
    from benchmarks.engine_comparison import expected_output, workload

    data = workload(rows, scenario="interval_join")
    expected = expected_output(data, "interval_join")
    columns, groups = prepare_gathers(data.table, expected)
    environment = worker_environment()
    samples = []
    for _ in range(12):
        start = time.perf_counter_ns()
        output = gather_projection(columns, groups)
        elapsed = time.perf_counter_ns() - start
        proof = validate_interval_output(output, expected, rows)
        samples.append({"elapsed_ns": elapsed, "proof": proof})
        del output
    elapsed_ns = [sample["elapsed_ns"] for sample in samples[2:]]
    return {
        "p50_ns": linear_percentile(elapsed_ns, 0.50),
        "p95_ns": linear_percentile(elapsed_ns, 0.95),
        "case": "interval_join_precomputed_positions_pyarrow",
        "rows_per_side": rows,
        "output_batches": len(groups),
        "environment": environment,
        "warmups": samples[:2],
        "samples": samples[2:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, measure(args.rows), exclusive=True)


if __name__ == "__main__":
    main()
