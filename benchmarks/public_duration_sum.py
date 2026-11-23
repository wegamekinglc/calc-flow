"""Public single-duration SUM stream and an independent timestamp-prefix oracle."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pyarrow as pa

import calc_flow as cf
from benchmarks.warm_stream import BASE_MICROS, SCHEMA

WINDOWS = (60,)
INPUT = SCHEMA.set(3, pa.field("price", pa.float64(), nullable=True))
OUTPUT = pa.schema(
    [
        INPUT.field("sequence"),
        INPUT.field("symbol"),
        *((f"sum_{s}", pa.float64()) for s in WINDOWS),
    ]
)


def fixture(path: Path):
    i = np.arange(100_000, dtype=np.int64)
    entity = i % 1_000
    prices = ((i * 17) % 193 - 96).astype(np.float64) / 4
    price = pa.array(prices, mask=((i % 43 == 0) | (entity == 999)))
    symbols = pa.array([f"E{n:04d}" for n in range(1_000)]).take(pa.array(entity))
    table = pa.Table.from_arrays(
        [
            pa.array(BASE_MICROS + i * 1_000, type=INPUT.field(0).type),
            pa.array(i, type=pa.uint64()),
            symbols,
            price,
        ],
        schema=INPUT,
    )
    with path.open("xb") as out, pa.ipc.new_stream(out, INPUT) as writer:
        for batch in table.to_batches(max_chunksize=1000):
            writer.write_batch(batch)


def load(path: Path):
    with pa.memory_map(str(path), "r") as src:
        batches = tuple(pa.Table.from_batches([b]) for b in pa.ipc.open_stream(src))
    if len(batches) != 100 or any(
        b.num_rows != 1000 or not b.schema.equals(INPUT, check_metadata=True)
        for b in batches
    ):
        raise ValueError("incorrect input boundaries or schema")
    return batches


def declaration():
    ticks = cf.table_input(
        "ticks",
        schema=INPUT,
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    return ticks.select(
        "sequence",
        "symbol",
        **{
            f"sum_{n}": cf.ts.sum(ticks["price"], window=cf.duration(n * 1_000_000))
            for n in WINDOWS
        },
    )


def plan(expr):
    nodes = (
        cf.Program("duration-reference", mode="stream", outputs={"result": expr})
        .to_project()
        .root["graph"]["nodes"]
    )
    specs = [
        node["operator"]["spec"]
        for node in nodes
        if node["operator"]["kind"] == "rolling"
    ]
    if len(specs) != 1:
        raise ValueError("requires one rolling operator")
    spec = specs[0]
    if (
        spec["partition_by"] != ["symbol"]
        or spec["event_time"] != "event_time"
        or spec["sequence_by"] != ["sequence"]
        or [
            (o["kind"], o["input"], o["frame"], o["min_periods"], o["output"])
            for o in spec["outputs"]
        ]
        != [
            (
                "sum",
                "price",
                {"kind": "duration", "micros": s * 1_000_000},
                1,
                f"sum_{s}",
            )
            for s in WINDOWS
        ]
    ):
        raise ValueError("rolling-stage input, frame, identity or output differs")
    return spec


def expected(table):
    times = table["event_time"].cast(pa.int64()).to_numpy()
    prices = table["price"].fill_null(0).to_numpy()
    valid = table["price"].is_valid().to_numpy()
    _, codes = np.unique(table["symbol"].to_numpy(), return_inverse=True)
    order = np.argsort(codes, kind="stable")
    boundaries = np.flatnonzero(np.diff(codes[order])) + 1
    sums = {s: np.empty(len(times), np.float64) for s in WINDOWS}
    nulls = {s: np.empty(len(times), np.bool_) for s in WINDOWS}
    for indices in np.split(order, boundaries):
        stamps = times[indices]
        if np.any(stamps[1:] < stamps[:-1]):
            raise ValueError("oracle requires in-order events per entity")
        total = np.r_[0.0, np.cumsum(prices[indices])]
        counts = np.r_[0, np.cumsum(valid[indices], dtype=np.int64)]
        for seconds in WINDOWS:
            lower = np.searchsorted(stamps, stamps - seconds * 1_000_000, side="right")
            sums[seconds][indices] = total[1:] - total[lower]
            nulls[seconds][indices] = counts[1:] == counts[lower]
    return pa.Table.from_arrays(
        [
            table["sequence"],
            table["symbol"],
            *(pa.array(sums[s], mask=nulls[s]) for s in WINDOWS),
        ],
        schema=OUTPUT,
    )


async def observe(batches):
    async def source():
        for batch in batches:
            yield batch

    start = perf_counter_ns()
    expr = declaration()
    async with expr.stream(
        source(), config=cf.StreamRuntimeConfig(checkpointing=False)
    ) as stream:
        outputs = [x async for x in stream]
    return perf_counter_ns() - start, outputs


def validate(outputs, wanted):
    result = pa.concat_tables(outputs)
    if not result.equals(wanted, check_metadata=True):
        raise ValueError(
            f"public rolling output differs ({result.num_rows} rows, {result.schema})"
        )
    return [result.num_rows, *[result[f"sum_{s}"].null_count for s in WINDOWS]]


class DurationCase:
    """One isolated arm with a fixed fixture and an independently checked plan."""

    def __init__(self, case, _root):
        import asyncio

        self.batches = load(Path(case["fixture"]))
        self.expected = expected(pa.concat_tables(self.batches))
        self.plan = plan(declaration())
        self.loop = asyncio.new_event_loop()

    def sample(self):
        elapsed, outputs = self.loop.run_until_complete(observe(self.batches))
        counts = validate(outputs, self.expected)
        return {
            "seconds": elapsed / 1e9,
            "verified_row_count_and_nulls": counts,
            "correctness": {"rows": counts[0], "passed": True},
        }

    def finish(self):
        return {"state": "completed", "rolling_stage": self.plan}

    def close(self):
        self.loop.close()


if __name__ == "__main__":
    import sys

    fixture(Path(sys.argv[1]))
