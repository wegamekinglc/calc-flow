"""Match trades to recent quotes after both streams finalize their event time.

Tolerance bounds quote age, not wall-clock waiting. Explicit watermarks permit
out-of-order input; idle does not finalize ASOF. These iterable sources have no
replay and convenience-stream checkpoints are temporary. Durable recovery and
external exactly-once delivery require capable sources/sinks and a stable
managed checkpoint root.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pyarrow as pa

import calc_flow as cf

SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("sequence", pa.uint64(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
    ]
)


def source(name: str) -> cf.TableExpr:
    return cf.table_input(
        name,
        schema=SCHEMA,
        entity_by=["symbol"],
        event_time="time",
        sequence_by=["sequence"],
    )


def rows(times: list[int], prices: list[float]) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "symbol": ["AAA"] * len(times),
            "time": times,
            "sequence": list(range(len(times))),
            "price": prices,
        },
        schema=SCHEMA,
    )


def watermark(micros: int) -> cf.Watermark:
    return cf.Watermark(
        datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=micros)
    )


async def events(batch: pa.Table, advance: asyncio.Event):
    yield batch
    yield watermark(105)
    await advance.wait()
    yield watermark(122)


async def main() -> None:
    matched = source("trades").stream_asof_join(
        source("quotes"),
        tolerance=timedelta(microseconds=10),
        limits=cf.AsofStateLimits(100_000, 64 * 1024 * 1024),
        prefixes=("trade", "quote"),
        late_policy="error",
    )
    advance = asyncio.Event()
    inputs = {
        "trades": events(rows([105, 121], [10.4, 10.6]), advance),
        "quotes": events(rows([90, 100, 110], [10.0, 10.2, 10.5]), advance),
    }
    policies = {name: cf.SourceProvidedWatermarks() for name in inputs}
    async with (
        asyncio.timeout(10),
        matched.stream(inputs, watermarks=policies) as result,
    ):
        pending = asyncio.create_task(anext(result))
        while True:
            statuses = result.job.status()["stream_asof_joins"]
            status = next(iter(statuses.values()), None)
            if status is not None and all(
                status[side]["watermark_micros"] == 105 for side in ("left", "right")
            ):
                break
            await asyncio.sleep(0.001)
        if pending.done():
            raise RuntimeError("Equal watermarks cannot finalize the trade at 105")
        print("Both watermarks equal 105: no final output yet.")
        advance.set()
        batches = [await pending, *[batch async for batch in result]]
    output = pa.concat_tables(batches)
    if output["quote__price"].to_pylist() != [10.2, None]:
        raise RuntimeError("Expected the recent quote and an unmatched trade")
    print(output.select(["trade__time", "trade__price", "quote__price"]).to_pydict())


if __name__ == "__main__":
    asyncio.run(main())
