from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import calc_flow as cf
from calc_flow import _native

case_dir = Path(__file__).resolve().parent
assert Path(_native.__file__).resolve().parent == case_dir / "runtime/calc_flow"
spec = pa.schema([
    pa.field("key", pa.int64(), False),
    pa.field("time", pa.timestamp("us", tz="UTC"), False),
    pa.field("seq", pa.int64(), False),
    pa.field("value", pa.int64(), False),
])


quote_spec = spec.set(0, pa.field("key", pa.int64(), True))


def data(rows, schema=spec):
    return pa.table([list(col) for col in zip(*rows)], schema=schema)


def watermark(micros):
    return cf.Watermark(datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=micros))


async def feed(queue):
    while (item := await queue.get()) is not None:
        yield item


def query():
    facts, quotes, rates = (cf.table_input(name, schema=quote_spec if name == "quotes" else spec, event_time="time", entity_by=["key"], sequence_by=["seq"]) for name in ("facts", "quotes", "rates"))
    joined = cf.table.stream_join(
        facts, quotes, left_keys=["key"], right_keys=["key"],
        left_event_time="time", right_event_time="time",
        bounds=cf.JoinTimeBounds(timedelta(microseconds=2), timedelta()),
        limits=cf.JoinStateLimits(64, 16 << 20, 4096),
        output_entity_by=["left__key"], output_event_time="left__time",
        output_sequence_by=["left__seq", "right__seq"],
    )
    matched = joined.stream_asof_join(rates, tolerance=timedelta(microseconds=2), limits=cf.AsofStateLimits(64, 16 << 20))
    return matched.changelog_sql("SELECT left__left__key AS k, SUM(right__value) AS total FROM input GROUP BY left__left__key")


async def at(stream, label, predicate):
    async with asyncio.timeout(15):
        while True:
            status = stream.job.status()
            assert status["state"] in ("running", "draining"), (label, status)
            (join,) = status["stream_joins"].values()
            (asof,) = status["stream_asof_joins"].values()
            if predicate(join, asof):
                return join, asof
            await asyncio.sleep(0.005)


async def run(mixed):
    queues = {name: asyncio.Queue() for name in ("facts","quotes","rates")}
    output = query()
    current = {}
    trace = []
    config = cf.StreamRuntimeConfig(checkpointing=False)
    async with output.stream({k:feed(q) for k,q in queues.items()}, watermarks={k:cf.SourceProvidedWatermarks() for k in queues}, config=config) as stream:
        queues["facts"].put_nowait(data([(1,10,1,10),(2,12,2,20),(1,20,3,30)]))
        queues["facts"].put_nowait(watermark(20))
        await at(stream, "left admission", lambda j,a: j["left"]["watermark_micros"] == 20 and j["left"]["retained_rows"] == 3)
        quote_rows = [(1,9,1,100),(1,24,2,200),(2,10,3,300),(2,25,4,400),(1,20,5,500),(9,28,6,600)]
        if mixed:
            quote_rows.insert(1, (None, 23, 7, 7900))
            quote_rows.insert(4, (None, 25, 8, 8900))
        queues["quotes"].put_nowait(data(quote_rows, schema=quote_spec))
        j, a = await at(stream, "right mixed retention", lambda j,a: j["emitted_match_rows"] == 3 and a["pending_left_rows"] == 3)
        assert j["right"]["retained_rows"] == 4, j
        assert j["right"]["null_key_rows"] == (2 if mixed else 0), j
        queues["facts"].put_nowait(data([(1,26,4,40),(2,27,5,50),(9,28,6,60)]))
        queues["facts"].put_nowait(None)
        queues["quotes"].put_nowait(None)
        await at(stream, "six intermediate matches", lambda j,a: j["emitted_match_rows"] == 6 and a["left"]["ended"] and a["pending_left_rows"] == 6)
        queues["rates"].put_nowait(data([(1,8,1,2),(1,8,2,3),(2,11,1,5),(1,19,1,7),(1,25,1,11),(2,27,1,13)]))
        for cut, em, operations, expected in [
            (20,2, [(1,3,"upsert"),(2,5,"upsert")], {1:3,2:5}),
            (21,3, [(1,3,"retract"),(1,10,"upsert")], {1:10,2:5}),
            (28,5, [(1,10,"retract"),(1,21,"upsert"),(2,5,"retract"),(2,18,"upsert")], {1:21,2:18}),
            (29,6, [(9,None,"upsert")], {1:21,2:18,9:None}),
        ]:
            queues["rates"].put_nowait(watermark(cut))
            received = []
            async with asyncio.timeout(15):
                while len(received)<len(operations):
                    table = await stream.__anext__()
                    assert table.column_names == ["k","total","_cf_op"]
                    for row in table.to_pylist():
                        k,v,op = row["k"],row["total"],row["_cf_op"]
                        if op == "retract":
                            assert k in current and current.pop(k) == v, (k,v,current)
                        else:
                            assert op == "upsert" and k not in current, (k,v,op,current)
                            current[k] = v
                        received.append((k,v,op))
            assert sorted(received) == sorted(operations), (cut,received,operations)
            assert current == expected, (cut,current,expected)
            _, state = await at(stream, f"cut {cut}", lambda j,a: a["right"]["watermark_micros"] == cut and a["emitted_left_rows"] == em)
            assert state["pending_left_rows"] == 6 - em, (cut,state)
            trace.append({"watermark":cut,"pending":state["pending_left_rows"],"rows":received,"reconstructed":current.copy()})
        queues["rates"].put_nowait(None)
        async with asyncio.timeout(15):
            async for extra in stream:
                assert extra.num_rows == 0, extra
        status = stream.job.status()
        (join,) = status["stream_joins"].values()
        (asof,) = status["stream_asof_joins"].values()
        assert status["state"] == "completed", status
        assert (join["emitted_match_rows"],asof["emitted_left_rows"],asof["matched_rows"],asof["unmatched_rows"]) == (6,6,5,1)
    print(json.dumps({"native":_native.__file__,"pyarrow":pa.__version__,"join_matches":6,"asof_matches":5,"asof_missing":1,"input_geometry":"mixed_null_keys" if mixed else "dense_admission", "right_input_rows":len(quote_rows), "null_keys":join["right"]["null_key_rows"], "right_retained":4,"changes":trace}, sort_keys=True))

for mixed in (False, True):
    asyncio.run(run(mixed))
