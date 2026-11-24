"""Check the public warm ASOF delivery benchmark on an active stream."""

from __future__ import annotations

import asyncio

import pyarrow as pa
import pytest

import calc_flow as cf
from benchmarks import warm_asof_latency as latency


def test_clock_only_covers_prefix_submission_and_delivery(monkeypatch):
    events = []
    ticks = iter([1_000, 1_300])
    original = latency.check_rows
    put = asyncio.Queue.put_nowait
    receive = latency._receive
    frontier = latency.await_asof_frontier

    async def published(job, micros):
        status = await frontier(job, micros)
        events.append(f"published {micros}")
        return status

    def submit(queue, item):
        if isinstance(item, pa.Table) and item.num_rows == 4:
            events.append("right batch")
        if isinstance(item, cf.Watermark) and item.at.microsecond == 16:
            events.append("right cut")
        return put(queue, item)

    async def read(stream, rows):
        tables = await receive(stream, rows)
        if rows == 8:
            events.append("prefix delivered")
        return tables

    def clock():
        events.append("clock")
        return next(ticks)

    def check(tables, expected):
        events.append(expected.num_rows)
        return original(tables, expected)

    monkeypatch.setattr(asyncio.Queue, "put_nowait", submit)
    monkeypatch.setattr(latency, "await_asof_frontier", published)
    monkeypatch.setattr(latency, "_receive", read)
    monkeypatch.setattr(latency, "perf_counter_ns", clock)
    monkeypatch.setattr(latency, "check_rows", check)
    result = asyncio.run(asyncio.wait_for(latency.observe(), 8))
    assert events == [
        1,
        "published 0",
        "clock",
        "right batch",
        "right cut",
        "prefix delivered",
        "clock",
        8,
        "published 15",
        1,
    ]
    assert result == {
        "delivery_ns": 300,
        "proof": {"rows": 8, "matched": 5, "unmatched": 3, "pending_at_cut": 1},
    }


@pytest.mark.parametrize("row,wrong_sequence", [(0, 1), (2, None), (4, 4)])
def test_oracle_rejects_wrong_tie_tolerance_or_missing_key(row, wrong_sequence):
    expected = latency.prefix_oracle()
    field = expected.schema.field("right__seq")
    values = expected[field.name].to_pylist()
    values[row] = wrong_sequence
    corrupt = expected.set_column(
        expected.schema.get_field_index(field.name), field, pa.array(values, field.type)
    )
    with pytest.raises(ValueError, match="ASOF"):
        latency.check_rows([corrupt], expected)
