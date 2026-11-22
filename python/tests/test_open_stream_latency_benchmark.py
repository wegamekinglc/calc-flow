"""Contract checks for the public event-time stream latency diagnostic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pyarrow as pa
import pytest

from benchmarks import open_stream_latency as latency


def test_seeded_schedule_alternates_order_and_replays_offsets() -> None:
    plan = latency.arrival_schedule(4, 217)
    assert [phase for phase, _ in plan] == [
        "immediate",
        "seeded",
        "seeded",
        "immediate",
        "immediate",
        "seeded",
        "seeded",
        "immediate",
    ]
    assert all(delay == 0 for phase, delay in plan if phase == "immediate")
    jitter = [delay for phase, delay in plan if phase == "seeded"]
    assert len(set(jitter)) == 4 and all(0 < delay <= 100_000_000 for delay in jitter)
    assert plan == latency.arrival_schedule(4, 217)
    assert plan != latency.arrival_schedule(4, 218)


@pytest.mark.parametrize("offset", [0, 50_000_000])
def test_open_stream_reports_wait_and_delivery_separately(monkeypatch, offset) -> None:
    query, rows = latency.workload()
    ticks = iter([1_000, 3_000, 3_100 + offset, 3_500 + offset])
    monkeypatch.setattr(latency, "perf_counter_ns", lambda: next(ticks))
    pause = AsyncMock()
    monkeypatch.setattr(latency, "sleep", pause)
    result = asyncio.run(latency.observe(query, rows, offset))
    assert result == {
        "requested_offset_ns": offset,
        "actual_offset_ns": offset + 100,
        "eligible_to_arrow_ns": 400,
        "input_to_arrow_ns": offset + 2_500,
    }
    if offset:
        pause.assert_awaited_once_with(offset / 1e9)
    else:
        pause.assert_not_awaited()


@pytest.mark.parametrize(
    "malformed",
    [
        pa.table({"sequence": [1], "mean": [1.5]}),
        pa.table(
            {"sequence": [1], "mean": [1.0]},
            schema=pa.schema(
                [
                    pa.field("sequence", pa.uint64(), nullable=False),
                    ("mean", pa.float64()),
                ]
            ),
        ),
    ],
)
def test_oracle_rejects_wrong_schema_or_rolling_value(malformed) -> None:
    with pytest.raises(ValueError, match="expected"):
        latency.check_output(malformed, 1)
