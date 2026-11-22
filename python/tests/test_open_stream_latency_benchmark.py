"""Contract checks for the public event-time stream latency diagnostic."""

from __future__ import annotations

import asyncio
import json
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
    assert json.loads(json.dumps(jitter)) == [
        73_929_008,
        85_857_836,
        98_986_776,
        72_506_624,
    ]
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


@pytest.mark.parametrize(
    "schema",
    [
        latency.OUTPUT.with_metadata({b"origin": b"other"}),
        latency.OUTPUT.set(
            1, latency.OUTPUT.field("mean").with_metadata({b"unit": b"seconds"})
        ),
    ],
)
def test_oracle_rejects_unexpected_metadata(schema) -> None:
    result = pa.table({"sequence": [1], "mean": [1.5]}, schema=schema)
    with pytest.raises(ValueError, match="expected"):
        latency.check_output(result, 1)


def test_environment_failure_precedes_sampling(monkeypatch, tmp_path) -> None:
    def unavailable():
        raise OSError("worker environment unavailable")

    monkeypatch.setattr("sys.argv", ["probe", "--build", "build", "--output", "result"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(latency, "release_identity", lambda _: {})
    monkeypatch.setattr(latency, "worker_environment", unavailable)
    collect = AsyncMock(return_value={})
    monkeypatch.setattr(latency, "collect", collect)
    with pytest.raises(OSError, match="worker environment unavailable"):
        latency.main()
    collect.assert_not_awaited()
