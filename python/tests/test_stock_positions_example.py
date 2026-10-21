from __future__ import annotations

import asyncio
import itertools
import runpy
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import calc_flow as cf


@pytest.fixture
def example(monkeypatch):
    directory = Path(__file__).parents[2] / "examples"
    monkeypatch.syspath_prepend(str(directory))
    return runpy.run_path(str(directory / "30_streaming_stock_positions.py"))


def test_stock_position_feed_has_ten_million_rows_and_requested_dimensions(
    example,
) -> None:
    batches = example["demo_batches"]()
    initial = pa.concat_tables(list(itertools.islice(batches, 10)))
    assert initial.num_rows == 10_000
    assert pc.count_distinct(initial["account"]).as_py() == 100
    assert pc.count_distinct(initial["stock"]).as_py() == 100
    assert pc.all(pc.equal(initial["quantity"], 0)).as_py()
    batch_count = 0
    business_rows = 0
    for batch in batches:
        assert batch.num_rows == 1_000
        business_rows += batch.num_rows
        batch_count += 1
    assert business_rows == 10_000_000
    assert batch_count == 10_000


def test_stock_positions_stream_updates_before_eof_and_closes_source(example) -> None:
    async def run() -> list[pa.Table]:
        waiting = asyncio.Event()
        closed = asyncio.Event()

        async def feed():
            try:
                for batch in example["demo_batches"](
                    row_count=80, account_count=2, stock_count=2, batch_size=4
                ):
                    yield batch
                waiting.set()
                await asyncio.Event().wait()
            finally:
                closed.set()

        program = example["make_program"]()
        assert program.engine == "streaming"
        results = program.stream(
            {"trades": feed()}, config=cf.StreamRuntimeConfig(checkpointing=False)
        )

        async def collect() -> list[pa.Table]:
            snapshots = []
            async for output in results:
                assert output.name == "positions"
                snapshots.append(
                    output.table.sort_by(
                        [("account", "ascending"), ("stock", "ascending")]
                    )
                )
                if len(snapshots) == 21:
                    return snapshots
            raise AssertionError("stream ended before all snapshots arrived")

        async with results:
            snapshots = await asyncio.wait_for(collect(), 5)
            await asyncio.wait_for(waiting.wait(), 5)
            assert not closed.is_set()
        assert closed.is_set()
        assert results.job.status()["task_count"] == 0
        return snapshots

    snapshots = asyncio.run(run())
    assert snapshots[0]["position"].equals(pa.chunked_array([[0] * 4], type=pa.int64()))
    assert snapshots[1]["position"].equals(
        pa.chunked_array([[200] * 4], type=pa.int64())
    )
    assert snapshots[-1]["position"].equals(
        pa.chunked_array([[800] * 4], type=pa.int64())
    )


def test_stock_positions_cli_prints_only_final_snapshot_and_elapsed(
    example, monkeypatch, capsys
):
    clock = iter((10.0, 12.5))
    monkeypatch.setitem(
        example["main"].__globals__, "perf_counter", lambda: next(clock)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "30_streaming_stock_positions.py",
            "--rows",
            "80",
            "--accounts",
            "2",
            "--stocks",
            "2",
            "--batch-size",
            "4",
        ],
    )
    example["main"]()
    output = capsys.readouterr().out
    assert output.count("pyarrow.Table") == 1
    assert "position: [[800,800,800,800]]" in output
    assert output.rstrip().endswith("total elapsed: 2.500 s")


def test_stock_positions_cli_rejects_engine_selection(example, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "30_streaming_stock_positions.py",
            "--engine",
            "sql",
            "--rows",
            "0",
            "--accounts",
            "1",
            "--stocks",
            "1",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        example["main"]()
    assert caught.value.code == 2
