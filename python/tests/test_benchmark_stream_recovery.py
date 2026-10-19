from __future__ import annotations

import asyncio
import json

import pytest

from benchmarks import engine_stream
from benchmarks.engine_comparison import EngineCase
from calc_flow.runtime import StreamingJob


async def _wait_for(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


def _hold_eof(source, entered, release):
    push = source.push

    async def gated(event):
        if event is None:
            entered.set()
            await release.wait()
        await push(event)

    source.push = gated


@pytest.mark.parametrize("first", ["reference.input", "quotes.input"])
def test_periodic_manual_checkpoint_cancel_restore_and_two_source_eof(
    tmp_path, monkeypatch, first
):
    case = {
        "family": "engines",
        "backend": "calc-flow-stream",
        "scenario": "asof_join",
        "rows": 128,
        "batch_rows": 32,
        "checkpoint_interval_millis": 100,
    }
    engine = EngineCase(case, tmp_path)
    started = []
    epochs = []
    start = engine_stream._start_replay_job
    trigger = StreamingJob.trigger_checkpoint_async

    async def manual(job):
        await _wait_for(
            lambda: (job.status()["checkpoint"]["last_completed_epoch"] or 0) >= 1
        )
        periodic = job.status()["checkpoint"]["last_completed_epoch"]
        epoch = await trigger(job)
        epochs.append((periodic, epoch))
        return epoch

    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()

        async def gated_start(*args, **kwargs):
            if started:
                assert started[0][2].status()["state"] == "cancelled"
            result = await start(*args, **kwargs)
            if started:
                sources = result[0]
                # Keep the first EOF consumed while its peer is still unqueued.
                sources_ordered = {
                    first: sources[first],
                    **{k: v for k, v in sources.items() if k != first},
                }
                second = next(name for name in sources_ordered if name != first)
                _hold_eof(sources_ordered[second], entered, release)
                result = (sources_ordered, *result[1:])
            started.append(result)
            return result

        monkeypatch.setattr(engine_stream, "_start_replay_job", gated_start)
        task = asyncio.create_task(
            engine_stream.run_variant_stream(
                lambda: engine_stream.stream_plan(
                    "asof_join", engine.data.table, engine.data.dimension, batch_rows=32
                ),
                engine.streams,
                tmp_path,
                engine.expected.num_rows,
                case=case,
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 5)
            sources, _, job = started[1]
            await _wait_for(
                lambda: sources[first].position == len(sources[first].events)
            )
            assert job.status()["state"] != "completed"
            assert not task.done()
            release.set()
            result = await asyncio.wait_for(task, 5)
            assert engine.validate(result[0])["passed"]
            assert job.status()["state"] == "completed"
            assert all(
                source.position == len(source.events) for source in sources.values()
            )
            assert (
                _require_terminal_manifest(tmp_path)
                == job.status()["checkpoint"]["last_completed_epoch"]
            )
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    monkeypatch.setattr(StreamingJob, "trigger_checkpoint_async", manual)
    try:
        engine.loop.run_until_complete(exercise())
    finally:
        engine.close()
    assert epochs and all(manual > periodic >= 1 for periodic, manual in epochs)


def _require_terminal_manifest(root):
    manifests = [json.loads(path.read_text()) for path in root.rglob("manifest-*.json")]
    terminal = max(manifests, key=lambda manifest: manifest["epoch"])
    assert all(source["ended"] for source in terminal["sources"].values())
    assert all(
        source["cursor"]["payload"]["rows"] == 128
        for source in terminal["sources"].values()
    )
    return terminal["epoch"]
