from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import calc_flow
from benchmarks.warm_stream import ScenarioConfig, WarmSessionScenario
from calc_flow import _native


async def run():
    expected = Path(sys.argv[1]).resolve()
    assert Path(calc_flow.__file__).resolve().parent == expected
    assert Path(_native.__file__).resolve().parent == expected
    config = ScenarioConfig(
        history_rows=4096,
        append_rows=1,
        entities=1,
        indicator="dual_sma_spread",
        exact_projection=True,
    )
    session = await WarmSessionScenario.start(config)
    try:
        sample = await session.sample(collect_gc=True)
        assert sample["start_row"] == 4096 and sample["correctness"]["passed"]
        assert session.phase_probe["installed"] and sample["phases_seconds"]
        finish = await session.finish()
        assert finish["state"] == "completed" and not any(
            finish["finish_rows"].values()
        )
        return {
            "sample": sample,
            "warm_output_digest": session.warm_output_digest,
            "plan": session.plan_identity,
            "phase_probe": session.phase_probe,
            "cpu": sorted(os.sched_getaffinity(0)),
            "finish": finish["finish_rows"],
        }
    finally:
        await session.cancel()


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run())))
