"""One isolated wheel and one active workload per process; JSON-lines IPC."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path


def environment() -> dict:
    import polars as pl

    from scripts.profile_warm_stream import _worker_environment

    return {
        **_worker_environment(),
        "packages": {
            name: version(name)
            for name in ("datafusion", "polars", "TA-Lib", "jax", "jaxlib")
        },
        "polars_threads": pl.thread_pool_size(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "POLARS_MAX_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
    }


class WarmCase:
    """Own a continuous scenario across increments, with an untimed loop bridge."""

    def __init__(self, case: dict, root: Path) -> None:
        from benchmarks.warm_stream import ScenarioConfig, WarmScenario

        self.loop = asyncio.new_event_loop()
        config = ScenarioConfig(
            history_rows=case["history_rows"],
            append_rows=case["rows"],
            entities=1,
            indicator="rolling_mean"
            if case["scenario"] == "sma20"
            else "dual_sma_spread",
        )
        try:
            self.scenario = self.loop.run_until_complete(
                WarmScenario.start(config, root)
            )
        except BaseException:
            self.loop.close()
            raise
        self.finished = False

    def sample(self) -> dict:
        return self.loop.run_until_complete(self.scenario.sample(collect_gc=False))

    def finish(self) -> dict:
        result = self.loop.run_until_complete(self.scenario.finish())
        self.finished = True
        return {"state": result["state"], "warm_seconds": result["warm_seconds"]}

    def close(self) -> None:
        try:
            if not self.finished:
                self.loop.run_until_complete(self.scenario.job.cancel_async())
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        finally:
            self.loop.close()


def prepare_case(case: dict, root: Path):
    from benchmarks.engine_comparison import EngineCase

    factory = WarmCase if case["family"] == "warm" else EngineCase
    active = factory(case, root)
    try:
        warmup = active.sample()
    except BaseException:
        active.close()
        raise
    return {"warmup": warmup, "case": case}, active


def finish_case(active) -> dict:
    outcome = (
        active.finish() if isinstance(active, WarmCase) else {"state": "completed"}
    )
    active.close()
    return outcome


def _require_prepared(operation: str, active) -> None:
    if operation in ("sample", "finish") and active is None:
        raise ValueError("no active benchmark; prepare a case first")


def dispatch(message: dict, active, root: Path):
    operation = message["operation"]
    _require_prepared(operation, active)
    match operation:
        case "hello":
            return environment(), active
        case "prepare":
            if active is not None:
                raise ValueError("a benchmark is already active")
            return prepare_case(message["case"], root)
        case "sample":
            return active.sample(), active
        case "finish":
            return finish_case(active), None
        case _:
            raise ValueError("unknown worker request")


def main(root: Path) -> None:
    active = None
    try:
        for line in sys.stdin:
            response, active = dispatch(json.loads(line), active, root)
            print(json.dumps(response, allow_nan=False), flush=True)
    except Exception as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}), flush=True)
        raise
    finally:
        if active is not None:
            active.close()
