"""Diagnostic-only phase clocks; overlapping stages are retained separately."""

from __future__ import annotations

import cProfile
import json
import pstats
import time
from collections import defaultdict
from pathlib import Path

from common import save


class Profile:
    """Own diagnostic counters for one worker; never shared across workers."""

    def __init__(self, directory: Path, enabled: bool):
        self.directory = directory
        self.enabled = enabled
        self.clocks = defaultdict(list)
        self.details = {}
        self.cpu = cProfile.Profile()
        if enabled:
            self.cpu.enable()

    def call(self, name, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        start = time.perf_counter_ns()
        try:
            return function(*args, **kwargs)
        finally:
            self.clocks[name].append(time.perf_counter_ns() - start)

    async def await_call(self, name, function, *args, **kwargs):
        if not self.enabled:
            return await function(*args, **kwargs)
        start = time.perf_counter_ns()
        try:
            return await function(*args, **kwargs)
        finally:
            self.clocks[name].append(time.perf_counter_ns() - start)

    def dump(self):
        if not self.enabled:
            return
        self.cpu.disable()
        self.cpu.dump_stats(str(self.directory / "python-main.pstats"))
        stats = pstats.Stats(self.cpu)
        ordered = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)
        top = [
            {
                "function": str(key),
                "primitive_calls": v[0],
                "calls": v[1],
                "self_s": v[2],
                "cumulative_s": v[3],
            }
            for key, v in ordered[:100]
        ]
        save(
            self.directory / "profile.json",
            {
                "stage_ns": dict(self.clocks),
                "details": self.details,
                "cprofile_main_thread_only": top,
                "warning": (
                    "await/dispatch/callback durations can overlap; "
                    "do not sum as exclusive wall time"
                ),
            },
        )


def callback_trace(job, profile: Profile) -> None:
    if not profile.enabled:
        return
    try:
        profile.details["native_callbacks"] = json.loads(
            job._inner._take_callback_profile()
        )
    except Exception as error:
        profile.details["native_callback_error"] = repr(error)


def bridge_diagnostic(tables, profile: Profile) -> None:
    if not profile.enabled:
        return
    import calc_flow as cf

    for table in tables:
        wrapped = profile.call("bridge.from_pyarrow", cf.Batch.from_pyarrow, table)
        exported = profile.call("bridge.to_pyarrow", wrapped.to_pyarrow)
        if not exported.equals(table):
            raise ValueError("Arrow bridge roundtrip changed values")
