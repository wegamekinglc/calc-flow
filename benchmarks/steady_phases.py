"""Observe the public one-row session clock and its single native-call boundary."""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from time import perf_counter_ns

from benchmarks import steady_session, warm_stream
from calc_flow._native import _StreamSession
from calc_flow.session import StreamSession
from scripts.toolkit import worker_environment, write_json


class _PhaseClock:
    def __init__(self, probe):
        self.probe = probe
        self.started = None
        self.phases = []

    def __call__(self):
        if self.started is None:
            self.probe.reset()
            self.started = perf_counter_ns()
            return self.started
        finished = perf_counter_ns()
        split = self.probe.phases(self.started, finished)
        if isinstance(split, str):
            raise ValueError(split)
        parts = [
            self.probe.called_ns - self.started,
            self.probe.returned_ns - self.probe.called_ns,
            finished - self.probe.returned_ns,
        ]
        self.phases.append(parts)
        self.started = None
        return finished


def observe(fixture, mode):
    if mode == "plain":
        result = steady_session.observe(fixture)
        return {**result, "phases_ns": [], "calibration_ns": None}
    if mode != "probed":
        raise ValueError("expected plain or probed")
    probe = warm_stream._NativeCallProbe(
        (StreamSession.push.__code__,),
        warm_stream._native_method_matcher(_StreamSession, "push"),
    )
    error = probe.install()
    if error is not None:
        raise RuntimeError(error)
    clock = _PhaseClock(probe)
    try:
        result = steady_session.observe(fixture, clock=clock)
    finally:
        probe.close()
    overhead = warm_stream._probe_overhead_seconds()
    if overhead is None:
        raise ValueError("native probe calibration unavailable")
    return {**result, "phases_ns": clock.phases, "calibration_ns": overhead * 1e9}


if __name__ == "__main__":
    if not gc.isenabled() or sys.getprofile() is not None or sys.gettrace() is not None:
        raise ValueError("requires ordinary GC without tracing or profiling")
    mode, fixture, target = sys.argv[1:]
    write_json(
        Path(target),
        {
            **observe(Path(fixture), mode),
            "mode": mode,
            "gc_enabled": True,
            "environment": worker_environment(),
        },
        exclusive=True,
    )
