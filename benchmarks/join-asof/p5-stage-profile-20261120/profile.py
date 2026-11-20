from __future__ import annotations

import argparse
import json
import runpy
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch


def summarize_trace(trace: dict) -> dict:
    stages = defaultdict(lambda: {"count": 0, "rows": 0, "wall_ns": 0, "cpu_ns": 0})
    markers = []
    for event in trace["events"]:
        if event["operator"] == "runtime":
            markers.append(event)
            continue
        if event["end_ns"] is None:
            raise ValueError("unsettled Join stage")
        stage = stages[(event["operator"], event["stage"])]
        stage["count"] += 1
        stage["rows"] += event["rows"]
        stage["wall_ns"] += event["end_ns"] - event["start_ns"]
        stage["cpu_ns"] += event["thread_cpu_ns"] or 0
    ordered = sorted(
        (
            {
                "operator": operator,
                "stage": name,
                **values,
            }
            for (operator, name), values in stages.items()
        ),
        key=lambda stage: (-stage["wall_ns"], stage["operator"], stage["stage"]),
    )
    return {
        "schema": trace["schema"],
        "capacity": trace["capacity"],
        "overflowed": trace["overflowed"],
        "max_active_workers": trace["max_active_workers"],
        "stages": ordered,
        "markers": markers,
    }


async def _profile_case_body(engine, case: dict, root: Path) -> dict:
    from benchmarks import engine_stream as stream

    jobs = []
    measure = stream._measure_ready
    runner = stream.StreamingRunner

    class ProfileRunner(runner):
        async def start_async(self):
            job = await super().start_async()
            job._inner._benchmark_start_join_cpu_trace()
            jobs.append(job)
            return job

    async def measured(sources, sink, timed, job, **kwargs):
        job._inner._benchmark_mark_join_cpu_trace("ready_timer_start")
        result = await measure(sources, sink, timed, job, **kwargs)
        job._inner._benchmark_mark_join_cpu_trace("ready_timer_end")
        return result

    with (
        patch.object(stream, "StreamingRunner", ProfileRunner),
        patch.object(stream, "_measure_ready", measured),
    ):
        table, seconds = await stream.run_stream(
            stream.stream_plan(
                case["scenario"], engine.data.table, engine.data.dimension
            ),
            engine.streams,
            root / "sample-1",
            engine.expected.num_rows,
            static_join=case["scenario"] == "join",
        )
    if len(jobs) != 1:
        raise RuntimeError("expected one profiled runner")
    job = jobs[0]
    trace = json.loads(job._inner._benchmark_take_join_cpu_trace())
    root.mkdir(parents=True, exist_ok=True)
    (root / "join-cpu-trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    return {
        "seconds": seconds,
        "correctness": engine.validate(table),
        "completion": job.status(),
        "trace": summarize_trace(trace),
    }


def profile_case(case: dict, root: Path) -> dict:
    from benchmarks.engine_comparison import EngineCase

    engine = EngineCase(case, root)
    try:
        return engine.loop.run_until_complete(_profile_case_body(engine, case, root))
    finally:
        engine.close()


def native_evidence(wheel: Path) -> dict:
    path = Path(__file__).resolve().parents[1] / "p5-morsel-state-20261120/measure.py"
    return runpy.run_path(str(path))["native_evidence"](wheel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("join", "interval_join"), required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    case = {
        "backend": "calc-flow-stream",
        "scenario": args.scenario,
        "rows": args.rows,
    }
    result = profile_case(case, args.root)
    report = {
        "schema": "calc-flow.join-stage-profile-result.v1",
        "case": case,
        "candidate": native_evidence(args.wheel),
        "wall_seconds": time.perf_counter() - started,
        **result,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
