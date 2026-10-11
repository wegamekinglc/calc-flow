"""Owned runtime samples for the fixed Join/ASOF acceptance inventory."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import time
from contextlib import suppress
from datetime import timedelta
from pathlib import Path

from benchmarks.checkpoint_cycles import (
    CycleValidationError,
    _cancel_owned_job,
    _require_terminal,
    _settle_owned,
    checkpoint_evidence,
)
from benchmarks.engine_lifecycle import run_with_completion
from benchmarks.sustained_join import checkpoint_windows
from scripts.benchmark_suite.issue406_acceptance import checkpoint_settings


def runtime_config_arguments(mode: str) -> dict:
    """Avoid passing the candidate-only disable keyword to baseline main."""
    settings = checkpoint_settings(mode)
    return {
        "checkpoint_interval": timedelta(
            milliseconds=settings["checkpoint_interval_millis"] or 86_400_000
        ),
        **({"checkpointing": False} if mode == "off" else {}),
    }


def checkpoint_observation(job) -> dict:
    before = time.perf_counter_ns()
    checkpoint = job.status()["checkpoint"]
    return {
        "before_ns": before,
        "after_ns": time.perf_counter_ns(),
        "checkpoint": checkpoint,
    }


async def _observe_checkpoints(job, observations: list[dict], stop: asyncio.Event):
    while not stop.is_set():
        observations.append(checkpoint_observation(job))
        with suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), 0.01)


def _trace_control_header(trace: dict, case: dict, output: str, frontier: int) -> bool:
    return not (
        trace.get("overflowed") is not False
        or trace.get("terminal_excluded") is not True
        or trace.get("allow_barriers") is not (case["mode"] == "on")
        or trace.get("completed_frontiers", {}).get(output, frontier - 1) < frontier
    )


def _validate_trace_header(
    trace: dict, events, case: dict, output: str, frontier: int
) -> None:
    if (
        trace.get("schema") != "calc-flow.asof-dispatch-trace.v1"
        or trace.get("capacity") != 8192
        or not isinstance(events, list)
        or not events
        or trace.get("record_count") != len(events)
        or not _trace_control_header(trace, case, output, frontier)
    ):
        raise ValueError("invalid or incomplete ASOF processing dispatch trace")


def _validate_trace_controls(events: list[dict], case: dict) -> None:
    allowed = {"data", "watermark"} | ({"barrier"} if case["mode"] == "on" else set())
    if any(
        event.get("kind") not in allowed
        or event.get("ingress") not in ("left", "right")
        for event in events
    ):
        raise ValueError("ASOF processing trace contains an unsupported control")


def _trace_operator_counts(status: dict) -> dict:
    fields = (
        "input_batches",
        "input_rows",
        "fanned_out_batches",
        "fanned_out_rows",
        "errors",
        "ended",
    )
    operators = {
        node: {field: value[field] for field in fields}
        for node, value in status["operators"].items()
    }
    if not operators or any(
        value["errors"] != 0 or value["ended"] is not False
        for value in operators.values()
    ):
        raise ValueError("ASOF trace was captured after an error or terminal callback")
    return operators


def _dispatch_trace(value: str, case: dict, status: dict, frontier: int, sink) -> dict:
    """Validate captured controls and retain actual runtime data-count evidence."""
    trace = json.loads(value)
    events = trace.get("events")
    output = case["dispatch_output_node_id"]
    _validate_trace_header(trace, events, case, output, frontier)
    _validate_trace_controls(events, case)
    operators = _trace_operator_counts(status)
    states = status.get("stream_asof_joins", {})
    if set(states) != {trace.get("node_id")}:
        raise ValueError("ASOF trace lacks the selected operator's final state")
    return {
        **trace,
        "operator_data_counts": operators,
        "asof_state": states[trace["node_id"]],
        "output_record_batch_rows": [
            [batch.num_rows for batch in table.to_batches()] for table in sink.tables
        ],
        "output_node_id": output,
        "required_output_frontier_micros": frontier,
    }


async def measure_acceptance_job(job, sources, sink, streams, *, case: dict) -> tuple:
    """Retain processing evidence even when baseline EOF has a known failure."""
    from benchmarks import engine_stream as stream

    evidence = {"checkpoint_observations": []}
    observations = evidence["checkpoint_observations"]
    stop = asyncio.Event()
    observer = None
    table = None
    outcome = None
    started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()

    async def process_and_end():
        nonlocal table, observer
        observations.append(checkpoint_observation(job))
        observer = asyncio.create_task(_observe_checkpoints(job, observations, stop))
        capture_trace = case["workload"] == "asof" and case["batch_rows"] == 1024
        if capture_trace:
            evidence["dispatch_trace_setup_seconds"] = await _start_processing_trace(
                job, sources, sink, case
            )
        table, seconds, clocks = await stream._measure_ready_clock(
            sources,
            sink,
            streams,
            job,
            join_mode={"join_sealed": "sealed", "join_retained": "retained"}.get(
                case["workload"]
            ),
        )
        evidence.update(clocks)
        evidence["processing_seconds"] = seconds
        if capture_trace:
            await _finish_processing_trace(job, streams, sink, case, evidence)
        observations.append(checkpoint_observation(job))
        if not capture_trace:
            evidence["before_eof"] = job.status()
        evidence["terminal_started_ns"] = time.perf_counter_ns()
        await _end_acceptance_sources(sources, streams, case)

    async def complete():
        nonlocal outcome
        outcome = await job.wait_async()
        return outcome

    failure = None
    observer_failure = None
    try:
        await run_with_completion(process_and_end(), complete())
        evidence["after_eof"] = job.status()
        _require_terminal(outcome, evidence["after_eof"], sink.rows, sink.expected_rows)
    except (Exception, asyncio.CancelledError) as error:
        failure = error
        evidence["after_eof"] = job.status()
        evidence["terminal_error"] = str(error) or type(error).__name__
    finally:
        (
            observer_failure,
            cancellation,
            cleanup_error,
            cleanup_started,
        ) = await _finish_acceptance_job(
            job, evidence, observer, stop, (started, cpu_started)
        )
    failure = cancellation or cleanup_error or failure
    _require_acceptance_outcome(failure, observer_failure, table, evidence, case)
    evidence["lifecycle"] = "completed" if failure is None else "failed"
    if failure is not None:
        evidence["lifecycle_error"] = evidence["terminal_error"]
    evidence["terminal_seconds"] = (
        cleanup_started - evidence["terminal_started_ns"]
    ) / 1e9
    evidence.update(_processing_checkpoint_evidence(observations, evidence))
    return table, evidence


def acceptance_checkpoint_evidence(
    root: Path, status: dict, *, checkpointing: bool
) -> dict:
    """Retain exact source cursor progress for completed processing epochs."""
    from scripts.toolkit import fingerprint_json

    evidence = checkpoint_evidence(root, status, checkpointing=checkpointing)
    manifests = []
    for item in evidence["manifests"]:
        document = json.loads((root / item["path"]).read_text())
        sources = {}
        for name, source in document["sources"].items():
            cursor = source["cursor"]
            rows = None if cursor is None else cursor["payload"].get("rows")
            if rows is not None and (type(rows) is not int or rows < 0):
                raise ValueError("acceptance source cursor has invalid cumulative rows")
            sources[name] = {
                "rows": rows,
                "ended": source["ended"],
                "cursor_sha256": fingerprint_json(cursor),
            }
        manifests.append({**item, "sources": sources})
    return {**evidence, "manifests": manifests}


async def _inspect_sample_checkpoints(
    state_root: Path, evidence: dict, case: dict, scenario: str
) -> dict:
    result = {}
    try:
        inspection = asyncio.create_task(
            asyncio.to_thread(
                acceptance_checkpoint_evidence,
                state_root,
                evidence["after_eof"]["checkpoint"],
                checkpointing=case["mode"] != "off",
            )
        )
        cancellation = await _settle_owned(inspection)
        if cancellation is not None:
            raise cancellation
        checkpoint = inspection.result()
        result["checkpoint_evidence"] = checkpoint
        return {
            "checkpoint_evidence": checkpoint,
            "advancing_processing_epochs": _advancing_processing_epochs(
                checkpoint, evidence, scenario
            ),
        }
    except Exception as error:
        if evidence["lifecycle"] == "completed":
            raise
        return {**result, "checkpoint_evidence_error": str(error)}


def _advancing_processing_epochs(
    checkpoint: dict, evidence: dict, scenario: str
) -> list:
    quote_binding = (
        "left"
        if scenario == "join"
        else "quotes.input"
        if scenario == "asof_join"
        else "input"
    )
    return [
        item["epoch"]
        for item in checkpoint["manifests"]
        if item["epoch"] in evidence["completed_processing_epochs"]
        and not item["terminal"]
        and (item["sources"].get(quote_binding, {}).get("rows") or 0) > 0
    ]


class AcceptanceEngineCase:
    """Reuse suite fixtures with fresh jobs and explicit checkpoint policies."""

    def __init__(self, case: dict, root: Path) -> None:
        from benchmarks import engine_stream as stream
        from benchmarks.engine_comparison import expected_output, workload

        if case["workload"] not in (
            "join_sealed",
            "join_retained",
            "asof",
            "projection",
        ):
            raise ValueError("acceptance runtime workload is unsupported")
        checkpoint_settings(case["mode"])
        self.case, self.root, self.count = dict(case), root, 0
        self.scenario = {
            "join_sealed": "join",
            "join_retained": "join",
            "asof": "asof_join",
            "projection": "projection",
        }[case["workload"]]
        self.data = workload(case["rows"], scenario=self.scenario)
        self.expected = expected_output(self.data, self.scenario)
        events = stream.stream_events(
            self.data.table, self.data.entities, batch_rows=case["batch_rows"]
        )
        self.streams = (
            {
                "right": stream.dimension_events(
                    stream.stream_dimension(self.data.dimension), self.data.table
                ),
                "left": events,
            }
            if self.scenario == "join"
            else {"reference.input": events, "quotes.input": events}
            if self.scenario == "asof_join"
            else {"input": events}
        )
        self.program = stream.stream_program(
            self.scenario,
            self.data.table,
            self.data.dimension,
            batch_rows=case["batch_rows"],
            retained_rows=case["rows"] if case["workload"] == "join_retained" else None,
        )
        self.fixture = None
        self.output_node_id = None
        if self.scenario == "asof_join" and case["batch_rows"] == 1024:
            from benchmarks.join_asof_acceptance import (
                export_runtime_fixture,
                final_projection_node,
            )

            fixture_root = Path(case.get("fixture_root", root / "matched-fixture"))
            self.fixture = export_runtime_fixture(
                fixture_root, self.program, self.streams, self.expected, case
            )
            self.output_node_id = final_projection_node(
                self.program.to_project().model_dump(mode="json")
            )
        self.loop = asyncio.new_event_loop()
        self.active_task = None
        self.previous_signals = {
            signum: signal.signal(signum, self._signal)
            for signum in (signal.SIGTERM, signal.SIGINT)
        }

    def _signal(self, signum, _frame) -> None:
        if self.active_task is None or self.active_task.done():
            raise SystemExit(128 + signum)
        self.loop.call_soon_threadsafe(self.active_task.cancel)

    async def _sample(self, state_root: Path) -> tuple:
        from benchmarks import engine_stream as stream

        plan = self.program.compile_stream(stream.Runtime())
        plan_fingerprint = plan.fingerprint
        timed = stream._validated_timed_streams(plan, self.streams)
        sources = {
            name: stream._ReplaySource(events, batch_rows=self.case["batch_rows"])
            for name, events in self.streams.items()
        }
        sink = stream._CollectSink(self.expected.num_rows)
        config = stream.StreamRuntimeConfig(
            **runtime_config_arguments(self.case["mode"]),
            edge_budget=stream.EdgeBudget(
                max_rows=self.case["batch_rows"], max_bytes=64 << 20
            ),
        )
        startup_started = time.perf_counter_ns()
        job = await stream.StreamingRunner(
            plan,
            {
                name: stream.SourceBinding(
                    source, watermark_policy=stream.SourceProvidedWatermarks()
                )
                for name, source in sources.items()
            },
            {"output": [stream.SinkBinding.ordinary("suite", sink)]},
            None
            if self.case["mode"] == "off"
            else stream.ManagedCheckpointRuntime(state_root),
            config=config,
        ).start_async()
        startup_seconds = (time.perf_counter_ns() - startup_started) / 1e9
        table, evidence = await measure_acceptance_job(
            job,
            sources,
            sink,
            timed,
            case={**self.case, "dispatch_output_node_id": self.output_node_id},
        )
        if self.fixture is not None and (
            evidence["dispatch_trace"]["compiled_plan_fingerprint"] != plan_fingerprint
        ):
            raise CycleValidationError(
                "dispatch trace belongs to a different plan", evidence
            )
        evidence["startup_seconds"] = startup_seconds
        evidence.update(
            await _inspect_sample_checkpoints(
                state_root, evidence, self.case, self.scenario
            )
        )
        return table, evidence

    def sample(self) -> dict:
        from benchmarks.join_asof_acceptance import validate_one_to_one_output

        self.count += 1
        state_root = self.root / f"state-{self.count}"
        started = time.perf_counter_ns()
        evidence = {}
        try:
            self.active_task = self.loop.create_task(self._sample(state_root))
            table, evidence = self.loop.run_until_complete(self.active_task)
            evidence["oracle"] = validate_one_to_one_output(table, self.expected)
            evidence["oracle_passed"] = True
            evidence["fixture"] = self.fixture
            evidence["cpu_affinity"] = sorted(os.sched_getaffinity(0))
            evidence["mode"] = self.case["mode"]
            evidence["product_commit"] = self.case.get("product_commit")
            if self.fixture is not None:
                from scripts.benchmark_suite.issue406_acceptance import (
                    matched_runtime_identity,
                )

                evidence["match_identity"] = matched_runtime_identity(
                    self.fixture, self.case["mode"], evidence["cpu_affinity"]
                )
        except BaseException as error:
            from scripts.toolkit import write_json

            self.root.mkdir(parents=True, exist_ok=True)
            write_json(
                self.root / f"failed-sample-{self.count}.json",
                {
                    **evidence,
                    **getattr(error, "evidence", {}),
                    "failure": str(error) or type(error).__name__,
                    "sample_wall_seconds": (time.perf_counter_ns() - started) / 1e9,
                },
            )
            raise
        finally:
            self.active_task = None
            if state_root.exists():
                shutil.rmtree(state_root)
        evidence["sample_wall_seconds"] = (time.perf_counter_ns() - started) / 1e9
        return evidence

    def finish(self) -> dict:
        return {"state": "completed", "samples": self.count}

    def close(self) -> None:
        for signum, handler in self.previous_signals.items():
            signal.signal(signum, handler)
        if not self.loop.is_closed():
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.run_until_complete(self.loop.shutdown_default_executor())
            self.loop.close()


async def _start_processing_trace(job, sources, sink, case: dict) -> float:
    from benchmarks import engine_stream as stream

    trace_setup = time.perf_counter_ns()
    await asyncio.wait_for(
        asyncio.gather(*(source.ready.wait() for source in sources.values())),
        30,
    )
    stream._require_ready_sources(sources, sink)
    job._inner._benchmark_start_dispatch_trace(case["mode"] == "on")
    return (time.perf_counter_ns() - trace_setup) / 1e9


async def _finish_processing_trace(
    job, streams, sink, case: dict, evidence: dict
) -> None:
    from benchmarks import engine_stream as stream
    from benchmarks.join_asof_acceptance import await_asof_frontier

    watermark = streams["quotes.input"][-1].at
    micros = stream.BASE_MICROS + (watermark - stream.BASE) // timedelta(microseconds=1)
    # ASOF publishes a closed output frontier one microsecond before
    # the minimum input watermark. Publication follows callback return.
    evidence["final_frontier_status"] = await asyncio.wait_for(
        await_asof_frontier(
            job,
            int(micros) - 1,
            output_node_id=case["dispatch_output_node_id"],
        ),
        30,
    )
    settled = time.perf_counter_ns()
    evidence["settled_processing_seconds"] = (
        settled - evidence["processing_started_ns"]
    ) / 1e9
    evidence["settled_processing_ended_ns"] = settled
    evidence["processing_settled"] = True
    capture_started = time.perf_counter_ns()
    evidence["dispatch_trace_capture_started_ns"] = capture_started
    raw_trace = job._inner._benchmark_take_dispatch_trace()
    evidence["before_eof"] = job.status()
    evidence["dispatch_trace"] = _dispatch_trace(
        raw_trace, case, evidence["before_eof"], int(micros) - 1, sink
    )
    evidence["dispatch_trace_capture_seconds"] = (
        time.perf_counter_ns() - capture_started
    ) / 1e9


def _nonterminal_epochs(observations: list[dict], evidence: dict) -> list:
    return sorted(
        {
            epoch
            for observation in observations
            if evidence["processing_started_ns"]
            <= observation["after_ns"]
            <= evidence["terminal_started_ns"]
            for epoch in (
                observation["checkpoint"]["current_epoch"],
                observation["checkpoint"]["last_completed_epoch"],
            )
            if epoch is not None
        }
    )


def _processing_checkpoint_free(observations: list[dict], evidence: dict) -> bool:
    return all(
        observation["checkpoint"][key] is None
        for observation in observations
        if observation["before_ns"] <= evidence["terminal_started_ns"]
        for key in ("current_epoch", "last_completed_epoch")
    ) and all(
        evidence["before_eof"]["checkpoint"][key] is None
        for key in ("current_epoch", "last_completed_epoch")
    )


def _processing_checkpoint_evidence(observations: list[dict], evidence: dict) -> dict:
    result = {}
    windows = checkpoint_windows(
        observations,
        start_ns=evidence["processing_started_ns"],
        end_ns=evidence.get(
            "settled_processing_ended_ns", evidence["processing_ended_ns"]
        ),
    )
    result["checkpoint_windows"] = windows
    result["completed_processing_epochs"] = [
        item["epoch"] for item in windows if item["inside_steady"]
    ]
    result["nonterminal_epochs"] = _nonterminal_epochs(observations, evidence)
    result["processing_checkpoint_free"] = _processing_checkpoint_free(
        observations, evidence
    )
    result["observer_policy"] = "all-modes-status-every-10ms-plus-final-frontier-1ms"
    return result


async def _finish_acceptance_job(
    job, evidence: dict, observer, stop, started: tuple
) -> tuple:
    started, cpu_started = started
    observer_failure = None
    stop.set()
    if observer is not None:
        observer_cancel = await _settle_owned(observer)
        observer_failure = observer_cancel or (
            asyncio.CancelledError() if observer.cancelled() else observer.exception()
        )
    cleanup_started = time.perf_counter_ns()
    cancellation, cleanup_error = await _cancel_owned_job(job, evidence)
    stopped = time.perf_counter_ns()
    evidence["cleanup_seconds"] = (stopped - cleanup_started) / 1e9
    evidence["job_seconds"] = (stopped - started) / 1e9
    evidence["job_cpu_seconds"] = (time.process_time_ns() - cpu_started) / 1e9
    if "processing_started_ns" in evidence:
        evidence["full_cycle_seconds"] = (
            stopped - evidence["processing_started_ns"]
        ) / 1e9
    evidence["after_cleanup"] = job.status()
    evidence["cleanup_complete"] = evidence.pop("cleanup_completed")
    return observer_failure, cancellation, cleanup_error, cleanup_started


def _invalid_failed_sample(table, evidence: dict, case: dict) -> bool:
    return (
        case.get("side") != "baseline"
        or table is None
        or "terminal_started_ns" not in evidence
        or not evidence["cleanup_complete"]
    )


def _require_acceptance_outcome(
    failure, observer_failure, table, evidence: dict, case: dict
) -> None:
    if observer_failure is not None:
        if isinstance(observer_failure, asyncio.CancelledError):
            raise observer_failure
        raise RuntimeError(
            "checkpoint observer failed after owned cleanup"
        ) from observer_failure
    if isinstance(failure, asyncio.CancelledError):
        failure.evidence = evidence
        raise failure
    if failure is not None and _invalid_failed_sample(table, evidence, case):
        raise CycleValidationError("invalid acceptance sample", evidence) from failure


async def _end_acceptance_sources(sources, streams, case: dict) -> None:
    from benchmarks import engine_stream as stream

    if case["workload"] == "join_retained":
        for event in streams["right"]:
            if isinstance(event, stream.Watermark):
                await sources["right"].push(event)
    for source in sources.values():
        await source.push(None)
