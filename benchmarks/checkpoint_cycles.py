"""Finite stream timing that includes terminal checkpoints and owned cleanup.

This diagnostic is separate from the pause/recovery benchmark. Fixture creation,
planning, startup, the Arrow oracle and manifest inspection are outside its timer;
the caller must still charge them to the overall measurement budget.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from benchmarks.engine_lifecycle import interleaved_events, run_with_completion


@dataclass(frozen=True)
class CycleCallbacks:
    """Output assembly and clocks used by the measurement window."""

    concat: Callable[[list], object]
    clock: Callable[[], int] = time.perf_counter_ns
    cpu_clock: Callable[[], int] = time.process_time_ns


@dataclass(frozen=True)
class CycleOptions:
    """Checkpoint policy and input sizing for one finite stream run."""

    mode: str
    batch_rows: int


class CycleValidationError(RuntimeError):
    """An invalid sample retaining its diagnostic timings and terminal status."""

    def __init__(self, message: str, evidence: dict):
        super().__init__(message)
        self.evidence = evidence


def _require_natural_end(outcome) -> None:
    if outcome.state != "completed" or outcome.cause != "natural_end":
        raise RuntimeError("stream must complete with natural_end")
    if outcome.errors:
        raise RuntimeError(f"stream completed with errors: {outcome.errors}")


def _require_terminal(outcome, status: dict, rows: int, expected_rows: int) -> None:
    _require_natural_end(outcome)
    if status["state"] != "completed" or status["terminal_cause"] != "natural_end":
        raise RuntimeError("terminal status must be completed/natural_end")
    if status["task_errors"]:
        raise RuntimeError("terminal status contains task errors")
    if status["metrics_overflowed"]:
        raise RuntimeError("terminal metrics overflowed")
    checkpoint = status["checkpoint"]
    if checkpoint["current_epoch"] is not None:
        raise RuntimeError("terminal status contains an in-flight checkpoint")
    if checkpoint["failure_category"] is not None:
        raise RuntimeError("terminal status contains a checkpoint failure")
    if rows != expected_rows:
        raise RuntimeError("stream output row count changed during completion")


async def _settle_owned(task):
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
        except Exception:
            break
    return cancellation


async def _cancel_owned_job(job, evidence: dict):
    cleanup = asyncio.create_task(job.cancel_async())
    cancellation = await _settle_owned(cleanup)
    failure = asyncio.CancelledError() if cleanup.cancelled() else cleanup.exception()
    evidence["cleanup_completed"] = failure is None
    if failure is not None:
        evidence["cleanup_error"] = str(failure) or type(failure).__name__
    return cancellation, failure


async def _wait_ready(sources: dict, sink) -> None:
    await asyncio.wait_for(
        asyncio.gather(*(source.ready.wait() for source in sources.values())),
        30,
    )
    if (
        any(not source.opened.is_set() for source in sources.values())
        or not sink.opened.is_set()
        or sink.rows
    ):
        raise RuntimeError("stream must be ready with empty state before timing")


def _elapsed_seconds(started, stopped):
    return None if started is None else (stopped - started) / 1e9


def _raise_cycle_error(error, evidence: dict) -> None:
    if error is None:
        return
    if isinstance(error, asyncio.CancelledError):
        error.evidence = evidence
        raise error
    raise CycleValidationError(str(error), evidence) from error


async def measure_cycle(
    sources: dict,
    sink,
    streams: dict[str, tuple],
    job,
    *,
    callbacks: CycleCallbacks,
) -> tuple[object, float, dict]:
    """Time ready-to-final-Arrow delivery, including EOF and job settlement.

    Failed and cancelled samples retain ``evidence`` on their exception. An
    external ``asyncio.wait_for`` timeout retains it on the exception's cause.
    Cancellation still propagates after the owned cleanup settles.
    """

    concat, clock, cpu_clock = callbacks.concat, callbacks.clock, callbacks.cpu_clock
    evidence = {}
    started = None
    cpu_started = None
    outcome = None

    async def feed_and_end():
        nonlocal started, cpu_started
        await _wait_ready(sources, sink)
        evidence["before_data"] = job.status()
        cpu_started = cpu_clock()
        started = clock()
        for name, event in interleaved_events(streams):
            await sources[name].push(event)
        await asyncio.wait_for(sink.complete.wait(), 600)
        evidence["data_seconds"] = (clock() - started) / 1e9
        evidence["before_eof"] = job.status()
        if sink.rows != sink.expected_rows:
            raise RuntimeError("stream output row count differs before EOF")
        for source in sources.values():
            await source.push(None)

    async def completion():
        nonlocal outcome
        outcome = await job.wait_async()
        return outcome

    error = None
    table = None
    try:
        await run_with_completion(feed_and_end(), completion())
        evidence["completion_seconds"] = (clock() - started) / 1e9
        evidence["after_eof"] = job.status()
        _require_terminal(outcome, evidence["after_eof"], sink.rows, sink.expected_rows)
        table = concat(sink.tables)
    except (Exception, asyncio.CancelledError) as failure:
        error = failure
    finally:
        cancellation, cleanup_error = await _cancel_owned_job(job, evidence)
        stopped = clock()
        cpu_stopped = cpu_clock()
    evidence["seconds"] = _elapsed_seconds(started, stopped)
    evidence["cpu_seconds"] = _elapsed_seconds(cpu_started, cpu_stopped)
    evidence["after_cleanup"] = job.status()
    if error is not None:
        evidence["operation_error"] = str(error) or type(error).__name__
    error = cancellation or error or cleanup_error
    if error is not None:
        evidence.setdefault("after_eof", evidence["after_cleanup"])
        _raise_cycle_error(error, evidence)
    return table, evidence["seconds"], evidence


def _join_retained_rows(operators: dict) -> int:
    retained_rows = 0
    for operator in operators.values():
        metadata = operator["inline_metadata"]
        spec = metadata.get("spec", {})
        if "left_keys" in spec and "right_keys" in spec:
            retained_rows += sum(
                metadata["metrics"][side]["retained_rows"] for side in ("left", "right")
            )
    return retained_rows


def _committed_manifest(path: Path, root: Path, last_completed: int | None):
    if not path.name.startswith("manifest-") or path.suffix != ".json":
        return None
    document = path.read_bytes()
    manifest = json.loads(document)
    epoch = manifest["epoch"]
    if last_completed is None or epoch > last_completed:
        return None
    if manifest["format_version"] != 3 or not manifest["sources"]:
        raise ValueError("checkpoint evidence requires a v3 source manifest")
    terminal = all(source["ended"] for source in manifest["sources"].values())
    retained_rows = _join_retained_rows(manifest["operators"])
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(document).hexdigest(),
        "epoch": epoch,
        "terminal": terminal,
        "join_retained_rows": retained_rows,
    }


def _committed_manifests(files: list[Path], root: Path, last_completed: int | None):
    manifests = []
    for path in files:
        manifest = _committed_manifest(path, root, last_completed)
        if manifest is not None:
            manifests.append(manifest)
    return manifests


def _manifest_epochs(manifests: list[dict]):
    nonterminal = [item["epoch"] for item in manifests if not item["terminal"]]
    nonempty = [
        item["epoch"]
        for item in manifests
        if not item["terminal"] and item["join_retained_rows"] > 0
    ]
    terminal = [item["epoch"] for item in manifests if item["terminal"]]
    return nonterminal, nonempty, terminal


def _checkpoint_summary(manifests: list[dict], last_completed, checkpointing: bool):
    nonterminal, nonempty, terminal = _manifest_epochs(manifests)
    if checkpointing and last_completed not in terminal:
        raise ValueError("completed enabled stream lacks its terminal manifest")
    return {
        "count_scope": "retained_committed_manifests_only",
        "nonterminal_epochs": nonterminal,
        "nonempty_nonterminal_epochs": nonempty,
        "terminal_epochs": terminal,
        "coverage": "disabled"
        if not checkpointing
        else "nonempty_nonterminal_observed"
        if nonempty
        else "insufficient_nonterminal_coverage",
        "manifests": manifests,
    }


def checkpoint_evidence(root: Path, status: dict, *, checkpointing: bool) -> dict:
    """Read retained, committed manifests after settlement, outside the timer.

    Retention can remove earlier epochs. These lists prove observed coverage;
    neither their lengths nor the last epoch are a total nonterminal count.
    """

    files = sorted(path for path in root.rglob("*") if path.is_file())
    last_completed = status["last_completed_epoch"]
    if not checkpointing and (files or last_completed is not None):
        raise ValueError("disabled checkpoints produced state files or an epoch")
    manifests = _committed_manifests(files, root, last_completed)
    return {
        **_checkpoint_summary(manifests, last_completed, checkpointing),
        "state_file_count": len(files),
        "state_bytes": sum(path.stat().st_size for path in files),
    }


async def _inspect_checkpoints(root: Path, evidence: dict, *, checkpointing: bool):
    inspection = asyncio.create_task(
        asyncio.to_thread(
            checkpoint_evidence,
            root,
            evidence["after_eof"]["checkpoint"],
            checkpointing=checkpointing,
        )
    )
    cancellation = await _settle_owned(inspection)
    failure = None
    try:
        manifest_evidence = inspection.result()
    except (Exception, asyncio.CancelledError) as error:
        failure = error
        evidence = {**evidence, "manifest_error": str(error) or type(error).__name__}
    _raise_cycle_error(cancellation or failure, evidence)
    return {**evidence, "checkpoint_evidence": manifest_evidence}


async def run_cycle(
    plan,
    streams: dict[str, tuple],
    root: Path,
    expected_rows: int,
    *,
    options: CycleOptions,
) -> tuple[object, float, dict]:
    """Run one interval-Join sample with enabled, low-frequency or no checkpoints.

    Supply EngineCase's interval-Join plan, streams and oracle row count. Plans
    and roots are single-use. Validate the returned Arrow table with that case's
    oracle before accepting the sample. No mode pauses, restores, triggers a
    checkpoint or waits merely to increase checkpoint coverage.
    """

    from benchmarks import engine_stream as stream

    mode, batch_rows = options.mode, options.batch_rows
    if mode not in ("low", "on", "off"):
        raise ValueError("checkpoint mode must be low, on or off")
    timed = stream._validated_timed_streams(plan, streams)
    sources = {
        name: stream._ReplaySource(events, batch_rows=batch_rows)
        for name, events in streams.items()
    }
    sink = stream._CollectSink(expected_rows)
    config = stream.StreamRuntimeConfig(
        checkpoint_interval=timedelta(milliseconds=100)
        if mode == "on"
        else timedelta(hours=24),
        edge_budget=stream.EdgeBudget(max_rows=batch_rows, max_bytes=64 << 20),
        **({"checkpointing": False} if mode == "off" else {}),
    )
    job = await stream.StreamingRunner(
        plan,
        {
            name: stream.SourceBinding(
                source, watermark_policy=stream.SourceProvidedWatermarks()
            )
            for name, source in sources.items()
        },
        {"output": [stream.SinkBinding.ordinary("suite", sink)]},
        None if mode == "off" else stream.ManagedCheckpointRuntime(root),
        config=config,
    ).start_async()
    table, seconds, evidence = await measure_cycle(
        sources, sink, timed, job, callbacks=CycleCallbacks(stream.pa.concat_tables)
    )
    evidence = await _inspect_checkpoints(root, evidence, checkpointing=mode != "off")
    return table, seconds, evidence
