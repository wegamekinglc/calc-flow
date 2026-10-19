"""Best-effort evidence outside measured regions, owned by one worker request."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path

from scripts.toolkit import sha256_file, write_json

_TRACE: ContextVar[dict | None] = ContextVar("benchmark_request_trace", default=None)


def request_contexts(
    case: dict, releases: dict, *, round_index: int | None = None
) -> dict:
    return {
        side: {
            "side": side,
            "round": round_index,
            "case_id": case.get("id"),
            "release": release,
        }
        for side, release in releases.items()
    }


def persist(path: Path, value: dict, error: BaseException | None = None) -> None:
    """A failed diagnostic must never replace the failure being investigated."""
    try:
        write_json(path, value)
    except Exception as diagnostic_error:
        note = (
            f"diagnostic {path}: {type(diagnostic_error).__name__}: {diagnostic_error}"
        )
        if error is not None:
            error.add_note(note)
        with suppress(Exception):
            print(note, file=sys.stderr, flush=True)


@contextmanager
def trace_request(root: Path, message: dict):
    trace = {"context": message.get("context", {}), "phases": []}
    token = _TRACE.set(trace)
    path = root / f"worker-request-{message.get('request_id', 0):04d}.json"
    try:
        yield trace
    except BaseException as error:
        persist(path, {**trace, "error": f"{type(error).__name__}: {error}"}, error)
        raise
    else:
        persist(path, trace)
    finally:
        _TRACE.reset(token)


def mark_phase(phase: str) -> None:
    """Keep milestones in memory; flush only after timing or on failure."""
    trace = _TRACE.get()
    if trace is not None:
        trace["phases"].append({"phase": phase, "monotonic_ns": time.monotonic_ns()})


def observe(operation) -> object:
    try:
        return operation()
    except Exception as error:
        return {"diagnostic_error": f"{type(error).__name__}: {error}"}


def checkpoint_summary(root: Path) -> list[dict]:
    paths = sorted(
        root.rglob("manifest-*.json"), key=lambda path: path.stat().st_mtime_ns
    )
    return [observe(lambda path=path: _manifest(path, root)) for path in paths[-8:]]


def _manifest(path: Path, root: Path) -> dict:
    if path.stat().st_size > 1 << 20:
        raise ValueError("checkpoint manifest exceeds diagnostic size bound")
    return {
        "path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "manifest": json.loads(path.read_text(encoding="utf-8")),
    }


def _source_state(source) -> dict:
    events = getattr(source, "events", ())
    return {
        "position": getattr(source, "position", None),
        "pushed": getattr(source, "_pushed", None),
        "event_count": len(events),
        "queued": source._events.qsize(),
        "opened": source.opened.is_set(),
        "ready": source.ready.is_set(),
        "cursor": _source_cursor(source, events),
        "eof_push_started": getattr(source, "_pushed", None) == len(events)
        if events
        else None,
        "eof_consumed": getattr(source, "position", None) == len(events)
        if events
        else None,
    }


def _source_cursor(source, events) -> dict | None:
    position = getattr(source, "position", 0)
    for event in reversed(events[:position]):
        cursor = getattr(event, "cursor", None)
        if cursor is not None:
            return {"order": cursor.order.hex(), "payload": dict(cursor.payload)}
    return None


def stream_failure(root: Path, job, sources: dict, sink, error: BaseException) -> None:
    trace = _TRACE.get() or {}
    record = {
        "context": trace.get("context", {}),
        "phases": list(trace.get("phases", [])),
        "error": f"{type(error).__name__}: {error}",
        "job": observe(lambda: job.status()),
        "sources": {
            name: observe(lambda source=source: _source_state(source))
            for name, source in sources.items()
        },
        "sink": observe(
            lambda: {"rows": sink.rows, "expected_rows": sink.expected_rows}
        ),
        "checkpoint": observe(lambda: checkpoint_summary(root)),
    }
    persist(root / "failure-before-cancel.json", record, error)


def report_failure(callback, error: BaseException) -> None:
    if callback is None:
        return
    try:
        callback(error)
    except Exception as diagnostic_error:
        error.add_note(
            f"failure snapshot: {type(diagnostic_error).__name__}: {diagnostic_error}"
        )


def close_after_failure(active, error: BaseException) -> None:
    try:
        active.close()
    except BaseException as cleanup_error:
        error.add_note(f"worker close: {type(cleanup_error).__name__}: {cleanup_error}")
