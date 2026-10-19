"""Own benchmark work and the stream job's terminal result together."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterator
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, TypeVar

from scripts.benchmark_suite.diagnostics import mark_phase, report_failure

_T = TypeVar("_T")
COMPLETION_TIMEOUT = 600


@dataclass(frozen=True, slots=True)
class AcceptanceBindings:
    """Owned I/O resources for one running acceptance sample."""

    job: object
    sources: dict
    sink: object
    streams: dict


def interleaved_events(streams: dict[str, tuple]) -> Iterator[tuple[str, object]]:
    missing = object()
    for events in zip_longest(*streams.values(), fillvalue=missing):
        for name, event in zip(streams, events):
            if event is not missing:
                yield name, event


def _require_completed(outcome: Any) -> None:
    if outcome.state != "completed":
        raise RuntimeError(f"stream failed: {outcome.errors}")


async def run_with_completion(
    operation: Awaitable[_T], completion: Awaitable[Any], *, on_failure=None
) -> _T:
    running = asyncio.ensure_future(operation)
    finished = asyncio.ensure_future(completion)
    try:
        done, _ = await asyncio.wait(
            (running, finished), return_when=asyncio.FIRST_COMPLETED
        )
        if finished in done:
            _require_completed(await finished)
            return await running
        result = await running
        mark_phase("terminal-awaited")
        _require_completed(
            await asyncio.wait_for(asyncio.shield(finished), timeout=COMPLETION_TIMEOUT)
        )
        return result
    except BaseException as error:
        report_failure(on_failure, error)
        raise
    finally:
        for task in (running, finished):
            if not task.done():
                task.cancel()
        await asyncio.gather(running, finished, return_exceptions=True)
