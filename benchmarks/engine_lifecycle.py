"""Own benchmark work and the stream job's terminal result together."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterator
from itertools import zip_longest
from typing import Any, TypeVar

_T = TypeVar("_T")


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
    operation: Awaitable[_T], completion: Awaitable[Any]
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
        _require_completed(await asyncio.wait_for(finished, timeout=600))
        return result
    finally:
        for task in (running, finished):
            if not task.done():
                task.cancel()
        await asyncio.gather(running, finished, return_exceptions=True)
