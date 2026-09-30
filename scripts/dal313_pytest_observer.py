from __future__ import annotations

import asyncio
import faulthandler
import functools
import inspect
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

TARGETS = frozenset(
    {
        "python/tests/test_continuous_runtime.py::test_checkpoint_completion_does_not_wait_for_next_source_poll",
        "python/tests/test_continuous_runtime.py::test_cancelling_terminal_observer_preserves_owner_cleanup[shutdown]",
        "python/tests/test_example_validation.py::test_example_rejects_incorrect_results_with_python_optimization[01_datafusion_pipeline.py-quantity]",
    }
)


def await_chain(value: object) -> list[dict[str, object]]:
    chain: list[dict[str, object]] = []
    seen: set[int] = set()
    while value is not None and id(value) not in seen and len(chain) < 16:
        seen.add(id(value))
        frame = getattr(value, "cr_frame", None) or getattr(value, "gi_frame", None)
        record: dict[str, object] = {"kind": type(value).__name__}
        if frame is not None:
            record |= {
                "file": frame.f_code.co_filename,
                "line": frame.f_lineno,
                "function": frame.f_code.co_name,
            }
            record["fixture_events"] = {
                name: event.is_set()
                for name in (
                    "polling",
                    "written",
                    "polled",
                    "close_entered",
                    "close_release",
                )
                if isinstance(event := frame.f_locals.get(name), asyncio.Event)
            }
        chain.append(record)
        value = getattr(value, "cr_await", None) or getattr(value, "gi_yieldfrom", None)
    return chain


class Observer:
    def __init__(self, node: str, output: Path) -> None:
        self.node = node
        self.output = output
        self.started = time.perf_counter_ns()
        self.lock = threading.RLock()
        self.events: list[dict[str, object]] = []
        self.jobs: weakref.WeakSet[Any] = weakref.WeakSet()
        self.callbacks: dict[int, dict[str, Any]] = {}
        self.dropped = 0
        self.sample_ns = 0

    def emit(self, event: str, **data: object) -> None:
        with self.lock:
            if len(self.events) >= 2048:
                self.dropped += 1
                return
            self.events.append(
                {
                    "sequence": len(self.events),
                    "elapsed_ns": time.perf_counter_ns() - self.started,
                    "thread": threading.get_ident(),
                    "event": event,
                    "data": data,
                }
            )

    def callback_factory(
        self, original: Callable[..., Any], label: str
    ) -> Callable[..., Any]:
        @functools.wraps(original)
        def create(binding: object, *args: object, **kwargs: object) -> object:
            coroutine = original(binding, *args, **kwargs)
            with self.lock:
                key = len(self.callbacks)
                self.callbacks[key] = {
                    "label": label,
                    "weak": weakref.ref(coroutine),
                    "binding": id(binding),
                }
            self.emit(
                "callback_created",
                callback_id=key,
                callback=label,
                binding=id(binding),
                creation_stack=traceback.format_stack(limit=12)
                if label.endswith("_native_close")
                else None,
                allocation_stack="creation stack only; tracemalloc is disabled",
            )
            weakref.finalize(
                coroutine,
                self.emit,
                "callback_collected",
                callback_id=key,
                callback=label,
            )
            # Return the same coroutine; do not add a second coroutine or lease.
            return coroutine

        return create

    def snapshot(self, label: str) -> None:
        started = time.perf_counter_ns()
        try:
            jobs = [
                {
                    "job": job.id,
                    "status": job.status(),
                    "completed_callbacks": json.loads(
                        job._inner._take_callback_profile()
                    ),
                }
                for job in list(self.jobs)
            ]
            tasks = [
                {
                    "name": task.get_name(),
                    "cancelling": task.cancelling(),
                    "await_chain": await_chain(task.get_coro()),
                }
                for task in asyncio.all_tasks()
            ]
            with self.lock:
                callbacks = list(self.callbacks.items())
            pending = []
            for key, record in callbacks:
                if (coroutine := record["weak"]()) is not None:
                    pending.append(
                        {
                            "id": key,
                            "callback": record["label"],
                            "binding": record["binding"],
                            "state": inspect.getcoroutinestate(coroutine),
                            "await_chain": await_chain(coroutine),
                        }
                    )
            self.emit(
                "snapshot",
                label=label,
                jobs=jobs,
                tasks=tasks,
                live_python_coroutines=pending,
                native_pending_counter="unavailable",
                native_callback_lease="unavailable",
                idle_boundary=(
                    "terminal return follows native wait_idle; no native pending probe"
                ),
                python_thread_stacks={
                    str(key): traceback.format_stack(frame, limit=12)
                    for key, frame in sys._current_frames().items()
                },
            )
        except Exception as error:
            self.emit("observation_error", label=label, error=repr(error))
        finally:
            self.sample_ns += time.perf_counter_ns() - started

    def install_runtime(self, patch: pytest.MonkeyPatch) -> None:
        from calc_flow import runtime

        original_init = runtime.StreamingJob.__init__

        @functools.wraps(original_init)
        def initialize(job: object, *args: object, **kwargs: object) -> None:
            original_init(job, *args, **kwargs)
            self.jobs.add(job)
            job._inner._enable_callback_profiling()
            self.emit("job_created", job=job.id)

        patch.setattr(runtime.StreamingJob, "__init__", initialize)
        for cls, name in (
            (runtime.SourceBinding, "_native_next"),
            (runtime.SourceBinding, "_native_close"),
            (runtime.SinkBinding, "_native_close"),
        ):
            patch.setattr(
                cls,
                name,
                self.callback_factory(getattr(cls, name), cls.__name__ + "." + name),
            )
        original_wait_for = asyncio.wait_for

        async def wait_for(future: object, timeout: float | None) -> object:
            if timeout != 1:
                return await original_wait_for(future, timeout)
            loop = asyncio.get_running_loop()
            sample = loop.call_later(0.8, self.snapshot, "before_original_1s_deadline")
            stack = (self.output / "before-original-1s-deadline-stacks.log").open(
                "w", encoding="utf-8"
            )
            faulthandler.dump_traceback_later(0.9, file=stack, exit=False)
            self.emit("deadline_enter", timeout=timeout)
            try:
                result = await original_wait_for(future, timeout)
                self.emit("deadline_return", timeout=timeout, result=str(result))
                return result
            except BaseException as error:
                self.emit("deadline_error", timeout=timeout, error=repr(error))
                self.snapshot("after_original_wait_for_error")
                raise
            finally:
                sample.cancel()
                faulthandler.cancel_dump_traceback_later()
                stack.close()

        patch.setattr(asyncio, "wait_for", wait_for)
        for name in ("shutdown_async", "cancel_async", "wait_async"):
            original = getattr(runtime.StreamingJob, name)

            def wrap(method: Callable[..., Any], label: str) -> Callable[..., Any]:
                @functools.wraps(method)
                async def operation(
                    job: object, *args: object, **kwargs: object
                ) -> object:
                    self.emit("terminal_observer_enter", operation=label, job=job.id)
                    try:
                        result = await method(job, *args, **kwargs)
                        self.emit(
                            "terminal_observer_return",
                            operation=label,
                            job=job.id,
                            outcome=str(result),
                            status=job.status(),
                            completed_callbacks=json.loads(
                                job._inner._take_callback_profile()
                            ),
                        )
                        return result
                    except BaseException as error:
                        self.emit(
                            "terminal_observer_error",
                            operation=label,
                            job=job.id,
                            error=repr(error),
                            status=job.status(),
                        )
                        raise

                return operation

            patch.setattr(runtime.StreamingJob, name, wrap(original, name))

    def install_child(self, patch: pytest.MonkeyPatch) -> None:
        original = subprocess.run

        def run(command: object, *args: object, **kwargs: object) -> object:
            selected = (
                isinstance(command, list)
                and len(command) == 6
                and command[1:3] == ["-O", "-c"]
                and command[4:] == ["examples/01_datafusion_pipeline.py", "quantity"]
            )
            if not selected:
                return original(command, *args, **kwargs)
            if (
                kwargs.get("timeout") != 60
                or not kwargs.get("capture_output")
                or not kwargs.get("text")
            ):
                raise ValueError(
                    "example command differs from original timeout/capture contract"
                )
            prefix = "import dal313_child_observer as _dal313; _dal313.start();\n"
            observed = [*command[:3], prefix + command[3], *command[4:]]
            parent = (self.output / "example-parent-stacks.log").open(
                "w", encoding="utf-8"
            )
            faulthandler.dump_traceback_later(50, file=parent, exit=False)
            self.emit(
                "child_enter",
                original_command=command,
                timeout=60,
                parent_dump_at_seconds=50,
                child_dump_at_seconds=45,
            )
            try:
                result = original(observed, *args, **kwargs)
                self.save_output(result.stdout, result.stderr)
                self.emit("child_return", returncode=result.returncode)
                return result
            except subprocess.TimeoutExpired as error:
                self.save_output(error.stdout, error.stderr)
                self.emit("child_timeout", timeout=error.timeout)
                raise
            finally:
                faulthandler.cancel_dump_traceback_later()
                parent.close()

        patch.setattr(subprocess, "run", run)

    def save_output(
        self, stdout: str | bytes | None, stderr: str | bytes | None
    ) -> None:
        for name, value in (
            ("example-partial-stdout.txt", stdout),
            ("example-partial-stderr.txt", stderr),
        ):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            (self.output / name).write_text(value or "", encoding="utf-8")

    def save(self) -> None:
        with self.lock:
            data = {
                "node": self.node,
                "worker": os.environ.get("PYTEST_XDIST_WORKER"),
                "events": self.events,
                "dropped_events": self.dropped,
                "snapshot_ns": self.sample_ns,
                "tracemalloc": False,
                "continuous_profiler": False,
            }
            (self.output / "python-events.json").write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item) -> object:
    if item.nodeid not in TARGETS:
        yield
        return
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    output = (
        Path(os.environ["DAL313_DIAGNOSTIC_OUTPUT"])
        / worker
        / item.name.replace("[", "-").replace("]", "")
    )
    output.mkdir(parents=True, exist_ok=True)
    observer = Observer(item.nodeid, output)
    with pytest.MonkeyPatch.context() as patch:
        if "test_example" in item.nodeid:
            patch.setenv("DAL313_CHILD_OUTPUT", str(output))
            observer.install_child(patch)
        else:
            observer.install_runtime(patch)
        observer.emit("case_enter")
        try:
            outcome = yield
            observer.emit(
                "case_exit",
                exception=repr(outcome.excinfo) if outcome.excinfo else None,
            )
        finally:
            observer.save()
