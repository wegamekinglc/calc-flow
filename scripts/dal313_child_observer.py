from __future__ import annotations

import atexit
import faulthandler
import json
import os
import runpy
import sys
import time
from collections.abc import Callable
from pathlib import Path

_STREAM: object | None = None


def start() -> None:
    global _STREAM
    root = Path(os.environ["DAL313_CHILD_OUTPUT"])
    errors: list[dict[str, str]] = []

    def unhealthy(operation: str, error: Exception) -> None:
        if len(errors) < 64:
            errors.append({"operation": operation, "error": repr(error)})
        try:
            print(
                "DAL313 child observation unhealthy: "
                + json.dumps({"healthy": False, "pid": os.getpid(), "errors": errors}),
                file=sys.stderr,
                flush=True,
            )
        except (OSError, ValueError) as fallback_error:
            if len(errors) < 64:
                errors.append(
                    {"operation": "child-health.stderr", "error": repr(fallback_error)}
                )

    def observe(operation: str, action: Callable[[], object]) -> bool:
        try:
            action()
            return True
        except (OSError, ValueError) as error:
            unhealthy(operation, error)
            return False

    _STREAM = None
    try:
        _STREAM = (root / "example-child-stacks.log").open("w", encoding="utf-8")
    except (OSError, ValueError) as error:
        unhealthy("child-stack.open", error)
    enabled = False
    timer = False
    if _STREAM is not None:
        enabled = observe(
            "child-stack.enable",
            lambda: faulthandler.enable(file=_STREAM, all_threads=True),
        )
        timer = observe(
            "child-stack.timer",
            lambda: faulthandler.dump_traceback_later(45, file=_STREAM, exit=False),
        )
    stages = root / "example-child-stages.log"

    def emit(stage: str) -> None:
        def write() -> None:
            with stages.open("a", encoding="utf-8") as stream:
                stream.write(f"{time.time_ns()} pid={os.getpid()} {stage}\n")

        observe("child-stage." + stage, write)

    def cancel_timer() -> None:
        nonlocal timer
        if timer and observe(
            "child-stack.cancel_timer", faulthandler.cancel_dump_traceback_later
        ):
            timer = False

    original = runpy.run_path

    def run_path(*args: object, **kwargs: object) -> object:
        emit("example_run_path_enter")
        try:
            result = original(*args, **kwargs)
            emit("example_run_path_return")
            return result
        except BaseException as error:
            emit(f"example_run_path_error={type(error).__name__}")
            raise
        finally:
            cancel_timer()
            if _STREAM is not None:
                observe("child-stack.flush", _STREAM.flush)

    runpy.run_path = run_path
    emit("child_started_before_original_corrupted_input")

    def close() -> None:
        cancel_timer()
        if enabled:
            observe("child-stack.disable", faulthandler.disable)
        if _STREAM is not None:
            observe("child-stack.close", _STREAM.close)

        def save_health() -> None:
            (root / "example-child-health.json").write_text(
                json.dumps(
                    {"healthy": not errors, "pid": os.getpid(), "errors": errors},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        observe("child-health.save", save_health)

    atexit.register(close)
