from __future__ import annotations

import atexit
import faulthandler
import os
import runpy
import time
from pathlib import Path

_STREAM: object | None = None


def start() -> None:
    global _STREAM
    root = Path(os.environ["DAL313_CHILD_OUTPUT"])
    _STREAM = (root / "example-child-stacks.log").open("w", encoding="utf-8")
    faulthandler.enable(file=_STREAM, all_threads=True)
    faulthandler.dump_traceback_later(45, file=_STREAM, exit=False)
    stages = root / "example-child-stages.log"

    def emit(stage: str) -> None:
        with stages.open("a", encoding="utf-8") as stream:
            stream.write(f"{time.time_ns()} pid={os.getpid()} {stage}\n")

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
            faulthandler.cancel_dump_traceback_later()
            _STREAM.flush()

    runpy.run_path = run_path
    emit("child_started_before_original_corrupted_input")

    def close() -> None:
        faulthandler.cancel_dump_traceback_later()
        faulthandler.disable()
        _STREAM.close()

    atexit.register(close)
