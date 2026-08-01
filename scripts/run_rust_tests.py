from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
from collections.abc import Sequence

DEFAULT_PYTHON_TIMEOUT_SECONDS = 300.0


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()


def _run(command: Sequence[str], *, timeout: float | None = None) -> int:
    print(f"+ {shlex.join(command)}", flush=True)
    process = subprocess.Popen(command, start_new_session=os.name == "posix")
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(
            f"{shlex.join(command)} timed out after {timeout:g} seconds",
            file=sys.stderr,
            flush=True,
        )
        _terminate(process)
        return 124
    except BaseException:
        if process.poll() is None:
            _terminate(process)
        raise


def _exit_for_signal(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Rust matrix with the PyO3 lib test isolated and bounded."
    )
    parser.add_argument("--cargo", default="cargo", help=argparse.SUPPRESS)
    parser.add_argument(
        "--python-timeout-seconds",
        type=_positive_float,
        default=DEFAULT_PYTHON_TIMEOUT_SECONDS,
        help="runtime-only timeout for each compiled calc_flow_python lib test run",
    )
    parser.add_argument(
        "--python-stress-runs",
        type=_positive_int,
        default=1,
        help="number of isolated serial calc_flow_python lib test runs",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    core_status = _run(
        [
            options.cargo,
            "test",
            "-p",
            "calc-flow",
            "--all-targets",
            "--all-features",
        ]
    )
    if core_status != 0:
        return core_status

    compile_status = _run(
        [
            options.cargo,
            "test",
            "-p",
            "calc-flow-python",
            "--lib",
            "--all-features",
            "--no-run",
        ]
    )
    if compile_status != 0:
        return compile_status

    python_command = [
        options.cargo,
        "test",
        "-p",
        "calc-flow-python",
        "--lib",
        "--all-features",
        "--",
        "--test-threads=1",
    ]
    for _ in range(options.python_stress_runs):
        python_status = _run(
            python_command,
            timeout=options.python_timeout_seconds,
        )
        if python_status != 0:
            return python_status
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _exit_for_signal)
    raise SystemExit(main())
