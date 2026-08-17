from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

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


def _popen_group_options() -> dict[str, object]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate(
    process: subprocess.Popen[bytes] | subprocess.Popen[str],
) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        with suppress(ProcessLookupError):
            process.wait()
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with suppress(ProcessLookupError):
            process.wait()


def _python_test_environment() -> dict[str, str]:
    environment = dict(os.environ)
    library_directory = sysconfig.get_config_var("LIBDIR")
    if not isinstance(library_directory, str) or not library_directory:
        return environment
    if os.name == "nt":
        variable = "PATH"
    elif sys.platform == "darwin":
        variable = "DYLD_LIBRARY_PATH"
    else:
        variable = "LD_LIBRARY_PATH"
    existing = environment.get(variable)
    environment[variable] = (
        library_directory
        if not existing
        else f"{library_directory}{os.pathsep}{existing}"
    )
    return environment


def _run(
    command: Sequence[str],
    *,
    timeout: float | None = None,
    environment: dict[str, str] | None = None,
) -> int:
    print(f"+ {shlex.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        env=environment,
        **_popen_group_options(),
    )
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


def _compile_python_test(cargo: str) -> tuple[int, Path | None]:
    command = [
        cargo,
        "test",
        "-p",
        "calc-flow-python",
        "--lib",
        "--all-features",
        "--no-run",
        "--message-format=json",
    ]
    print(f"+ {shlex.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        text=True,
        **_popen_group_options(),
    )
    executables: set[Path] = set()
    try:
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    print(line, end="")
                    continue
                rendered = (message.get("message") or {}).get("rendered")
                if rendered:
                    print(rendered, end="", file=sys.stderr)
                executable = message.get("executable")
                if (
                    message.get("reason") == "compiler-artifact"
                    and (message.get("target") or {}).get("name") == "calc_flow_python"
                    and (message.get("profile") or {}).get("test") is True
                    and isinstance(executable, str)
                ):
                    executables.add(Path(executable))
        status = process.wait()
    except BaseException:
        if process.poll() is None:
            _terminate(process)
        raise
    if status != 0:
        return status, None
    if len(executables) != 1:
        print(
            "cargo did not report exactly one calc_flow_python test executable",
            file=sys.stderr,
            flush=True,
        )
        return 1, None
    return 0, next(iter(executables))


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
            "--lib",
            "--bins",
            "--tests",
            "--examples",
            "--all-features",
        ]
    )
    if core_status != 0:
        return core_status

    connectors_status = _run(
        [
            options.cargo,
            "test",
            "--locked",
            "-p",
            "calc-flow-connectors",
            "--all-features",
        ],
    )
    if connectors_status != 0:
        return connectors_status

    benchmark_status = _run(
        [
            options.cargo,
            "test",
            "--locked",
            "-p",
            "calc-flow",
            "--bench",
            "core",
            "--all-features",
        ]
    )
    if benchmark_status != 0:
        return benchmark_status

    compile_status, python_executable = _compile_python_test(options.cargo)
    if compile_status != 0:
        return compile_status
    assert python_executable is not None

    python_command = [
        str(python_executable),
        "--test-threads=1",
    ]
    for _ in range(options.python_stress_runs):
        python_status = _run(
            python_command,
            timeout=options.python_timeout_seconds,
            environment=_python_test_environment(),
        )
        if python_status != 0:
            return python_status
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _exit_for_signal)
    raise SystemExit(main())
