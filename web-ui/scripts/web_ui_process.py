#!/usr/bin/env python3
"""Start and stop the local Calc Flow API and Vite development server."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
WEB_UI = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIRECTORY = Path(
    os.environ.get("CALC_FLOW_WEB_RUNTIME_DIR", ROOT / ".calc-flow-web")
)
STATE_FILE_NAME = "processes.json"
EXPECTED_SERVICES = frozenset({"api", "studio"})


class WebUiProcessError(RuntimeError):
    """Raised when the managed local services cannot be started or stopped."""


@dataclass(frozen=True, slots=True)
class ServiceRecord:
    name: str
    pid: int
    process_group: int
    start_token: str | None
    command: tuple[str, ...]
    url: str
    log_path: Path

    @classmethod
    def from_json(cls, name: str, value: object) -> ServiceRecord:
        if not isinstance(value, dict):
            raise WebUiProcessError(f"invalid process record for {name!r}")
        try:
            pid = value["pid"]
            process_group = value["process_group"]
            start_token = value["start_token"]
            command = value["command"]
            url = value["url"]
            log_path = value["log_path"]
        except KeyError as error:
            raise WebUiProcessError(
                f"incomplete process record for {name!r}"
            ) from error
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 1
            or isinstance(process_group, bool)
            or not isinstance(process_group, int)
            or process_group <= 1
            or (start_token is not None and not isinstance(start_token, str))
            or not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) for part in command)
            or not isinstance(url, str)
            or not isinstance(log_path, str)
        ):
            raise WebUiProcessError(f"invalid process record for {name!r}")
        return cls(
            name=name,
            pid=pid,
            process_group=process_group,
            start_token=start_token,
            command=tuple(command),
            url=url,
            log_path=Path(log_path),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "process_group": self.process_group,
            "start_token": self.start_token,
            "command": list(self.command),
            "url": self.url,
            "log_path": str(self.log_path),
        }


def _state_path(runtime_directory: Path) -> Path:
    return runtime_directory / STATE_FILE_NAME


def _load_services(runtime_directory: Path) -> dict[str, ServiceRecord]:
    path = _state_path(runtime_directory)
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WebUiProcessError(
            f"could not read process state at {path}; inspect or remove it manually"
        ) from error
    if (
        not isinstance(document, dict)
        or document.get("version") != 1
        or not isinstance(document.get("services"), dict)
    ):
        raise WebUiProcessError(f"unsupported process state at {path}")
    return {
        name: ServiceRecord.from_json(name, value)
        for name, value in document["services"].items()
    }


def _write_services(
    runtime_directory: Path, services: dict[str, ServiceRecord]
) -> None:
    runtime_directory.mkdir(parents=True, exist_ok=True)
    destination = _state_path(runtime_directory)
    temporary = destination.with_suffix(f".{uuid4().hex}.tmp")
    document = {
        "version": 1,
        "services": {
            name: service.to_json() for name, service in sorted(services.items())
        },
    }
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _process_identity(pid: int) -> tuple[str | None, str | None]:
    """Return Linux process state and start time, or empty values elsewhere."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None, None
    closing_parenthesis = stat.rfind(")")
    if closing_parenthesis < 0:
        return None, None
    fields = stat[closing_parenthesis + 2 :].split()
    if len(fields) <= 19:
        return None, None
    return fields[0], fields[19]


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _service_is_running(service: ServiceRecord) -> bool:
    if not _pid_exists(service.pid):
        return False
    process_state, start_token = _process_identity(service.pid)
    if process_state == "Z":
        return False
    if service.start_token is not None and start_token is not None:
        return service.start_token == start_token
    return True


def _signal_service(service: ServiceRecord, requested_signal: signal.Signals) -> None:
    if not _service_is_running(service):
        return
    try:
        current_group = os.getpgid(service.pid)
        if current_group == service.process_group:
            os.killpg(service.process_group, requested_signal)
        else:
            os.kill(service.pid, requested_signal)
    except ProcessLookupError:
        return
    except PermissionError as error:
        raise WebUiProcessError(
            f"permission denied while stopping {service.name} (PID {service.pid})"
        ) from error


def _running_services(services: list[ServiceRecord]) -> list[ServiceRecord]:
    return [service for service in services if _service_is_running(service)]


def _stop_services(services: dict[str, ServiceRecord], timeout: float) -> None:
    active = _running_services(list(reversed(tuple(services.values()))))
    if not active:
        return

    for service in active:
        _signal_service(service, signal.SIGTERM)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        active = _running_services(active)
        if not active:
            return
        time.sleep(0.1)

    for service in active:
        _signal_service(service, signal.SIGKILL)

    kill_deadline = time.monotonic() + 2.0
    while time.monotonic() < kill_deadline:
        active = _running_services(active)
        if not active:
            return
        time.sleep(0.1)
    names = ", ".join(service.name for service in active)
    raise WebUiProcessError(f"services did not stop: {names}")


def _require_command(command: str) -> str:
    path = shutil.which(command)
    if path is None:
        raise WebUiProcessError(
            f"{command!r} is required; install it and ensure it is on PATH"
        )
    return path


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    temporary_directory = Path(tempfile.gettempdir())
    environment.setdefault(
        "UV_CACHE_DIR", str(temporary_directory / "calc-flow-web-uv-cache")
    )
    environment.setdefault(
        "npm_config_cache", str(temporary_directory / "calc-flow-npm-cache")
    )
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return environment


def _install_frontend_dependencies(npm: str, environment: dict[str, str]) -> None:
    if (WEB_UI / "node_modules" / ".bin" / "vite").is_file():
        return
    print("Installing locked web UI dependencies with npm ci ...")
    try:
        subprocess.run(
            [npm, "ci"],
            cwd=WEB_UI,
            env=environment,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise WebUiProcessError("npm ci failed") from error


def _spawn_service(
    *,
    name: str,
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    url: str,
    log_path: Path,
) -> tuple[subprocess.Popen[bytes], ServiceRecord]:
    try:
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as error:
        raise WebUiProcessError(f"could not start {name}: {error}") from error

    _, start_token = _process_identity(process.pid)
    return process, ServiceRecord(
        name=name,
        pid=process.pid,
        process_group=os.getpgid(process.pid),
        start_token=start_token,
        command=tuple(command),
        url=url,
        log_path=log_path,
    )


def _url_is_ready(url: str) -> bool:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "calc-flow-web-ui-process-manager"},
    )
    try:
        with urllib.request.urlopen(request, timeout=0.5) as response:
            return 200 <= response.status < 400
    except (OSError, urllib.error.URLError):
        return False


def _wait_until_ready(
    processes: dict[str, subprocess.Popen[bytes]],
    services: dict[str, ServiceRecord],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for name, process in processes.items():
            return_code = process.poll()
            if return_code is not None:
                raise WebUiProcessError(
                    f"{name} exited with status {return_code}; "
                    f"see {services[name].log_path}"
                )
        if all(_url_is_ready(service.url) for service in services.values()):
            return
        time.sleep(0.2)
    waiting_for = ", ".join(
        service.name for service in services.values() if not _url_is_ready(service.url)
    )
    raise WebUiProcessError(
        f"timed out waiting for {waiting_for}; logs are in "
        f"{next(iter(services.values())).log_path.parent}"
    )


def start(runtime_directory: Path, timeout: float) -> None:
    existing = _load_services(runtime_directory)
    if existing:
        all_expected = set(existing) == EXPECTED_SERVICES
        all_running = all(_service_is_running(service) for service in existing.values())
        if all_expected and all_running:
            print("Calc Flow Studio is already running.")
            for service in existing.values():
                print(f"  {service.name}: {service.url} (PID {service.pid})")
            return
        print("Cleaning up an incomplete or stale Calc Flow Studio start ...")
        _stop_services(existing, timeout=5.0)
        _state_path(runtime_directory).unlink(missing_ok=True)

    uv = _require_command("uv")
    npm = _require_command("npm")
    environment = _child_environment()
    _install_frontend_dependencies(npm, environment)

    runtime_directory.mkdir(parents=True, exist_ok=True)
    api_log = runtime_directory / "api.log"
    studio_log = runtime_directory / "studio.log"
    api_log.write_text("", encoding="utf-8")
    studio_log.write_text("", encoding="utf-8")

    services: dict[str, ServiceRecord] = {}
    processes: dict[str, subprocess.Popen[bytes]] = {}
    try:
        api_process, api_service = _spawn_service(
            name="api",
            command=[uv, "run", "--package", "calc-flow-studio", "calc-flow-web"],
            cwd=ROOT,
            environment=environment,
            url="http://127.0.0.1:8765/api/v1/catalog",
            log_path=api_log,
        )
        processes["api"] = api_process
        services["api"] = api_service
        _write_services(runtime_directory, services)

        studio_process, studio_service = _spawn_service(
            name="studio",
            command=[
                npm,
                "run",
                "dev",
                "--",
                "--host",
                "127.0.0.1",
                "--port",
                "5173",
                "--strictPort",
            ],
            cwd=WEB_UI,
            environment=environment,
            url="http://127.0.0.1:5173",
            log_path=studio_log,
        )
        processes["studio"] = studio_process
        services["studio"] = studio_service
        _write_services(runtime_directory, services)
        _wait_until_ready(processes, services, timeout)
    except BaseException:
        _stop_services(services, timeout=5.0)
        _state_path(runtime_directory).unlink(missing_ok=True)
        raise

    print("Calc Flow Studio is ready.")
    print("  studio: http://127.0.0.1:5173")
    print("  api:    http://127.0.0.1:8765")
    print(f"  logs:   {runtime_directory}")


def stop(runtime_directory: Path, timeout: float) -> None:
    services = _load_services(runtime_directory)
    if not services:
        print("Calc Flow Studio is already stopped.")
        return
    print("Stopping Calc Flow Studio ...")
    _stop_services(services, timeout)
    _state_path(runtime_directory).unlink(missing_ok=True)
    print("Calc Flow Studio stopped.")
    print(f"Logs remain in {runtime_directory}.")


def status(runtime_directory: Path) -> int:
    services = _load_services(runtime_directory)
    if not services:
        print("Calc Flow Studio is stopped.")
        return 1
    all_running = set(services) == EXPECTED_SERVICES
    for name, service in services.items():
        state = "running" if _service_is_running(service) else "stopped"
        print(f"{name}: {state} (PID {service.pid}, {service.url})")
        all_running = all_running and state == "running"
    return 0 if all_running else 1


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("timeout must be greater than zero")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME_DIRECTORY,
        help="directory for PID state and logs",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        help="startup or shutdown timeout in seconds",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    runtime_directory = arguments.runtime_dir.expanduser().resolve()
    try:
        if arguments.action == "start":
            start(runtime_directory, arguments.timeout or 45.0)
            return 0
        if arguments.action == "stop":
            stop(runtime_directory, arguments.timeout or 5.0)
            return 0
        return status(runtime_directory)
    except WebUiProcessError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
