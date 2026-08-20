"""Run the workspace Rust coverage gate with service-backed connectors.

The connector implementations are part of the workspace coverage denominator,
while their real I/O paths live in opt-in container tests.  This runner keeps
one llvm-cov profile set across the ordinary workspace suite and those gated
tests, then enforces the unchanged 90 percent line floor on the combined data.
"""

from __future__ import annotations

import os
import shlex
import subprocess  # nosec B404 -- fixed, module-owned cargo commands only
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "rust-lcov.info"
SHOW_ENV_COMMAND = ("cargo", "llvm-cov", "show-env", "--sh")
CLEAN_COMMAND = ("cargo", "llvm-cov", "clean", "--workspace")


def coverage_commands() -> tuple[tuple[str, ...], ...]:
    """Return the deterministic command plan for one combined coverage run."""
    connector = (
        "cargo",
        "test",
        "-p",
        "calc-flow-connectors",
        "--all-features",
    )
    return (
        ("cargo", "test", "--workspace", "--all-features"),
        (
            *connector,
            "--test",
            "kafka_connector",
            "--",
            "--ignored",
            "--test-threads=1",
        ),
        (
            *connector,
            "--test",
            "postgresql_connector",
            "--test",
            "postgresql_cdc",
            "--",
            "--ignored",
            "--test-threads=1",
        ),
        (
            *connector,
            "--test",
            "clickhouse_connector",
            "--",
            "--ignored",
            "--test-threads=1",
        ),
        (
            "cargo",
            "llvm-cov",
            "report",
            "--workspace",
            "--fail-under-lines",
            "90",
            "--lcov",
            "--output-path",
            str(OUTPUT_PATH),
        ),
    )


def instrumented_environment(environment: dict[str, str]) -> dict[str, str]:
    """Return a copy extended with cargo-llvm-cov's reviewed shell exports."""
    completed = subprocess.run(  # nosec B603  # nosemgrep
        SHOW_ENV_COMMAND,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    instrumented = dict(environment)
    for line in completed.stdout.splitlines():
        if not line.startswith("export ") or "=" not in line:
            raise SystemExit("cargo llvm-cov show-env returned an invalid export")
        name, encoded = line.removeprefix("export ").split("=", 1)
        values = shlex.split(encoded)
        if not name.isidentifier() or len(values) != 1:
            raise SystemExit("cargo llvm-cov show-env returned an invalid export")
        instrumented[name] = values[0]
    required = {"LLVM_PROFILE_FILE", "CARGO_LLVM_COV", "CARGO_LLVM_COV_TARGET_DIR"}
    if not required.issubset(instrumented):
        raise SystemExit("cargo llvm-cov show-env omitted required exports")
    return instrumented


def require_connector_environment(environment: dict[str, str]) -> None:
    """Fail before compilation when the three service gates are not enabled."""
    required = {
        "CALC_FLOW_CONNECTOR_CONTAINERS": "1",
        "CALC_FLOW_KAFKA_BOOTSTRAP": "",
        "CALC_FLOW_PG_TEST_URL": "",
        "CH_TEST_URL": "",
    }
    missing = [
        name
        for name, exact in required.items()
        if name not in environment
        or not environment[name]
        or (exact and environment[name] != exact)
    ]
    if missing:
        joined = ", ".join(sorted(missing))
        raise SystemExit(f"connector coverage environment is incomplete: {joined}")


def run(environment: dict[str, str]) -> None:
    """Execute the combined coverage plan and propagate the first failure."""
    require_connector_environment(environment)
    environment = instrumented_environment(environment)
    subprocess.run(  # nosec B603  # nosemgrep
        CLEAN_COMMAND, cwd=ROOT, env=environment, check=True
    )
    for command in coverage_commands():
        # The tuple comes exclusively from coverage_commands and shell
        # expansion is disabled.
        subprocess.run(  # nosec B603  # nosemgrep
            command, cwd=ROOT, env=environment, check=True
        )


def main() -> None:
    run(dict(os.environ))


if __name__ == "__main__":
    main()
