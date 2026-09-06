"""Run the production Rust coverage gate with service-backed connectors.

The connector implementations and Rust/PyO3 binding stay in the coverage
denominator. Their public behavior spans Python and opt-in container tests, so
this runner keeps one llvm-cov profile set across those surfaces. The dedicated
streaming soak test module is test harness rather than production source and is
excluded from the unchanged 90 percent production-line floor.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess  # nosec B404 -- fixed, module-owned cargo commands only
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "rust-lcov.info"
NATIVE_LIBRARY = ROOT / "target" / "debug" / "libcalc_flow_python.so"
SHOW_ENV_COMMAND = ("cargo", "llvm-cov", "show-env", "--sh")
CLEAN_COMMAND = ("cargo", "llvm-cov", "clean", "--workspace")
REPORT_IGNORE_ARGUMENTS = (
    "--ignore-filename-regex",
    r"/runtime/streaming/soak\.rs$",
)
PYTHON_TEST_COMMAND = (
    "uv",
    "run",
    "--no-sync",
    "pytest",
    "python/tests",
    "-q",
    "-p",
    "no:benchmark",
)


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
        ("uv", "sync", "--extra", "dev", "--no-install-project"),
        (
            "cargo",
            "build",
            "-p",
            "calc-flow-python",
        ),
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
            *connector,
            "--test",
            "mysql_connector",
            "--",
            "--ignored",
            "--test-threads=1",
        ),
        (
            "cargo",
            "llvm-cov",
            "report",
            *REPORT_IGNORE_ARGUMENTS,
            "--lcov",
            "--output-path",
            str(OUTPUT_PATH),
        ),
        (
            "cargo",
            "llvm-cov",
            "report",
            *REPORT_IGNORE_ARGUMENTS,
            "--fail-under-lines",
            "90",
        ),
    )


def run_python_suite(environment: dict[str, str]) -> None:
    """Run Python tests against the unstripped, instrumented debug extension."""
    if not NATIVE_LIBRARY.is_file():
        raise SystemExit(f"instrumented Python extension is missing: {NATIVE_LIBRARY}")
    target = ROOT / "target"
    with tempfile.TemporaryDirectory(prefix="python-rust-cov-", dir=target) as root:
        staged = Path(root) / "calc_flow"
        shutil.copytree(
            ROOT / "python" / "calc_flow",
            staged,
            ignore=shutil.ignore_patterns("_native*.so", "__pycache__"),
        )
        shutil.copy2(NATIVE_LIBRARY, staged / "_native.abi3.so")
        python_environment = dict(environment)
        inherited = python_environment.get("PYTHONPATH")
        python_environment["PYTHONPATH"] = (
            f"{root}{os.pathsep}{inherited}" if inherited else root
        )
        subprocess.run(  # nosec B603  # nosemgrep
            PYTHON_TEST_COMMAND,
            cwd=ROOT,
            env=python_environment,
            check=True,
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
    """Fail before compilation when the four service gates are not enabled."""
    required = {
        "CALC_FLOW_CONNECTOR_CONTAINERS": "1",
        "CALC_FLOW_KAFKA_BOOTSTRAP": "",
        "CALC_FLOW_PG_TEST_URL": "",
        "CALC_FLOW_MYSQL_TEST_URL": "",
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
    for index, command in enumerate(coverage_commands()):
        # The tuple comes exclusively from coverage_commands and shell
        # expansion is disabled.
        subprocess.run(  # nosec B603  # nosemgrep
            command, cwd=ROOT, env=environment, check=True
        )
        if index == 2:
            run_python_suite(environment)


def main() -> None:
    run(dict(os.environ))


if __name__ == "__main__":
    main()
