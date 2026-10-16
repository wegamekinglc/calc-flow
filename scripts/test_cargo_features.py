from __future__ import annotations

import subprocess
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch

if __package__:
    from scripts.toolkit import command_output, require_executable
else:
    from toolkit import command_output, require_executable

ROOT = Path(__file__).resolve().parents[1]
KAFKA_PACKAGES = {"rdkafka", "rdkafka-sys"}


def _version_note(name: str) -> str:
    try:
        version = command_output([require_executable(name), "--version"], cwd=ROOT)
    except Exception as error:
        return f"{name} --version: unavailable ({type(error).__name__}: {error})"
    return f"{name} --version: {version}"


def _cargo_tree_output(arguments: list[str]) -> str:
    try:
        return command_output(arguments, cwd=ROOT)
    except subprocess.CalledProcessError as error:
        error.add_note(f"Cargo feature query stdout:\n{error.stdout or '<empty>'}")
        error.add_note(f"Cargo feature query stderr:\n{error.stderr or '<empty>'}")
        error.add_note("\n".join(_version_note(name) for name in ("cargo", "rustc")))
        raise


def _python_test_dependencies(*feature_options: str) -> set[str]:
    output = _cargo_tree_output(
        [
            require_executable("cargo"),
            "tree",
            "--locked",
            "-p",
            "calc-flow-python",
            "--edges",
            "normal,build,dev",
            "--prefix",
            "none",
            "--format",
            "{p}",
            *feature_options,
        ],
    )
    return {line.split()[0] for line in output.splitlines() if line.strip()}


class PythonConnectorFeatureTests(unittest.TestCase):
    def test_python_tests_without_kafka_exclude_native_kafka_dependencies(self) -> None:
        for options in [(), ("--no-default-features",)]:
            with self.subTest(options=options):
                dependencies = _python_test_dependencies(*options)
                self.assertFalse(KAFKA_PACKAGES & dependencies)

    def test_python_kafka_feature_includes_native_dependencies(self) -> None:
        dependencies = _python_test_dependencies(
            "--no-default-features", "--features", "connector-kafka"
        )
        self.assertLessEqual(KAFKA_PACKAGES, dependencies)


class CargoFeatureDiagnosticsTests(unittest.TestCase):
    def test_failed_query_exposes_streams_and_versions_in_traceback(self) -> None:
        error = subprocess.CalledProcessError(
            101,
            ["cargo", "tree"],
            output="partial dependency graph",
            stderr="registry unavailable",
        )
        with (
            patch(
                f"{__name__}.require_executable",
                side_effect=lambda name: f"/tools/{name}",
            ),
            patch(
                f"{__name__}.command_output",
                side_effect=[error, "cargo 1.88.0", "rustc 1.88.0"],
            ),
            self.assertRaises(subprocess.CalledProcessError) as caught,
        ):
            _python_test_dependencies(
                "--no-default-features", "--features", "connector-kafka"
            )
        self.assertIs(caught.exception, error)
        self.assertEqual(caught.exception.returncode, 101)
        rendered = "".join(traceback.format_exception(caught.exception))
        self.assertIn("stdout:\npartial dependency graph", rendered)
        self.assertIn("stderr:\nregistry unavailable", rendered)
        self.assertIn("cargo 1.88.0", rendered)
        self.assertIn("rustc 1.88.0", rendered)

    def test_version_probe_failures_preserve_original_cargo_error(self) -> None:
        error = subprocess.CalledProcessError(
            101, ["cargo", "tree"], output="", stderr="original cargo failure"
        )
        with (
            patch(
                f"{__name__}.require_executable",
                side_effect=lambda name: f"/tools/{name}",
            ),
            patch(
                f"{__name__}.command_output",
                side_effect=[
                    error,
                    FileNotFoundError("cargo missing"),
                    RuntimeError("rustc missing"),
                ],
            ),
            self.assertRaises(subprocess.CalledProcessError) as caught,
        ):
            _python_test_dependencies(
                "--no-default-features", "--features", "connector-kafka"
            )
        self.assertIs(caught.exception, error)
        self.assertEqual(caught.exception.returncode, 101)
        rendered = "".join(traceback.format_exception(caught.exception))
        self.assertIn("stderr:\noriginal cargo failure", rendered)
        self.assertIn("cargo --version: unavailable", rendered)
        self.assertIn("rustc --version: unavailable", rendered)


if __name__ == "__main__":
    unittest.main()
