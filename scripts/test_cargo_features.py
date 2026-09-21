from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAFKA_PACKAGES = {"rdkafka", "rdkafka-sys"}


def _python_test_dependencies(*feature_options: str) -> set[str]:
    result = subprocess.run(
        [
            "cargo",
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
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return {line.split()[0] for line in result.stdout.splitlines() if line.strip()}


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


if __name__ == "__main__":
    unittest.main()
