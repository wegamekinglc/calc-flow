from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_examples


class RunExamplesTests(unittest.TestCase):
    def test_python_surface_runs_numbered_and_named_examples_in_order(self) -> None:
        with (
            patch.dict(run_examples.os.environ, {}, clear=True),
            patch.object(run_examples.subprocess, "run") as run,
        ):
            exit_code = run_examples.main(["--surface", "python", "--include-services"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                [sys.executable, str(path)]
                for path in sorted(Path("examples").glob("[0-9][0-9]_*.py"))
            ]
            + [[sys.executable, "examples/symbolic_event_window.py"]],
        )
        self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))
        self.assertTrue(
            all(
                call.kwargs["env"]["JAX_PLATFORMS"] == "cpu"
                for call in run.call_args_list
            )
        )

    def test_default_python_surface_skips_service_examples(self) -> None:
        with patch.object(run_examples.subprocess, "run") as run:
            exit_code = run_examples.main(["--surface", "python"])

        self.assertEqual(exit_code, 0)
        paths = [call.args[0][1] for call in run.call_args_list]
        self.assertIn("examples/15_file_source.py", paths)
        for path in (
            *run_examples.SERVICE_PYTHON_EXAMPLES,
            "examples/22_kafka_sink.py",
            "examples/23_postgresql_sink.py",
            "examples/24_mysql_sink.py",
            "examples/25_clickhouse_sink.py",
        ):
            self.assertNotIn(path, paths)

    def test_rust_surface_runs_user_examples_but_not_schema_generators(self) -> None:
        with (
            patch.dict(run_examples.os.environ, {}, clear=True),
            patch.object(run_examples.subprocess, "run") as run,
        ):
            exit_code = run_examples.main(["--surface", "rust"])

        self.assertEqual(exit_code, 0)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            commands,
            [
                ["cargo", "run", "-p", "calc-flow", "--example", name]
                for name in (
                    "expression_pipeline",
                    "sql_join",
                    "continuous_runtime",
                    "windowed_streaming",
                )
            ],
        )
        self.assertFalse(
            any("schema" in part for command in commands for part in command)
        )

    def test_failed_example_exit_code_is_preserved(self) -> None:
        failure = run_examples.subprocess.CalledProcessError(
            returncode=17,
            cmd=[sys.executable, "examples/01_datafusion_pipeline.py"],
        )
        with (
            patch.dict(run_examples.os.environ, {}, clear=True),
            patch.object(run_examples.subprocess, "run", side_effect=failure),
        ):
            exit_code = run_examples.main(["--surface", "python"])

        self.assertEqual(exit_code, 17)


if __name__ == "__main__":
    unittest.main()
