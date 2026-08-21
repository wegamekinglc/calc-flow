from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_examples


class RunExamplesTests(unittest.TestCase):
    def test_python_surface_runs_every_numbered_example_in_order(self) -> None:
        with (
            patch.dict(run_examples.os.environ, {}, clear=True),
            patch.object(run_examples.subprocess, "run") as run,
        ):
            exit_code = run_examples.main(["--surface", "python"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [call.args[0] for call in run.call_args_list],
            [
                [sys.executable, str(path)]
                for path in sorted(Path("examples").glob("[0-9][0-9]_*.py"))
            ],
        )
        self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))
        self.assertTrue(
            all(
                call.kwargs["env"]["JAX_PLATFORMS"] == "cpu"
                for call in run.call_args_list
            )
        )

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
