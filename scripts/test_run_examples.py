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
        self.assertIn("examples/26_late_side_output.py", paths)
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
        names = [command[command.index("--example") + 1] for command in commands]
        self.assertIn("datafusion_pipeline", names)
        self.assertIn("streaming_account_market_values", names)
        self.assertIn("file_source", names)
        self.assertIn("late_output_recovery", names)
        for name in run_examples.SERVICE_RUST_EXAMPLES:
            self.assertNotIn(name, names)
        self.assertFalse(
            any("schema" in part for command in commands for part in command)
        )

    def test_rust_service_inventory_covers_every_python_example(self) -> None:
        commands = run_examples._commands("rust", include_services=True)
        names = [command[command.index("--example") + 1] for command in commands]
        for path in run_examples.PYTHON_EXAMPLES:
            stem = Path(path).stem
            name = stem.split("_", 1)[1] if stem[:2].isdigit() else stem
            self.assertIn(name, names, path)
        self.assertEqual(len(names), len(set(names)))
        for command in commands:
            package = command[command.index("-p") + 1]
            name = command[command.index("--example") + 1]
            self.assertTrue(Path(f"crates/{package}/examples/{name}.rs").is_file())
            if package == "calc-flow-connectors":
                self.assertIn("--features", command)
        self.assertIn("kafka_custom_decoder", names)
        for command in commands:
            if command[command.index("--example") + 1] in {
                "kafka_protobuf_source",
                "kafka_custom_decoder",
            }:
                self.assertEqual(command[-2:], ["--", "--consume"])

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
