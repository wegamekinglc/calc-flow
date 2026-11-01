from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_examples


class RunExamplesTests(unittest.TestCase):
    def test_python_surface_runs_numbered_examples_in_order(self) -> None:
        with (
            patch.dict(run_examples.os.environ, {}, clear=True),
            patch.object(run_examples.subprocess, "run") as run,
        ):
            exit_code = run_examples.main(["--surface", "python", "--include-services"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            [call.args[0][:2] for call in run.call_args_list],
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
        self.assertIn("01_datafusion_pipeline", names)
        self.assertIn("31_streaming_account_market_values", names)
        self.assertIn("15_file_source", names)
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
            self.assertIn(stem, names, path)
        self.assertEqual(len(names), len(set(names)))
        for command in commands:
            package = command[command.index("-p") + 1]
            name = command[command.index("--example") + 1]
            self.assertTrue(Path(f"crates/{package}/examples/{name}.rs").is_file())
            if package == "calc-flow-connectors":
                self.assertIn("--features", command)
        self.assertIn("28_kafka_custom_decoder", names)
        for command in commands:
            if command[command.index("--example") + 1] in {
                "27_kafka_protobuf_source",
                "28_kafka_custom_decoder",
            }:
                self.assertEqual(command[-2:], ["--", "--consume"])

    def test_python_and_rust_pairs_have_unique_contiguous_numbers(self) -> None:
        stems = [Path(path).stem for path in run_examples.PYTHON_EXAMPLES]
        self.assertEqual(
            [stem.split("_", 1)[0] for stem in stems],
            [f"{number:02d}" for number in range(1, 37)],
        )
        rust_names = {
            command[command.index("--example") + 1]
            for command in run_examples._commands("rust", include_services=True)
        }
        self.assertEqual(
            {name for name in rust_names if name[:2].isdigit()}, set(stems)
        )

    def test_financial_examples_use_small_matching_inputs_on_both_surfaces(
        self,
    ) -> None:
        for surface in ("python", "rust"):
            commands = run_examples._commands(surface)
            financial = [
                command
                for command in commands
                if any(
                    "30_streaming_stock_positions" in part
                    or "31_streaming_account_market_values" in part
                    for part in command
                )
            ]
            self.assertEqual(len(financial), 2)
            for command in financial:
                self.assertEqual(command[command.index("--rows") + 1], "30")
                self.assertEqual(command[command.index("--accounts") + 1], "2")
                self.assertEqual(command[command.index("--stocks") + 1], "3")
                self.assertEqual(command[command.index("--batch-size") + 1], "4")
            self.assertEqual(financial[1][-2:], ["--quote-rows", "9"])

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
