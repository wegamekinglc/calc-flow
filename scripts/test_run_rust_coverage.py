from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_rust_coverage import (
    CLEAN_COMMAND,
    OUTPUT_PATH,
    ROOT,
    SHOW_ENV_COMMAND,
    coverage_commands,
    instrumented_environment,
    require_connector_environment,
    run,
)


def connector_environment() -> dict[str, str]:
    return {
        "CALC_FLOW_CONNECTOR_CONTAINERS": "1",
        "CALC_FLOW_KAFKA_BOOTSTRAP": "localhost:9092",
        "CALC_FLOW_PG_TEST_URL": "postgresql://localhost/postgres",
        "CH_TEST_URL": "http://localhost:8123",
    }


class RustCoverageRunnerTests(unittest.TestCase):
    def test_plan_accumulates_all_real_connector_tests_before_enforcement(self) -> None:
        commands = coverage_commands()

        self.assertEqual(len(commands), 5)
        self.assertEqual(
            commands[0], ("cargo", "test", "--workspace", "--all-features")
        )
        self.assertEqual(
            [
                target
                for command in commands[1:4]
                for index, target in enumerate(command)
                if index > 0 and command[index - 1] == "--test"
            ],
            [
                "kafka_connector",
                "postgresql_connector",
                "postgresql_cdc",
                "clickhouse_connector",
            ],
        )
        for command in commands[1:4]:
            self.assertIn("--ignored", command)
        self.assertEqual(commands[-1][0:3], ("cargo", "llvm-cov", "report"))
        self.assertNotIn("--all-features", commands[-1])
        self.assertNotIn("--workspace", commands[-1])
        self.assertIn("90", commands[-1])
        self.assertEqual(commands[-1][-1], str(OUTPUT_PATH))

    def test_environment_guard_names_every_missing_service(self) -> None:
        with self.assertRaisesRegex(
            SystemExit,
            "CALC_FLOW_CONNECTOR_CONTAINERS, CALC_FLOW_KAFKA_BOOTSTRAP, "
            "CALC_FLOW_PG_TEST_URL, CH_TEST_URL",
        ):
            require_connector_environment({})

    def test_environment_guard_rejects_disabled_container_tests(self) -> None:
        environment = connector_environment()
        environment["CALC_FLOW_CONNECTOR_CONTAINERS"] = "0"
        with self.assertRaisesRegex(SystemExit, "CALC_FLOW_CONNECTOR_CONTAINERS"):
            require_connector_environment(environment)

    @patch("scripts.run_rust_coverage.subprocess.run")
    def test_instrumented_environment_parses_only_valid_exports(self, execute) -> None:
        execute.return_value = SimpleNamespace(
            stdout=(
                "export LLVM_PROFILE_FILE='/repo-%p.profraw'\n"
                "export CARGO_LLVM_COV=1\n"
                "export CARGO_LLVM_COV_TARGET_DIR='/repo/target'\n"
            ),
        )
        base = connector_environment()

        instrumented = instrumented_environment(base)

        self.assertEqual(instrumented["LLVM_PROFILE_FILE"], "/repo-%p.profraw")
        self.assertEqual(instrumented["CARGO_LLVM_COV"], "1")
        self.assertEqual(instrumented["CARGO_LLVM_COV_TARGET_DIR"], "/repo/target")
        self.assertNotIn("LLVM_PROFILE_FILE", base)

        for invalid in ["plain text\n", "export INVALID\n", "export A='one' 'two'\n"]:
            execute.return_value = SimpleNamespace(stdout=invalid)
            with self.assertRaisesRegex(SystemExit, "invalid export"):
                instrumented_environment(base)

        execute.return_value = SimpleNamespace(
            stdout="export LLVM_PROFILE_FILE='/repo-%p.profraw'\n",
        )
        with self.assertRaisesRegex(SystemExit, "omitted required exports"):
            instrumented_environment(base)

    @patch("scripts.run_rust_coverage.subprocess.run")
    def test_runner_executes_the_exact_plan_in_the_repository(self, execute) -> None:
        environment = connector_environment()
        execute.return_value = SimpleNamespace(
            stdout=(
                "export LLVM_PROFILE_FILE='/repo-%p.profraw'\n"
                "export CARGO_LLVM_COV=1\n"
                "export CARGO_LLVM_COV_TARGET_DIR='/repo/target'\n"
            ),
        )

        run(environment)

        self.assertEqual(execute.call_count, 7)
        self.assertEqual(execute.call_args_list[0].args, (SHOW_ENV_COMMAND,))
        self.assertEqual(
            execute.call_args_list[0].kwargs,
            {
                "cwd": ROOT,
                "env": environment,
                "check": True,
                "capture_output": True,
                "text": True,
            },
        )
        instrumented = {
            **environment,
            "LLVM_PROFILE_FILE": "/repo-%p.profraw",
            "CARGO_LLVM_COV": "1",
            "CARGO_LLVM_COV_TARGET_DIR": "/repo/target",
        }
        self.assertEqual(execute.call_args_list[1].args, (CLEAN_COMMAND,))
        self.assertEqual(
            execute.call_args_list[1].kwargs,
            {"cwd": ROOT, "env": instrumented, "check": True},
        )
        for call, command in zip(
            execute.call_args_list[2:],
            coverage_commands(),
            strict=True,
        ):
            self.assertEqual(call.args, (command,))
            self.assertEqual(
                call.kwargs, {"cwd": ROOT, "env": instrumented, "check": True}
            )


if __name__ == "__main__":
    unittest.main()
