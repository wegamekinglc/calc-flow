from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.benchmark_suite.process import Worker, child_environment, command


class CommandOwnershipTests(unittest.IsolatedAsyncioTestCase):
    async def test_spawn_and_stop_complete_before_repeated_cancel_propagates(self):
        created, return_handle = asyncio.Event(), asyncio.Event()
        stopping, finish_stop = asyncio.Event(), asyncio.Event()
        child = SimpleNamespace(returncode=None)
        outputs = []

        async def spawn(*args, **kwargs):
            outputs.append(kwargs["stdout"])
            created.set()
            await return_handle.wait()
            self.assertFalse(outputs[0].closed)
            return child

        async def settle(owned):
            self.assertIs(owned, child)
            stopping.set()
            await finish_stop.wait()
            self.assertFalse(outputs[0].closed)
            child.returncode = -15

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "scripts.benchmark_suite.process.shutil.which",
                    return_value="/fake/bench",
                ),
                patch(
                    "scripts.benchmark_suite.process.asyncio.create_subprocess_exec",
                    side_effect=spawn,
                ),
                patch(
                    "scripts.benchmark_suite.process.stop", side_effect=settle
                ) as stop,
            ):
                task = asyncio.create_task(
                    command(["bench"], cwd=root, log=root / "run.log")
                )
                try:
                    await created.wait()
                    task.cancel()
                    await asyncio.sleep(0)
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(
                        task.done(),
                        "creation must retain ownership until handle return",
                    )
                    return_handle.set()
                    await asyncio.wait_for(stopping.wait(), 1)
                    task.cancel()
                    await asyncio.sleep(0)
                    self.assertFalse(
                        task.done(), "repeated cancellation must not interrupt stop"
                    )
                    finish_stop.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    stop.assert_awaited_once_with(child)
                    self.assertEqual(child.returncode, -15)
                    self.assertTrue(outputs[0].closed)
                    record = json.loads((root / "run.command.json").read_text())
                    self.assertIsNone(record["exit_code"])
                    self.assertTrue(record["error"].startswith("CancelledError:"))
                finally:
                    return_handle.set()
                    finish_stop.set()
                    if not task.done():
                        task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_timeout_remains_primary_when_owned_stop_fails(self):
        async def wait():
            raise TimeoutError("primary wait deadline")

        child = SimpleNamespace(returncode=None, wait=wait)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "scripts.benchmark_suite.process.shutil.which",
                    return_value="/fake/bench",
                ),
                patch(
                    "scripts.benchmark_suite.process.asyncio.create_subprocess_exec",
                    return_value=child,
                ),
                patch(
                    "scripts.benchmark_suite.process.stop",
                    side_effect=OSError("cleanup failed"),
                ),
                self.assertRaisesRegex(TimeoutError, "primary wait deadline") as raised,
            ):
                await command(["bench"], cwd=root, log=root / "run.log")
            self.assertIsInstance(raised.exception.__cause__, OSError)
            record = json.loads((root / "run.command.json").read_text())
            self.assertIn("cleanup failed", record["cleanup_error"])
            self.assertIsNone(record["exit_code"])

    async def test_spawn_failure_preserves_primary_without_self_cause(self):
        primary = OSError("spawn failed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "scripts.benchmark_suite.process.shutil.which",
                    return_value="/fake/bench",
                ),
                patch(
                    "scripts.benchmark_suite.process.asyncio.create_subprocess_exec",
                    side_effect=primary,
                ),
                patch("scripts.benchmark_suite.process.stop") as stop,
            ):
                with self.assertRaises(OSError) as raised:
                    await command(["bench"], cwd=root, log=root / "run.log")
                stop.assert_not_called()
            self.assertIs(raised.exception, primary)
            self.assertIsNot(raised.exception.__cause__, primary)
            record = json.loads((root / "run.command.json").read_text())
            self.assertIsNone(record["exit_code"])
            self.assertTrue(record["ownership_unsettled"])

    async def test_private_creation_cancellation_does_not_claim_zero_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "scripts.benchmark_suite.process.shutil.which",
                    return_value="/fake/bench",
                ),
                patch(
                    "scripts.benchmark_suite.process.asyncio.create_subprocess_exec",
                    side_effect=asyncio.CancelledError,
                ),
                patch("scripts.benchmark_suite.process.stop") as stop,
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await command(["bench"], cwd=root, log=root / "run.log")
                stop.assert_not_called()
            record = json.loads((root / "run.command.json").read_text())
            self.assertIs(record["ownership_unsettled"], True)
            self.assertIsNone(record["exit_code"])

    async def test_cancel_remains_primary_when_final_journal_fails(self):
        from scripts.benchmark_suite import process

        write_json = process.write_json
        writes = 0

        def journal(path, value):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise OSError("journal failed")
            write_json(path, value)

        async def wait():
            raise asyncio.CancelledError("primary cancellation")

        child = SimpleNamespace(returncode=None, wait=wait)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch(
                    "scripts.benchmark_suite.process.shutil.which",
                    return_value="/fake/bench",
                ),
                patch(
                    "scripts.benchmark_suite.process.asyncio.create_subprocess_exec",
                    return_value=child,
                ),
                patch("scripts.benchmark_suite.process.stop") as stop,
                patch(
                    "scripts.benchmark_suite.process.write_json", side_effect=journal
                ),
            ):
                with self.assertRaises(asyncio.CancelledError) as raised:
                    await command(["bench"], cwd=root, log=root / "run.log")
                stop.assert_awaited_once_with(child)
            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertIn("journal failed", str(raised.exception.__cause__))

    async def test_normal_command_keeps_argv_stdio_exit_and_journal(self):
        async def wait():
            return 0

        child = SimpleNamespace(returncode=0, wait=wait)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {"TOKIO_WORKER_THREADS": "32"}
            with (
                patch(
                    "scripts.benchmark_suite.process.shutil.which",
                    return_value="/fake/bench",
                ),
                patch(
                    "scripts.benchmark_suite.process.asyncio.create_subprocess_exec",
                    return_value=child,
                ) as spawn,
                patch("scripts.benchmark_suite.process.stop") as stop,
            ):
                await command(
                    ["bench", "case", "--bench"],
                    cwd=root,
                    log=root / "run.log",
                    env=env,
                    timeout=0.5,
                )
                stop.assert_not_called()
            self.assertEqual(spawn.call_args.args, ("/fake/bench", "case", "--bench"))
            self.assertEqual(spawn.call_args.kwargs["env"], env)
            self.assertIs(spawn.call_args.kwargs["shell"], False)
            self.assertEqual(
                spawn.call_args.kwargs["stderr"], asyncio.subprocess.STDOUT
            )
            self.assertTrue(spawn.call_args.kwargs["stdout"].closed)
            record = json.loads((root / "run.command.json").read_text())
            self.assertEqual(record["exit_code"], 0)
            self.assertEqual(record["argv"], ["bench", "case", "--bench"])
            self.assertNotIn("error", record)


class BenchmarkProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_runs_workload_from_its_release_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / "site"
            site.mkdir()
            for side in ("baseline", "candidate"):
                source = root / side
                package = source / "scripts" / "benchmark_suite"
                package.mkdir(parents=True)
                (source / "scripts" / "__init__.py").write_text("")
                (package / "__init__.py").write_text("")
                (package / "__main__.py").write_text(
                    "import json\n"
                    "import sys\n"
                    "for line in sys.stdin:\n"
                    f"    print(json.dumps({{'source': {side!r}}}), flush=True)\n"
                )
                worker = await Worker.start(site, root / "runs" / side, source=source)
                try:
                    self.assertEqual(
                        await worker.request(operation="hello"), {"source": side}
                    )
                finally:
                    await worker.close()

    async def test_python_command_preserves_the_managed_interpreter_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "python.log"
            options = ["-S"] if sys.flags.no_site else []
            await command(
                [sys.executable, *options, "-c", "import sys; print(sys.prefix)"],
                cwd=root,
                log=log,
            )
            self.assertEqual(log.read_text(encoding="utf-8").strip(), sys.prefix)

    async def test_executable_resolution_preserves_dispatch_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shim = root / "cargo"
            shim.symlink_to(sys.executable)
            with (
                patch(
                    "scripts.benchmark_suite.process.shutil.which",
                    return_value=str(shim),
                ),
                patch(
                    "scripts.benchmark_suite.process.asyncio.create_subprocess_exec"
                ) as spawn,
            ):
                spawn.return_value.wait.return_value = 0
                await command(["cargo", "--version"], cwd=root, log=root / "run.log")
            self.assertEqual(spawn.call_args.args[0], str(shim))

    async def test_executable_is_resolved_without_a_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "scripts.benchmark_suite.process.asyncio.create_subprocess_exec"
            ) as spawn:
                spawn.return_value.wait.return_value = 0
                await command(["git", "--version"], cwd=root, log=root / "run.log")
            self.assertTrue(Path(spawn.call_args.args[0]).is_absolute())
            self.assertIs(spawn.call_args.kwargs["shell"], False)

    async def test_failed_command_retains_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "run.log"
            with self.assertRaisesRegex(RuntimeError, "exited 2"):
                await command(
                    [
                        sys.executable,
                        "-c",
                        "print('failure evidence'); raise SystemExit(2)",
                    ],
                    cwd=root,
                    log=log,
                )
            self.assertIn("failure evidence", log.read_text(encoding="utf-8"))
            record = json.loads(log.with_suffix(".command.json").read_text())
            self.assertEqual(record["exit_code"], 2)
            self.assertEqual(record["argv"][0], sys.executable)
            self.assertEqual(record["cwd"], str(root))

    async def test_command_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(
                    command(
                        [sys.executable, "-c", "import time; time.sleep(60)"],
                        cwd=root,
                        log=root / "run.log",
                        timeout=0.05,
                    ),
                    timeout=5,
                )

    def test_worker_environment_has_explicit_thread_and_cache_boundaries(self):
        environment = child_environment(Path(__file__).resolve().parent / "test-site")
        self.assertEqual(environment["TOKIO_WORKER_THREADS"], "32")
        self.assertEqual(environment["POLARS_MAX_THREADS"], "32")
        self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "1")
        self.assertTrue(environment["npm_config_cache"].endswith("target/npm-cache"))

    def test_single_thread_polars_environment_preserves_other_pool_sizes(self):
        environment = child_environment(polars_threads=1)
        self.assertEqual(environment["POLARS_MAX_THREADS"], "1")
        self.assertEqual(environment["TOKIO_WORKER_THREADS"], "32")
        self.assertEqual(environment["OPENBLAS_NUM_THREADS"], "1")


if __name__ == "__main__":
    unittest.main()
