from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class ContractTests(unittest.TestCase):
    def test_plan_is_only_two_fixed_cases_and_three_fixed_comparisons(self):
        from scripts.dal301_groupby import contract

        plan = contract.plan()
        self.assertEqual([case["rows"] for case in plan["cases"]], [10_000, 100_000])
        self.assertEqual(plan["comparisons"], [["A", "A"], ["B", "B"], ["A", "B"]])
        self.assertEqual(plan["rounds"], 2)
        self.assertEqual(plan["pairs"], 10)
        self.assertEqual(plan["threshold_percent"], 5.0)
        self.assertEqual(
            plan["order"], [["baseline", "candidate"], ["candidate", "baseline"]] * 5
        )

    def test_preflight_rejects_wsl_affinity_substitutes_and_wrong_cpu_count(self):
        from scripts.dal301_groupby import contract

        valid = {
            "system": "Linux",
            "release": "6.8-generic",
            "cpus": 4,
            "affinity": [0, 1, 2, 3],
            "runner_environment": "github-hosted",
        }
        contract.require_host(valid)
        for patch in (
            {"release": "microsoft-standard-WSL2"},
            {"cpus": 32},
            {"affinity": [0, 1]},
            {"runner_environment": "self-hosted"},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                contract.require_host({**valid, **patch})

    def test_release_claim_cannot_substitute_current_pr_or_another_native(self):
        from scripts.dal301_groupby import contract

        original = contract.SEALS["A"]
        with mock.patch.object(contract, "load_release", return_value=original):
            self.assertEqual(contract.sealed("A", "unused"), original)
        for patch in (
            {"git_sha": "1" * 40},
            {"native_sha256": "f" * 64},
            {"wheel_sha256": "e" * 64},
        ):
            with (
                mock.patch.object(
                    contract, "load_release", return_value={**original, **patch}
                ),
                self.assertRaises(ValueError),
            ):
                contract.sealed("A", "unused")


class WorkflowTests(unittest.TestCase):
    def test_failed_manual_command_retains_both_streams_and_original_exit(self):
        import subprocess  # nosec B404  # fixed workflow snippet and local fixture

        source = (
            Path(__file__).resolve().parents[1] / ".github/workflows/benchmarks.yml"
        ).read_text()
        for name, directory in (
            ("Build separately identified profile wheel", "dal301-profile"),
            (
                "Run only fixed group_by controls, comparison and separate profiles",
                "dal301-evidence",
            ),
        ):
            step = source.split(f"      - name: {name}\n", 1)[1].split("      - ", 1)[0]
            directive, script = step.split("        run: ", 1)[1].split("\n", 1)
            separator = " " if directive == ">-" else "\n"
            script = separator.join(line[10:] for line in script.splitlines())
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                python = root / "target/diagnostic-venv/bin/python"
                python.parent.mkdir(parents=True)
                python.write_text(
                    '#!/bin/bash\nprintf "out\\n"\nprintf "err\\n" >&2\nexit 42\n'
                )
                python.chmod(0o755)
                result = subprocess.run(  # nosec B603  # actual checked-in step, fixed fixture only
                    ["/bin/bash", "-e", "-o", "pipefail", "-c", script],
                    cwd=root,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 42)
                self.assertEqual(
                    (root / f"target/{directory}/stdout.log").read_text(), "out\n"
                )
                self.assertEqual(
                    (root / f"target/{directory}/stderr.log").read_text(), "err\n"
                )
                self.assertEqual(
                    (root / f"target/{directory}/exit-code.txt").read_text().strip(),
                    "42",
                )

    def test_manual_mode_gates_every_regular_matrix_and_always_keeps_evidence(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / ".github/workflows/benchmarks.yml"
        ).read_text()
        self.assertIn("dal301-groupby", source)
        self.assertEqual(source.count("inputs.mode != 'dal301-groupby'"), 2)
        self.assertIn("artifact-ids: 10636839917", source)
        self.assertIn("artifact-ids: 10637665772", source)
        self.assertIn("run-id: 35596885420", source)
        self.assertIn("name: dal301-evidence-", source)
        self.assertIn("python -m scripts.dal301_groupby run", source)


class FailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_command_keeps_original_signal_exit_and_full_log(self):
        from scripts.dal301_groupby import profile

        async def killed(*args, **kwargs):
            kwargs["log"].write_text("raw failure\n")
            raise RuntimeError("command exited -9")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(profile, "command", side_effect=killed),
                self.assertRaises(RuntimeError),
            ):
                await profile.logged(["perf", "--version"], root, "probe")
            self.assertEqual(
                json.loads((root / "probe.command.json").read_text())["exit_code"], -9
            )
            self.assertEqual((root / "probe.log").read_text(), "raw failure\n")

    async def test_profile_stops_at_fixed_window_without_expanding_workload(self):
        from scripts.dal301_groupby.profile import profile_workload

        worker = SimpleNamespace(
            request=mock.AsyncMock(
                return_value={"seconds": 1.0, "correctness": {"passed": True}}
            )
        )
        profiler = SimpleNamespace(done=mock.Mock(side_effect=[False, True]))
        self.assertEqual(await profile_workload(worker, profiler), 1)
        worker.request.assert_awaited_once_with(operation="sample")

    async def test_malformed_worker_stdout_and_nonzero_exit_are_retained(self):
        from scripts.dal301_groupby.runtime import AuditWorker

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = SimpleNamespace(
                pid=123,
                returncode=42,
                stdin=SimpleNamespace(write=mock.Mock(), drain=mock.AsyncMock()),
                stdout=SimpleNamespace(
                    readline=mock.AsyncMock(return_value=b"broken\n")
                ),
            )
            worker = AuditWorker(process, (root / "stderr.log").open("wb"))
            with self.assertRaises(ValueError):
                await worker.request(operation="hello")
            with self.assertRaisesRegex(RuntimeError, "42"):
                await worker.close()
            self.assertEqual((root / "stdout.log").read_bytes(), b"broken\n")
            self.assertEqual(
                json.loads((root / "exit.json").read_text())["exit_code"], 42
            )

    async def test_failed_profile_permission_stops_before_profile_samples(self):
        from scripts.dal301_groupby import profile

        worker = SimpleNamespace(request=mock.AsyncMock(), close=mock.AsyncMock())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "calc_flow").mkdir()
            (root / "calc_flow/_native.so").write_bytes(b"fixture")
            with (
                mock.patch.object(profile.AuditWorker, "start", return_value=worker),
                mock.patch.object(profile, "_prepare", new_callable=mock.AsyncMock),
                mock.patch.object(
                    profile,
                    "perf_preflight",
                    side_effect=ValueError("permission denied"),
                ),
                mock.patch.object(
                    profile, "record_profile", new_callable=mock.AsyncMock
                ) as record,
                self.assertRaisesRegex(ValueError, "permission denied"),
            ):
                await profile.profile_case({}, {}, root, root)
            record.assert_not_called()
            worker.close.assert_awaited_once()

    async def test_pairing_keeps_two_rounds_ten_pairs_and_rejects_input_drift(self):
        from scripts.dal301_groupby import controller
        from scripts.dal301_groupby.contract import cases

        events = []

        async def request(side, **message):
            events.append((side, message["operation"]))
            if message["operation"] == "sample":
                return {"seconds": 1.0, "correctness": {"passed": True}}
            return {"state": "completed"}

        workers = {
            side: SimpleNamespace(
                input_hashes={"table": "same"},
                request=mock.AsyncMock(),
                close=mock.AsyncMock(),
            )
            for side in ("baseline", "candidate")
        }
        from functools import partial

        for side, worker in workers.items():
            worker.request.side_effect = partial(request, side)

        async def start(site, root):
            return workers[site]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(controller.AuditWorker, "start", side_effect=start),
                mock.patch.object(
                    controller, "_prepare", return_value={"fixture": True}
                ),
            ):
                result = await controller.paired(
                    cases()[0], {s: s for s in workers}, {}, root
                )
                samples = [side for side, operation in events if operation == "sample"]
                self.assertEqual(
                    samples, ["baseline", "candidate", "candidate", "baseline"] * 10
                )
                self.assertEqual(result["baseline"], [[1.0] * 10] * 2)
                self.assertEqual(len(result["evidence"]), 2)
                workers["baseline"].input_hashes = {"table": "different"}
                events.clear()
                with self.assertRaisesRegex(ValueError, "input IPC"):
                    await controller.round_(
                        cases()[0], {s: s for s in workers}, {}, root
                    )
                self.assertEqual(events, [])

    async def test_failed_preflight_writes_failure_and_never_starts_measurement(self):
        import json
        import tempfile
        from pathlib import Path

        from scripts.dal301_groupby import controller

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(controller, "host", return_value={}),
                mock.patch.object(
                    controller, "require_host", side_effect=ValueError("wrong host")
                ),
                mock.patch.object(
                    controller, "collect", new_callable=mock.AsyncMock
                ) as collect,
            ):
                code = await controller.run(root, root / "releases", None)
            self.assertEqual(code, 1)
            collect.assert_not_called()
            self.assertIn(
                "wrong host", json.loads((root / "outcome.json").read_text())["error"]
            )


class ProfileTests(unittest.TestCase):
    def test_profile_rejects_lost_throttled_or_unresolved_traces(self):
        from scripts.dal301_groupby.profile import validate_trace

        symbols = "calc_flow::operator::window::execute"
        validate_trace("PERF_RECORD_SAMPLE", symbols)
        for raw, decoded in (
            ("PERF_RECORD_LOST", symbols),
            ("PERF_RECORD_LOST_SAMPLES", symbols),
            ("PERF_RECORD_THROTTLE", symbols),
            ("PERF_RECORD_SAMPLE", "[unknown]"),
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_trace(raw, decoded)

    def test_profile_rejects_drift_in_declared_build_settings(self):
        from scripts.dal301_groupby.profile import (
            PROFILE_ENV,
            ROOT,
            validate_build_settings,
        )
        from scripts.toolkit import sha256_file

        valid = {
            "env": PROFILE_ENV,
            "rustc_release": "1.88.0",
            "dependency_lock": sha256_file(ROOT / "benchmarks/requirements.lock"),
        }
        validate_build_settings(valid)
        for field in valid:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_build_settings({**valid, field: "different"})

    def test_profile_manifest_cannot_claim_original_release_or_missing_symbols(self):
        from scripts.dal301_groupby.contract import SEALS
        from scripts.dal301_groupby.profile import validate_profile

        valid = {
            "contract": "dal301-profile-only-v1",
            "source_ref": SEALS["A"]["git_sha"],
            "native_sha256": "c" * 64,
            "build_id": "d" * 40,
            "symbols": True,
        }
        validate_profile("A", valid)
        for patch in (
            {"native_sha256": SEALS["A"]["native_sha256"]},
            {"symbols": False},
            {"build_id": ""},
            {"source_ref": "e" * 40},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                validate_profile("A", {**valid, **patch})


if __name__ == "__main__":
    unittest.main()
