from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_suite.frontend import dependency_metadata
from scripts.benchmark_suite.legacy import _frontend_run, block_problem, measure_legacy


def frontend_lock(version: str = "4.0.0") -> dict:
    return {
        "name": "calc-flow-web-ui",
        "version": version,
        "lockfileVersion": 3,
        "requires": True,
        "packages": {
            "": {
                "name": "calc-flow-web-ui",
                "version": version,
                "dependencies": {"react": "19.2.7"},
                "devDependencies": {"vitest": "4.1.10"},
            },
            "node_modules/vitest": {
                "version": "4.1.10",
                "resolved": "https://registry.npmjs.org/vitest/-/vitest-4.1.10.tgz",
                "integrity": "sha512-fixture",
                "dev": True,
                "dependencies": {"vite": "8.1.4"},
            },
        },
    }


class FrontendFingerprintTests(unittest.TestCase):
    def test_dependency_changes_remain_part_of_identity(self):
        original = frontend_lock()
        baseline = dependency_metadata(json.dumps(original).encode())
        mutations = [
            (("name",), "renamed-app"),
            (("requires",), False),
            (("packages", "", "dependencies", "react"), "20.0.0"),
            (("packages", "", "devDependencies", "vitest"), "5.0.0"),
            (("packages", "node_modules/vitest", "version"), "5.0.0"),
            (
                ("packages", "node_modules/vitest", "resolved"),
                "https://example.invalid",
            ),
            (("packages", "node_modules/vitest", "integrity"), "sha512-changed"),
            (("packages", "node_modules/vitest", "dependencies", "vite"), "9.0.0"),
            (("packages", "node_modules/vitest", "engines"), {"node": ">=99"}),
            (("packages", "node_modules/vitest", "optional"), True),
            (("packages", "node_modules/vitest", "os"), ["linux"]),
            (("packages", "node_modules/new"), {"version": "1.0.0"}),
        ]
        for path, value in mutations:
            changed = copy.deepcopy(original)
            cursor = changed
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            with self.subTest(path=path):
                self.assertNotEqual(
                    dependency_metadata(json.dumps(changed).encode())[
                        "dependency_fingerprint"
                    ],
                    baseline["dependency_fingerprint"],
                )
        removed = frontend_lock()
        del removed["packages"]["node_modules/vitest"]
        self.assertNotEqual(
            dependency_metadata(json.dumps(removed).encode())["dependency_fingerprint"],
            baseline["dependency_fingerprint"],
        )
        self.assertEqual(original, frontend_lock())

    def test_encoding_is_deterministic_and_protocol_is_hashed(self):
        raw = json.dumps(frontend_lock()).encode()
        baseline = dependency_metadata(raw)
        reordered = json.dumps(frontend_lock(), sort_keys=True, indent=4).encode()
        self.assertEqual(
            dependency_metadata(reordered)["dependency_fingerprint"],
            baseline["dependency_fingerprint"],
        )
        self.assertNotEqual(
            dependency_metadata(reordered)["package_lock_sha256"],
            baseline["package_lock_sha256"],
        )
        with patch(
            "scripts.benchmark_suite.frontend.FINGERPRINT_PROTOCOL", "future-v2"
        ):
            self.assertNotEqual(
                dependency_metadata(raw)["dependency_fingerprint"],
                baseline["dependency_fingerprint"],
            )

    def test_invalid_lockfile_structures_fail_explicitly(self):
        valid = frontend_lock()
        invalid = [
            None,
            [],
            {},
            *[{**valid, "lockfileVersion": value} for value in (None, 2, 4, True, 3.0)],
            *[{**valid, "version": value} for value in (None, 5, "")],
            *[{**valid, "packages": value} for value in (None, [], {}, {"": []})],
            {**valid, "packages": {"": {}}},
            {**valid, "packages": {"": {"version": 5}}},
            {**valid, "packages": {"": {"version": ""}}},
            {**valid, "packages": {**valid["packages"], "node_modules/bad": []}},
            {key: value for key, value in valid.items() if key != "version"},
            {**valid, "extra": float("nan")},
        ]
        raw_values = [json.dumps(value).encode() for value in invalid]
        raw_values.extend(
            [
                b"not-json",
                b"\xff",
                (json.dumps(valid)[:-1] + ', "version": "5.0.0"}').encode(),
            ]
        )
        for raw in raw_values:
            with (
                self.subTest(raw=raw),
                self.assertRaisesRegex(ValueError, "lockfile|JSON"),
            ):
                dependency_metadata(raw)

    def test_root_versions_do_not_change_identity_but_keep_raw_provenance(self):
        original = frontend_lock()
        raw = json.dumps(original).encode()
        metadata = dependency_metadata(raw)
        for location in ((), ("packages", "")):
            changed = frontend_lock()
            cursor = changed
            for key in location:
                cursor = cursor[key]
            cursor["version"] = "5.0.0"
            changed_raw = json.dumps(changed).encode()
            with self.subTest(location=location):
                result = dependency_metadata(changed_raw)
                self.assertEqual(
                    result["dependency_fingerprint"], metadata["dependency_fingerprint"]
                )
                self.assertNotEqual(
                    result["package_lock_sha256"], metadata["package_lock_sha256"]
                )
                self.assertEqual(
                    result["package_lock_sha256"],
                    hashlib.sha256(changed_raw).hexdigest(),
                )
                self.assertEqual(result["project_version"], changed["version"])
                self.assertEqual(
                    result["root_package_version"], changed["packages"][""]["version"]
                )
        self.assertEqual(
            metadata["dependency_fingerprint_protocol"], "frontend-npm-lock-v1"
        )
        self.assertEqual(raw, json.dumps(original).encode())


def frontend_blocks(baseline: dict, candidate: dict) -> dict:
    return {
        side: [
            {
                "case": {
                    "rows": None,
                    "scope": "vitest-native-boundary",
                    "metadata": metadata,
                }
            }
            for _ in range(2)
        ]
        for side, metadata in (("baseline", baseline), ("candidate", candidate))
    }


class FrontendBlockTests(unittest.TestCase):
    def test_blocks_reject_missing_unknown_and_mixed_protocols(self):
        metadata = dependency_metadata(json.dumps(frontend_lock()).encode())
        invalid = [
            {
                key: value
                for key, value in metadata.items()
                if key != "dependency_fingerprint_protocol"
            },
            {**metadata, "dependency_fingerprint_protocol": "frontend-npm-lock-v2"},
            {"dependency_fingerprint": metadata["package_lock_sha256"]},
        ]
        for candidate in invalid:
            for baseline in (metadata, candidate):
                with self.subTest(baseline=baseline, candidate=candidate):
                    problem = block_problem(
                        "case", frontend_blocks(baseline, candidate)
                    )
                    self.assertIsNotNone(problem)
                    self.assertIn("fingerprint protocol", problem)

    def test_blocks_require_identity_and_raw_provenance_on_both_sides(self):
        metadata = dependency_metadata(json.dumps(frontend_lock()).encode())
        for key in (
            "dependency_fingerprint",
            "package_lock_sha256",
            "project_version",
            "root_package_version",
        ):
            for value in (
                None,
                "",
                "invalid" if "fingerprint" in key or "sha256" in key else 5,
            ):
                invalid = {**metadata, key: value}
                for baseline, candidate in (
                    (metadata, invalid),
                    (invalid, metadata),
                    (invalid, invalid),
                ):
                    with self.subTest(key=key, value=value):
                        problem = block_problem(
                            "case", frontend_blocks(baseline, candidate)
                        )
                        self.assertIsNotNone(problem)
                        self.assertIn("provenance", problem)

    def test_blocks_allow_root_versions_but_reject_dependency_drift_and_missing_blocks(
        self,
    ):
        baseline = dependency_metadata(json.dumps(frontend_lock()).encode())
        candidate = dependency_metadata(json.dumps(frontend_lock("5.0.0")).encode())
        self.assertIsNone(block_problem("case", frontend_blocks(baseline, candidate)))
        changed = frontend_lock("5.0.0")
        changed["packages"]["node_modules/vitest"]["version"] = "5.0.0"
        drift = dependency_metadata(json.dumps(changed).encode())
        self.assertIn(
            "dependency_fingerprint changed",
            block_problem("case", frontend_blocks(baseline, drift)),
        )
        incomplete = frontend_blocks(baseline, candidate)
        incomplete["candidate"].pop()
        self.assertIn("confirmation block", block_problem("case", incomplete))


def write_frontend_checkout(root: Path, version: str) -> None:
    frontend = root / "web-ui"
    frontend.mkdir(parents=True)
    (frontend / "package-lock.json").write_text(json.dumps(frontend_lock(version)))
    generated = root / "target/benchmark-suite/vitest.json"
    generated.parent.mkdir(parents=True)
    generated.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "groups": [
                            {
                                "fullName": "compare",
                                "benchmarks": [
                                    {
                                        "name": f"compare/{size}_cases",
                                        "samples": [1.0, 2.0],
                                    }
                                    for size in (100, 1000)
                                ],
                            }
                        ]
                    }
                ]
            }
        )
    )


class BenchmarkFrontendTests(unittest.IsolatedAsyncioTestCase):
    async def test_same_candidate_harness_records_both_versions_and_raw_locks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, candidate = root / "baseline", root / "candidate"
            write_frontend_checkout(baseline, "4.0.0")
            write_frontend_checkout(candidate, "5.0.0")
            controller = candidate / "scripts/benchmark_suite/frontend.mjs"
            controller.parent.mkdir(parents=True)
            controller.write_text("// candidate harness fixture\n")
            output = root / "output"
            output.mkdir()
            with (
                patch("scripts.benchmark_suite.legacy.ROOT", candidate),
                patch("scripts.benchmark_suite.legacy.validate_sources"),
                patch("scripts.benchmark_suite.legacy._setup", return_value={}),
                patch(
                    "scripts.benchmark_suite.legacy.observe_workers", return_value={}
                ),
                patch("scripts.benchmark_suite.legacy.command") as command,
            ):
                report = await measure_legacy(
                    {"id": "frontend", "family": "frontend"}, {}, output, baseline
                )
            self.assertEqual(report["errors"], [])
            self.assertEqual(len(report["cases"]), 2)
            for case in report["cases"]:
                self.assertEqual(case["status"], "ok", case)
                self.assertEqual(case["comparison"], "suite-blocks")
                self.assertEqual(case["result"]["verdict"], "informational")
            self.assertEqual(
                [call.kwargs["cwd"] for call in command.call_args_list],
                [
                    side / "web-ui"
                    for side in (baseline, candidate, candidate, baseline)
                ],
            )
            blocks = json.loads((output / "blocks.json").read_text())
            fingerprints = set()
            raw_hashes = set()
            for side, source, version in (
                ("baseline", baseline, "4.0.0"),
                ("candidate", candidate, "5.0.0"),
            ):
                runner = source / "web-ui/node_modules/.cache/calc-flow-benchmark.mjs"
                self.assertEqual(runner.read_bytes(), controller.read_bytes())
                self.assertEqual(len(blocks[side]), 2)
                raw = (source / "web-ui/package-lock.json").read_bytes()
                for block in blocks[side]:
                    for row in block.values():
                        metadata = row["metadata"]
                        self.assertEqual(metadata["group"], "compare")
                        self.assertEqual(
                            metadata["dependency_fingerprint_protocol"],
                            "frontend-npm-lock-v1",
                        )
                        self.assertEqual(metadata["project_version"], version)
                        self.assertEqual(metadata["root_package_version"], version)
                        self.assertEqual(
                            metadata["package_lock_sha256"],
                            hashlib.sha256(raw).hexdigest(),
                        )
                        fingerprints.add(metadata["dependency_fingerprint"])
                        raw_hashes.add(metadata["package_lock_sha256"])
            self.assertEqual(len(fingerprints), 1)
            self.assertEqual(len(raw_hashes), 2)

    async def test_invalid_lock_fails_before_starting_frontend_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            write_frontend_checkout(source, "5.0.0")
            (source / "web-ui/package-lock.json").write_text("{}")
            output = source / "output"
            output.mkdir()
            with (
                patch("scripts.benchmark_suite.legacy.command") as command,
                self.assertRaisesRegex(ValueError, "lockfile"),
            ):
                await _frontend_run(source, output)
            command.assert_not_awaited()

    async def test_failed_legacy_blocks_retain_each_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("scripts.benchmark_suite.legacy.validate_sources"),
                patch("scripts.benchmark_suite.legacy._setup", return_value={}),
                patch(
                    "scripts.benchmark_suite.legacy.observe_workers", return_value={}
                ),
                patch(
                    "scripts.benchmark_suite.legacy._frontend_run",
                    side_effect=RuntimeError("missing benchmark dependency"),
                ),
            ):
                report = await measure_legacy(
                    {"id": "frontend", "family": "frontend"}, {}, root, root
                )
            self.assertEqual(report["cases"], [])
            self.assertEqual(report["expected_case_ids"], [])
            self.assertEqual(
                sum(
                    "missing benchmark dependency" in error
                    for error in report["errors"]
                ),
                4,
            )

    async def test_uses_checkout_local_static_runner_and_archives_its_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            frontend = source / "web-ui"
            generated = source / "target/benchmark-suite/vitest.json"
            generated.parent.mkdir(parents=True)
            frontend.mkdir()
            generated.write_text('{"files": []}', encoding="utf-8")
            (frontend / "package-lock.json").write_text(
                json.dumps(frontend_lock()), encoding="utf-8"
            )
            output = source / "archived"
            output.mkdir()
            with (
                patch("scripts.benchmark_suite.legacy.command") as run,
                patch("scripts.benchmark_suite.legacy.vitest_rows", return_value={}),
            ):
                await _frontend_run(source, output)
            self.assertEqual(run.call_args.kwargs["cwd"], frontend)
            argv = run.call_args.args[0]
            self.assertEqual(len(argv), 2)
            runner = Path(argv[1])
            self.assertTrue(runner.is_relative_to(frontend / "node_modules"))
            self.assertIn('from "vitest/node"', runner.read_text(encoding="utf-8"))
            self.assertEqual(
                (output / "vitest.json").read_bytes(), generated.read_bytes()
            )
