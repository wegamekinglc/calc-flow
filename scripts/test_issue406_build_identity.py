from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.benchmark_suite.issue406_build_identity import validate_build_harness


class BuildIdentityTests(unittest.TestCase):
    def fixture(self, root):
        source, execution = root / "build", root / "execution"
        name = "scripts/benchmark_suite/issue406_fresh_v3.py"
        for folder, contents in ((source, "before"), (execution, "after")):
            path = folder / name
            path.parent.mkdir(parents=True)
            path.write_text(contents)

        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        witness = root / "witness.json"
        witness.write_text(
            json.dumps(
                {
                    "schema": "calc-flow.issue406-execution-repair.v1",
                    "build_harness_git_sha": "a" * 40,
                    "execution_git_sha": "b" * 40,
                    "patch_sha256": hashlib.sha256(b"exact reviewed patch").hexdigest(),
                    "changed_files": {
                        name: {
                            "before_sha256": digest(source / name),
                            "after_sha256": digest(execution / name),
                        }
                    },
                }
            )
        )
        plan = {
            "harness_git_sha": "b" * 40,
            "build_harness_git_sha": "a" * 40,
            "build_harness_source": str(source),
            "execution_repair_witness": str(witness),
        }
        files = {str(p): digest(p) for p in (source / name, execution / name, witness)}

        def command(arguments, **kwargs):
            return (
                name
                if "ls-files" in arguments or "--name-only" in arguments
                else "exact reviewed patch"
            )

        return plan, files, execution, source, command

    def test_exact_sealed_controller_repair_preserves_original_build(self):
        with TemporaryDirectory() as directory:
            plan, files, execution, source, command = self.fixture(Path(directory))
            with (
                patch(
                    "scripts.benchmark_suite.issue406_build_identity.ROOT", execution
                ),
                patch(
                    "scripts.benchmark_suite.issue406_build_identity.command_output",
                    side_effect=command,
                ),
                patch(
                    "scripts.benchmark_suite.issue406_build_identity._clean_revision",
                    return_value="a" * 40,
                ),
            ):
                validate_build_harness(plan, files)
                for name in (
                    str(source / "scripts/benchmark_suite/issue406_fresh_v3.py"),
                    plan["execution_repair_witness"],
                ):
                    with self.subTest(missing=name), self.assertRaises(ValueError):
                        validate_build_harness(
                            plan, {k: v for k, v in files.items() if k != name}
                        )
                with self.assertRaises(ValueError):
                    validate_build_harness(
                        {**plan, "build_harness_git_sha": "c" * 40}, files
                    )
                with (
                    patch(
                        "scripts.benchmark_suite.issue406_build_identity._clean_revision",
                        side_effect=ValueError("dirty"),
                    ),
                    self.assertRaises(ValueError),
                ):
                    validate_build_harness(plan, files)

    def test_altered_witness_or_product_path_is_rejected(self):
        with TemporaryDirectory() as directory:
            plan, files, execution, _, command = self.fixture(Path(directory))
            with (
                patch(
                    "scripts.benchmark_suite.issue406_build_identity.ROOT", execution
                ),
                patch(
                    "scripts.benchmark_suite.issue406_build_identity.command_output",
                    side_effect=command,
                ),
                patch(
                    "scripts.benchmark_suite.issue406_build_identity._clean_revision",
                    return_value="a" * 40,
                ),
            ):
                path = Path(plan["execution_repair_witness"])
                data = json.loads(path.read_text())
                path.write_text(json.dumps({**data, "patch_sha256": "f" * 64}))
                files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    validate_build_harness(plan, files)
                with (
                    patch(
                        "scripts.benchmark_suite.issue406_build_identity.command_output",
                        side_effect=lambda args, **kw: (
                            command(args, **kw)
                            if "ls-files" in args
                            else "crates/calc-flow/src/lib.rs"
                        ),
                    ),
                    self.assertRaises(ValueError),
                ):
                    validate_build_harness(plan, files)

    def test_unsplit_identity_needs_no_reuse_witness(self):
        validate_build_harness({"harness_git_sha": "a" * 40}, {})


if __name__ == "__main__":
    unittest.main()
