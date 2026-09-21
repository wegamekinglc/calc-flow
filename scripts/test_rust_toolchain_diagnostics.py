from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "scripts/rust_toolchain_log.sh"


@unittest.skipUnless(os.name == "posix", "Rust core parity uses Linux bash")
class InstallLogTests(unittest.TestCase):
    def test_hook_does_not_hide_missing_rustup_from_action_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                ["/bin/bash", "-c", "command -v rustup"],
                env={**os.environ, "PATH": temporary, "BASH_ENV": str(HOOK)},
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)

    def test_logging_failure_and_non_install_commands_keep_original_status(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "rustup"
            fake.write_text(
                '#!/bin/bash\nprintf "diagnostic\\n"\nexit 37\n', encoding="utf-8"
            )
            fake.chmod(0o755)
            output = root / "not-a-directory"
            output.write_bytes(b"unchanged")
            env = {
                **os.environ,
                "PATH": f"{root}:{os.environ['PATH']}",
                "BASH_ENV": str(HOOK),
                "RUST_TOOLCHAIN_EVIDENCE": str(output),
            }
            for command in ("rustup toolchain install 1.88.0", "rustup --version"):
                with self.subTest(command=command):
                    result = subprocess.run(
                        ["bash", "-e", "-o", "pipefail", "-c", command],
                        env=env,
                        capture_output=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 37)
                    self.assertIn(b"diagnostic", result.stdout)
                    self.assertEqual(output.read_bytes(), b"unchanged")

    def test_install_exit_and_complete_output_survive_logging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake = root / "rustup"
            fake.write_text(
                '#!/bin/bash\nprintf "out\\n"\nprintf "err\\n" >&2\n'
                'exit "$FAKE_EXIT"\n',
                encoding="utf-8",
            )
            fake.chmod(0o755)
            for code in (0, 42):
                with self.subTest(code=code):
                    output = root / str(code)
                    env = {
                        **os.environ,
                        "PATH": f"{root}:{os.environ['PATH']}",
                        "BASH_ENV": str(HOOK),
                        "RUST_TOOLCHAIN_EVIDENCE": str(output),
                        "FAKE_EXIT": str(code),
                    }
                    result = subprocess.run(
                        [
                            "bash",
                            "-e",
                            "-o",
                            "pipefail",
                            "-c",
                            "rustup toolchain install 1.88.0",
                        ],
                        env=env,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, code)
                    self.assertTrue(
                        (output / "install.log").exists(),
                        "install log was not retained",
                    )
                    self.assertEqual(
                        (output / "install.log").read_bytes(), b"out\nerr\n"
                    )
                    self.assertEqual(
                        (output / "install-exit-code.txt").read_text().strip(),
                        str(code),
                    )
                    self.assertIn(
                        "rustup toolchain install 1.88.0",
                        (output / "install-command.txt").read_text(),
                    )


class SnapshotTests(unittest.TestCase):
    def test_snapshot_bounds_manifest_bytes_and_marks_missing_files(self) -> None:
        from scripts import rust_toolchain_diagnostics as diagnostics

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest"
            path.write_bytes(b"abcdefghijklmnop")
            with (
                mock.patch.object(diagnostics, "TEXT_LIMIT", 8),
                mock.patch.object(diagnostics, "HASH_LIMIT", 12),
            ):
                record = diagnostics.file_record(path, text=True)
            self.assertEqual(record["text"], "abcdefgh")
            self.assertTrue(record["truncated"])
            self.assertTrue(record["hash_limit_exceeded"])
            self.assertNotIn("sha256", record)
            self.assertIn("error", diagnostics.file_record(path / "absent"))

    @unittest.skipUnless(os.name == "posix", "Linux symlink ownership")
    def test_symlink_is_recorded_without_following_or_copying_contents(self) -> None:
        from scripts import rust_toolchain_diagnostics as diagnostics

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "binary"
            target.write_bytes(b"binary")
            link = root / "cargo-clippy"
            link.symlink_to(target)
            record = diagnostics.file_record(link, text=True)
            self.assertEqual(record["type"], "symlink")
            self.assertEqual(record["link_target"], str(target))
            self.assertNotIn("text", record)

    def test_command_errors_exit_codes_and_output_limits_are_evidence(self) -> None:
        from scripts import rust_toolchain_diagnostics as diagnostics

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            missing = diagnostics.command_result(
                [str(directory / "missing")], directory
            )
            self.assertIn("error", missing)
            with mock.patch.object(diagnostics, "TEXT_LIMIT", 4):
                failed = diagnostics.command_result(
                    [sys.executable, "-c", "print('bounded'); raise SystemExit(23)"],
                    directory,
                )
            self.assertEqual(failed["exit_code"], 23)
            self.assertEqual(failed["output"], "boun")
            self.assertTrue(failed["truncated"])

    def test_snapshot_preserves_toolchain_and_records_bounded_ownership_evidence(
        self,
    ) -> None:
        from scripts import rust_toolchain_diagnostics as diagnostics

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rustup_home = root / "rustup"
            toolchain = rustup_home / "toolchains/1.88.0-x86_64-unknown-linux-gnu"
            manifests = toolchain / "lib/rustlib"
            manifests.mkdir(parents=True)
            manifest = manifests / "manifest-clippy-preview-x86_64-unknown-linux-gnu"
            manifest.write_text("file:bin/cargo-clippy\n", encoding="utf-8")
            (manifests / "components").write_text("clippy-preview\n", encoding="utf-8")
            binary = toolchain / "bin/cargo-clippy"
            binary.parent.mkdir()
            binary.write_bytes(b"clippy binary")
            secret = rustup_home / "unrelated-cache"
            secret.write_text("not collected", encoding="utf-8")
            env = {
                "RUSTUP_HOME": str(rustup_home),
                "CARGO_HOME": str(root / "cargo"),
                "RUST_DIAGNOSTICS_HEAD": "head-source",
                "RUST_DIAGNOSTICS_BASE": "base-source",
                "RUST_DIAGNOSTICS_UV": json.dumps(
                    {"outcome": "success", "outputs": {"cache-hit": "true"}}
                ),
            }
            with (
                mock.patch.dict(os.environ, env),
                mock.patch.object(
                    diagnostics,
                    "command_result",
                    return_value={"exit_code": 0, "output": "fixture"},
                ),
            ):
                diagnostics.snapshot(root / "evidence", "before")
            result = json.loads((root / "evidence/before.json").read_text())
            files = {item["path"]: item for item in result["files"]}
            self.assertEqual(files[str(binary)]["size"], 13)
            self.assertEqual(files[str(binary)]["uid"], binary.stat().st_uid)
            self.assertEqual(files[str(binary)]["type"], "regular")
            self.assertIn("file:bin/cargo-clippy", files[str(manifest)]["text"])
            self.assertNotIn(str(secret), files)
            self.assertEqual(result["identity"]["head"], "head-source")
            self.assertEqual(result["identity"]["base"], "base-source")
            self.assertEqual(result["uv_restore"]["outcome"], "success")
            self.assertEqual(binary.read_bytes(), b"clippy binary")
            self.assertEqual(manifest.read_text(), "file:bin/cargo-clippy\n")


class WorkflowTests(unittest.TestCase):
    def test_failed_install_keeps_snapshots_exit_code_and_artifact_upload(self) -> None:
        workflow = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
        job = workflow.split("  rust-core:\n", 1)[1].split("  rust-coverage:\n", 1)[0]
        self.assertIn("scripts/rust_toolchain_diagnostics.py before", job)
        self.assertIn(
            "RUST_TOOLCHAIN_EVIDENCE: ${{ runner.temp }}/rust-toolchain-install", job
        )
        self.assertIn("path: ${{ runner.temp }}/rust-toolchain-install/", job)
        install = job.index(
            "uses: dtolnay/rust-toolchain@4e529fb27e59237866a6523e61ab248308c068b4"
        )
        self.assertLess(
            job.index("scripts/rust_toolchain_diagnostics.py before"), install
        )
        self.assertGreater(
            job.index("scripts/rust_toolchain_diagnostics.py after"), install
        )
        self.assertIn(
            "BASH_ENV: ${{ github.workspace }}/scripts/rust_toolchain_log.sh", job
        )
        self.assertIn("components: clippy,rustfmt", job)
        self.assertIn("if: ${{ always() }}", job)
        self.assertIn("toJSON(steps.rust_uv)", job)
        self.assertIn("toJSON(steps.rust_cache)", job)
        self.assertIn("toJSON(steps.rust_install)", job)
        self.assertIn(
            "rust-toolchain-install-${{ github.run_id }}-${{ github.run_attempt }}", job
        )
        self.assertIn(
            "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a", job
        )
        self.assertIn(
            "cargo clippy --workspace --all-targets --all-features -- -D warnings", job
        )
        install_step = job.split("      - id: rust_install\n", 1)[1].split(
            "      - name:", 1
        )[0]
        self.assertNotIn("continue-on-error", install_step)
        for name in (
            "Snapshot Rust installation after setup",
            "Record Rust installation and cache step results",
            "Upload Rust installation evidence",
        ):
            step = job.split(f"      - name: {name}\n", 1)[1].split("      - ", 1)[0]
            self.assertIn("if: ${{ always() }}", step)


if __name__ == "__main__":
    unittest.main()
