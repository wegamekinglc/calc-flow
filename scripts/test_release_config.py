from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseConfigTests(unittest.TestCase):
    def test_release_versions_are_final_and_aligned(self) -> None:
        workspace = tomllib.loads((ROOT / "Cargo.toml").read_text())
        binding = tomllib.loads(
            (ROOT / "crates/calc-flow-python/Cargo.toml").read_text()
        )
        package = tomllib.loads((ROOT / "pyproject.toml").read_text())
        studio = tomllib.loads((ROOT / "web-ui/backend/pyproject.toml").read_text())
        frontend = json.loads((ROOT / "web-ui/package.json").read_text())
        frontend_lock = json.loads((ROOT / "web-ui/package-lock.json").read_text())

        self.assertEqual(workspace["workspace"]["package"]["version"], "2.0.0")
        self.assertEqual(binding["dependencies"]["calc-flow"]["version"], "=2.0.0")
        self.assertEqual(package["project"]["version"], "2.0.0")
        self.assertEqual(studio["project"]["version"], "2.0.0")
        self.assertIn("calc-flow>=2.0.0,<3", studio["project"]["dependencies"])
        self.assertEqual(frontend["version"], "2.0.0")
        self.assertEqual(frontend_lock["version"], "2.0.0")
        self.assertEqual(frontend_lock["packages"][""]["version"], "2.0.0")
        self.assertIn(
            '__version__ = "2.0.0"',
            (ROOT / "python/calc_flow/__init__.py").read_text(),
        )
        self.assertIn(
            'version="2.0.0"',
            (ROOT / "web-ui/backend/src/calc_flow_studio/app.py").read_text(),
        )
        openapi = json.loads((ROOT / "web-ui/openapi.json").read_text())
        self.assertEqual(openapi["info"]["version"], "2.0.0")

        release_text = "\n".join(
            (ROOT / path).read_text()
            for path in (".github/workflows/ci.yml", ".github/workflows/release.yml")
        )
        self.assertNotIn(">=2.0.0a1", release_text)
        self.assertIn(">=2.0.0", release_text)

    def test_python_projects_ship_license_files(self) -> None:
        for project in (ROOT, ROOT / "web-ui/backend"):
            with self.subTest(project=project):
                config = tomllib.loads((project / "pyproject.toml").read_text())
                self.assertEqual(config["project"]["license-files"], ["LICENSE"])
                license_text = (project / "LICENSE").read_text()
                self.assertIn("Apache License", license_text)
                self.assertIn("Version 2.0, January 2004", license_text)

    def test_crate_excludes_integration_tests_and_ships_license(self) -> None:
        crate = ROOT / "crates/calc-flow"
        config = tomllib.loads((crate / "Cargo.toml").read_text())
        self.assertIn("tests/**", config["package"]["exclude"])
        license_text = (crate / "LICENSE").read_text()
        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0, January 2004", license_text)

    def test_release_maturin_actions_pin_tool_and_rust_versions(self) -> None:
        workflow = (ROOT / ".github/workflows/release.yml").read_text()
        action_count = workflow.count("uses: PyO3/maturin-action@")
        self.assertEqual(action_count, 3)
        self.assertEqual(workflow.count("maturin-version: v1.14.1"), action_count)
        self.assertEqual(workflow.count('rust-toolchain: "1.88.0"'), action_count)

    def test_rust_core_ci_sets_python_313_before_all_features(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        rust_core = workflow.split("  rust-core:\n", 1)[1].split(
            "  rust-supply-chain:\n", 1
        )[0]

        setup_python = (
            "uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1"
        )
        install_pyarrow = 'python -m pip install "pyarrow==24.0.0"'
        self.assertIn(setup_python, rust_core)
        self.assertIn("python-version-file: .python-version", rust_core)
        self.assertIn(install_pyarrow, rust_core)
        self.assertIn("RUST_TEST_THREADS: 1", rust_core)
        self.assertLess(
            rust_core.index(setup_python),
            rust_core.index("cargo clippy --workspace --all-targets --all-features"),
        )
        self.assertLess(
            rust_core.index(install_pyarrow),
            rust_core.index("cargo test --workspace --all-targets --all-features"),
        )

    def test_python_package_excludes_unsupported_pyarrow_25(self) -> None:
        for project in (ROOT, ROOT / "web-ui/backend"):
            with self.subTest(project=project):
                package = tomllib.loads((project / "pyproject.toml").read_text())
                self.assertIn("pyarrow>=24.0.0,<25", package["project"]["dependencies"])

    def test_final_release_error_docs_do_not_claim_alpha_status(self) -> None:
        stale_claims = {
            "crates/calc-flow/src/error.rs": (
                "Public v2-alpha error surface",
                "while the Rust API is alpha",
            ),
            "crates/calc-flow-python/src/error.rs": ("non-exhaustive during alpha",),
        }

        for path, claims in stale_claims.items():
            text = (ROOT / path).read_text()
            with self.subTest(path=path):
                for claim in claims:
                    self.assertNotIn(claim, text)

    def test_batch_metadata_docs_match_runtime_contract(self) -> None:
        introduction = (ROOT / "docs/introduction.md").read_text()
        self.assertIn(
            "metadata contains a source identifier, non-negative sequence, and\n"
            "  JSON-compatible attributes.",
            introduction,
        )

        rust_api = (ROOT / "docs/rust-api.md").read_text()
        self.assertIn(
            "`BatchMetadata::new(source, sequence, attributes)` validates the source "
            "and stores\n"
            "JSON-compatible attributes. Its sequence is descriptive batch metadata.\n"
            "`MicroBatchRunner` checkpoint ordering uses `SourceItem.sequence`, while\n"
            "`StreamingRunner` maintains its own sequence counter; both are distinct "
            "from\n"
            "`BatchMetadata.sequence`.",
            rust_api,
        )


if __name__ == "__main__":
    unittest.main()
