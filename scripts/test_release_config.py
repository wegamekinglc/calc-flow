from __future__ import annotations

import ast
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
        rust_test_command = (
            "python3.13 scripts/run_rust_tests.py --python-stress-runs 3"
        )

        setup_python = (
            "uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1"
        )
        install_test_dependencies = (
            'python -m pip install "numpy>=2.0.0" "pyarrow==24.0.0"'
        )
        self.assertIn(setup_python, rust_core)
        self.assertIn("python-version-file: .python-version", rust_core)
        self.assertIn(install_test_dependencies, rust_core)
        rust_core_header = rust_core.split("    steps:\n", 1)[0]
        self.assertIn(
            "    env:\n      RUST_TEST_THREADS: 1\n",
            rust_core_header,
        )
        self.assertLess(
            rust_core.index(setup_python),
            rust_core.index("cargo clippy --workspace --all-targets --all-features"),
        )
        self.assertLess(
            rust_core.index(install_test_dependencies),
            rust_core.index(rust_test_command),
        )
        rust_test_step = rust_core.split("      - name: Run Rust tests\n", 1)[1].split(
            "      - name:", 1
        )[0]
        self.assertIn("timeout-minutes: 30", rust_test_step)
        self.assertIn(f"run: {rust_test_command}", rust_test_step)

    def test_rust_core_ci_reclaims_disk_around_coverage(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        rust_core = workflow.split("  rust-core:\n", 1)[1].split(
            "  rust-supply-chain:\n", 1
        )[0]

        rust_tests = "python3.13 scripts/run_rust_tests.py --python-stress-runs 3"
        clean_tests = "cargo clean"
        coverage = "cargo llvm-cov --workspace --all-features --fail-under-lines 90"
        clean_coverage = "cargo llvm-cov clean --workspace"
        rustdoc = (
            'RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps'
        )

        self.assertLess(rust_core.index(rust_tests), rust_core.index(clean_tests))
        self.assertLess(rust_core.index(clean_tests), rust_core.index(coverage))
        self.assertLess(
            rust_core.index(coverage),
            rust_core.index(clean_coverage),
        )
        self.assertLess(
            rust_core.index(clean_coverage),
            rust_core.index(rustdoc),
        )

    def test_benchmark_smoke_runs_every_supported_scale(self) -> None:
        support_tree = ast.parse((ROOT / "benchmarks/support.py").read_text())
        scales_assignment = next(
            node
            for node in support_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "SCALES"
                for target in node.targets
            )
        )
        self.assertIsInstance(scales_assignment.value, ast.Dict)
        scales = [ast.literal_eval(key) for key in scales_assignment.value.keys]

        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        benchmark_job = workflow.split("  benchmark-smoke:\n", 1)[1].split(
            "  rust-core:\n", 1
        )[0]

        self.assertEqual(scales, ["overhead", "small", "standard", "nightly"])
        self.assertIn("fail-fast: false", benchmark_job)
        self.assertIn(f"scale: [{', '.join(scales)}]", benchmark_job)
        self.assertIn("CALC_FLOW_BENCHMARK_SCALE: ${{ matrix.scale }}", benchmark_job)
        self.assertIn("JAX_PLATFORMS: cpu", benchmark_job)
        self.assertIn(
            '--benchmark-json="benchmark-results/${CALC_FLOW_BENCHMARK_SCALE}.json"',
            benchmark_job,
        )
        self.assertIn("name: benchmark-smoke-${{ matrix.scale }}", benchmark_job)
        self.assertIn("path: benchmark-results/${{ matrix.scale }}.json", benchmark_job)

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

    def test_getting_started_covers_packages_source_and_studio_platforms(
        self,
    ) -> None:
        guide_path = ROOT / "docs/getting-started.md"
        self.assertTrue(guide_path.is_file())
        guide = guide_path.read_text()

        for heading in (
            "## Choose an installation path",
            "## Prerequisites",
            "## Install published packages",
            "## Build and install from source",
            "## Start and stop Studio",
            "## Verify the installation",
            "## Troubleshooting",
            "### Linux and WSL",
            "### Windows PowerShell",
        ):
            self.assertIn(heading, guide)

        for command in (
            "uv tool install calc-flow-studio",
            "cargo build --workspace --all-features --release",
            "maturin==1.14.1",
            "UV_TOOL_DIR",
            "-delete",
            "Remove-Item",
            "uv run --no-sync --package calc-flow-studio calc-flow-web",
            "./web-ui/scripts/start_web_ui.sh",
            r".\web-ui\scripts\start_web_ui.ps1",
        ):
            self.assertIn(command, guide)

        readme = (ROOT / "README.md").read_text()
        self.assertIn("[getting started](docs/getting-started.md)", readme)


if __name__ == "__main__":
    unittest.main()
