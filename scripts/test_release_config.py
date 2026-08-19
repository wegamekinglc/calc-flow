from __future__ import annotations

import ast
import json
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _is_read_text_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_text"
    )


def _has_literal_utf8_encoding(node: ast.Call) -> bool:
    encoding = next(
        (keyword.value for keyword in node.keywords if keyword.arg == "encoding"),
        None,
    )
    return (
        isinstance(encoding, ast.Constant)
        and isinstance(encoding.value, str)
        and encoding.value.lower() == "utf-8"
    )


def _non_utf8_read_text_calls(tree: ast.AST) -> list[int]:
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if _is_read_text_call(node)
        and isinstance(node, ast.Call)
        and not _has_literal_utf8_encoding(node)
    )


class ReleaseConfigTests(unittest.TestCase):
    def test_text_read_guard_rejects_non_utf8_encoding(self) -> None:
        tree = ast.parse('Path("fixture").read_text(encoding="latin-1")')
        self.assertEqual(_non_utf8_read_text_calls(tree), [1])

    def test_text_read_guard_sorts_multiple_violations(self) -> None:
        tree = ast.parse(
            'Path("first").read_text()\nPath("second").read_text(encoding="latin-1")'
        )
        tree.body.reverse()
        self.assertEqual(_non_utf8_read_text_calls(tree), [1, 2])

    def test_release_config_text_reads_pin_utf8(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertEqual(_non_utf8_read_text_calls(tree), [])

    def test_release_versions_are_final_and_aligned(self) -> None:
        workspace = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
        binding = tomllib.loads(
            (ROOT / "crates/calc-flow-python/Cargo.toml").read_text(encoding="utf-8")
        )
        package = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        studio = tomllib.loads(
            (ROOT / "web-ui/backend/pyproject.toml").read_text(encoding="utf-8")
        )
        frontend = json.loads(
            (ROOT / "web-ui/package.json").read_text(encoding="utf-8")
        )
        frontend_lock = json.loads(
            (ROOT / "web-ui/package-lock.json").read_text(encoding="utf-8")
        )

        self.assertEqual(workspace["workspace"]["package"]["version"], "3.0.0")
        self.assertEqual(binding["dependencies"]["calc-flow"]["version"], "=3.0.0")
        self.assertEqual(package["project"]["version"], "3.0.0")
        self.assertEqual(studio["project"]["version"], "3.0.0")
        self.assertIn("calc-flow>=3.0.0,<4", studio["project"]["dependencies"])
        self.assertEqual(frontend["version"], "3.0.0")
        self.assertEqual(frontend_lock["version"], "3.0.0")
        self.assertEqual(frontend_lock["packages"][""]["version"], "3.0.0")
        self.assertIn(
            '__version__ = "3.0.0"',
            (ROOT / "python/calc_flow/__init__.py").read_text(encoding="utf-8"),
        )
        self.assertIn(
            'version="3.0.0"',
            (ROOT / "web-ui/backend/src/calc_flow_studio/app.py").read_text(
                encoding="utf-8"
            ),
        )
        openapi = json.loads((ROOT / "web-ui/openapi.json").read_text(encoding="utf-8"))
        self.assertEqual(openapi["info"]["version"], "3.0.0")

        release_text = "\n".join(
            (ROOT / path).read_text(encoding="utf-8")
            for path in (
                ".github/workflows/ci-linux.yml",
                ".github/workflows/release.yml",
            )
        )
        self.assertNotIn(">=2.0.0a1", release_text)
        self.assertIn(">=3.0.0", release_text)

        release_workflow = (ROOT / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('- "v3.*"', release_workflow)
        self.assertNotIn('- "v2.*"', release_workflow)
        self.assertIn('assert "/api/v3/catalog"', release_workflow)
        self.assertIn('and ">=3.0.0" in requirement', release_workflow)
        self.assertIn('and "<4" in requirement', release_workflow)
        self.assertEqual(release_workflow.count("--save-baseline exact-"), 2)
        self.assertIn("--criterion-dir", release_workflow)
        self.assertIn("--criterion-baseline exact-baseline", release_workflow)
        self.assertIn("--criterion-candidate exact-candidate", release_workflow)
        self.assertEqual(release_workflow.count("provenance.json"), 3)
        self.assertIn('baseline_sha="$(git rev-parse', release_workflow)
        self.assertIn('candidate_sha="$(git rev-parse', release_workflow)
        self.assertIn('test "${baseline_sha}" != "${candidate_sha}"', release_workflow)
        self.assertLess(
            release_workflow.index("cargo bench --manifest-path"),
            release_workflow.index("scripts/verify_perf_gates.py"),
        )

    def test_python_projects_ship_license_files(self) -> None:
        for project in (ROOT, ROOT / "web-ui/backend"):
            with self.subTest(project=project):
                config = tomllib.loads(
                    (project / "pyproject.toml").read_text(encoding="utf-8")
                )
                self.assertEqual(config["project"]["license-files"], ["LICENSE"])
                license_text = (project / "LICENSE").read_text(encoding="utf-8")
                self.assertIn("Apache License", license_text)
                self.assertIn("Version 2.0, January 2004", license_text)

    def test_crate_excludes_integration_tests_and_ships_license(self) -> None:
        crate = ROOT / "crates/calc-flow"
        config = tomllib.loads((crate / "Cargo.toml").read_text(encoding="utf-8"))
        self.assertIn("tests/**", config["package"]["exclude"])
        license_text = (crate / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0, January 2004", license_text)

    def test_codacy_excludes_only_the_frozen_allocation_harness(self) -> None:
        config = (ROOT / ".codacy.yml").read_text(encoding="utf-8")
        harness_path = "crates/calc-flow/benches/allocation_regression.rs"
        frozen_harness = f'  - "{harness_path}"'

        self.assertEqual(config.count(frozen_harness), 1)
        self.assertNotIn('  - "crates/calc-flow/benches/**"', config)
        self.assertTrue((ROOT / harness_path).is_file())
        legacy_issue_slug = "_".join(("dal", "38"))
        self.assertFalse(
            (
                ROOT / f"crates/calc-flow/benches/{legacy_issue_slug}_allocation.rs"
            ).exists()
        )
        for path in (
            ".github/workflows/ci-linux.yml",
            "crates/calc-flow/Cargo.toml",
            harness_path,
        ):
            with self.subTest(path=path):
                self.assertNotRegex(
                    (ROOT / path).read_text(encoding="utf-8"),
                    r"(?i)dal(?:[_-]?38)",
                )

        harness = (ROOT / harness_path).read_text(encoding="utf-8")
        self.assertIn(
            'env::var("ALLOCATION_REGRESSION_BACKGROUND_LOAD_POLICY")',
            harness,
        )

    def test_release_maturin_actions_pin_tool_and_rust_versions(self) -> None:
        workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        action_count = workflow.count("uses: PyO3/maturin-action@")
        self.assertEqual(action_count, 3)
        self.assertEqual(workflow.count("maturin-version: v1.14.1"), action_count)
        self.assertEqual(workflow.count('rust-toolchain: "1.88.0"'), action_count)

    def test_workflow_actions_are_sha_pinned(self) -> None:
        for name in (
            "benchmarks.yml",
            "ci-linux.yml",
            "ci-windows.yml",
            "release.yml",
        ):
            workflow = (ROOT / ".github" / "workflows" / name).read_text(
                encoding="utf-8"
            )
            for line in workflow.splitlines():
                step = line.strip()
                if not step.startswith("uses:") or "@" not in step:
                    continue
                action, _, reference = step.removeprefix("uses:").strip().partition("@")
                reference = reference.split()[0]
                with self.subTest(workflow=name, action=action):
                    self.assertEqual(len(reference), 40)
                    self.assertTrue(
                        all(character in "0123456789abcdef" for character in reference)
                    )

    def test_rust_core_ci_sets_python_313_before_all_features(self) -> None:
        workflow = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
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
        self.assertIn("      RUST_TEST_THREADS: 1\n", rust_core_header)
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
        workflow = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
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

    def test_rust_core_ci_isolates_frozen_allocation_harness(self) -> None:
        workflow = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
        rust_core = workflow.split("  rust-core:\n", 1)[1].split(
            "  rust-supply-chain:\n", 1
        )[0]

        harness_sha = "fe34d7dcd5bfd66c9e97c79d540380f58ee1a04d"
        rust_test_harness = (
            "python3.13 scripts/run_rust_tests.py --python-stress-runs 3"
        )
        harness_self_test = (
            "cargo test --locked -p calc-flow --bench "
            "allocation_regression --all-features"
        )

        self.assertIn("fetch-depth: 0", rust_core)
        self.assertIn(f'FROZEN_ALLOCATION_HARNESS_SHA: "{harness_sha}"', rust_core)
        self.assertIn("id: frozen_allocation_harness", rust_core)
        self.assertIn(
            'git merge-base --is-ancestor "$FROZEN_ALLOCATION_HARNESS_SHA" HEAD',
            rust_core,
        )
        self.assertIn(
            "if: steps.frozen_allocation_harness.outputs.enabled == 'true'",
            rust_core,
        )
        self.assertIn(rust_test_harness, rust_core)
        self.assertIn(
            'git worktree add --detach "$FROZEN_ALLOCATION_HARNESS_WORKTREE" '
            '"$FROZEN_ALLOCATION_HARNESS_SHA"',
            rust_core,
        )
        self.assertIn(harness_self_test, rust_core)
        self.assertIn(
            "if: always() && steps.frozen_allocation_harness.outputs.enabled == 'true'",
            rust_core,
        )
        self.assertIn(
            'git worktree remove --force "$FROZEN_ALLOCATION_CANDIDATE_WORKTREE"',
            rust_core,
        )
        self.assertIn(
            'git worktree remove --force "$FROZEN_ALLOCATION_HARNESS_WORKTREE"',
            rust_core,
        )
        self.assertIn("git worktree prune", rust_core)
        self.assertIn(
            "github.event.pull_request.base.sha == env.FROZEN_ALLOCATION_HARNESS_SHA",
            rust_core,
        )
        self.assertIn(
            "FROZEN_ALLOCATION_CANDIDATE_SHA: "
            "${{ github.event.pull_request.head.sha }}",
            rust_core,
        )
        self.assertIn("--role baseline", rust_core)
        self.assertIn("--role candidate", rust_core)
        self.assertIn(
            '--compare "$FROZEN_ALLOCATION_BASELINE_REPORT" '
            '"$FROZEN_ALLOCATION_CANDIDATE_REPORT"',
            rust_core,
        )
        self.assertLess(
            rust_core.index(rust_test_harness), rust_core.index(harness_self_test)
        )
        self.assertLess(
            rust_core.index(harness_self_test), rust_core.index("--role baseline")
        )

    def test_ci_and_release_execute_rust_test_harness_unit_tests(self) -> None:
        command = (
            "python -m unittest scripts.test_run_rust_tests "
            "scripts.test_inspect_wheel scripts.test_release_config "
            "scripts.test_verify_perf_gates scripts.test_verify_security_gates"
        )
        windows_test = (
            "scripts.test_run_rust_tests.RustTestHarnessTests."
            "test_timeout_cleans_up_the_test_binary_process_tree_on_windows"
        )

        for path in (
            ".github/workflows/ci-linux.yml",
            ".github/workflows/release.yml",
        ):
            with self.subTest(path=path):
                workflow = (ROOT / path).read_text(encoding="utf-8")
                self.assertIn(command, workflow)

        windows_ci = (ROOT / ".github/workflows/ci-windows.yml").read_text(
            encoding="utf-8"
        )
        runner_tests = (ROOT / "scripts/test_run_rust_tests.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("name: Windows CI", windows_ci)
        self.assertIn("WINDOWS_PROCESS_TREE_EVIDENCE", runner_tests)
        windows_job = windows_ci.split("  process-tree:\n", 1)[1].split(
            "  rust-tests:\n", 1
        )[0]
        self.assertIn("runs-on: windows-latest", windows_job)
        self.assertIn("timeout-minutes: 10", windows_job)
        self.assertIn("WINDOWS_RUNNER_EVIDENCE", windows_job)
        self.assertIn(windows_test, windows_job)

        rust_job = windows_ci.split("  rust-tests:\n", 1)[1].split(
            "  python-tests:\n", 1
        )[0]
        self.assertIn("runs-on: windows-latest", rust_job)
        self.assertIn('CARGO_PROFILE_DEV_DEBUG: "0"', rust_job)
        self.assertIn("python scripts/run_rust_tests.py", rust_job)
        # The harness serializes the embedded-Python lib test itself, so the
        # core crate must not run with RUST_TEST_THREADS forced to one.
        self.assertNotIn("RUST_TEST_THREADS", rust_job)

        python_job = windows_ci.split("  python-tests:\n", 1)[1]
        self.assertIn("runs-on: windows-latest", python_job)
        self.assertIn('CARGO_PROFILE_DEV_DEBUG: "0"', python_job)
        self.assertIn("uv run pytest", python_job)

    def test_agents_rust_runner_uses_synced_python_dependencies(self) -> None:
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        sync = "uv sync --extra dev"
        runner = "uv run python scripts/run_rust_tests.py"

        self.assertIn("NumPy and PyArrow", agents)
        self.assertIn(runner, agents)
        self.assertLess(agents.index(sync), agents.index(runner))

    def test_benchmark_smoke_runs_every_supported_scale(self) -> None:
        support_tree = ast.parse(
            (ROOT / "benchmarks/support.py").read_text(encoding="utf-8")
        )
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

        workflow = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
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

    def test_linux_ci_reports_parallel_coverage_to_coveralls(self) -> None:
        workflow = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")

        self.assertEqual(workflow.count("uses: coverallsapp/github-action@"), 4)
        self.assertEqual(workflow.count("parallel: true"), 3)
        self.assertEqual(workflow.count("format: cobertura"), 2)
        for flag, report in (
            ("python", "file: coverage.xml"),
            ("studio", "file: web-ui/backend/coverage.xml"),
            ("rust", "file: rust-lcov.info"),
        ):
            with self.subTest(flag=flag):
                self.assertIn(f"flag-name: {flag}", workflow)
                self.assertIn(report, workflow)

        finish = workflow.split("  coveralls-finish:\n", 1)[1]
        self.assertIn("- lint-and-test", finish)
        self.assertIn("- studio-backend", finish)
        self.assertIn("- rust-core", finish)
        self.assertIn("parallel-finished: true", finish)

    def test_scheduled_rust_benchmark_targets_only_core_harness(self) -> None:
        workflow = (ROOT / ".github/workflows/benchmarks.yml").read_text(
            encoding="utf-8"
        )
        rust_benchmark = workflow.split("  rust-benchmark:\n", 1)[1].split(
            "  benchmark:\n", 1
        )[0]

        cargo_bench_commands = [
            line.strip()
            for line in rust_benchmark.splitlines()
            if line.strip().startswith("run: cargo bench ")
        ]
        self.assertEqual(
            cargo_bench_commands,
            ["run: cargo bench --locked -p calc-flow --bench core"],
        )

    def test_python_package_excludes_unsupported_pyarrow_25(self) -> None:
        for project in (ROOT, ROOT / "web-ui/backend"):
            with self.subTest(project=project):
                package = tomllib.loads(
                    (project / "pyproject.toml").read_text(encoding="utf-8")
                )
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
            text = (ROOT / path).read_text(encoding="utf-8")
            with self.subTest(path=path):
                for claim in claims:
                    self.assertNotIn(claim, text)

    def test_normative_docs_use_final_package_and_project_versions(self) -> None:
        documentation = {
            "README.md": ("Calc Flow 3.0", 'calc-flow = "3.0.0"'),
            "docs/api-reference.md": (
                "Calc Flow 3.0 API reference",
                "`calc-flow==3.0.0`",
                "Project format version `3`",
            ),
            "docs/getting-started.md": ("cargo add calc-flow@3.0.0",),
            "docs/python-api.md": ("`calc-flow==3.0.0`",),
            "docs/rust-api.md": ("Calc Flow 3.0", "cargo add calc-flow@3.0.0"),
        }
        stale_package_claims = (
            "Calc Flow 2.0",
            'calc-flow = "2.0.0"',
            "calc-flow==2.0.0",
            "calc-flow@2.0.0",
        )

        for path, required in documentation.items():
            text = (ROOT / path).read_text(encoding="utf-8")
            with self.subTest(path=path):
                for claim in required:
                    self.assertIn(claim, text)
                for claim in stale_package_claims:
                    self.assertNotIn(claim, text)

    def test_batch_metadata_docs_match_runtime_contract(self) -> None:
        introduction = (ROOT / "docs/introduction.md").read_text(encoding="utf-8")
        self.assertIn(
            "metadata contains a source identifier, non-negative sequence, and\n"
            "  JSON-compatible attributes.",
            introduction,
        )

        rust_api = (ROOT / "docs/rust-api.md").read_text(encoding="utf-8")
        self.assertIn(
            "`BatchMetadata::new(source, sequence, attributes)` validates the source "
            "and stores\n"
            "JSON-compatible attributes. Its sequence is descriptive batch metadata "
            "and is\n"
            "independent of continuous source cursors and checkpoint epochs.",
            rust_api,
        )

    def test_getting_started_covers_packages_source_and_studio_platforms(
        self,
    ) -> None:
        guide_path = ROOT / "docs/getting-started.md"
        self.assertTrue(guide_path.is_file())
        guide = guide_path.read_text(encoding="utf-8")

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

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("[getting started](docs/getting-started.md)", readme)


if __name__ == "__main__":
    unittest.main()
