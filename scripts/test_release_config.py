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
    def test_warm_stream_regressions_run_in_both_python_ci_jobs(self) -> None:
        for name in ("ci-linux.yml", "ci-windows.yml"):
            workflow = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
            self.assertTrue(
                "python/tests benchmarks/test_warm_stream.py" in workflow,
                f"{name} must execute warm streaming scenario tests",
            )
            self.assertTrue(
                "python -m unittest scripts.test_profile_warm_stream" in workflow,
                f"{name} must execute warm profiling controller tests",
            )

    def test_current_python_surfaces_do_not_use_removed_compile_method(self) -> None:
        paths = [
            *sorted((ROOT / "benchmarks").glob("*.py")),
            *sorted((ROOT / "examples").glob("*.py")),
            ROOT / "README.md",
            ROOT / "examples/README.md",
            ROOT / "scripts/smoke_wheel.py",
            *(
                ROOT / path
                for path in (
                    "docs/api-reference.md",
                    "docs/getting-started.md",
                    "docs/introduction.md",
                    "docs/python-api.md",
                )
            ),
        ]
        for absolute_path in paths:
            path = absolute_path.relative_to(ROOT)
            with self.subTest(path=path):
                source = absolute_path.read_text(encoding="utf-8")
                self.assertNotIn(".compile()", source)
                self.assertNotIn(".compile(runtime)", source)
                self.assertNotIn('["pipeline"]', source)

    def test_generated_contracts_are_pinned_to_lf(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        for path in (
            "schemas/project-v3.schema.json",
            "web-ui/openapi.json",
            "web-ui/src/api/schema.d.ts",
        ):
            with self.subTest(path=path):
                self.assertIn(f"{path} text eol=lf", attributes.splitlines())

    def test_kafka_ci_bounds_stale_apt_index_recovery(self) -> None:
        workflow = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
        self.assertEqual(
            workflow.count("timeout 5m sudo apt-get"),
            3,
        )
        self.assertEqual(
            workflow.count("timeout 2m sudo apt-get"),
            3,
        )
        self.assertEqual(workflow.count("if ! install_libcurl_headers; then"), 3)

    def test_rustsec_waivers_are_consistent_and_scoped(self) -> None:
        advisory = "RUSTSEC-2026-0235"
        for path in (
            "AGENTS.md",
            ".github/workflows/ci-linux.yml",
            ".github/workflows/release.yml",
            "scripts/verify_security_gates.py",
        ):
            with self.subTest(path=path):
                self.assertIn(
                    advisory,
                    (ROOT / path).read_text(encoding="utf-8"),
                )

        audit_path = (
            Path("docs/superpowers/audits")
            / "2026-08-19-continuous-streaming-v3-current-main.md"
        )
        audit = (ROOT / audit_path).read_text(encoding="utf-8")
        self.assertIn(advisory, audit)
        self.assertIn("lockfile-only", audit.lower())
        self.assertIn("cargo tree --workspace --all-features -i rkyv", audit)

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

        self.assertEqual(workspace["workspace"]["package"]["version"], "4.0.0")
        self.assertEqual(binding["dependencies"]["calc-flow"]["version"], "=4.0.0")
        self.assertEqual(package["project"]["version"], "4.0.0")
        self.assertEqual(package["project"]["name"], "calc-flow-python")
        self.assertEqual(package["tool"]["maturin"]["module-name"], "calc_flow._native")
        self.assertEqual(studio["project"]["version"], "4.0.0")
        self.assertIn("calc-flow-python>=4.0.0,<5", studio["project"]["dependencies"])
        self.assertEqual(frontend["version"], "4.0.0")
        self.assertEqual(frontend_lock["version"], "4.0.0")
        self.assertEqual(frontend_lock["packages"][""]["version"], "4.0.0")
        self.assertIn(
            '__version__ = "4.0.0"',
            (ROOT / "python/calc_flow/__init__.py").read_text(encoding="utf-8"),
        )
        self.assertIn(
            'version="4.0.0"',
            (ROOT / "web-ui/backend/src/calc_flow_studio/app.py").read_text(
                encoding="utf-8"
            ),
        )
        openapi = json.loads((ROOT / "web-ui/openapi.json").read_text(encoding="utf-8"))
        self.assertEqual(openapi["info"]["version"], "4.0.0")

        release_text = "\n".join(
            (ROOT / path).read_text(encoding="utf-8")
            for path in (
                ".github/workflows/ci-linux.yml",
                ".github/workflows/release.yml",
            )
        )
        self.assertNotIn(">=2.0.0a1", release_text)
        self.assertIn(">=4.0.0", release_text)

        release_workflow = (ROOT / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn('- "v4.*"', release_workflow)
        self.assertNotIn('- "v3.*"', release_workflow)
        self.assertIn('assert "/api/v3/catalog"', release_workflow)
        self.assertIn('and ">=4.0.0" in requirement', release_workflow)
        self.assertIn('and "<5" in requirement', release_workflow)
        self.assertEqual(release_workflow.count("--save-baseline exact-"), 4)
        self.assertIn("--criterion-dir", release_workflow)
        self.assertIn("--criterion-baseline exact-baseline", release_workflow)
        self.assertIn("--criterion-candidate exact-candidate", release_workflow)
        self.assertIn("--scenario rolling_kernel_sma20", release_workflow)
        self.assertIn("--scenario rolling_kernel_dual_sma_5_20", release_workflow)
        self.assertIn('--expected-commit "${candidate_sha}"', release_workflow)
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

    def test_python_projects_publish_nonempty_readmes(self) -> None:
        for project in (ROOT, ROOT / "web-ui/backend"):
            with self.subTest(project=project):
                config = tomllib.loads(
                    (project / "pyproject.toml").read_text(encoding="utf-8")
                )
                readme = project / config["project"]["readme"]
                self.assertTrue(readme.read_text(encoding="utf-8").strip())

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
        self.assertEqual(config.count('      - "web-ui/**"'), 2)
        self.assertIn("Biome's default Qwik", config)
        self.assertIn("sandbox does not install", config)
        self.assertEqual(config.count('  - "web-ui/src/api/schema.d.ts"'), 1)
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

    def test_python_release_verifies_exact_artifacts_before_oidc_publish(self) -> None:
        workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")

        self.assertIn("  prepare-python-release:\n", workflow)
        self.assertIn("python scripts/verify_python_release.py", workflow)
        self.assertIn("git fetch --no-tags origin main", workflow)
        self.assertIn('git cat-file -t "${GITHUB_REF}"', workflow)
        self.assertIn("  verify-python-release:\n", workflow)
        self.assertIn("name: verified-python-release", workflow)
        self.assertEqual(workflow.count("sha256sum --check release-manifest.txt"), 1)
        self.assertEqual(workflow.count("uses: pypa/gh-action-pypi-publish@"), 1)
        self.assertEqual(workflow.count("id-token: write"), 1)
        self.assertIn("name: pypi\n", workflow)
        self.assertNotIn("name: pypi-studio\n", workflow)
        self.assertIn("url: https://pypi.org/project/calc-flow-python/", workflow)
        self.assertNotIn("  publish-python-studio:\n", workflow)
        self.assertIn("packages-dir: release-dist/core", workflow)
        self.assertNotIn("packages-dir: release-dist/studio", workflow)
        self.assertIn(
            "if: github.event_name == 'push' && github.ref_type == 'tag'", workflow
        )
        self.assertIn("python scripts/release_baseline.py", workflow)
        self.assertIn("initial-baseline:", workflow)
        self.assertNotIn("skip-existing", workflow)

        verify_job = workflow.split("  verify-python-release:\n", 1)[1].split(
            "  publish-python-core:\n", 1
        )[0]
        self.assertIn(
            "uses: actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1",
            verify_job,
        )
        self.assertIn('python-version: "3.13"', verify_job)

    def test_python_release_guide_covers_rehearsal_and_trusted_publishers(self) -> None:
        guide = (ROOT / "docs/python-release.md").read_text(encoding="utf-8")

        for expected in (
            "python scripts/build_python_release.py --clean",
            "python scripts/verify_python_release.py",
            "`pypi`",
            "Studio is not uploaded",
            "release.yml",
            "git tag -a v<version>",
            "PyPI versions and files are immutable",
        ):
            self.assertIn(expected, guide)

        release_table = guide.split("tagged release run:\n\n", 1)[1].split("\n\n", 1)[0]
        pipe_positions = {
            tuple(index for index, character in enumerate(line) if character == "|")
            for line in release_table.splitlines()
        }
        self.assertEqual(len(pipe_positions), 1)

    def test_frontend_module_stems_do_not_collide_on_windows(self) -> None:
        modules = sorted((ROOT / "web-ui/src").rglob("*.ts")) + sorted(
            (ROOT / "web-ui/src").rglob("*.tsx")
        )
        paths_by_stem: dict[str, list[str]] = {}
        for module in modules:
            key = str(module.relative_to(ROOT).with_suffix("")).casefold()
            paths_by_stem.setdefault(key, []).append(str(module.relative_to(ROOT)))
        collisions = {
            stem: paths for stem, paths in paths_by_stem.items() if len(paths) > 1
        }

        self.assertEqual(collisions, {})

    def test_public_package_tables_have_full_width_separators(self) -> None:
        for path in ("docs/api-reference.md", "docs/python-release.md"):
            source = (ROOT / path).read_text(encoding="utf-8")
            table = next(
                block for block in source.split("\n\n") if block.startswith("|")
            )
            rows = table.splitlines()
            with self.subTest(path=path):
                self.assertEqual(
                    len(
                        {
                            tuple(
                                index for index, char in enumerate(row) if char == "|"
                            )
                            for row in rows
                        }
                    ),
                    1,
                )
                self.assertTrue(
                    all(set(cell) == {"-"} for cell in rows[1].split("|")[1:-1])
                )

    def test_workflow_actions_are_sha_pinned(self) -> None:
        for name in (
            "benchmarks.yml",
            "benchmark-suite.yml",
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
            "  rust-coverage:\n", 1
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
        self.assertNotIn("RUST_TEST_THREADS", rust_core_header)
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

    def test_rust_tests_and_coverage_run_in_parallel_jobs(self) -> None:
        workflow = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
        rust_core = workflow.split("  rust-core:\n", 1)[1].split(
            "  rust-coverage:\n", 1
        )[0]
        rust_coverage = workflow.split("  rust-coverage:\n", 1)[1].split(
            "  rust-supply-chain:\n", 1
        )[0]

        rust_tests = "python3.13 scripts/run_rust_tests.py --python-stress-runs 3"
        coverage = "python3.13 scripts/run_rust_coverage.py"
        rustdoc = (
            'RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps'
        )

        self.assertIn(rust_tests, rust_core)
        self.assertIn(rustdoc, rust_core)
        self.assertNotIn(coverage, rust_core)
        self.assertNotIn("services:", rust_core)
        self.assertIn("services:", rust_coverage)
        self.assertIn(coverage, rust_coverage)
        self.assertNotIn(rust_tests, rust_coverage)
        self.assertNotIn("cargo clean", workflow)
        self.assertNotIn("FROZEN_ALLOCATION", workflow)
        self.assertNotIn("RUST_TEST_THREADS", workflow)

    def test_ci_uses_docs_only_classification_and_stable_gates(self) -> None:
        linux = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
        windows = (ROOT / ".github/workflows/ci-windows.yml").read_text(
            encoding="utf-8"
        )

        for workflow, gate, next_job in (
            (linux, "linux-gate", "docs-check"),
            (windows, "windows-gate", "process-tree"),
        ):
            with self.subTest(gate=gate):
                changes = workflow.split("  changes:\n", 1)[1].split(
                    f"  {next_job}:\n", 1
                )[0]
                self.assertIn("  changes:\n", workflow)
                self.assertIn("scripts/classify_ci_changes.py", workflow)
                self.assertIn(
                    "docs_only: ${{ steps.classify.outputs.docs_only }}", workflow
                )
                self.assertIn("uses: actions/setup-python@", changes)
                self.assertIn("python-version-file: .python-version", changes)
                self.assertIn("cancel-in-progress: true", workflow)
                self.assertIn(f"  {gate}:\n", workflow)
                self.assertIn("if: always()", workflow.split(f"  {gate}:\n", 1)[1])

        self.assertIn("  docs-check:\n", linux)
        self.assertIn('run: git diff --check "$BASE_SHA" "$HEAD_SHA"', linux)
        docs_check = linux.split("  docs-check:\n", 1)[1].split(
            "  connector-containers:\n", 1
        )[0]
        self.assertNotIn("if: needs.changes.outputs.docs_only", docs_check)

        linux_gate = linux.split("  linux-gate:\n", 1)[1]
        docs_result_check = 'test "$DOCS_RESULT" = success'
        self.assertIn(docs_result_check, linux_gate)
        self.assertLess(
            linux_gate.index(docs_result_check),
            linux_gate.index('if [ "$DOCS_ONLY" = true ]; then'),
        )

    def test_linux_ci_builds_the_core_wheel_once_for_consumers(self) -> None:
        workflow = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
        package = workflow.split("  package:\n", 1)[1].split("  studio-package:\n", 1)[
            0
        ]
        studio_package = workflow.split("  studio-package:\n", 1)[1].split(
            "  benchmark-smoke:\n", 1
        )[0]

        self.assertIn("needs: changes", package)
        self.assertNotIn("needs: lint-and-test", package)
        self.assertIn("name: python-distributions", package)
        self.assertEqual(workflow.count("name: Download core distributions"), 4)
        self.assertGreaterEqual(workflow.count("--no-install-workspace"), 3)
        self.assertGreaterEqual(workflow.count("uv run --no-sync"), 5)
        self.assertIn("name: Download core distributions", studio_package)
        self.assertNotIn("\n          uv build\n", studio_package)

    def test_ci_and_release_execute_script_unit_tests(self) -> None:
        command = (
            "python -m unittest scripts.test_generate_rolling_kernel_manifest "
            "scripts.test_run_rust_tests "
            "scripts.test_run_rust_coverage "
            "scripts.test_classify_ci_changes "
            "scripts.test_build_python_release scripts.test_inspect_wheel "
            "scripts.test_release_config scripts.test_verify_python_release "
            "scripts.test_verify_perf_gates "
            "scripts.test_verify_stream_lifecycle_evidence "
            "scripts.test_verify_symbolic_milestone_perf "
            "scripts.test_write_criterion_provenance "
            "scripts.test_verify_complexity_gates "
            "scripts.test_verify_security_gates "
            "scripts.test_verify_sql_datafusion_performance "
            "scripts.test_analyze_sql_datafusion_attribution "
            "scripts.test_run_sql_datafusion_matrix"
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

    def _legacy_scale_names(self) -> tuple[str, ...]:
        support = ast.parse(
            (ROOT / "benchmarks/support.py").read_text(encoding="utf-8")
        )
        scales = next(
            node.value
            for node in support.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "SCALES"
                for target in node.targets
            )
        )
        return tuple(ast.literal_eval(key) for key in scales.keys)

    def test_pr_and_schedule_run_the_same_complete_catalog(self) -> None:
        from scripts.benchmark_suite.catalog import LEGACY_SCALES, ROW_SCALES, shards

        names = self._legacy_scale_names()
        automated = tuple(name for name in names if name != "nightly")
        self.assertEqual(automated, LEGACY_SCALES)
        self.assertNotIn("nightly", LEGACY_SCALES)
        self.assertEqual(ROW_SCALES, tuple(10**n for n in range(1, 8)))
        for name in ("ci-linux.yml", "benchmarks.yml"):
            workflow = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
            self.assertIn("uses: ./.github/workflows/benchmark-suite.yml", workflow)
        self.assertEqual(
            {s["scale"] for s in shards() if s["family"] == "python"},
            set(automated),
        )

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
        self.assertIn("- rust-coverage", finish)
        self.assertIn(
            "if: always() && needs.changes.outputs.docs_only != 'true'",
            finish,
        )
        self.assertIn("parallel-finished: true", finish)

    def test_performance_workflows_cover_p1_and_p2_evidence(self) -> None:
        from scripts.benchmark_suite.catalog import shards
        from scripts.benchmark_suite.legacy import pytest_arguments
        from scripts.benchmark_suite.rust import bench_targets

        families = {shard["family"] for shard in shards()}
        self.assertTrue({"rust", "studio", "frontend", "lifecycle"} <= families)
        self.assertEqual(
            set(bench_targets(ROOT)),
            {
                "core",
                "m4_state_window",
                "stream_join_perf",
                "allocation_regression",
                "sql_datafusion_performance",
            },
        )
        self.assertEqual(
            pytest_arguments("lifecycle"),
            [
                "benchmarks/test_symbolic_baseline.py::test_stream_window_checkpoint_and_recovery"
            ],
        )
        self.assertEqual(
            pytest_arguments("studio"),
            ["web-ui/backend/benchmarks/test_performance.py"],
        )
        self.assertIn("not stream_lifecycle", pytest_arguments("python"))
        core = (ROOT / "crates/calc-flow/benches/core.rs").read_text(encoding="utf-8")
        for case in (
            "stream/channel_fanin_2",
            "stream/channel_fanin_4",
            "stream/channel_fanin_8",
            "stream/backpressure_saturated",
        ):
            self.assertIn(case, core)

    def test_linux_ci_executes_sql_datafusion_smoke_benchmark(self) -> None:
        workflow = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
        smoke = workflow.split(
            "      - name: Validate SQL/DataFusion comparison smoke\n", 1
        )[1].split("      - run: RUSTDOCFLAGS=", 1)[0]

        self.assertIn(
            "cargo bench --locked -p calc-flow --bench sql_datafusion_performance --",
            smoke,
        )
        self.assertNotIn("cargo test", smoke)
        evidence_path = (
            '"${GITHUB_WORKSPACE}/benchmark-results/sql-datafusion-smoke.json"'
        )
        self.assertIn('mkdir -p "${GITHUB_WORKSPACE}/benchmark-results"', smoke)
        self.assertEqual(smoke.count(evidence_path), 2)
        self.assertIn("scripts/verify_sql_datafusion_performance.py", smoke)

    def test_release_budget_preserves_cold_builds_and_full_soaks(self) -> None:
        release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        acceptance = release.split("  acceptance-gates:\n", 1)[1].split(
            "  crate:\n", 1
        )[0]

        self.assertIn("    timeout-minutes: 180\n", acceptance)
        for soak in (
            "twenty_minute_two_source_slow_sink",
            "twenty_minute_epoch_checkpoint_restart",
        ):
            with self.subTest(soak=soak):
                self.assertIn(f"runtime::streaming::soak::{soak}", acceptance)
        self.assertEqual(acceptance.count("-- --ignored --exact --nocapture"), 3)
        self.assertNotIn("continue-on-error:", acceptance)

    def test_pr_and_release_isolate_stream_lifecycle_evidence(self) -> None:
        linux = (ROOT / ".github/workflows/ci-linux.yml").read_text(encoding="utf-8")
        from scripts.benchmark_suite.catalog import shards
        from scripts.benchmark_suite.legacy import pytest_arguments

        self.assertIn("not stream_lifecycle", pytest_arguments("python"))
        self.assertIn({"id": "lifecycle", "family": "lifecycle"}, shards())
        linux_gate = linux.split("  linux-gate:\n", 1)[1]
        self.assertIn("- benchmark-smoke", linux_gate)
        self.assertIn('"$BENCHMARK_RESULT"', linux_gate)
        legacy = (ROOT / "scripts/benchmark_suite/legacy.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("scripts/verify_stream_lifecycle_evidence.py", legacy)

        release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
        exact_gate = release.split(
            "      - name: Run paired exact-ref Rust and Python performance gate\n", 1
        )[1].split("      - name:", 1)[0]
        self.assertEqual(exact_gate.count("--bench stream_join_perf"), 2)
        self.assertEqual(
            exact_gate.count(
                "benchmarks/test_symbolic_baseline.py::"
                "test_stream_window_checkpoint_and_recovery"
            ),
            2,
        )
        self.assertEqual(
            exact_gate.count('-k "not test_stream_window_checkpoint_and_recovery"'),
            2,
        )
        self.assertIn("--require-stream-lifecycle", exact_gate)
        self.assertIn("allow-dependency-drift:", release)
        self.assertIn("inputs['allow-dependency-drift']", exact_gate)
        self.assertIn("--allow-dependency-drift", exact_gate)
        self.assertIn('"${perf_gate_extra_args[@]}"', exact_gate)

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
            "README.md": ("Calc Flow 4.0", 'calc-flow = "4.0.0"'),
            "docs/api-reference.md": (
                "Calc Flow 4.0 API reference",
                "`calc-flow-python==4.0.0`",
                "Project format version `3`",
            ),
            "docs/getting-started.md": ("cargo add calc-flow@4.0.0",),
            "docs/python-api.md": ("`calc-flow-python==4.0.0`",),
            "docs/rust-api.md": ("Calc Flow 4.0", "cargo add calc-flow@4.0.0"),
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
