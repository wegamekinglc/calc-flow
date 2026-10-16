from __future__ import annotations

import hashlib
import json
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.benchmark_suite import issue406_fresh as fresh
from scripts.benchmark_suite import issue406_harness as harness
from scripts.benchmark_suite import issue406_reuse as reuse
from scripts.test_benchmark_rust_harness import checkout
from scripts.toolkit import command_output, sha256_file, write_json


def commit(root: Path) -> None:
    command_output(["git", "add", "."], cwd=root)
    command_output(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Fixture source",
        ],
        cwd=root,
    )


def clock_lock_fixture() -> str:
    return (
        'version = 4\n[[package]]\nname = "calc-flow"\nversion = "1"\n'
        'dependencies = [\n "parking_lot",\n]\n'
        '[[package]]\nname = "nix"\nversion = "0.27.1"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        'checksum = "2eb04e9c688eff1c89d72b407f168cf79b'
        'b9e867a9d3323ed6c01519eb9cc053"\n'
    )


def fixtures(root: Path) -> tuple[Path, Path]:
    baseline, candidate = root / "baseline", root / "candidate"
    for source, label in ((baseline, "main"), (candidate, "candidate")):
        checkout(source, b"old bench", lock=clock_lock_fixture())
        for relative in harness.COPIED_FILES:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"// {label} {relative}\n")
        for relative in harness.INSERTIONS:
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                '[package]\nname = "calc-flow"\n'
                if relative.endswith(".toml")
                else f"// {label} production implementation\n"
            )
        for relative, patches in harness.LITERAL_PATCHES.items():
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(before for before, _after in patches))
        (source / "production.rs").write_text(f"// {label} must remain intact\n")
        commit(source)
    return baseline, candidate


RUSTDOC_PATH = "crates/calc-flow/src/benchmark_internals.rs"
RUSTDOC_OLD = (
    b"/// This correctness-only diagnostic observes reservations in the unchanged\n"
    b"/// one-GiB runtime pool. It does not report performance measurements."
)
RUSTDOC_NEW = (
    b"/// This correctness-only diagnostic observes reservations and refunds in the\n"
    b"/// default unbounded runtime pool. It does not report performance measurements."
)


def rustdoc_sources() -> tuple[bytes, bytes, bytes, bytes]:
    new = (
        Path(__file__).parent / "fixtures/issue406-rustdoc/benchmark_internals.rs.txt"
    ).read_bytes()
    old = new.replace(RUSTDOC_NEW, RUSTDOC_OLD, 1)
    start = old.index(b"/// Checks the fixed retained Join checkpoint lifecycle")
    end = old.index(b"\n}\n\n", start) + 4
    addition = old[start:end]
    historical = old.replace(addition, b"", 1)
    return new, old, historical, addition


def git_blob(contents: bytes) -> str:
    # Git blob IDs require SHA-1; witness content is separately sealed with SHA-256.
    # nosemgrep: insecure-hash-algorithm-sha1
    return hashlib.sha1(
        b"blob " + str(len(contents)).encode() + b"\0" + contents,
        usedforsecurity=False,
    ).hexdigest()


def rustdoc_witness(candidate, recipe_path, historical, addition):
    prefix = (
        (candidate / recipe_path)
        .read_text()
        .split("\ndef validate_acceptance_harness(")[0]
    )
    frozen_files = {
        path: (candidate / path).read_bytes() for path in harness.COPIED_FILES
    }
    historical_files = {**frozen_files, RUSTDOC_PATH: historical}
    return {
        "recipe": {
            "path": recipe_path,
            "boundary": "\ndef validate_acceptance_harness(",
            "historical_prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
            "candidate_prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        },
        "copied_files": {
            path: {
                "historical_git_blob": git_blob(historical_files[path]),
                "historical_sha256": hashlib.sha256(historical_files[path]).hexdigest(),
                "candidate_build_git_blob": git_blob(contents),
                "candidate_build_sha256": hashlib.sha256(contents).hexdigest(),
                "exact_additions": [addition.decode()] if path == RUSTDOC_PATH else [],
            }
            for path, contents in frozen_files.items()
        },
    }


@contextmanager
def rustdoc_fixture(root: Path):
    """Exercise the three existing provenance entries using source-only fixtures."""
    baseline, candidate = fixtures(root)
    new, old, historical, addition = rustdoc_sources()
    copied = candidate / RUSTDOC_PATH
    recipe_path = "scripts/benchmark_suite/issue406_harness.py"
    recipe = candidate / recipe_path
    recipe.parent.mkdir(parents=True, exist_ok=True)
    recipe.write_text(Path(harness.__file__).read_text())
    copied.write_bytes(historical)
    commit(candidate)
    historical_revision = command_output(["git", "rev-parse", "HEAD"], cwd=candidate)
    with ExitStack() as stack:
        stack.enter_context(patch.object(harness, "ROOT", candidate))
        historical_output = root / "historical-output"
        with harness.common_acceptance_harness(
            baseline, historical_output, "stream_asof_e2e"
        ):
            pass
        copied.write_bytes(old)
        commit(candidate)
        frozen_revision = command_output(["git", "rev-parse", "HEAD"], cwd=candidate)
        frozen_output = root / "frozen-output"
        with harness.common_acceptance_harness(
            baseline, frozen_output, "stream_asof_e2e"
        ):
            pass
        witness = rustdoc_witness(candidate, recipe_path, historical, addition)
        witness_path = root / "witness.json"
        write_json(witness_path, witness)
        stack.enter_context(
            patch.object(reuse, "WITNESS_SHA256", sha256_file(witness_path))
        )
        stack.enter_context(patch.object(reuse, "HISTORICAL", historical_revision))
        stack.enter_context(patch.object(reuse, "FROZEN", frozen_revision))
        stack.enter_context(
            patch.object(
                fresh,
                "MAIN",
                command_output(["git", "rev-parse", "HEAD"], cwd=baseline),
            )
        )
        reconstruction = reuse.helper_reconstruction(witness_path)
        effective = harness._effective
        traced_output = root / "traced-output"
        with (
            patch.object(
                harness,
                "_effective",
                side_effect=lambda source: reuse.restore_copied(
                    effective(source), witness["copied_files"]
                ),
            ),
            harness.common_acceptance_harness(
                baseline, traced_output, "calc_flow_python"
            ),
        ):
            pass
        traced_path = traced_output / "build-calc_flow_python.acceptance-harness.json"
        write_json(
            traced_path,
            {
                **json.loads(traced_path.read_text()),
                "helper_reconstruction": reconstruction,
            },
        )
        (candidate / "python_before_pool.py").write_text("source_only = True\n")
        commit(candidate)
        other_build = command_output(["git", "rev-parse", "HEAD"], cwd=candidate)
        copied.write_bytes(new)
        commit(candidate)
        pool = command_output(["git", "rev-parse", "HEAD"], cwd=candidate)
        stack.enter_context(patch.object(reuse, "RUSTDOC_PRODUCT", pool, create=True))
        (candidate / "python_after_pool.py").write_text("controller_only = True\n")
        commit(candidate)
        current = command_output(["git", "rev-parse", "HEAD"], cwd=candidate)
        builds = {
            "frozen": (frozen_output, "stream_asof_e2e", {}),
            "historical": (
                historical_output,
                "stream_asof_e2e",
                {"compatibility": witness_path},
            ),
            "traced": (
                traced_output,
                "calc_flow_python",
                {"reconstruction": reconstruction},
            ),
        }
        yield {
            "candidate": candidate,
            "baseline": baseline,
            "new": new,
            "old": old,
            "historical": historical,
            "witness": witness_path,
            "frozen": frozen_revision,
            "pool": pool,
            "current": current,
            "other_build": other_build,
            "builds": builds,
        }


class RustdocReuseTests(unittest.TestCase):
    def test_exact_rustdoc_edit_restores_all_three_historical_builds(self):
        with TemporaryDirectory() as raw, rustdoc_fixture(Path(raw)) as fixture:
            self.assertEqual(
                hashlib.sha256(fixture["new"]).hexdigest(),
                "924fe08cef76b74164c8e7aedce011b1762ad10077c47b98e2317c955f2315da",
            )
            self.assertEqual(
                hashlib.sha256(fixture["old"]).hexdigest(),
                "5d900ce4399b115e6dbdd30948c62abf0f67502b6944dd96140e87a8b95b28fe",
            )
            for name, (output, target, options) in fixture["builds"].items():
                with self.subTest(entry=name):
                    path = output / f"build-{target}.acceptance-harness.json"
                    before = path.read_bytes()
                    record = harness.validate_acceptance_harness(
                        output, target, **options
                    )
                    expected = (
                        fixture["old"] if name == "frozen" else fixture["historical"]
                    )
                    self.assertEqual(
                        record["effective_files"][RUSTDOC_PATH],
                        hashlib.sha256(expected).hexdigest(),
                    )
                    self.assertEqual(path.read_bytes(), before)
                    self.assertEqual(
                        (fixture["candidate"] / RUSTDOC_PATH).read_bytes(),
                        fixture["new"],
                    )
            proof = harness.native_harness_reuse(fixture["frozen"])
            self.assertEqual(
                proof["copied_file_git_blobs"][RUSTDOC_PATH], git_blob(fixture["new"])
            )
            self.assertEqual(proof["execution_harness_git_sha"], fixture["current"])
            self.assertEqual(
                proof["rustdoc_compatibility"]["product_ancestor"], fixture["pool"]
            )
            self.assertEqual(
                proof["rustdoc_compatibility"]["restored_git_blob"],
                git_blob(fixture["old"]),
            )

    def test_current_build_preserves_actual_new_helper_bytes(self):
        with TemporaryDirectory() as raw, rustdoc_fixture(Path(raw)) as fixture:
            output = Path(raw) / "current-output"
            with harness.common_acceptance_harness(
                fixture["baseline"], output, "stream_asof_e2e"
            ) as mirror:
                self.assertEqual((mirror / RUSTDOC_PATH).read_bytes(), fixture["new"])
            record = harness.validate_acceptance_harness(output, "stream_asof_e2e")
            self.assertEqual(record["harness_git_sha"], fixture["current"])
            self.assertEqual(
                record["effective_files"][RUSTDOC_PATH],
                hashlib.sha256(fixture["new"]).hexdigest(),
            )

    def test_rustdoc_reuse_rejects_every_unreviewed_copied_source_edit(self):
        with TemporaryDirectory() as raw, rustdoc_fixture(Path(raw)) as fixture:
            original = fixture["new"]
            edits = {
                "body": original.replace(
                    b"    crate::operator::run_checkpoint_history_diagnostic().await",
                    b"    return "
                    b"crate::operator::run_checkpoint_history_diagnostic().await;",
                    1,
                ),
                "extra-comment": original + b"\n// another comment\n",
                "duplicate-literal": original + RUSTDOC_NEW + b"\n",
                "missing-literal": original.replace(RUSTDOC_NEW, b"", 1),
            }
            for label, contents in edits.items():
                self.assertNotEqual(contents, original)
                (fixture["candidate"] / RUSTDOC_PATH).write_bytes(contents)
                commit(fixture["candidate"])
                for name, (output, target, options) in fixture["builds"].items():
                    with (
                        self.subTest(change=label, entry=name),
                        self.assertRaisesRegex(ValueError, "attestation"),
                    ):
                        harness.validate_acceptance_harness(output, target, **options)
            (fixture["candidate"] / RUSTDOC_PATH).write_bytes(original)
            other = fixture["candidate"] / harness.COPIED_FILES[0]
            other.write_bytes(other.read_bytes() + b"// different helper\n")
            commit(fixture["candidate"])
            for name, (output, target, options) in fixture["builds"].items():
                with (
                    self.subTest(change="other-helper", entry=name),
                    self.assertRaisesRegex(ValueError, "attestation"),
                ):
                    harness.validate_acceptance_harness(output, target, **options)

    def test_rustdoc_reuse_requires_the_exact_build_and_pool_ancestor(self):
        with TemporaryDirectory() as raw, rustdoc_fixture(Path(raw)) as fixture:
            with self.assertRaises(ValueError):
                harness.native_harness_reuse(fixture["other_build"])
            with patch.object(reuse, "RUSTDOC_PRODUCT", "f" * 40, create=True):
                for name, (output, target, options) in fixture["builds"].items():
                    with (
                        self.subTest(entry=name),
                        self.assertRaisesRegex(ValueError, "attestation"),
                    ):
                        harness.validate_acceptance_harness(output, target, **options)

    def test_rustdoc_reuse_rejects_an_altered_original_witness(self):
        with TemporaryDirectory() as raw, rustdoc_fixture(Path(raw)) as fixture:
            path = fixture["witness"]
            path.write_bytes(path.read_bytes() + b"\n")
            for name in ("historical", "traced"):
                output, target, options = fixture["builds"][name]
                with (
                    self.subTest(entry=name),
                    self.assertRaisesRegex(ValueError, "attestation"),
                ):
                    harness.validate_acceptance_harness(output, target, **options)


class AcceptanceHarnessTests(unittest.TestCase):
    def test_python_correction_reuses_only_unchanged_historical_native_overlay(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            recipe = candidate / "scripts/benchmark_suite/issue406_harness.py"
            recipe.parent.mkdir(parents=True, exist_ok=True)
            recipe.write_text(Path(harness.__file__).read_text())
            commit(candidate)
            output = root / "output"
            with patch.object(harness, "ROOT", candidate):
                with harness.common_acceptance_harness(
                    baseline, output, "stream_asof_e2e"
                ):
                    pass
                original = harness.validate_acceptance_harness(
                    output, "stream_asof_e2e"
                )
                (candidate / "python_driver.py").write_text(
                    "fingerprint_before_run = True\n"
                )
                commit(candidate)
                reused = harness.validate_acceptance_harness(output, "stream_asof_e2e")
                self.assertEqual(reused, original)
                copied = candidate / harness.COPIED_FILES[0]
                copied.write_text(copied.read_text() + "// changed native work\n")
                commit(candidate)
                with self.assertRaisesRegex(ValueError, "attestation"):
                    harness.validate_acceptance_harness(output, "stream_asof_e2e")

    def test_historical_reuse_rejects_recipe_drift_and_missing_history(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            recipe = candidate / "scripts/benchmark_suite/issue406_harness.py"
            recipe.parent.mkdir(parents=True, exist_ok=True)
            recipe.write_text(Path(harness.__file__).read_text())
            commit(candidate)
            output = root / "output"
            with patch.object(harness, "ROOT", candidate):
                with harness.common_acceptance_harness(
                    baseline, output, "stream_asof_e2e"
                ):
                    pass
                record = harness.validate_acceptance_harness(output, "stream_asof_e2e")
                recipe.write_text(
                    recipe.read_text().replace(
                        "Closed, benchmark-only", "Changed, benchmark-only", 1
                    )
                )
                commit(candidate)
                with self.assertRaisesRegex(ValueError, "attestation"):
                    harness.validate_acceptance_harness(output, "stream_asof_e2e")
                evidence = output / "build-stream_asof_e2e.acceptance-harness.json"
                evidence.write_text(json.dumps({**record, "harness_git_sha": "0" * 40}))
                with self.assertRaisesRegex(ValueError, "attestation"):
                    harness.validate_acceptance_harness(output, "stream_asof_e2e")

    def test_traced_wheel_requires_both_actual_release_compiler_invocations(self):
        def invocation(crate, features):
            flags = " ".join(f"--cfg 'feature=\"{name}\"'" for name in features)
            return f"Running `rustc --crate-name {crate} -C opt-level=3 {flags}`"

        record = {
            "actual_core_invocation": invocation("calc_flow", ["benchmark-internals"]),
            "actual_binding_invocation": invocation(
                "calc_flow_python", ["benchmark-internals", "connector-file", "default"]
            ),
        }
        harness.validate_traced_wheel_invocations(record)
        for mutation in (
            {**record, "actual_core_invocation": invocation("calc_flow", [])},
            {
                **record,
                "actual_binding_invocation": record[
                    "actual_binding_invocation"
                ].replace("opt-level=3", "opt-level=0"),
            },
            {
                **record,
                "actual_core_invocation": record["actual_core_invocation"].replace(
                    " -C", " --cfg test -C"
                ),
            },
            {
                **record,
                "actual_core_invocation": invocation(
                    "calc_flow", ["benchmark-internals", "benchmark-checkpoint-off"]
                ),
            },
        ):
            with (
                self.subTest(mutation=mutation),
                self.assertRaisesRegex(ValueError, "wheel"),
            ):
                harness.validate_traced_wheel_invocations(mutation)

    def test_lock_identity_ignores_dependency_order_but_preserves_versions(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / "Cargo.lock"
            prefix = 'version = 4\n[[package]]\nname = "calc-flow"\nversion = "1"\n'
            path.write_text(prefix + 'dependencies = ["fs2 1", "flatbuffers 2"]\n')
            original = harness._lock_identity(root)
            path.write_text(prefix + 'dependencies = ["flatbuffers 2", "fs2 1"]\n')
            self.assertEqual(harness._lock_identity(root), original)
            path.write_text(prefix + 'dependencies = ["flatbuffers 3", "fs2 1"]\n')
            self.assertNotEqual(harness._lock_identity(root), original)

    def test_only_closed_benchmark_overlay_changes_product_mirror(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            output = root / "output"
            with patch.object(harness, "ROOT", candidate):
                with harness.common_acceptance_harness(
                    baseline, output, "stream_asof_e2e"
                ) as mirror:
                    self.assertEqual(
                        (mirror / "production.rs").read_bytes(),
                        (baseline / "production.rs").read_bytes(),
                    )
                    for relative in (
                        harness.INSERTIONS.keys() | harness.LITERAL_PATCHES.keys()
                    ):
                        restored = (mirror / relative).read_text()
                        for before, after in reversed(
                            harness.LITERAL_PATCHES.get(relative, ())
                        ):
                            restored = restored.replace(after, before, 1)
                        insertion = harness.INSERTIONS.get(relative)
                        if insertion is not None:
                            restored = restored.replace(insertion, "", 1)
                        self.assertEqual(
                            restored,
                            (baseline / relative).read_text(),
                        )
                    for relative in harness.COPIED_FILES:
                        self.assertEqual(
                            (mirror / relative).read_bytes(),
                            (candidate / relative).read_bytes(),
                        )
                self.assertFalse(mirror.exists())
                record = harness.validate_acceptance_harness(output, "stream_asof_e2e")
                self.assertTrue(record["product_files_verified"])
                self.assertEqual(
                    record["product_git_sha"],
                    command_output(["git", "rev-parse", "HEAD"], cwd=baseline),
                )
                evidence = output / "build-stream_asof_e2e.acceptance-harness.json"
                invalid = {**record, "effective_files": {}}
                evidence.write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError, "attestation"):
                    harness.validate_acceptance_harness(output, "stream_asof_e2e")

    def test_unapproved_mutation_rejects_build_and_removes_mirror(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            with patch.object(harness, "ROOT", candidate):
                with (
                    self.assertRaisesRegex(ValueError, "product files"),
                    harness.common_acceptance_harness(
                        baseline, root / "output", "stream_join_perf"
                    ) as mirror,
                ):
                    (mirror / "production.rs").write_text("changed algorithm")
                self.assertFalse(mirror.exists())

    def test_failed_build_never_publishes_verified_attestation(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            baseline, candidate = fixtures(root)
            output = root / "output"
            with patch.object(harness, "ROOT", candidate):
                with (
                    self.assertRaisesRegex(RuntimeError, "build failed"),
                    harness.common_acceptance_harness(
                        baseline, output, "stream_join_perf"
                    ) as mirror,
                ):
                    raise RuntimeError("build failed")
                self.assertFalse(mirror.exists())
                with self.assertRaisesRegex(ValueError, "attestation"):
                    harness.validate_acceptance_harness(output, "stream_join_perf")

    def test_compiled_core_must_use_release_without_test_configuration(self):
        base = {
            "package": {"workspace_package": "crates/calc-flow"},
            "target": {"kind": ["lib"], "name": "calc_flow"},
            "features": ["benchmark-internals"],
            "profile": {"opt_level": "3", "test": False},
        }
        harness.validate_native_profile([base], checkpoint_off=False)
        for invalid in (
            {**base, "profile": {"opt_level": "3", "test": True}},
            {**base, "features": []},
            {**base, "features": ["benchmark-internals", "benchmark-checkpoint-off"]},
        ):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(ValueError, "release core"),
            ):
                harness.validate_native_profile([invalid], checkpoint_off=False)
