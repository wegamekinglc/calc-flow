"""Regression coverage for the Rust suite's compiled dependency boundary."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.benchmark_suite.legacy import combine_blocks
from scripts.benchmark_suite.rust import _stamp_fingerprints, _with_fingerprints
from scripts.benchmark_suite.rust_provenance import (
    compiled_dependencies,
    with_compiled_dependencies,
)

REGISTRY = "registry+https://github.com/rust-lang/crates.io-index"


def artifact(root: Path, *, dependency: bool) -> dict:
    return {
        "reason": "compiler-artifact",
        "package_id": f"{REGISTRY}#arrow@1.0.0"
        if dependency
        else "path+file:///core#4.0.0",
        "manifest_path": str(
            Path("/registry/arrow/Cargo.toml")
            if dependency
            else root / "crates/calc-flow/Cargo.toml"
        ),
        "target": {
            "name": "arrow" if dependency else "core",
            "kind": ["lib"] if dependency else ["bench"],
            "crate_types": ["lib"] if dependency else ["bin"],
            "edition": "2024",
        },
        "features": ["default"],
        "profile": {"opt_level": "3", "test": False},
        "fresh": True,
    }


def write_inputs(root: Path) -> tuple[Path, list[dict]]:
    root.joinpath("Cargo.lock").write_text(
        'version = 4\n[[package]]\nname = "arrow"\nversion = "1.0.0"\n'
        f'source = "{REGISTRY}"\nchecksum = "{"a" * 64}"\n',
        encoding="utf-8",
    )
    messages = [
        artifact(root, dependency=True),
        artifact(root, dependency=False),
        {"reason": "build-finished", "success": True},
    ]
    return root / "build-core.jsonl", messages


def measure(root: Path, log: Path, messages: list[dict]) -> dict:
    log.write_text("\n".join(json.dumps(row) for row in messages), encoding="utf-8")
    return compiled_dependencies(root, {"core": log})


def add_vendored_dependency(root: Path, messages: list[dict]) -> list[dict]:
    root.joinpath("Cargo.toml").write_text(
        '[patch.crates-io]\narrow-ipc = { path = "vendor/arrow-ipc" }\n',
        encoding="utf-8",
    )
    vendor = root / "vendor/arrow-ipc"
    (vendor / "src").mkdir(parents=True)
    (vendor / "Cargo.toml").write_text(
        '[package]\nname = "arrow-ipc"\nversion = "59.2.0"\n', encoding="utf-8"
    )
    (vendor / "src/lib.rs").write_text("pub fn patched() {}\n", encoding="utf-8")
    with (root / "Cargo.lock").open("a", encoding="utf-8") as lock:
        lock.write('\n[[package]]\nname = "arrow-ipc"\nversion = "59.2.0"\n')
    dependency = {
        **messages[0],
        "package_id": f"path+{vendor.as_uri()}#59.2.0",
        "manifest_path": str(vendor / "Cargo.toml"),
        "target": {**messages[0]["target"], "name": "arrow_ipc"},
    }
    return [*messages[:-1], dependency, messages[-1]]


class CompiledDependencyTests(unittest.TestCase):
    def test_vendored_patch_records_relative_source_and_content_fingerprint(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            log, messages = write_inputs(root)
            messages = add_vendored_dependency(root, messages)
            original = copy.deepcopy(messages)
            result = measure(root, log, messages)
            package = next(
                row["package"]
                for row in result["core"]
                if row["package"].get("name") == "arrow-ipc"
            )
            self.assertEqual(package["source"], "path+vendor/arrow-ipc")
            self.assertEqual(package["version"], "59.2.0")
            self.assertRegex(package["source_sha256"], r"^[a-f0-9]{64}$")
            self.assertEqual(messages, original)
            messages[-2]["package_id"] = messages[-2]["package_id"].replace(
                "#59.2.0", "#arrow-ipc@59.2.0"
            )
            self.assertEqual(measure(root, log, messages), result)

    def test_vendor_identity_is_stable_across_checkouts_and_removed_build_mirrors(self):
        with TemporaryDirectory() as left, TemporaryDirectory() as right:
            identities = []
            for raw in (left, right):
                root = Path(raw)
                log, messages = write_inputs(root)
                messages = add_vendored_dependency(root, messages)
                identities.append(measure(root, log, messages))
                mirror = root / "removed-mirror/source"
                for row in messages[:-1]:
                    manifest = Path(row["manifest_path"])
                    if manifest.is_relative_to(root):
                        relocated = mirror / manifest.relative_to(root)
                        row["manifest_path"] = str(relocated)
                        row["package_id"] = f"path+{relocated.parent.as_uri()}#59.2.0"
                log.write_text("\n".join(json.dumps(row) for row in messages))
                identities.append(
                    compiled_dependencies(
                        root, {"core": log}, build_roots={"core": mirror}
                    )
                )
            self.assertTrue(all(value == identities[0] for value in identities))

    def test_vendor_source_changes_invalidate_dependency_comparisons(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            log, messages = write_inputs(root)
            messages = add_vendored_dependency(root, messages)
            identity = {
                "dependency_identity": {"rustc": "1.94", "cargo": "1.94"},
                "machine_fingerprint": "a" * 64,
                "scoped_workload_fingerprints": {"core": "b" * 64},
            }
            measure(root, log, messages)
            baseline = with_compiled_dependencies(identity, root, {"core": log})
            source = root / "vendor/arrow-ipc/src/lib.rs"
            source.write_text("pub fn changed_patch() {}\n", encoding="utf-8")
            candidate = with_compiled_dependencies(identity, root, {"core": log})
            stamps = _stamp_fingerprints(
                {"baseline": baseline, "candidate": candidate}, {}
            )
            blocks = {
                side: [
                    {
                        "core/case": {
                            "rows": 1,
                            "scope": "native",
                            "samples": [1.0] * 20,
                            "metadata": targets["core"],
                        }
                    }
                ]
                * 2
                for side, targets in stamps.items()
            }
            case = combine_blocks({"id": "rust", "family": "rust"}, blocks)[0]
            self.assertEqual(case["status"], "error")
            self.assertIn("dependency_fingerprint changed", case["error"])

    def test_vendor_file_inventory_changes_and_symlinks_cannot_reuse_identity(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            log, messages = write_inputs(root)
            messages = add_vendored_dependency(root, messages)
            before = measure(root, log, messages)
            vendor = root / "vendor/arrow-ipc"
            extra = vendor / "build.rs"
            extra.write_text("fn main() {}\n", encoding="utf-8")
            self.assertNotEqual(measure(root, log, messages), before)
            extra.unlink()
            self.assertEqual(measure(root, log, messages), before)
            try:
                extra.symlink_to(root / "Cargo.lock")
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "symlink"):
                measure(root, log, messages)

    def test_undeclared_or_mismatched_local_dependencies_are_rejected(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            log, messages = write_inputs(root)
            messages = add_vendored_dependency(root, messages)
            for field, value in (
                ("package_id", "path+file:///external#59.2.0"),
                ("package_id", "path+vendor/arrow-ipc"),
                ("package_id", f"path+{(root / 'vendor/arrow-ipc').as_uri()}#60.0.0"),
                ("manifest_path", str(root / "vendor/unknown/Cargo.toml")),
                ("manifest_path", str(root / "vendor/arrow-ipc/other.toml")),
            ):
                changed = copy.deepcopy(messages)
                changed[-2][field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    measure(root, log, changed)
            for path, text in (
                ("Cargo.toml", "[patch.crates-io]\n"),
                (
                    "Cargo.lock",
                    (root / "Cargo.lock").read_text().replace("59.2.0", "60.0.0"),
                ),
            ):
                file = root / path
                original = file.read_text()
                file.write_text(text)
                with self.subTest(path=path), self.assertRaises(ValueError):
                    measure(root, log, messages)
                file.write_text(original)

    def test_comparison_uses_compiled_identity_and_retains_the_full_lock_hash(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            log, messages = write_inputs(root)
            measure(root, log, messages)
            identity = {
                "dependency_identity": {
                    "cargo_lock_sha256": "full-lock",
                    "rustc": "rustc 1.88.0",
                    "cargo": "cargo 1.88.0",
                },
                "dependency_fingerprint": "full-lock-fingerprint",
                "machine_fingerprint": "a" * 64,
                "workload_fingerprint": "workload",
            }
            original = copy.deepcopy(identity)
            result = with_compiled_dependencies(identity, root, {"core": log})
            self.assertEqual(identity, original)
            self.assertEqual(
                result["dependency_identity"], identity["dependency_identity"]
            )
            rows = _with_fingerprints(
                {"one": {"metadata": {}}},
                {"dependency_fingerprint": result["compiled_dependency_fingerprint"]},
            )
            self.assertEqual(
                rows["one"]["metadata"]["dependency_fingerprint"],
                result["compiled_dependency_fingerprint"],
            )
            self.assertEqual(result["dependency_fingerprint"], "full-lock-fingerprint")

    def test_added_target_preserves_shared_dependency_stamp_and_new_coverage(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            log, messages = write_inputs(root)
            measure(root, log, messages)
            identity = {
                "dependency_identity": {"rustc": "1.88", "cargo": "1.88"},
                "machine_fingerprint": "a" * 64,
                "scoped_workload_fingerprints": {"core": "b" * 64},
            }
            baseline = with_compiled_dependencies(identity, root, {"core": log})
            added = copy.deepcopy(messages)
            added[1]["target"]["name"] = "new_target"
            new_log = root / "new.jsonl"
            new_log.write_text("\n".join(json.dumps(row) for row in added))
            candidate = with_compiled_dependencies(
                {
                    **identity,
                    "scoped_workload_fingerprints": {
                        "core": "b" * 64,
                        "new_target": "c" * 64,
                    },
                },
                root,
                {"core": log, "new_target": new_log},
            )
            original = copy.deepcopy(candidate)
            stamps = _stamp_fingerprints(
                {"baseline": baseline, "candidate": candidate}, {}
            )
            self.assertEqual(stamps["baseline"]["core"], stamps["candidate"]["core"])
            self.assertNotIn("new_target", stamps["baseline"])
            self.assertIn("new_target", stamps["candidate"])
            self.assertNotEqual(
                baseline["compiled_dependency_fingerprint"],
                candidate["compiled_dependency_fingerprint"],
            )
            self.assertEqual(candidate, original)
            blocks = {
                side: [
                    {
                        f"{target}/case": {
                            "rows": 1,
                            "scope": "native",
                            "samples": [1.0] * 20,
                            "metadata": stamp,
                        }
                        for target, stamp in targets.items()
                    }
                ]
                * 2
                for side, targets in stamps.items()
            }
            cases = {
                case["scenario"]: case
                for case in combine_blocks({"id": "rust", "family": "rust"}, blocks)
            }
            self.assertEqual(cases["core/case"]["status"], "ok")
            self.assertEqual(
                cases["new_target/case"]["result"]["verdict"], "new-coverage"
            )

    def test_target_stamps_fail_closed_on_dependency_changes_or_missing_identity(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            log, messages = write_inputs(root)
            measure(root, log, messages)
            identity = with_compiled_dependencies(
                {
                    "dependency_identity": {"rustc": "1.88", "cargo": "1.88"},
                    "machine_fingerprint": "a" * 64,
                    "scoped_workload_fingerprints": {"core": "b" * 64},
                },
                root,
                {"core": log},
            )
            before = _stamp_fingerprints({"candidate": identity}, {})
            for field in ("rustc", "cargo", "builds"):
                changed = copy.deepcopy(identity)
                compiled = changed["compiled_dependency_identity"]
                if field == "builds":
                    compiled["builds"]["core"][0]["features"] = ["changed"]
                else:
                    compiled[field] = "changed"
                with self.subTest(field=field):
                    self.assertNotEqual(
                        _stamp_fingerprints({"candidate": changed}, {}), before
                    )
            for field in ("rustc", "cargo", "builds", "schema"):
                changed = copy.deepcopy(identity)
                del changed["compiled_dependency_identity"][field]
                with (
                    self.subTest(missing=field),
                    self.assertRaises((ValueError, KeyError)),
                ):
                    _stamp_fingerprints({"candidate": changed}, {})
            for field in ("features", "profile"):
                changed = copy.deepcopy(identity)
                del changed["compiled_dependency_identity"]["builds"]["core"][0][field]
                with (
                    self.subTest(missing_artifact=field),
                    self.assertRaises((ValueError, KeyError)),
                ):
                    _stamp_fingerprints({"candidate": changed}, {})
            for builds in ({}, {"core": []}, {"core": [{}]}):
                changed = copy.deepcopy(identity)
                changed["compiled_dependency_identity"]["builds"] = builds
                with (
                    self.subTest(builds=builds),
                    self.assertRaises((ValueError, KeyError)),
                ):
                    _stamp_fingerprints({"candidate": changed}, {})

    def test_unused_lock_entries_do_not_invalidate_comparable_benchmarks(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            log, messages = write_inputs(root)
            before = measure(root, log, messages)
            with root.joinpath("Cargo.lock").open("a", encoding="utf-8") as lock:
                lock.write('\n[[package]]\nname = "mysql"\nversion = "1.0.0"\n')
            self.assertEqual(measure(root, log, messages), before)

    def test_used_dependency_and_build_changes_remain_incomparable(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            log, messages = write_inputs(root)
            before = measure(root, log, messages)
            for key, value in (
                ("features", ["simd"]),
                ("profile", {"opt_level": "0", "test": False}),
                ("target", {**messages[0]["target"], "kind": ["proc-macro"]}),
            ):
                with self.subTest(key=key):
                    changed = copy.deepcopy(messages)
                    changed[0][key] = value
                    self.assertNotEqual(measure(root, log, changed), before)
            lock = root / "Cargo.lock"
            lock.write_text(lock.read_text().replace("a" * 64, "b" * 64))
            self.assertNotEqual(measure(root, log, messages), before)
            lock.write_text(lock.read_text().replace("1.0.0", "2.0.0"))
            messages[0]["package_id"] = f"{REGISTRY}#arrow@2.0.0"
            self.assertNotEqual(measure(root, log, messages), before)

    def test_checkout_location_order_and_cache_hits_do_not_change_identity(self):
        with TemporaryDirectory() as left, TemporaryDirectory() as right:
            identities = []
            for raw in (left, right):
                root = Path(raw)
                log, messages = write_inputs(root)
                if raw == right:
                    messages[0]["fresh"] = False
                    messages[0]["filenames"] = ["/another/cache/arrow.rlib"]
                    messages = [messages[1], messages[0], messages[2]]
                identities.append(measure(root, log, messages))
            self.assertEqual(identities[0], identities[1])

    def test_incomplete_failed_or_unrecognized_builds_are_rejected(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            log, messages = write_inputs(root)
            invalid = [messages[:-1], messages[:1] + messages[-1:], messages[1:]]
            failed = copy.deepcopy(messages)
            failed[-1]["success"] = False
            invalid.append(failed)
            unknown = copy.deepcopy(messages)
            unknown[0]["package_id"] = "path+file:///external#1.0.0"
            invalid.append(unknown)
            for rows in invalid:
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    measure(root, log, rows)

    def test_core_features_are_included_and_inputs_are_not_mutated(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            log, messages = write_inputs(root)
            original = copy.deepcopy(messages)
            before = measure(root, log, messages)
            self.assertEqual(messages, original)
            messages[1]["features"] = ["new-feature"]
            self.assertNotEqual(measure(root, log, messages), before)


if __name__ == "__main__":
    unittest.main()
