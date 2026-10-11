from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.benchmark_suite.issue406_reuse import (
    load_witness,
    restore_copied,
    validate_ordinary_release,
)


class HistoricalReuseTests(unittest.TestCase):
    def test_ordinary_binding_requires_exact_python_313_cfgs(self):
        from scripts.benchmark_suite.issue406_reuse import _ordinary_invocation

        features = {"connector-file", "default"}
        line = (
            "Running `rustc --crate-name calc_flow_python -C opt-level=3 "
            "-C embed-bitcode=no -C strip=debuginfo "
            "--cfg 'feature=\"connector-file\"' --cfg 'feature=\"default\"' "
            "--cfg Py_3_7 --cfg Py_3_8 --cfg Py_3_9 --cfg Py_3_10 "
            "--cfg Py_3_11 --cfg Py_3_12 --cfg Py_3_13`"
        )
        _ordinary_invocation(line, "calc_flow_python", features)
        for altered in (
            line.replace("--cfg Py_3_7 ", ""),
            line[:-1] + " --cfg Py_3_14`",
            line[:-1] + " --cfg hidden_override`",
        ):
            with self.assertRaisesRegex(ValueError, "invocation"):
                _ordinary_invocation(altered, "calc_flow_python", features)

    def test_ordinary_audit_binds_fresh_inventory_and_actual_release_invocations(self):
        from scripts.benchmark_suite.issue406_reuse import validate_ordinary_audit
        from scripts.toolkit import fingerprint_json, sha256_file, write_json

        with TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            descriptor = {
                "build_metadata": str(root / "build.json"),
                "native_sha256": "a" * 64,
                "wheel_sha256": "b" * 64,
            }
            profile = {
                "debug_assertions": False,
                "debuginfo": 0,
                "opt_level": "3",
                "overflow_checks": False,
                "test": False,
            }
            artifacts = [
                {
                    "package": {"workspace_package": f"crates/{package}"},
                    "target": {"name": crate, "kind": [kind]},
                    "features": features,
                    "profile": profile,
                }
                for package, crate, kind, features in (
                    ("calc-flow", "calc_flow", "lib", []),
                    (
                        "calc-flow-python",
                        "calc_flow_python",
                        "cdylib",
                        ["connector-file", "default"],
                    ),
                    ("calc-flow-connectors", "calc_flow_connectors", "lib", []),
                )
            ] + [{"package": {"checksum": "dependency"}}]
            artifacts.sort(key=fingerprint_json)
            replay = root / "wheel-cache-audit.jsonl"
            messages = [
                {"reason": "compiler-artifact", "fresh": True} for _ in artifacts
            ]
            messages.append({"reason": "build-finished", "success": True})
            replay.write_text("\n".join(map(json.dumps, messages)) + "\n")
            argv = [
                "cargo",
                "rustc",
                "--profile",
                "release",
                "--message-format",
                "json-render-diagnostics",
                "-vv",
                "--locked",
                "--offline",
                "--manifest-path",
                str(source / "crates/calc-flow-python/Cargo.toml"),
                "--lib",
            ]
            assignments = {
                "PYO3_BUILD_EXTENSION_MODULE": "1",
                "PYO3_ENVIRONMENT_SIGNATURE": "cpython-3.13-64bit",
                "PYO3_PYTHON": "python",
                "PYTHON_SYS_EXECUTABLE": "python",
            }
            core = (
                "Running `rustc --crate-name calc_flow -C opt-level=3 "
                "-C embed-bitcode=no -C strip=debuginfo`"
            )
            binding = (
                "Running `rustc --crate-name calc_flow_python -C opt-level=3 "
                "-C embed-bitcode=no -C strip=debuginfo "
                "--cfg 'feature=\"connector-file\"' --cfg 'feature=\"default\"' "
                "--cfg Py_3_7 --cfg Py_3_8 --cfg Py_3_9 --cfg Py_3_10 "
                "--cfg Py_3_11 --cfg Py_3_12 --cfg Py_3_13`"
            )
            log = root / "candidate-build.log"
            log.write_text(
                core
                + "\n"
                + binding
                + "\n"
                + "maturin::compile: Running env -u CARGO "
                + " ".join(f"{key}={value}" for key, value in assignments.items())
                + " "
                + " ".join(argv)
                + "\n"
            )
            audit = {
                "argv": argv,
                "environment_assignments": assignments,
                "actual_core_invocation": core,
                "actual_binding_invocation": binding,
                "native_sha256": "a" * 64,
                "compiled_before_sha256": "a" * 64,
                "compiled_after_sha256": "a" * 64,
                "wheel_before_sha256": "b" * 64,
                "wheel_after_sha256": "b" * 64,
                "replay_sha256": sha256_file(replay),
                "artifacts": artifacts,
                "dependency_fingerprint": fingerprint_json(artifacts),
            }
            path = root / "wheel-audit.json"
            write_json(path, audit)
            sealed = {str(p): sha256_file(p) for p in (path, replay, log)}
            build = {"source": str(source), "python": "python"}
            with patch(
                "scripts.benchmark_suite.issue406_reuse._wheel_artifact_identity",
                return_value=artifacts,
            ):
                self.assertEqual(
                    validate_ordinary_audit(descriptor, build, sealed), artifacts
                )
                with self.assertRaisesRegex(ValueError, "sealed"):
                    validate_ordinary_audit(descriptor, build, {})
                write_json(path, {**audit, "artifacts": artifacts[:-1]})
                sealed[str(path)] = sha256_file(path)
                with self.assertRaisesRegex(ValueError, "inventory"):
                    validate_ordinary_audit(descriptor, build, sealed)
                write_json(path, audit)
                sealed[str(path)] = sha256_file(path)
                log.write_text(log.read_text().replace("opt-level=3", "opt-level=2"))
                sealed[str(log)] = sha256_file(log)
                with self.assertRaisesRegex(ValueError, "invocation"):
                    validate_ordinary_audit(descriptor, build, sealed)
                hidden = core[:-1] + " --cfg hidden_override`"
                log.write_text(
                    log.read_text()
                    .replace("opt-level=2", "opt-level=3")
                    .replace(core, hidden)
                )
                write_json(path, {**audit, "actual_core_invocation": hidden})
                sealed.update({str(p): sha256_file(p) for p in (log, path)})
                with self.assertRaisesRegex(ValueError, "invocation"):
                    validate_ordinary_audit(descriptor, build, sealed)

    def test_traced_cache_audit_rejects_missing_or_nonfresh_build_evidence(self):
        from scripts.benchmark_suite.issue406_reuse import validate_traced_audit
        from scripts.toolkit import sha256_file, write_json

        with TemporaryDirectory() as raw:
            root = Path(raw)
            descriptor = {
                "build_output": str(root),
                "native_sha256": "a" * 64,
                "wheel_sha256": "b" * 64,
            }
            record = {
                "build_source": str(root / "source"),
                "product_source": str(root / "product"),
            }
            replay = root / "wheel-cache-audit.jsonl"
            replay.write_text(
                json.dumps({"reason": "compiler-artifact", "fresh": False})
                + "\n"
                + json.dumps({"reason": "build-finished", "success": True})
                + "\n"
            )
            audit = root / "wheel-cache-audit.json"
            write_json(
                audit,
                {
                    "native_before_sha256": "a" * 64,
                    "compiled_after_sha256": "a" * 64,
                    "wheel_before_sha256": "b" * 64,
                    "wheel_after_sha256": "b" * 64,
                    "replay_sha256": sha256_file(replay),
                    "artifacts": [],
                },
            )
            with self.assertRaisesRegex(ValueError, "sealed"):
                validate_traced_audit(descriptor, record, {})
            files = {str(path): sha256_file(path) for path in (audit, replay)}
            with self.assertRaisesRegex(ValueError, "fresh"):
                validate_traced_audit(descriptor, record, files)

    def test_reconstructed_build_retains_actual_helper_revision(self):
        from scripts.benchmark_suite.issue406_fresh import MAIN
        from scripts.benchmark_suite.issue406_reuse import (
            FROZEN,
            HISTORICAL,
            helper_reconstruction,
            reconstructed_overlay,
        )

        old, new = b"old\n", b"old\naddition\n"
        witness = {
            "copied_files": {
                "helper.rs": {
                    "historical_sha256": hashlib.sha256(old).hexdigest(),
                    "candidate_build_sha256": hashlib.sha256(new).hexdigest(),
                    "exact_additions": ["addition\n"],
                }
            }
        }
        with patch(
            "scripts.benchmark_suite.issue406_reuse.load_witness", return_value=witness
        ):
            reconstruction = helper_reconstruction(Path("/reviewed.json"))
            record = {
                "product_git_sha": MAIN,
                "harness_git_sha": FROZEN,
                "helper_reconstruction": reconstruction,
            }
            self.assertEqual(
                reconstructed_overlay({"helper.rs": new}, record, reconstruction),
                {"helper.rs": old},
            )
            self.assertEqual(record["harness_git_sha"], FROZEN)
            with self.assertRaisesRegex(ValueError, "reconstruction"):
                reconstructed_overlay(
                    {"helper.rs": new},
                    {**record, "harness_git_sha": HISTORICAL},
                    reconstruction,
                )
            with self.assertRaisesRegex(ValueError, "reconstruction"):
                reconstructed_overlay(
                    {"helper.rs": new},
                    record,
                    {**reconstruction, "product_commit": "x"},
                )

    def test_ordinary_release_revalidates_source_members_and_product(self):
        import zipfile

        from scripts.toolkit import sha256_file, write_json

        with TemporaryDirectory() as raw:
            root = Path(raw)
            source, site = root / "source", root / "site"
            source.mkdir()
            (source / "Cargo.lock").write_text("version = 4\n")
            (site / "calc_flow").mkdir(parents=True)
            native = site / "calc_flow/_native.so"
            native.write_bytes(b"synthetic native")
            wheel = root / "test.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.write(native, "calc_flow/_native.so")
            lock = sha256_file(source / "Cargo.lock")
            build = {
                "revision": "a" * 40,
                "failure": None,
                "source": str(source),
                "source_files_sha256": {"Cargo.lock": lock},
                "cargo_lock_after_sha256": lock,
                "native_path": str(native),
                "native_sha256": sha256_file(native),
                "wheel_path": str(wheel),
                "wheel_sha256": sha256_file(wheel),
                "environment": {"RUSTFLAGS": None},
                "python": "python",
                "rustc": "rustc\n",
                "commands": [
                    {
                        "label": "build",
                        "returncode": 0,
                        "command": [
                            "python",
                            "-m",
                            "maturin",
                            "build",
                            "--release",
                            "--locked",
                            "--offline",
                            "--strip",
                            "false",
                            "--out",
                            str(root),
                        ],
                    }
                ],
            }
            metadata = root / "build.json"
            write_json(metadata, build)
            release = {
                **{
                    key: build[key]
                    for key in (
                        "native_path",
                        "native_sha256",
                        "wheel_path",
                        "wheel_sha256",
                    )
                },
                "site": str(site),
                "product_commit": "a" * 40,
                "build_metadata": str(metadata),
            }
            files = {
                str(path): sha256_file(path)
                for path in (metadata, native, wheel, source / "Cargo.lock")
            }
            with patch(
                "scripts.benchmark_suite.issue406_reuse.validate_ordinary_audit",
                return_value=["compiled identity"],
            ):
                result = validate_ordinary_release(release, files)
            self.assertEqual(result["rustc"], "rustc")
            self.assertEqual(result["compiled_artifacts"], ["compiled identity"])
            with self.assertRaisesRegex(ValueError, "product"):
                validate_ordinary_release(
                    {**release, "product_commit": "b" * 40}, files
                )
            native.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError, "member"):
                validate_ordinary_release(release, files)

    def test_reconstruction_preserves_historical_bytes_and_other_product_overlay(self):
        old, added = b"old acceptance path\n", b"new correctness check\n"
        witness = {
            "helper.rs": {
                "historical_sha256": hashlib.sha256(old).hexdigest(),
                "candidate_build_sha256": hashlib.sha256(old + added).hexdigest(),
                "exact_additions": [added.decode()],
            }
        }
        overlay = {"helper.rs": old + added, "runtime.rs": b"product runtime"}
        result = restore_copied(overlay, witness)
        self.assertEqual(result, {"helper.rs": old, "runtime.rs": b"product runtime"})
        self.assertEqual(overlay["helper.rs"], old + added)
        for invalid in (old, old + added + added, b"changed acceptance path\n" + added):
            with self.assertRaisesRegex(ValueError, "source"):
                restore_copied({**overlay, "helper.rs": invalid}, witness)

    def test_witness_must_match_its_exact_reviewed_digest(self):
        with TemporaryDirectory() as raw:
            path = Path(raw) / "witness.json"
            path.write_text(json.dumps({"approved": False}))
            with self.assertRaisesRegex(ValueError, "witness"):
                load_witness(path)
