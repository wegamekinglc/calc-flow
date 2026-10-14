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
    def test_ordinary_release_profile_requires_matching_source_seals(self):
        from scripts.benchmark_suite.issue406_reuse import _ordinary_release_codegen
        from scripts.toolkit import sha256_file

        with TemporaryDirectory() as raw:
            source = Path(raw)
            manifest = source / "Cargo.toml"
            manifest.write_text('[profile.release]\nlto="thin"\ncodegen-units=1\n')
            digest = sha256_file(manifest)
            build = {
                "source": raw,
                "source_files_sha256": {"Cargo.toml": digest},
            }
            sealed = {str(manifest): digest}
            actual = _ordinary_release_codegen(build, sealed)
            self.assertEqual(actual["calc_flow"]["linker-plugin-lto"], None)
            self.assertEqual(actual["calc_flow_python"]["lto"], "thin")
            self.assertEqual(actual["calc_flow"]["codegen-units"], "1")
            with self.assertRaisesRegex(ValueError, "sealed"):
                _ordinary_release_codegen(build, {})
            with self.assertRaisesRegex(ValueError, "sealed"):
                _ordinary_release_codegen({"source": raw}, sealed)
            manifest.write_text("[workspace]\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                _ordinary_release_codegen(build, sealed)

    def test_ordinary_release_profile_rejects_other_codegen_policies(self):
        from scripts.benchmark_suite.issue406_reuse import _ordinary_release_codegen
        from scripts.toolkit import sha256_file

        with TemporaryDirectory() as raw:
            manifest = Path(raw) / "Cargo.toml"
            for source in (
                "profile=false\n",
                "[profile]\nrelease=false\n",
                '[profile.release]\nlto="fat"\ncodegen-units=1\n',
                '[profile.release]\nlto="thin"\ncodegen-units=true\n',
                '[profile.release]\nlto="thin"\ncodegen-units=16\n',
                '[profile.release]\nlto="thin"\ncodegen-units=1\nopt-level=2\n',
                "[profile.release.package.calc-flow]\ncodegen-units=1\n",
            ):
                manifest.write_text(source)
                digest = sha256_file(manifest)
                build = {
                    "source": raw,
                    "source_files_sha256": {"Cargo.toml": digest},
                }
                with (
                    self.subTest(source=source),
                    self.assertRaisesRegex(ValueError, "profile"),
                ):
                    _ordinary_release_codegen(build, {str(manifest): digest})

    def test_ordinary_invocation_rejects_ambiguous_codegen_options(self):
        from scripts.benchmark_suite.issue406_reuse import _ordinary_invocation

        line = (
            "Running `rustc --crate-name calc_flow -C opt-level=3 "
            "-C embed-bitcode=no -C strip=debuginfo`"
        )
        for altered in (
            line[:-1] + " -C opt-level=3`",
            line[:-1] + " -C`",
            line[:-1] + " -C hidden`",
            line[:-1] + " --codegen opt-level=2`",
            line[:-1] + " --codegen=lto=off`",
            line[:-1] + " -g`",
            line[:-1] + " -O`",
        ):
            with (
                self.subTest(line=altered),
                self.assertRaisesRegex(ValueError, "invocation"),
            ):
                _ordinary_invocation(altered, "calc_flow", set())

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
        self._assert_ordinary_audit(False)

    def test_ordinary_thin_lto_audit_requires_the_sealed_release_profile(self):
        self._assert_ordinary_audit(True)

    def _assert_ordinary_audit(self, thin_lto):
        from scripts.benchmark_suite.issue406_reuse import validate_ordinary_audit
        from scripts.toolkit import fingerprint_json, sha256_file, write_json

        with TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            source.mkdir()
            manifest = source / "Cargo.toml"
            manifest.write_text(
                '[profile.release]\nlto = "thin"\ncodegen-units = 1\n'
                if thin_lto
                else "[workspace]\n"
            )
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
            if thin_lto:
                core = core.replace(
                    "-C embed-bitcode=no", "-C linker-plugin-lto -C codegen-units=1"
                )
                binding = binding.replace(
                    "-C embed-bitcode=no", "-C lto=thin -C codegen-units=1"
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
            sealed = {str(p): sha256_file(p) for p in (path, replay, log, manifest)}
            build = {
                "source": str(source),
                "python": "python",
                "source_files_sha256": {"Cargo.toml": sha256_file(manifest)},
            }
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
            RUSTDOC_NEW,
            RUSTDOC_OLD,
            RUSTDOC_PATH,
            helper_reconstruction,
            reconstructed_overlay,
        )

        new_doc = (
            Path(__file__).parent
            / "fixtures/issue406-rustdoc/benchmark_internals.rs.txt"
        ).read_bytes()
        old_doc = new_doc.replace(RUSTDOC_NEW, RUSTDOC_OLD, 1)
        old, new = b"old\n", b"old\naddition\n"
        witness = {
            "copied_files": {
                "helper.rs": {
                    "historical_sha256": hashlib.sha256(old).hexdigest(),
                    "candidate_build_sha256": hashlib.sha256(new).hexdigest(),
                    "exact_additions": ["addition\n"],
                },
                RUSTDOC_PATH: {
                    "historical_sha256": hashlib.sha256(old_doc).hexdigest(),
                    "candidate_build_sha256": hashlib.sha256(old_doc).hexdigest(),
                    "exact_additions": [],
                },
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
                reconstructed_overlay(
                    {"helper.rs": new, RUSTDOC_PATH: new_doc}, record, reconstruction
                ),
                {"helper.rs": old, RUSTDOC_PATH: old_doc},
            )
            self.assertEqual(record["harness_git_sha"], FROZEN)
            with self.assertRaisesRegex(ValueError, "reconstruction"):
                reconstructed_overlay(
                    {"helper.rs": new, RUSTDOC_PATH: new_doc},
                    {**record, "harness_git_sha": HISTORICAL},
                    reconstruction,
                )
            with self.assertRaisesRegex(ValueError, "reconstruction"):
                reconstructed_overlay(
                    {"helper.rs": new, RUSTDOC_PATH: new_doc},
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
