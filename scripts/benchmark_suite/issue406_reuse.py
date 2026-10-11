"""The one reviewed historical native-source compatibility witness."""

from __future__ import annotations

import hashlib
import json
import shlex
import zipfile
from pathlib import Path

from scripts.toolkit import command_output, fingerprint_json, sha256_file

WITNESS_SHA256 = "337a245cf9bb68c18ce11c4f850449033e648833fc7a1dc3f0e45b91f2a1d9d6"
HISTORICAL = "a2f31da023dccb00518d2e043aa628d663aa93fb"
FROZEN = "121386713378459a4f16f63ee4848a9db7a5d466"


def load_witness(path: Path) -> dict:
    """Accept exactly the reviewed bytes, retaining their historical caveats."""
    if sha256_file(path) != WITNESS_SHA256:
        raise ValueError("historical compatibility witness differs from its review")
    return json.loads(path.read_text())


def restore_copied(overlay: dict, files: dict) -> dict:
    """Recover old actual helper bytes by removing only exact reviewed additions."""
    historical = {}
    for path, record in files.items():
        contents = overlay[path]
        if hashlib.sha256(contents).hexdigest() != record["candidate_build_sha256"]:
            raise ValueError("current copied source differs from reviewed witness")
        for addition in record["exact_additions"]:
            value = addition.encode()
            if contents.count(value) != 1:
                raise ValueError("reviewed source addition is missing or duplicated")
            contents = contents.replace(value, b"", 1)
        if hashlib.sha256(contents).hexdigest() != record["historical_sha256"]:
            raise ValueError("recovered source differs from historical bytes")
        historical[path] = contents
    return {**overlay, **historical}


def reviewed_reuse(root: Path, build: str, current: str, path: Path) -> dict:
    """Bind exact current and historical recipes, file blobs, and reviewed scope."""
    from scripts.benchmark_suite.issue406_harness import COPIED_FILES

    witness = load_witness(path)
    history = command_output(["git", "rev-list", current], cwd=root).splitlines()
    if build != HISTORICAL or any(
        revision not in history for revision in (HISTORICAL, FROZEN)
    ):
        raise ValueError("unreviewed historical native build")
    recipe = witness["recipe"]
    old = command_output(["git", "show", f"{build}:{recipe['path']}"], cwd=root).split(
        recipe["boundary"]
    )[0]
    active = (root / recipe["path"]).read_text().split(recipe["boundary"])[0]
    if (
        hashlib.sha256(old.encode()).hexdigest() != recipe["historical_prefix_sha256"]
        or hashlib.sha256(active.encode()).hexdigest()
        != recipe["candidate_prefix_sha256"]
        or set(witness["copied_files"]) != set(COPIED_FILES)
    ):
        raise ValueError("reviewed historical native recipe changed")
    blobs = {}
    for revision, label in (
        (build, "historical_git_blob"),
        (current, "candidate_build_git_blob"),
    ):
        values = command_output(
            ["git", "rev-parse", *(f"{revision}:{item}" for item in COPIED_FILES)],
            cwd=root,
        ).splitlines()
        blobs[revision] = dict(zip(COPIED_FILES, values, strict=True))
        if any(
            blobs[revision][item] != witness["copied_files"][item][label]
            for item in COPIED_FILES
        ):
            raise ValueError("reviewed copied native source changed")
    restore_copied(
        {item: (root / item).read_bytes() for item in COPIED_FILES},
        witness["copied_files"],
    )
    return {
        "original_build_harness_git_sha": build,
        "execution_harness_git_sha": current,
        "historical_recipe_prefix_sha256": recipe["historical_prefix_sha256"],
        "recipe_prefix_sha256": recipe["candidate_prefix_sha256"],
        "historical_copied_file_git_blobs": blobs[build],
        "copied_file_git_blobs": blobs[current],
        "compatibility_witness_sha256": WITNESS_SHA256,
    }


def historical_overlay(overlay: dict, witness_path: Path) -> dict:
    return restore_copied(overlay, load_witness(witness_path)["copied_files"])


def helper_reconstruction(path: Path) -> dict:
    from scripts.benchmark_suite.issue406_fresh import MAIN

    witness = load_witness(path)
    return {
        "schema": "calc-flow.issue406-historical-helper-reconstruction.v1",
        "product_commit": MAIN,
        "build_harness_commit": FROZEN,
        "historical_helper_commit": HISTORICAL,
        "witness_path": str(path),
        "witness_sha256": WITNESS_SHA256,
        "copied_files": {
            item: {
                "input_sha256": row["candidate_build_sha256"],
                "output_sha256": row["historical_sha256"],
                "removed_literals_sha256": [
                    hashlib.sha256(value.encode()).hexdigest()
                    for value in row["exact_additions"]
                ],
            }
            for item, row in witness["copied_files"].items()
        },
    }


def reconstructed_overlay(overlay: dict, record: dict, reconstruction: dict) -> dict:
    from scripts.benchmark_suite.issue406_fresh import MAIN

    path = Path(reconstruction["witness_path"])
    if (
        reconstruction != helper_reconstruction(path)
        or record.get("helper_reconstruction") != reconstruction
        or record.get("product_git_sha") != MAIN
        or record.get("harness_git_sha") != FROZEN
    ):
        raise ValueError("unreviewed main traced helper reconstruction")
    return historical_overlay(overlay, path)


def validate_traced_audit(release: dict, attestation: dict, sealed: dict) -> list:
    from scripts.benchmark_suite.issue406_harness import validate_native_profile
    from scripts.benchmark_suite.rust_provenance import _build_messages

    output = Path(release["build_output"])
    audit_path, replay = (
        output / "wheel-cache-audit.json",
        output / "wheel-cache-audit.jsonl",
    )
    if any(str(path) not in sealed for path in (audit_path, replay)):
        raise ValueError("traced dependency audit is not sealed")
    audit = json.loads(audit_path.read_text())
    messages = _build_messages("traced wheel", replay)
    compiled = [row for row in messages if row.get("reason") == "compiler-artifact"]
    if not compiled or any(row.get("fresh") is not True for row in compiled):
        raise ValueError("traced dependency replay is not entirely fresh")
    expected = {
        "native_before_sha256": release["native_sha256"],
        "compiled_after_sha256": release["native_sha256"],
        "wheel_before_sha256": release["wheel_sha256"],
        "wheel_after_sha256": release["wheel_sha256"],
        "replay_sha256": sha256_file(replay),
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("traced dependency audit artifact identities differ")
    source = Path(attestation["build_source"])
    argv = [
        "cargo",
        "rustc",
        "--profile",
        "release",
        "--features",
        "benchmark-internals",
        "--message-format",
        "json-render-diagnostics",
        "-vv",
        "--locked",
        "--offline",
        "--manifest-path",
        str(source / "crates/calc-flow-python/Cargo.toml"),
        "--lib",
    ]
    if audit.get("argv") != argv:
        raise ValueError("traced dependency audit command differs")
    artifacts = _wheel_artifact_identity(attestation, compiled)
    if sorted(audit["artifacts"], key=fingerprint_json) != artifacts:
        raise ValueError("traced dependency inventory differs from compiler messages")
    validate_native_profile(artifacts, checkpoint_off=False)
    binding = [
        row
        for row in artifacts
        if row["package"].get("workspace_package") == "crates/calc-flow-python"
        and row["target"]["kind"] == ["cdylib"]
    ]
    if (
        len(binding) != 1
        or binding[0]["profile"].get("test") is not False
        or binding[0]["profile"].get("opt_level") != "3"
        or binding[0]["features"]
        != ["benchmark-internals", "connector-file", "default"]
    ):
        raise ValueError("traced binding profile differs")
    return artifacts


def _wheel_artifact_identity(attestation: dict, messages: list) -> list:
    from scripts.benchmark_suite.rust_provenance import (
        _artifact_identity,
        _locked_packages,
    )

    source = Path(attestation["build_source"])
    locked = _locked_packages(Path(attestation["product_source"]))
    workspaces = {
        source / f"crates/{name}/Cargo.toml": f"crates/{name}"
        for name in ("calc-flow", "calc-flow-python", "calc-flow-connectors")
    }
    artifacts = {}
    for row in messages:
        workspace = workspaces.get(Path(row["manifest_path"]))
        if workspace is None:
            identity = _artifact_identity(source, row, locked)
        else:
            identity = {
                "package": {"workspace_package": workspace},
                "target": {
                    key: row["target"][key]
                    for key in ("name", "kind", "crate_types", "edition")
                },
                "features": sorted(row["features"]),
                "profile": row["profile"],
            }
        artifacts[fingerprint_json(identity)] = identity
    return [artifacts[key] for key in sorted(artifacts)]


def validate_ordinary_release(release: dict, sealed: dict) -> dict:
    """Validate a production wheel, its installed members and recorded source."""
    from scripts.benchmark_suite.issue406_harness import _lock_identity

    metadata = Path(release["build_metadata"])
    build = json.loads(metadata.read_text())
    revision = build.get("revision", build.get("candidate_revision"))
    if build.get("failure") is not None or revision != release["product_commit"]:
        raise ValueError("ordinary build product or completion differs")
    if str(metadata) not in sealed:
        raise ValueError("ordinary build metadata is not sealed")
    for kind in ("native", "wheel"):
        path, digest = release[f"{kind}_path"], release[f"{kind}_sha256"]
        if (
            build[f"{kind}_path"] != path
            or build[f"{kind}_sha256"] != digest
            or sealed.get(path) != digest
        ):
            raise ValueError("ordinary build artifact descriptor differs")
    _validate_ordinary_source(build, sealed)
    validate_wheel_members(release, sealed)
    commands = [item for item in build["commands"] if item["label"] == "build"]
    if len(commands) != 1 or commands[0]["returncode"] != 0:
        raise ValueError("ordinary release build command is incomplete")
    arguments = commands[0]["command"]
    if (
        arguments[:4] != [build["python"], "-m", "maturin", "build"]
        or not {
            "--release",
            "--locked",
            "--offline",
            "--strip",
            "false",
            "--out",
        }.issubset(arguments)
        or "--features" in arguments
        or "--profile" in arguments
    ):
        raise ValueError("ordinary release build configuration differs")
    return {
        "environment": {
            key: value
            for key, value in build["environment"].items()
            if key != "CARGO_TARGET_DIR"
        },
        "python": build["python"],
        "rustc": build["rustc"].rstrip("\r\n"),
        "lock_fingerprint": _lock_identity(Path(build["source"])),
        "compiled_artifacts": validate_ordinary_audit(release, build, sealed),
    }


def ordinary_audit_paths(release: dict) -> dict[str, Path]:
    """Resolve the recorded ordinary build's audit files without running Cargo."""
    root = Path(release["build_metadata"]).parent
    defaults = {
        "audit_path": root / "wheel-audit.json",
        "replay_path": root / "wheel-cache-audit.jsonl",
        "build_log": root / "candidate-build.log",
    }
    supplied = release.get("ordinary_audit", {})
    if set(supplied) - set(defaults):
        raise ValueError("unexpected ordinary audit descriptor field")
    return {key: Path(supplied.get(key, path)) for key, path in defaults.items()}


def validate_ordinary_audit(release: dict, build: dict, sealed: dict) -> list:
    """Bind unchanged wheel bytes to a fresh complete compiler artifact replay."""
    from scripts.benchmark_suite.rust_provenance import _build_messages

    paths = ordinary_audit_paths(release)
    if any(sealed.get(str(path)) != sha256_file(path) for path in paths.values()):
        raise ValueError("ordinary dependency audit is not sealed")
    audit = json.loads(paths["audit_path"].read_text())
    messages = _build_messages("ordinary wheel", paths["replay_path"])
    compiled = [row for row in messages if row.get("reason") == "compiler-artifact"]
    if not compiled or any(row.get("fresh") is not True for row in compiled):
        raise ValueError("ordinary dependency replay is not entirely fresh")
    expected = {
        "native_sha256": release["native_sha256"],
        "compiled_before_sha256": release["native_sha256"],
        "compiled_after_sha256": release["native_sha256"],
        "wheel_before_sha256": release["wheel_sha256"],
        "wheel_after_sha256": release["wheel_sha256"],
        "replay_sha256": sha256_file(paths["replay_path"]),
    }
    if any(audit.get(key) != value for key, value in expected.items()):
        raise ValueError("ordinary dependency audit artifact identities differ")
    _validate_ordinary_invocations(audit, build, paths["build_log"])
    artifacts = _wheel_artifact_identity(
        {"build_source": build["source"], "product_source": build["source"]},
        compiled,
    )
    if (
        len(artifacts) != len(compiled)
        or audit.get("artifacts") != artifacts
        or audit.get("dependency_fingerprint") != fingerprint_json(artifacts)
    ):
        raise ValueError("ordinary dependency inventory differs from compiler messages")
    _validate_ordinary_profiles(artifacts)
    return artifacts


def _validate_ordinary_profiles(artifacts: list) -> None:
    expected_profile = {
        "debug_assertions": False,
        "debuginfo": 0,
        "opt_level": "3",
        "overflow_checks": False,
        "test": False,
    }
    workspaces = {row["package"].get("workspace_package") for row in artifacts}
    if not {
        "crates/calc-flow",
        "crates/calc-flow-python",
        "crates/calc-flow-connectors",
    }.issubset(workspaces) or not any(
        "checksum" in row["package"] for row in artifacts
    ):
        raise ValueError("ordinary wheel compiled inventory is incomplete")
    for package, name, kind, features in (
        ("calc-flow", "calc_flow", ["lib"], []),
        (
            "calc-flow-python",
            "calc_flow_python",
            ["cdylib"],
            ["connector-file", "default"],
        ),
    ):
        found = [
            row
            for row in artifacts
            if row["package"].get("workspace_package") == f"crates/{package}"
            and row["target"]["name"] == name
            and row["target"]["kind"] == kind
        ]
        if (
            len(found) != 1
            or found[0]["profile"] != expected_profile
            or found[0]["features"] != features
        ):
            raise ValueError("ordinary wheel core or binding profile differs")


def _ordinary_invocation(line: str, crate: str, features: set) -> None:
    tokens = shlex.split(
        line[line.index("Running `") + len("Running `") : line.rindex("`")]
    )
    if any(
        token.startswith(("--cfg=", "--target="))
        or token.startswith("-C")
        and token != "-C"
        for token in tokens
    ):
        raise ValueError("ordinary compiler invocation has attached options")
    cfg = [tokens[index + 1] for index, token in enumerate(tokens) if token == "--cfg"]
    actual = {
        value.removeprefix('feature="').removesuffix('"')
        for value in cfg
        if value.startswith('feature="')
    }
    codegen = dict(
        tokens[index + 1].split("=", 1)
        for index, token in enumerate(tokens)
        if token == "-C"
    )
    normalized = {
        key: value
        for key, value in codegen.items()
        if key not in ("metadata", "extra-filename")
    }
    expected_cfg = {f'feature="{feature}"' for feature in features}
    if crate == "calc_flow_python":
        expected_cfg |= {f"Py_3_{minor}" for minor in range(7, 14)}
    if (
        tokens[tokens.index("--crate-name") + 1] != crate
        or actual != features
        or set(cfg) != expected_cfg
        or "--test" in tokens
        or "--target" in tokens
        or normalized != {"opt-level": "3", "embed-bitcode": "no", "strip": "debuginfo"}
    ):
        raise ValueError("ordinary compiler invocation configuration differs")


def _validate_ordinary_invocations(audit: dict, build: dict, log: Path) -> None:
    lines = log.read_text().splitlines()
    for key, crate, features in (
        ("actual_core_invocation", "calc_flow", set()),
        (
            "actual_binding_invocation",
            "calc_flow_python",
            {"connector-file", "default"},
        ),
    ):
        found = [
            line
            for line in lines
            if "Running `" in line and f"--crate-name {crate} " in line
        ]
        if len(found) != 1 or found[0] != audit.get(key):
            raise ValueError("ordinary compiler invocation differs from build log")
        _ordinary_invocation(found[0], crate, features)
    marker = "maturin::compile: Running "
    commands = [line.split(marker, 1)[1] for line in lines if marker in line]
    expected = [
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
        str(Path(build["source"]) / "crates/calc-flow-python/Cargo.toml"),
        "--lib",
    ]
    environment = {
        "PYO3_BUILD_EXTENSION_MODULE": "1",
        "PYO3_ENVIRONMENT_SIGNATURE": "cpython-3.13-64bit",
        "PYO3_PYTHON": build["python"],
        "PYTHON_SYS_EXECUTABLE": build["python"],
    }
    if len(commands) != 1:
        raise ValueError("ordinary Maturin invocation count differs")
    tokens = shlex.split(commands[0])
    cargo = tokens.index("cargo")
    if (
        tokens[:3] != ["env", "-u", "CARGO"]
        or tokens[cargo:] != expected
        or dict(token.split("=", 1) for token in tokens[3:cargo]) != environment
        or audit.get("argv") != expected
        or audit.get("environment_assignments") != environment
    ):
        raise ValueError("ordinary Maturin invocation or environment differs")


def _validate_ordinary_source(build: dict, sealed: dict) -> None:
    source = Path(build["source"])
    original = build["source_files_sha256"]
    if not original or "Cargo.lock" not in original:
        raise ValueError("ordinary build source inventory is incomplete")
    for relative, digest in original.items():
        path = source / relative
        expected = (
            build["cargo_lock_after_sha256"] if relative == "Cargo.lock" else digest
        )
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or sealed.get(str(path)) != expected
            or sha256_file(path) != expected
        ):
            raise ValueError(
                "ordinary build source differs from its recorded inventory"
            )


def validate_wheel_members(release: dict, sealed: dict) -> None:
    site = Path(release["site"])
    installed = {
        str(path.relative_to(site)): path
        for path in (site / "calc_flow").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    with zipfile.ZipFile(release["wheel_path"]) as archive:
        members = {
            name
            for name in archive.namelist()
            if name.startswith("calc_flow/") and not name.endswith("/")
        }
        if not members or members != set(installed):
            raise ValueError("ordinary installed member inventory differs from wheel")
        for name in members:
            with archive.open(name) as stream:
                expected = hashlib.file_digest(stream, "sha256").hexdigest()
            path = installed[name]
            if sealed.get(str(path)) != expected or sha256_file(path) != expected:
                raise ValueError("ordinary installed member differs from wheel")
