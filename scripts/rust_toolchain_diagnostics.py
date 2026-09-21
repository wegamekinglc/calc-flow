"""Read bounded Rust installation evidence without activating a toolchain."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path

TEXT_LIMIT = 64 * 1024
HASH_LIMIT = 64 * 1024 * 1024
TOOLCHAIN_LIMIT = 4
MANIFEST_LIMIT = 16


def command_result(command: list[str], directory: Path) -> dict[str, object]:
    with tempfile.TemporaryFile() as log:
        try:
            result = subprocess.run(
                command,
                cwd=directory,
                env={**os.environ, "RUSTUP_AUTO_INSTALL": "0"},
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=5,
                check=False,
            )
            status: dict[str, object] = {"exit_code": result.returncode}
        except (OSError, subprocess.TimeoutExpired) as error:
            status = {"error": str(error)}
        log.seek(0)
        content = log.read(TEXT_LIMIT + 1)
    return {
        "command": command,
        **status,
        "output": content[:TEXT_LIMIT].decode("utf-8", errors="replace"),
        "truncated": len(content) > TEXT_LIMIT,
    }


def regular_file(path: Path, size: int, *, text: bool) -> dict[str, object]:
    result: dict[str, object] = {"hash_limit_exceeded": size > HASH_LIMIT}
    if size <= HASH_LIMIT:
        with path.open("rb") as source:
            result["sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    if text:
        with path.open("rb") as source:
            content = source.read(TEXT_LIMIT + 1)
        result.update(
            text=content[:TEXT_LIMIT].decode("utf-8", errors="replace"),
            truncated=len(content) > TEXT_LIMIT,
        )
    return result


def file_record(path: Path, *, text: bool = False) -> dict[str, object]:
    try:
        info = path.lstat()
        result: dict[str, object] = {
            "path": str(path),
            "mode": stat.filemode(info.st_mode),
            "uid": info.st_uid,
            "gid": info.st_gid,
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "inode": info.st_ino,
            "device": info.st_dev,
            "type": "other",
        }
        if stat.S_ISLNK(info.st_mode):
            result.update(type="symlink", link_target=os.readlink(path))
        elif stat.S_ISREG(info.st_mode):
            result.update(type="regular", **regular_file(path, info.st_size, text=text))
        return result
    except OSError as error:
        return {"path": str(path), "error": str(error)}


def manifest_records(toolchain: Path) -> list[dict[str, object]]:
    directory = toolchain / "lib/rustlib"
    fixed = (
        "components",
        "rust-installer-version",
        "multirust-config.toml",
        "multirust-channel-manifest.toml",
    )
    manifests = sorted(
        itertools.islice(directory.glob("manifest-*"), MANIFEST_LIMIT + 1)
    )
    return [
        *(file_record(directory / name, text=True) for name in fixed),
        *(file_record(path, text=True) for path in manifests[:MANIFEST_LIMIT]),
        {
            "path": str(directory),
            "manifest_limit_exceeded": len(manifests) > MANIFEST_LIMIT,
        },
    ]


def toolchain_records(rustup_home: Path) -> list[dict[str, object]]:
    directory = rustup_home / "toolchains"
    paths = sorted(itertools.islice(directory.glob("1.88.0-*"), TOOLCHAIN_LIMIT + 1))
    records: list[dict[str, object]] = []
    for path in paths[:TOOLCHAIN_LIMIT]:
        records.append(file_record(path))
        records.extend(
            file_record(path / "bin" / name)
            for name in ("cargo-clippy", "clippy-driver")
        )
        records.extend(manifest_records(path))
    records.append(
        {
            "path": str(directory),
            "toolchain_limit_exceeded": len(paths) > TOOLCHAIN_LIMIT,
        }
    )
    return records


def cache_result() -> dict[str, object]:
    raw = os.environ.get("RUST_DIAGNOSTICS_UV", "{}")
    try:
        return {**json.loads(raw), "directory": os.environ.get("UV_CACHE_DIR")}
    except ValueError as error:
        return {"error": str(error)}


def snapshot(output: Path, phase: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rustup_home = Path(os.environ.get("RUSTUP_HOME", str(Path.home() / ".rustup")))
    cargo_home = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo")))
    rustup = shutil.which("rustup") or "rustup"
    files = [
        file_record(Path(rustup)),
        file_record(rustup_home / "settings.toml", text=True),
        file_record(cargo_home / "bin/rustup"),
        file_record(cargo_home / "bin/cargo-clippy"),
        *toolchain_records(rustup_home),
    ]
    # Avoid repository overrides during read-only rustup queries.
    with tempfile.TemporaryDirectory() as temporary:
        commands = [
            command_result([rustup, "--version"], Path(temporary)),
            command_result([rustup, "toolchain", "list", "-v"], Path(temporary)),
        ]
    result = {
        "phase": phase,
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "rustup_home": str(rustup_home),
        "cargo_home": str(cargo_home),
        "identity": {
            "head": os.environ.get("RUST_DIAGNOSTICS_HEAD"),
            "base": os.environ.get("RUST_DIAGNOSTICS_BASE"),
            "github_sha": os.environ.get("GITHUB_SHA"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "checkout": command_result(["git", "rev-parse", "HEAD"], Path.cwd()),
            "parents": command_result(
                ["git", "show", "-s", "--format=%P", "HEAD"], Path.cwd()
            ),
        },
        "uv_restore": cache_result(),
        "commands": commands,
        "files": files,
        "limits": {
            "text_bytes": TEXT_LIMIT,
            "hash_bytes": HASH_LIMIT,
            "toolchains": TOOLCHAIN_LIMIT,
            "manifests_per_toolchain": MANIFEST_LIMIT,
        },
    }
    (output / f"{phase}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("before", "after"))
    arguments = parser.parse_args()
    snapshot(Path(os.environ["RUST_TOOLCHAIN_EVIDENCE"]), arguments.phase)


if __name__ == "__main__":
    main()
