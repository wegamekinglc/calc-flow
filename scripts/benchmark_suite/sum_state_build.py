"""Build one sealed side of the focused grouped SUM comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from scripts.benchmark_suite.rust import clear_stale_product_library
from scripts.benchmark_suite.sum_state import HARNESS, sha
from scripts.toolkit import write_json

EXAMPLE = "grouped_sum_state_benchmark"


def compiled_binary(messages: list[dict]) -> Path:
    artifacts = [m for m in messages if m.get("reason") == "compiler-artifact"]
    libraries = [
        m
        for m in artifacts
        if m["target"]["name"] == "calc_flow" and "lib" in m["target"]["kind"]
    ]
    examples = [
        m for m in artifacts if m["target"]["name"] == EXAMPLE and m.get("executable")
    ]
    if (
        len(libraries) != 1
        or len(examples) != 1
        or libraries[0]["fresh"]
        or examples[0]["fresh"]
    ):
        raise ValueError("product and diagnostic example must actually rebuild")
    return Path(examples[0]["executable"])


def source_seals(source: Path) -> dict[str, str]:
    paths = {
        source / "Cargo.toml",
        source / "Cargo.lock",
        source / "crates/calc-flow/Cargo.toml",
        source / HARNESS,
    }
    for directory in (
        "crates/calc-flow/src",
        "crates/calc-flow/examples/support",
        "vendor",
    ):
        paths.update(p for p in (source / directory).rglob("*") if p.is_file())
    paths.update(
        p
        for p in (source / "rust-toolchain.toml", source / ".cargo/config.toml")
        if p.is_file()
    )
    return {str(p.relative_to(source)): sha(p) for p in sorted(paths)}


def build(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    harness_source = args.harness_source.resolve()
    output = args.output.resolve()
    target = args.target.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if source != harness_source:
        shutil.copyfile(harness_source / HARNESS, source / HARNESS)
        manifest = source / "crates/calc-flow/Cargo.toml"
        if f'name = "{EXAMPLE}"' not in manifest.read_text():
            with manifest.open("a") as stream:
                stream.write(
                    f'\n[[example]]\nname = "{EXAMPLE}"\n'
                    'required-features = ["benchmark-internals"]\n'
                )
    before = source_seals(source)
    write_json(output / "source-before.json", before)
    clear_stale_product_library(target)
    for stale in (target / "release/examples").glob(f"{EXAMPLE}*"):
        if stale.is_file():
            stale.unlink()
    command = [
        "cargo",
        "build",
        "--release",
        "--locked",
        "-p",
        "calc-flow",
        "--features",
        "benchmark-internals",
        "--example",
        EXAMPLE,
        "--message-format=json",
    ]
    environment = {
        **os.environ,
        "CARGO_TARGET_DIR": str(target),
        "CARGO_BUILD_JOBS": "2",
        "CARGO_INCREMENTAL": "0",
    }
    started = time.monotonic()
    with (
        (output / "compiler.jsonl").open("w") as stdout,
        (output / "build.log").open("w") as stderr,
    ):
        result = subprocess.run(
            command,
            cwd=source,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    elapsed = time.monotonic() - started
    write_json(
        output / "attempt.json",
        {
            "command": command,
            "source_root": str(source),
            "exit_code": result.returncode,
            "build_elapsed_seconds": elapsed,
        },
    )
    result.check_returncode()
    if source_seals(source) != before:
        raise ValueError("source changed during build")
    messages = [
        json.loads(line)
        for line in (output / "compiler.jsonl").read_text().splitlines()
    ]
    binary = output / EXAMPLE
    shutil.copy2(compiled_binary(messages), binary)
    receipt = {
        "binary": str(binary),
        "binary_sha256": sha(binary),
        "source_root": str(source),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source,
            text=True,
        ).strip(),
        "command": command,
        "rustc": subprocess.check_output(["rustc", "-Vv"], text=True),
        "cargo_lock_sha256": sha(source / "Cargo.lock"),
        "harness_sha256": sha(source / HARNESS),
        "features": ["benchmark-internals"],
        "source_sha256": before,
        "build_elapsed_seconds": elapsed,
        "build_environment": {
            key: environment.get(key)
            for key in (
                "CARGO_TARGET_DIR",
                "CARGO_BUILD_JOBS",
                "CARGO_INCREMENTAL",
                "RUSTFLAGS",
                "CARGO_ENCODED_RUSTFLAGS",
                "RUSTUP_TOOLCHAIN",
                "CARGO_PROFILE_RELEASE_LTO",
                "CARGO_PROFILE_RELEASE_OPT_LEVEL",
                "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
                "CARGO_PROFILE_RELEASE_DEBUG",
            )
        },
        "compiler_log_sha256": sha(output / "compiler.jsonl"),
        "product_recompiled": True,
        "example_recompiled": True,
    }
    write_json(output / "build.json", receipt)
    print(
        json.dumps({"receipt": str(output / "build.json"), "seconds": elapsed}),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--harness-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
