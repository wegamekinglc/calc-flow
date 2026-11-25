"""Build and qualify the exact runtime source and the same single-case adapters."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

out = Path(__file__).resolve().parent
checkout = Path(sys.argv[1]).resolve()
sha = sys.argv[2]
cases = sys.argv[3:]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=checkout, text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if git("rev-parse", "HEAD") != sha or git(
    "status", "--porcelain", "--untracked-files=no"
):
    raise RuntimeError("unexpected product or uncommitted changes")
subprocess.run([sys.executable, str(out / "prepare.py"), str(checkout)], check=True)
root = checkout / "crates/calc-flow"
(root / "src/lib.rs").touch()
for case in cases:
    (root / f"examples/stage_{case}.rs").touch()
cmd = ["cargo", "build", "--profile", "bench", "--locked", "-p", "calc-flow"]
for case in cases:
    cmd += ["--example", "stage_" + case]
cmd += ["--message-format=json"]
log = out / (sha[:10] + "-build.jsonl")
start = time.monotonic()
with log.open("w") as stdout, log.with_suffix(".stderr").open("w") as stderr:
    subprocess.run(cmd, cwd=checkout, stdout=stdout, stderr=stderr, check=True)
records = [json.loads(line) for line in log.read_text().splitlines()]
for case in cases:
    kept = {}
    for name, kind in [("calc_flow", "lib"), ("stage_" + case, "example")]:
        matches = [
            r
            for r in records
            if r.get("reason") == "compiler-artifact"
            and r["target"]["name"] == name
            and kind in r["target"]["kind"]
        ]
        (r,) = matches
        if (
            r["fresh"]
            or Path(r["manifest_path"]) != root / "Cargo.toml"
            or r["profile"]["opt_level"] != "3"
            or r["profile"]["debug_assertions"]
            or "benchmark-internals" in r["features"]
        ):
            raise RuntimeError(f"unqualified target {name}: {r}")
        kept[name] = r
    dest = out / (case + "-" + sha[:10])
    shutil.copy2(r["executable"], dest)
    meta = dict(
        case=case,
        commit=sha,
        tree=git("rev-parse", "HEAD^{tree}"),
        source=git("rev-parse", "HEAD:crates/calc-flow/src"),
        harness_sha256=digest(root / f"examples/stage_{case}.rs"),
        binary_sha256=digest(dest),
        lock_sha256=digest(checkout / "Cargo.lock"),
        rustc=subprocess.check_output(["rustc", "-vV"], text=True),
        cargo=cmd,
        wall_seconds=time.monotonic() - start,
        artifacts=kept,
        settings={
            k: os.environ.get(k)
            for k in (
                "CARGO_TARGET_DIR",
                "CARGO_BUILD_JOBS",
                "RUSTFLAGS",
                "CARGO_PROFILE_BENCH_LTO",
                "CARGO_ENCODED_RUSTFLAGS",
            )
        },
    )
    dest.with_suffix(".receipt.json").write_text(json.dumps(meta, indent=2))
    print(
        case,
        sha[:10],
        round(meta["wall_seconds"], 1),
        meta["binary_sha256"][:15],
        flush=True,
    )
