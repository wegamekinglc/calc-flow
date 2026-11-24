from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent


def require(valid, message):
    if not valid:
        raise ValueError(message)


def file_sha(path):
    with path.open("rb") as contents:
        return hashlib.file_digest(contents, "sha256").hexdigest()


def main(candidate):
    roots = {
        side: json.loads((ROOT / f"{side}.receipt.json").read_bytes())
        for side in ("baseline", "candidate")
    }
    previous, current = roots.values()
    require(previous["revision"] == "85af0b53e0a38e19cb472aff18c1ccfd84048912", "baseline revision")
    require(current["revision"] == candidate, "candidate revision")
    for key in ("environment", "files", "rustc", "source_directory"):
        require(previous[key] == current[key], key)
    require(previous["source_tree"] != current["source_tree"], "distinct product source")
    require(previous["binary_sha256"] != current["binary_sha256"], "distinct executable")
    for target in ("calc_flow", "stream_asof_perf"):
        for key in ("features", "profile", "manifest_path", "fresh", "package_id", "target"):
            require(previous["cargo_targets"][target][key] == current["cargo_targets"][target][key], f"{target} {key}")
    for side, receipt in roots.items():
        require(file_sha(ROOT / side) == receipt["binary_sha256"], side)
    subprocess.run(["git", "merge-base", "--is-ancestor", previous["revision"], candidate], check=True)
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], check=True)
    subprocess.run(["sha256sum", "-c", str(ROOT / "sources.sha256")], check=True)
    changed = subprocess.check_output(["git", "diff", "--name-only", previous["revision"], candidate], text=True).splitlines()
    arms = {
        "products": {side: {k: receipt[k] for k in ("revision", "tree", "source_tree", "binary_sha256", "build_seconds")} for side, receipt in roots.items()},
        "build_tools": previous["rustc"],
        "python": sys.version,
        "psutil": psutil.__version__,
        "product_diff": changed,
        "cpu_model": next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
        "precollection_source_hashes_sha256": file_sha(ROOT / "sources.sha256"),
    }
    with (ROOT / "arms.json").open("x") as out:
        print(json.dumps(arms, indent=2, sort_keys=True), file=out)


if __name__ == "__main__":
    main(sys.argv[1])
