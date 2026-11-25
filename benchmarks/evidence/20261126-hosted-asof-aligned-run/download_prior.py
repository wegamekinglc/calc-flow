"""Download the registered uninstrumented products with archive digest verification."""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

run = 45691470579
api = "repos/wegamekinglc/calc-flow/actions"
side = sys.argv[1]
root = Path(os.environ["GITHUB_WORKSPACE"]) / "target"
items = json.loads((Path(__file__).parent / "plain-artifacts.json").read_text())
for arm in ("release", "candidate"):
    item = items["asof-product-" + side + "-" + arm]
    assert item["workflow_run"]["id"] == run
    target = root / "products-plain" / arm
    target.mkdir(parents=True)
    archive = target.parent / (arm + ".zip")
    with archive.open("wb") as target_file:
        subprocess.run(
            ["gh", "api", f"{api}/artifacts/{item['id']}/zip"],
            stdout=target_file,
            check=True,
            timeout=120,
        )
    with archive.open("rb") as f:
        assert (
            "sha256:" + hashlib.file_digest(f, "sha256").hexdigest() == item["digest"]
        )
    with zipfile.ZipFile(archive) as z:
        assert all(Path(k).name == k for k in z.namelist())
        z.extractall(target)
