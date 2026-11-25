"""Download the registered uninstrumented products with archive digest verification."""

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

runs = {"release": 45691470579, "parent": 45704369193}
api = "repos/wegamekinglc/calc-flow/actions"
side, baseline = sys.argv[1:]
root = Path(os.environ["GITHUB_WORKSPACE"]) / "target"
items = json.loads((Path(__file__).parent / "plain-artifacts.json").read_text())
item = items[side + "/" + baseline]
assert item["workflow_run"]["id"] == runs[baseline]
target = root / "products" / baseline
target.mkdir(parents=True)
archive = target.parent / (baseline + ".zip")
with archive.open("wb") as target_file:
    subprocess.run(
        ["gh", "api", f"{api}/artifacts/{item['id']}/zip"],
        stdout=target_file,
        check=True,
        timeout=120,
    )
with archive.open("rb") as f:
    assert "sha256:" + hashlib.file_digest(f, "sha256").hexdigest() == item["digest"]
with zipfile.ZipFile(archive) as z:
    assert all(Path(k).name == k for k in z.namelist())
    z.extractall(target)
