"""Compare the frozen release and candidate on two CPU masks of the same VM."""
import gzip
import os
from pathlib import Path
import subprocess
import sys

from wheel_study import OUT, PRODUCTS, ROOT, WORK, download, git

OUT.mkdir(parents=True)
download()
for product in PRODUCTS.values():
    git("worktree", "add", "--quiet", "--detach", str(OUT / product), product, cwd=WORK)
worker = WORK / "target/asof-wheel-worker"
git("worktree", "add", "--quiet", "--detach", str(worker), PRODUCTS["main"], cwd=WORK)
patch = gzip.decompress((ROOT / "post-timer-worker.patch.gz").read_bytes())
git("apply", "--check", input=patch, cwd=worker)
git("apply", input=patch, cwd=worker)
for mask in ("runner", "one"):
    dest = ROOT / ("wheel-" + mask)
    command = [sys.executable, str(ROOT / "public_compare.py"), "asof_join"]
    for arg, side in (("baseline", "release"), ("candidate", "main")):
        command.extend(["--"+arg, str(OUT / ("plain-"+side) / "release.json"),
                        "--"+arg+"-source", str(OUT / PRODUCTS[side])])
    command.extend(["--baseline-sha", PRODUCTS["release"], "--sha", PRODUCTS["main"], "--output", str(dest)])
    with (ROOT / ("wheel-"+mask+".log")).open("w") as log:
        result = subprocess.run(command, cwd=WORK, env={**os.environ,"ASOF_CPU_MASK":mask},
                                stdout=log, stderr=subprocess.STDOUT)
    if not (dest / "outcome.json").is_file():
        raise RuntimeError(f"{mask} stopped before its outcome was recorded: {result.returncode}")
