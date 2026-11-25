"""Build traced wheels on their frozen product parents and compare installed arms."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from study import PRODUCTS, ROOT, git

WORK = Path(os.environ["GITHUB_WORKSPACE"])
OUT = WORK / "target" / "asof-wheel-products"
REFS = ROOT / "wheel-arms.json"
PROFILE = {s:ROOT/f"{s}.patch.gz" for s in PRODUCTS}
RUN = 45623973262
API = "repos/wegamekinglc/calc-flow"


def api(path):
    return json.loads(subprocess.check_output(["gh","api",API+path],timeout=45))


def build():
    checkout=WORK/"target/asof-wheel-source"
    cache=WORK/"target/asof-wheel-cargo"
    OUT.mkdir(parents=True,exist_ok=True);cache.mkdir()
    git("worktree","add","--detach",str(checkout),PRODUCTS["release"],cwd=WORK)
    arms={}
    for side,product in PRODUCTS.items():
        git("switch","--detach",product,cwd=checkout)
        src=(checkout/"crates/calc-flow/src/lib.rs");src.touch()
        patch=gzip.decompress(PROFILE[side].read_bytes())
        git("apply","--check",input=patch,cwd=checkout);git("apply",input=patch,cwd=checkout)
        git("add","crates/calc-flow/src/operator/asof",cwd=checkout);git("diff","--cached","--check",cwd=checkout)
        git("-c","user.name=ASOF benchmark","-c","user.email=benchmark@localhost","commit","-qm","Measure native ASOF watermark phases",cwd=checkout,env={**os.environ,"GIT_AUTHOR_DATE":"2026-11-25T00:00:00Z","GIT_COMMITTER_DATE":"2026-11-25T00:00:00Z"})
        oid=git("rev-parse","HEAD",cwd=checkout,capture_output=True,text=True).stdout.strip()
        assert git("rev-parse","HEAD^",cwd=checkout,capture_output=True,text=True).stdout.strip()==product
        dest=OUT/("trace-"+side);dest.mkdir();(dest/"cargo").symlink_to(cache)
        (checkout/"crates/calc-flow-python/src/lib.rs").touch()
        subprocess.run([sys.executable,"-m","scripts.benchmark_suite","build","--source",str(checkout),"--output",str(dest)],cwd=WORK,check=True)
        arms["trace-"+side]=oid;arms["plain-"+side]=product
    REFS.write_text(json.dumps(arms,indent=2)+"\n")
    for oid in arms.values():
        git("worktree","add","--quiet","--detach",str(OUT/oid),oid,cwd=WORK)
    worker=WORK/"target/asof-wheel-worker"
    git("worktree","add","--quiet","--detach",str(worker),PRODUCTS["main"],cwd=WORK)
    patch=gzip.decompress((ROOT/"post-timer-worker.patch.gz").read_bytes())
    git("apply","--check",input=patch,cwd=worker);git("apply",input=patch,cwd=worker)


def download():
    run=api(f"/actions/runs/{RUN}");assert run["head_sha"]==PRODUCTS["main"] and run["event"]=="workflow_dispatch"
    archives=api(f"/actions/runs/{RUN}/artifacts?per_page=100")["artifacts"]
    refs={}
    for side,tag in [("release","baseline"),("main","candidate")]:
        a,=[x for x in archives if x["name"]==f"benchmark-release-{tag}-{RUN}-1"]
        dest=OUT/("plain-"+side);dest.mkdir();z=dest/"original.zip"
        with z.open("wb") as f:subprocess.run(["gh","api",API+"/actions/artifacts/"+str(a["id"])+"/zip"],stdout=f,timeout=100,check=True)
        with z.open("rb") as f:assert "sha256:"+hashlib.file_digest(f,"sha256").hexdigest()==a["digest"]
        assert z.stat().st_size==a["size_in_bytes"]
        with zipfile.ZipFile(z) as f:
            assert all(Path(n).name==n for n in f.namelist())
            f.extractall(dest)
        assert json.loads((dest/"release.json").read_text())["git_sha"]==PRODUCTS[side]
        refs[side]=a
    (ROOT/"wheel-downloads.json").write_text(json.dumps(refs,indent=2)+"\n")


def measure():
    arms=json.loads(REFS.read_text());cpu=None
    for mode in ("plain","trace"):
        base,head=(arms[f"{mode}-{s}"] for s in ("release","main"));d=ROOT/("wheel-"+mode)
        argv=[sys.executable,str(ROOT/"public_compare.py"),"asof_join","--baseline",str(OUT/f"{mode}-release/release.json"),"--candidate",str(OUT/f"{mode}-main/release.json"),"--baseline-sha",base,"--baseline-source",str(OUT/base),"--candidate-source",str(OUT/head),"--sha",head,"--output",str(d)]
        if cpu is not None:argv=["taskset","-c",str(cpu),*argv]
        with (ROOT/("wheel-"+mode+".log")).open("w") as log:
            res=subprocess.run(argv,cwd=WORK,env={**os.environ,"ASOF_COLLECT_TRACE":str(int(mode=="trace"))},stdout=log,stderr=subprocess.STDOUT)
        assert (d/"outcome.json").exists(),res.returncode
        if mode=="plain" and (d/"resources.jsonl").is_file():
            q=json.loads((d/"resources.jsonl").read_text().splitlines()[0])["cores"]
            if len(q)==1 and q[0]["suitable"]:cpu=q[0]["cpu"]


if __name__=="__main__":
    {"build":build,"download":download,"measure":measure}[sys.argv[1]]()
