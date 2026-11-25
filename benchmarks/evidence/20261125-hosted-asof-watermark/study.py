"""Build frozen product and trace pairs, then run the two registered cohorts."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CHECKOUT = Path(os.environ["GITHUB_WORKSPACE"]) / "target" / "asof-study-source"
PRODUCTS = {"release":"7012b5b409553df04e29e6d7f9a11abf919cbf93",
            "main":"de20f8216c8dcb1bfce3e3811c3884a3080157df"}


def git(*args, cwd=CHECKOUT, **kwargs):
    return subprocess.run(["git",*args],cwd=cwd,check=True,**kwargs)


def rev(ref):
    return git("rev-parse",ref,capture_output=True,text=True).stdout.strip()


def build():
    CHECKOUT.parent.mkdir(exist_ok=True)
    git("worktree","add","--detach",str(CHECKOUT),PRODUCTS["release"],cwd=ROOT)
    arms={}
    for mode in ("plain","trace"):
        for side,product in PRODUCTS.items():
            git("switch","--detach",product)
            if mode=="trace":
                patch=gzip.decompress((ROOT/f"{side}.patch.gz").read_bytes())
                git("apply","--check",input=patch)
                git("apply",input=patch)
                git("add","crates/calc-flow/src/operator/asof")
                git("diff","--cached","--check")
                git("-c","user.name=ASOF benchmark","-c","user.email=benchmark@localhost",
                    "commit","-qm","Measure native ASOF watermark phases",
                    env={**os.environ,"GIT_AUTHOR_DATE":"2026-11-25T00:00:00Z","GIT_COMMITTER_DATE":"2026-11-25T00:00:00Z"})
                assert rev("HEAD^")==product
                (ROOT/f"applied-{side}.diff.gz").write_bytes(gzip.compress(git("diff","--binary","HEAD^","HEAD",capture_output=True).stdout,mtime=0))
            sha=rev("HEAD")
            subprocess.run([sys.executable,str(ROOT/"build.py"),str(CHECKOUT),sha,"job"],check=True)
            arms[f"{mode}-{side}"]=sha
    (ROOT/"arms.json").write_text(json.dumps(arms,indent=2)+"\n")


def measure():
    from verify import verify

    arms=json.loads((ROOT/"arms.json").read_text())
    summaries={}
    cpu=None
    for mode in ("plain","trace"):
        base,head=(arms[f"{mode}-{s}"] for s in ("release","main"))
        cmd=[sys.executable,str(ROOT/"measure.py"),"job",head,base]
        if cpu is not None:
            cmd=["taskset","-c",str(cpu),*cmd]
        with (ROOT/f"{mode}.log").open("w") as log:
            done=subprocess.run(cmd,cwd=ROOT,env={**os.environ,"ASOF_COLLECT_TRACE":str(int(mode=="trace"))},stdout=log,stderr=subprocess.STDOUT)
        d=ROOT/f"obs-job-{base[:10]}-{head[:10]}"
        verified=verify(d)
        raw=json.loads((d/"outcome.json").read_text())
        summaries[mode]={"exit_code":done.returncode,"verified":verified,"outcome":raw}
        if mode=="plain" and (d/"resources.jsonl").is_file():
            first=json.loads((d/"resources.jsonl").read_text().splitlines()[0])
            if first["suitable"]:cpu=first["cpu"]
    (ROOT/"cohorts.json").write_text(json.dumps(summaries,indent=2)+"\n")


if __name__=="__main__":
    {"build":build,"measure":measure}[sys.argv[1]]()
