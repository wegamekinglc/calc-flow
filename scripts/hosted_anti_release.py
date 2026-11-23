"""Seal the official main wheel and start the registered hosted budget."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from scripts.sparse_changelog_wheels import _api, seal
from scripts.toolkit import sha256_file, write_json

REPO = "wegamekinglc/calc-flow"
STUDY = Path("benchmarks/join-asof/20261123-anti-watermark")
OUT = Path("target/anti-watermark")
SOURCE = "37dd0ab7a61087f0bd799fc555e639ba55442fca"
ARTIFACT = 14822848922


def register():
    config = json.loads((STUDY / "hosted-registration.json").read_text())
    run, attempt = os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"]
    candidates = json.loads(
        _api(REPO, f"actions/runs/{run}/attempts/{attempt}/jobs?per_page=100")
    )["jobs"]
    (job,) = [j for j in candidates if j["name"] == "Active anti join watermark"]
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(
        OUT / "registration.json",
        dict(
            config,
            workflow=os.environ["GITHUB_SHA"],
            run=run,
            job=job["id"],
            start_epoch=datetime.fromisoformat(job["started_at"]).timestamp(),
        ),
        exclusive=True,
    )
    if any(sha256_file(Path(p)) != h for p, h in config["source_hashes"].items()):
        raise ValueError("hosted collector or protocol differs from registration")
    observed = seal(REPO, ARTIFACT, SOURCE, OUT / "release")
    if any(observed[k] != v for k, v in config["release"].items()):
        raise ValueError("wheel identity differs from registration")


if __name__ == "__main__":
    register()
