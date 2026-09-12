"""Resolve only a partial Coveralls baseline to its same-tree merged PR measurement."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess  # nosec B404 - fixed gh GET command and validated identifiers
import urllib.request
from pathlib import Path

FLAGS = frozenset(("rust", "python", "studio"))
AGGREGATES = frozenset(("coverage/coveralls", "coverage/coveralls (push)"))
FLAG_PREFIX = "Coveralls - "
COVERAGE = {
    "rust": (
        "Rust combined coverage",
        "Enforce combined Rust and connector line coverage",
        "rust-coverage",
    ),
    "python": (
        "Ruff + unit tests",
        "Run unit, property, and coverage tests",
        "python-coverage",
    ),
    "studio": (
        "Studio backend tests",
        "Run studio backend coverage",
        "studio-python-coverage",
    ),
}


def _gh_executable() -> str:
    located = shutil.which("gh")
    if located is None:
        raise RuntimeError("gh executable is missing from PATH")
    path = Path(located)
    if not path.is_absolute() or not path.is_file():
        raise RuntimeError("gh executable must be an absolute regular file")
    return str(path)


def _github_pages(endpoint: str) -> list:
    # Absolute gh, fixed read-only GET argv and validated repository/commit IDs.
    completed = subprocess.run(  # noqa: S603  # nosec B603  # nosemgrep
        [
            _gh_executable(),
            "api",
            "--hostname",
            "github.com",
            "--method",
            "GET",
            "--paginate",
            endpoint,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        shell=False,
    )
    if completed.returncode:
        raise RuntimeError(f"GitHub API {endpoint}: {completed.stderr.strip()}")
    pages, remaining = [], completed.stdout.strip()
    decoder = json.JSONDecoder()
    while remaining:
        page, end = decoder.raw_decode(remaining)
        pages.append(page)
        remaining = remaining[end:].lstrip()
    if not pages:
        raise ValueError(f"GitHub API {endpoint} returned no JSON")
    return pages


def _rows(endpoint: str, key: str | None = None) -> list:
    result = []
    for page in _github_pages(f"{endpoint}?per_page=100"):
        rows = page if key is None else page[key]
        if not isinstance(rows, list):
            raise ValueError(f"GitHub API {endpoint} did not return a list")
        result.extend(rows)
    return result


def _one(items: list, label: str):
    if len(items) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(items)}")
    return items[0]


def _github_object(endpoint: str) -> dict:
    result = _one(_github_pages(endpoint), endpoint)
    if not isinstance(result, dict):
        raise ValueError(f"GitHub API {endpoint} did not return an object")
    return result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        raise ValueError("Coveralls JSON redirects are not allowed")


def _coveralls_json(url: str) -> dict:
    if not re.fullmatch(r"https://coveralls\.io/(?:builds|jobs)/[0-9]+\.json", url):
        raise ValueError("untrusted Coveralls JSON URL")
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(url, timeout=30) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("Coveralls JSON did not return an object")
    return result


def _report_location(status: dict, kind: str) -> tuple[str, int]:
    match = re.fullmatch(
        rf"https://coveralls\.io/{kind}/([0-9]+)", status["target_url"]
    )
    if match is None:
        raise ValueError(f"invalid Coveralls {kind} status URL")
    return status["target_url"] + ".json", int(match[1])


def _run_id(repository: str, url: str) -> int:
    match = re.fullmatch(
        rf"https://github\.com/{re.escape(repository)}/actions/runs/([0-9]+)", url
    )
    if match is None:
        raise ValueError("Coveralls build run URL is outside the repository")
    return int(match[1])


def _latest_statuses(repository: str, sha: str) -> dict:
    rows = _rows(f"repos/{repository}/commits/{sha}/statuses")
    latest = {}
    for row in sorted(rows, key=lambda row: row["id"], reverse=True):
        context = row["context"]
        if context in AGGREGATES or context.startswith(FLAG_PREFIX):
            latest.setdefault(context, row)
    return latest


def _flag_statuses(statuses: dict) -> dict:
    flags = {
        context.removeprefix(FLAG_PREFIX): row
        for context, row in statuses.items()
        if context.startswith(FLAG_PREFIX)
    }
    if not flags.keys() <= FLAGS:
        raise ValueError("unexpected Coveralls flags")
    return flags


def _flag_report(repository: str, status: dict, run_id: int) -> dict:
    url, number = _report_location(status, "jobs")
    report = _coveralls_json(url)
    if report["id"] != number or report["repo_name"] != repository:
        raise ValueError("Coveralls flag report identity differs")
    full_number = report["full_number"]
    if not isinstance(full_number, str) or not re.fullmatch(
        rf"{run_id}\.[0-9]+", full_number
    ):
        raise ValueError("Coveralls flag report belongs to a different build run")
    return {"status": status, "report": report}


def _coverage_report(repository: str, statuses: dict) -> dict:
    flags = _flag_statuses(statuses)
    aggregate = _one(
        [row for name, row in statuses.items() if name in AGGREGATES],
        "Coveralls aggregate",
    )
    url, number = _report_location(aggregate, "builds")
    build = _coveralls_json(url)
    if build["id"] != number or build["repo_name"] != repository:
        raise ValueError("Coveralls build identity differs")
    run_id = _run_id(repository, build["url"])
    return {
        "aggregate": aggregate,
        "build": build,
        "run_id": run_id,
        "flags": {
            name: _flag_report(repository, status, run_id)
            for name, status in flags.items()
        },
    }


def _require_successful_statuses(statuses: dict) -> None:
    if any(row["state"] != "success" for row in statuses.values()):
        raise ValueError("all existing Coveralls statuses must be successful")


def _tree(repository: str, sha: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("measurement or head SHA is not a full commit SHA")
    commit = _github_object(f"repos/{repository}/git/commits/{sha}")
    if commit["sha"] != sha:
        raise ValueError("GitHub commit identity differs")
    tree = commit["tree"]["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise ValueError("GitHub full tree identity is missing")
    return tree


def _merged_pull(repository: str, base_sha: str) -> dict:
    associated = _rows(f"repos/{repository}/commits/{base_sha}/pulls")
    candidate = _one(
        [pr for pr in associated if pr["merge_commit_sha"] == base_sha],
        "merged PR for the base commit",
    )
    number = candidate["number"]
    if type(number) is not int or number <= 0:
        raise ValueError("invalid merged PR number")
    pull = _github_object(f"repos/{repository}/pulls/{number}")
    _check_merged_pull(pull, repository, base_sha, number)
    return pull


def _check_merged_pull(pull: dict, repository: str, base_sha: str, number: int) -> None:
    expected = {
        "number": number,
        "merge_commit_sha": base_sha,
        "merged": True,
        "state": "closed",
    }
    if (
        any(pull.get(key) != value for key, value in expected.items())
        or not pull["merged_at"]
    ):
        raise ValueError("associated PR is not the actual merged base PR")
    for side in ("head", "base"):
        if pull[side]["repo"]["full_name"] != repository:
            raise ValueError("merged PR belongs to a different repository")


def _original_run(repository: str, pull: dict, report: dict) -> dict:
    run = _github_object(f"repos/{repository}/actions/runs/{report['run_id']}")
    expected = {
        "id": report["run_id"],
        "head_sha": pull["head"]["sha"],
        "event": "pull_request",
        "path": ".github/workflows/ci-linux.yml",
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
        "html_url": report["build"]["url"],
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("candidate is not its successful original Linux CI attempt")
    for key in ("repository", "head_repository"):
        if (
            run[key]["full_name"] != repository
            or run[key]["id"] != pull["base"]["repo"]["id"]
        ):
            raise ValueError("original Linux run repository differs")
    return run


def _check_coverage_job(job: dict, run: dict, step_name: str) -> None:
    expected = {
        "run_id": run["id"],
        "head_sha": run["head_sha"],
        "status": "completed",
        "conclusion": "success",
    }
    if any(job.get(key) != value for key, value in expected.items()):
        raise ValueError("original coverage job did not succeed for this run/head")
    step = _one(
        [step for step in job["steps"] if step["name"] == step_name],
        "required coverage step",
    )
    if step["status"] != "completed" or step["conclusion"] != "success":
        raise ValueError("required coverage step did not succeed")


def _coverage_jobs(repository: str, run: dict) -> dict:
    jobs = _rows(f"repos/{repository}/actions/runs/{run['id']}/attempts/1/jobs", "jobs")
    result = {}
    for flag, (name, step, _) in COVERAGE.items():
        job = _one([job for job in jobs if job["name"] == name], f"{flag} coverage job")
        _check_coverage_job(job, run, step)
        result[flag] = job
    return result


def _check_artifact(artifact: dict, run: dict) -> None:
    if artifact["expired"] is not False or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", artifact.get("digest") or ""
    ):
        raise ValueError("coverage artifact is expired or has no SHA256 digest")
    if type(artifact["size_in_bytes"]) is not int or artifact["size_in_bytes"] <= 0:
        raise ValueError("coverage artifact is empty")
    expected = {
        "id": run["id"],
        "head_sha": run["head_sha"],
        "repository_id": run["repository"]["id"],
        "head_repository_id": run["head_repository"]["id"],
    }
    if any(
        artifact["workflow_run"].get(key) != value for key, value in expected.items()
    ):
        raise ValueError("coverage artifact belongs to another run/head/repository")


def _coverage_artifacts(repository: str, run: dict) -> dict:
    artifacts = _rows(
        f"repos/{repository}/actions/runs/{run['id']}/artifacts", "artifacts"
    )
    result = {}
    for flag, (_, _, name) in COVERAGE.items():
        artifact = _one(
            [item for item in artifacts if item["name"] == name],
            f"{flag} coverage artifact",
        )
        _check_artifact(artifact, run)
        result[flag] = artifact
    return result


def _equivalent_candidate(repository: str, base_sha: str, tree: str) -> dict:
    pull = _merged_pull(repository, base_sha)
    head = pull["head"]["sha"]
    if _tree(repository, head) != tree:
        raise ValueError("merged PR head full tree differs from the base")
    statuses = _latest_statuses(repository, head)
    if _flag_statuses(statuses).keys() != FLAGS:
        raise ValueError("candidate measurement must contain all three flags")
    _require_successful_statuses(statuses)
    report = _coverage_report(repository, statuses)
    measurement = report["build"]["commit_sha"]
    if _tree(repository, measurement) != tree:
        raise ValueError("actual measurement full tree differs from the base")
    run = _original_run(repository, pull, report)
    return {
        "pull_request": pull,
        "head_sha": head,
        "head_tree": tree,
        "measurement_sha": measurement,
        "measurement_tree": tree,
        "report": report,
        "run": run,
        "jobs": _coverage_jobs(repository, run),
        "artifacts": _coverage_artifacts(repository, run),
    }


def _validate_request(repository: str, base_sha: str) -> None:
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository, flags=re.ASCII):
        raise ValueError("repository must be owner/repo")
    if not re.fullmatch(r"[0-9a-f]{40}", base_sha):
        raise ValueError("base SHA must be a full hexadecimal commit SHA")


def resolve(repository: str, base_sha: str) -> dict:
    _validate_request(repository, base_sha)
    result = {
        "repository": repository,
        "base_sha": base_sha,
        "compare_sha": "",
    }
    if base_sha == "0" * 40:
        return {**result, "origin": "default_new_branch", "base_statuses": {}}
    statuses = _latest_statuses(repository, base_sha)
    flags = _flag_statuses(statuses)
    result = {**result, "base_statuses": statuses}
    if not flags:
        if statuses:
            raise ValueError("Coveralls aggregate exists without any flag reports")
        return {**result, "origin": "default_no_flags"}
    if flags.keys() != FLAGS:
        _require_successful_statuses(statuses)
    report = _coverage_report(repository, statuses)
    if report["build"]["commit_sha"] != base_sha:
        raise ValueError("base Coveralls build measured a different commit")
    if flags.keys() == FLAGS:
        return {**result, "origin": "default_complete_flags", "base_report": report}
    tree = _tree(repository, base_sha)
    candidate = _equivalent_candidate(repository, base_sha, tree)
    return {
        **result,
        "origin": "equivalent_merged_pr",
        "source_tree": tree,
        "base_report": report,
        "candidate": candidate,
        "compare_sha": candidate["measurement_sha"],
    }


def _write_provenance(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def _record_failure(args, error: Exception) -> None:
    record = {
        "repository": args.repository,
        "base_sha": args.base_sha,
        "origin": "resolution_error",
        "error": f"{type(error).__name__}: {error}",
        "request_url": getattr(error, "url", None),
    }
    try:
        _write_provenance(args.provenance, record)
    except OSError as save_error:
        error.add_note(f"Could not save failure provenance: {save_error}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = resolve(args.repository, args.base_sha)
    except Exception as error:
        _record_failure(args, error)
        raise
    _write_provenance(args.provenance, result)
    with args.github_output.open("a") as stream:
        stream.write(f"compare_sha={result['compare_sha']}\n")
    print(
        json.dumps({key: result[key] for key in ("origin", "base_sha", "compare_sha")})
    )


if __name__ == "__main__":
    main()
