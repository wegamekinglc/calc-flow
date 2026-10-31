"""Delete GitHub Actions caches that no later run will restore.

The repository shares one 10 GB Actions cache budget. Past it, GitHub evicts
the least recently used entries, which can cost the benchmark Rust shard its
build cache. Two kinds of entries only consume that budget:

* entries scoped to a pull request that is no longer open, since no other ref
  can restore them;
* entries in the same ref and key family as a newer entry, when no run has
  restored them since that replacement has existed for the grace period.

A key family is the key without its trailing environment and lockfile hashes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess  # nosec B404 - fixed gh argv and validated identifiers
from datetime import UTC, datetime, timedelta

try:
    from scripts.toolkit import require_executable
except ImportError:  # direct execution puts only scripts/ on sys.path
    from toolkit import require_executable

REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
PULL_REF = re.compile(r"refs/pull/(\d+)/(?:merge|head)")
TRAILING_HASHES = re.compile(r"(?:-[0-9a-f]{8,})+$")


def cache_family(key: str) -> str:
    return TRAILING_HASHES.sub("", key)


def _created(cache: dict) -> datetime:
    return datetime.fromisoformat(cache["created_at"])


def _closed_pull(cache: dict, open_pulls: set[int]) -> str | None:
    match = PULL_REF.fullmatch(cache["ref"])
    if match is None or int(match.group(1)) in open_pulls:
        return None
    return f"pull request #{match.group(1)} is closed"


def _replacement(cache: dict, caches: list[dict], matured: datetime) -> str | None:
    accessed = datetime.fromisoformat(cache["last_accessed_at"])
    replacement = min(
        (
            other
            for other in caches
            if other["ref"] == cache["ref"]
            and cache_family(other["key"]) == cache_family(cache["key"])
            and _created(cache) < _created(other) <= matured
            and accessed < _created(other)
        ),
        key=_created,
        default=None,
    )
    if replacement is None:
        return None
    return f"replaced by cache {replacement['id']} at {replacement['created_at']}"


def stale_caches(
    caches: list[dict], open_pulls: set[int], *, now: datetime, grace: timedelta
) -> list[tuple[dict, str]]:
    """Return each unrestorable cache with the reason it is stale."""
    matured = now - grace
    stale = []
    for cache in caches:
        reason = _closed_pull(cache, open_pulls) or _replacement(cache, caches, matured)
        if reason is not None:
            stale.append((cache, reason))
    return stale


def _gh(*arguments: str) -> str:
    # Absolute gh, fixed argv and validated repository/cache identifiers.
    completed = subprocess.run(  # noqa: S603  # nosec B603  # nosemgrep
        [require_executable("gh"), "api", "--hostname", "github.com", *arguments],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        shell=False,
    )
    if completed.returncode:
        raise RuntimeError(f"GitHub API {arguments[-1]}: {completed.stderr.strip()}")
    return completed.stdout


def _json_lines(text: str) -> list:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def list_caches(repository: str) -> list[dict]:
    return _json_lines(
        _gh(
            "--paginate",
            "--jq",
            ".actions_caches[]",
            f"repos/{repository}/actions/caches?per_page=100",
        )
    )


def list_open_pulls(repository: str) -> set[int]:
    return set(
        _json_lines(
            _gh(
                "--paginate",
                "--jq",
                ".[].number",
                f"repos/{repository}/pulls?state=open&per_page=100",
            )
        )
    )


def delete_cache(repository: str, identifier: int) -> None:
    _gh("--method", "DELETE", f"repos/{repository}/actions/caches/{identifier:d}")


def _mebibytes(size: int) -> str:
    return f"{size / 2**20:.0f} MiB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository", required=True)
    parser.add_argument("--grace-hours", type=float, default=24.0)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    if REPOSITORY.fullmatch(arguments.repository) is None:
        parser.error("--repository must be OWNER/NAME")

    # List caches before pull requests: a pull request opened in between is
    # then reported open, so its fresh caches are never deleted.
    caches = list_caches(arguments.repository)
    open_pulls = list_open_pulls(arguments.repository)
    stale = stale_caches(
        caches,
        open_pulls,
        now=datetime.now(UTC),
        grace=timedelta(hours=arguments.grace_hours),
    )
    action = "Would delete" if arguments.dry_run else "Deleting"
    for cache, reason in stale:
        print(
            f"{action} {cache['id']} {_mebibytes(cache['size_in_bytes'])} "
            f"{cache['ref']} {cache['key']}: {reason}"
        )
        if not arguments.dry_run:
            delete_cache(arguments.repository, cache["id"])
    total = sum(cache["size_in_bytes"] for cache in caches)
    freed = sum(cache["size_in_bytes"] for cache, _ in stale)
    print(
        f"{len(stale)} of {len(caches)} caches stale; "
        f"{_mebibytes(freed)} of {_mebibytes(total)} "
        f"{'reclaimable' if arguments.dry_run else 'reclaimed'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
