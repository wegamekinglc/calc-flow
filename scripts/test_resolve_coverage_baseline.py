"""Prove coverage aliases use only a complete measurement of the same full tree."""

from __future__ import annotations

import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import resolve_coverage_baseline as resolver

REPOSITORY = "owner/project"
BASE = "a" * 40
HEAD = "b" * 40
MEASUREMENT = "c" * 40
TREE = "d" * 40
PREFIX = f"repos/{REPOSITORY}"
COVERAGE = (
    (
        "rust",
        "Rust combined coverage",
        "Enforce combined Rust and connector line coverage",
        "rust-coverage",
    ),
    (
        "python",
        "Ruff + unit tests",
        "Run unit, property, and coverage tests",
        "python-coverage",
    ),
    (
        "studio",
        "Studio backend tests",
        "Run studio backend coverage",
        "studio-python-coverage",
    ),
)


def _status(context: str, number: int, *, state: str = "success") -> dict:
    kind = "builds" if context.startswith("coverage/") else "jobs"
    return {
        "id": number,
        "context": context,
        "state": state,
        "target_url": f"https://coveralls.io/{kind}/{number}",
    }


def _fixture() -> tuple[dict, dict]:
    base_statuses = [
        _status("coverage/coveralls (push)", 10),
        _status("Coveralls - python", 11),
        _status("Coveralls - studio", 12),
    ]
    head_statuses = [_status("coverage/coveralls", 20)]
    reports = {}
    for number, sha, run_id in ((10, BASE, 50), (20, MEASUREMENT, 60)):
        reports[f"https://coveralls.io/builds/{number}.json"] = {
            "id": number,
            "repo_name": REPOSITORY,
            "commit_sha": sha,
            "url": f"https://github.com/{REPOSITORY}/actions/runs/{run_id}",
        }
    for number, run_id in ((11, 50), (12, 50), (21, 60), (22, 60), (23, 60)):
        reports[f"https://coveralls.io/jobs/{number}.json"] = {
            "id": number,
            "repo_name": REPOSITORY,
            "full_number": f"{run_id}.{number}",
        }
    jobs, artifacts = [], []
    for number, (flag, job_name, step_name, artifact_name) in enumerate(COVERAGE, 21):
        head_statuses.append(_status(f"Coveralls - {flag}", number))
        jobs.append(
            {
                "id": number + 100,
                "name": job_name,
                "run_id": 60,
                "head_sha": HEAD,
                "status": "completed",
                "conclusion": "success",
                "steps": [
                    {"name": step_name, "status": "completed", "conclusion": "success"}
                ],
            }
        )
        artifacts.append(
            {
                "id": number + 200,
                "name": artifact_name,
                "digest": "sha256:" + "f" * 64,
                "size_in_bytes": 100,
                "expired": False,
                "workflow_run": {
                    "id": 60,
                    "head_sha": HEAD,
                    "repository_id": 500,
                    "head_repository_id": 500,
                },
            }
        )
    github = {
        f"{PREFIX}/commits/{BASE}/statuses?per_page=100": [
            base_statuses[:2],
            base_statuses[2:],
        ],
        f"{PREFIX}/commits/{HEAD}/statuses?per_page=100": [
            head_statuses[:2],
            head_statuses[2:],
        ],
        f"{PREFIX}/commits/{BASE}/pulls?per_page=100": [
            [],
            [{"number": 12, "merge_commit_sha": BASE}],
        ],
        f"{PREFIX}/pulls/12": [
            {
                "number": 12,
                "merged": True,
                "state": "closed",
                "merged_at": "2026-01-01",
                "merge_commit_sha": BASE,
                "base": {"repo": {"full_name": REPOSITORY, "id": 500}},
                "head": {"sha": HEAD, "repo": {"full_name": REPOSITORY, "id": 500}},
            }
        ],
        f"{PREFIX}/actions/runs/60": [
            {
                "id": 60,
                "head_sha": HEAD,
                "event": "pull_request",
                "run_attempt": 1,
                "path": ".github/workflows/ci-linux.yml",
                "status": "completed",
                "conclusion": "success",
                "html_url": f"https://github.com/{REPOSITORY}/actions/runs/60",
                "repository": {"full_name": REPOSITORY, "id": 500},
                "head_repository": {"full_name": REPOSITORY, "id": 500},
            }
        ],
        f"{PREFIX}/actions/runs/60/attempts/1/jobs?per_page=100": [
            {"jobs": jobs[:1]},
            {"jobs": jobs[1:]},
        ],
        f"{PREFIX}/actions/runs/60/artifacts?per_page=100": [
            {"artifacts": artifacts[:1]},
            {"artifacts": artifacts[1:]},
        ],
    }
    for sha in (BASE, HEAD, MEASUREMENT):
        github[f"{PREFIX}/git/commits/{sha}"] = [{"sha": sha, "tree": {"sha": TREE}}]
    return github, reports


def _resolve_fixture(github: dict, reports: dict) -> dict:
    with (
        patch.object(
            resolver,
            "_github_pages",
            side_effect=lambda path: copy.deepcopy(github[path]),
        ),
        patch.object(
            resolver,
            "_coveralls_json",
            side_effect=lambda url: copy.deepcopy(reports[url]),
            create=True,
        ),
    ):
        return resolver.resolve(REPOSITORY, BASE)


class DefaultBaselineTests(unittest.TestCase):
    def test_new_branch_zero_sha_uses_default_without_any_api(self) -> None:
        with (
            patch.object(resolver, "_github_pages") as github,
            patch.object(resolver, "_coveralls_json") as coveralls,
        ):
            result = resolver.resolve(REPOSITORY, "0" * 40)
        self.assertEqual(result["origin"], "default_new_branch")
        self.assertEqual(result["compare_sha"], "")
        github.assert_not_called()
        coveralls.assert_not_called()

    def test_aggregate_without_any_flags_is_not_a_docs_only_default(self) -> None:
        rows = [[_status("coverage/coveralls (push)", 10)]]
        with (
            patch.object(resolver, "_github_pages", return_value=rows),
            self.assertRaisesRegex(ValueError, "aggregate.*without.*flag"),
        ):
            resolver.resolve(REPOSITORY, BASE)

    def test_complete_flags_keep_default_even_when_a_status_failed(self) -> None:
        github, reports = _fixture()
        github[f"{PREFIX}/commits/{BASE}/statuses?per_page=100"].append(
            [
                _status("Coveralls - rust", 13, state="failure"),
            ]
        )
        reports["https://coveralls.io/jobs/13.json"] = {
            "id": 13,
            "repo_name": REPOSITORY,
            "full_number": "50.3",
        }
        result = _resolve_fixture(github, reports)
        self.assertEqual(result["compare_sha"], "")
        self.assertEqual(result["origin"], "default_complete_flags")
        self.assertEqual(
            result["base_statuses"]["Coveralls - rust"]["state"], "failure"
        )

    def test_old_run_flag_cannot_complete_the_current_partial_build(self) -> None:
        github, reports = _fixture()
        github[f"{PREFIX}/commits/{BASE}/statuses?per_page=100"].append(
            [
                _status("Coveralls - rust", 13),
            ]
        )
        reports["https://coveralls.io/jobs/13.json"] = {
            "id": 13,
            "repo_name": REPOSITORY,
            "full_number": "49.3",
        }
        with self.assertRaisesRegex(ValueError, "flag report"):
            _resolve_fixture(github, reports)

    def test_no_flags_keeps_docs_only_default(self) -> None:
        with patch.object(resolver, "_github_pages", return_value=[[]]):
            result = resolver.resolve(REPOSITORY, BASE)
        self.assertEqual(result["compare_sha"], "")
        self.assertEqual(result["origin"], "default_no_flags")

    def test_github_pagination_keeps_every_json_page(self) -> None:
        pages = [[{"id": 1}], [{"id": 2}]]
        completed = Mock(
            returncode=0,
            stdout="\n".join(json.dumps(page) for page in pages),
            stderr="",
        )
        with (
            patch.object(resolver.subprocess, "run", return_value=completed) as run,
            patch.object(shutil, "which", return_value=sys.executable),
        ):
            self.assertEqual(resolver._github_pages("repos/owner/project/test"), pages)
        self.assertIn("--paginate", run.call_args.args[0])
        self.assertNotIn("--slurp", run.call_args.args[0])

    def test_github_permission_failure_is_not_an_absent_baseline(self) -> None:
        completed = Mock(
            returncode=1, stdout="", stderr="HTTP 403: resource not accessible"
        )
        with (
            patch.object(resolver.subprocess, "run", return_value=completed),
            patch.object(shutil, "which", return_value=sys.executable),
            self.assertRaisesRegex(RuntimeError, "HTTP 403"),
        ):
            resolver.resolve(REPOSITORY, BASE)


class EquivalentMeasurementTests(unittest.TestCase):
    def test_partial_base_uses_original_same_tree_measurement_with_full_evidence(
        self,
    ) -> None:
        github, reports = _fixture()
        before = copy.deepcopy((github, reports))
        result = _resolve_fixture(github, reports)
        self.assertEqual(result["compare_sha"], MEASUREMENT)
        self.assertEqual(result["origin"], "equivalent_merged_pr")
        self.assertEqual(result["source_tree"], TREE)
        candidate = result["candidate"]
        self.assertEqual(candidate["head_sha"], HEAD)
        self.assertEqual(candidate["run"]["id"], 60)
        self.assertEqual(set(candidate["artifacts"]), {"rust", "python", "studio"})
        self.assertEqual(candidate["report"]["build"]["commit_sha"], MEASUREMENT)
        self.assertEqual((github, reports), before)

    def test_head_or_synthetic_measurement_tree_difference_is_rejected(self) -> None:
        for sha in (HEAD, MEASUREMENT):
            with self.subTest(sha=sha):
                github, reports = _fixture()
                github[f"{PREFIX}/git/commits/{sha}"][0]["tree"]["sha"] = "e" * 40
                with self.assertRaisesRegex(ValueError, "tree"):
                    _resolve_fixture(github, reports)

    def test_partial_base_existing_failure_cannot_find_an_alias(self) -> None:
        github, reports = _fixture()
        github[f"{PREFIX}/commits/{BASE}/statuses?per_page=100"][0][1]["state"] = (
            "failure"
        )
        with self.assertRaisesRegex(ValueError, "successful"):
            _resolve_fixture(github, reports)

    def test_latest_failed_candidate_status_does_not_select_older_success(self) -> None:
        github, reports = _fixture()
        rows = github[f"{PREFIX}/commits/{HEAD}/statuses?per_page=100"]
        rows.append([_status("Coveralls - rust", 999, state="failure")])
        with self.assertRaisesRegex(ValueError, "successful"):
            _resolve_fixture(github, reports)

    def test_candidate_partial_flags_are_not_a_complete_baseline(self) -> None:
        github, reports = _fixture()
        github[f"{PREFIX}/commits/{HEAD}/statuses?per_page=100"][1].pop()
        with self.assertRaisesRegex(ValueError, "three flags"):
            _resolve_fixture(github, reports)

    def test_flag_reports_from_a_different_run_or_repo_are_rejected(self) -> None:
        for job, field, value in (
            (11, "full_number", "999.1"),
            (21, "full_number", "999.1"),
            (21, "repo_name", "other/repo"),
        ):
            with self.subTest(job=job, field=field):
                github, reports = _fixture()
                reports[f"https://coveralls.io/jobs/{job}.json"][field] = value
                with self.assertRaisesRegex(ValueError, "flag report"):
                    _resolve_fixture(github, reports)

    def test_build_repo_and_run_url_must_match_the_requested_repository(self) -> None:
        for field, value in (
            ("repo_name", "other/repo"),
            ("url", "https://github.com/other/repo/actions/runs/60"),
        ):
            with self.subTest(field=field):
                github, reports = _fixture()
                reports["https://coveralls.io/builds/20.json"][field] = value
                with self.assertRaisesRegex(ValueError, "build|run URL"):
                    _resolve_fixture(github, reports)

    def test_wrong_head_failed_workflow_or_rerun_cannot_supply_reports(self) -> None:
        for field, value in (
            ("head_sha", BASE),
            ("conclusion", "failure"),
            ("run_attempt", 2),
        ):
            with self.subTest(field=field):
                github, reports = _fixture()
                github[f"{PREFIX}/actions/runs/60"][0][field] = value
                with self.assertRaisesRegex(ValueError, "original Linux"):
                    _resolve_fixture(github, reports)

    def test_combined_rust_gate_must_have_succeeded_even_if_job_succeeded(self) -> None:
        github, reports = _fixture()
        github[f"{PREFIX}/actions/runs/60/attempts/1/jobs?per_page=100"][0]["jobs"][0][
            "steps"
        ][0]["conclusion"] = "skipped"
        with self.assertRaisesRegex(ValueError, "coverage step"):
            _resolve_fixture(github, reports)

    def test_missing_expired_or_unsigned_artifact_cannot_supply_a_baseline(
        self,
    ) -> None:
        for field, value in (("expired", True), ("digest", None), ("name", "other")):
            with self.subTest(field=field):
                github, reports = _fixture()
                github[f"{PREFIX}/actions/runs/60/artifacts?per_page=100"][0][
                    "artifacts"
                ][0][field] = value
                with self.assertRaisesRegex(ValueError, "artifact"):
                    _resolve_fixture(github, reports)

    def test_only_the_unique_actual_merged_same_repo_pr_can_supply_a_head(self) -> None:
        for changed in ("unmerged", "wrong_merge", "fork", "ambiguous"):
            with self.subTest(changed=changed):
                github, reports = _fixture()
                pull = github[f"{PREFIX}/pulls/12"][0]
                if changed == "unmerged":
                    pull["merged"] = False
                elif changed == "wrong_merge":
                    pull["merge_commit_sha"] = HEAD
                elif changed == "fork":
                    pull["head"]["repo"]["full_name"] = "another/repo"
                else:
                    github[f"{PREFIX}/commits/{BASE}/pulls?per_page=100"][0].append(
                        {"number": 13, "merge_commit_sha": BASE}
                    )
                with self.assertRaisesRegex(ValueError, "PR"):
                    _resolve_fixture(github, reports)


class GhExecutableTests(unittest.TestCase):
    def test_missing_relative_and_non_file_gh_never_start_a_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for location in (
                None,
                "bin/gh",
                directory,
                str(Path(directory) / "missing"),
            ):
                with (
                    self.subTest(location=location),
                    patch.object(shutil, "which", return_value=location),
                    patch.object(
                        resolver.subprocess,
                        "run",
                        return_value=Mock(returncode=0, stdout="[]", stderr=""),
                    ) as run,
                ):
                    with self.assertRaisesRegex(RuntimeError, "gh"):
                        resolver._github_pages(f"{PREFIX}/commits/{BASE}/statuses")
                    run.assert_not_called()

    def test_only_absolute_gh_uses_fixed_get_argv_without_a_shell(self) -> None:
        endpoint = f"{PREFIX}/commits/{BASE}/statuses?per_page=100"
        completed = Mock(returncode=0, stdout="[]", stderr="")
        with (
            patch.object(shutil, "which", return_value=sys.executable) as which,
            patch.object(resolver.subprocess, "run", return_value=completed) as run,
        ):
            self.assertEqual(resolver._github_pages(endpoint), [[]])
        which.assert_called_once_with("gh")
        self.assertEqual(
            run.call_args.args[0],
            [
                sys.executable,
                "api",
                "--hostname",
                "github.com",
                "--method",
                "GET",
                "--paginate",
                endpoint,
            ],
        )
        self.assertIs(run.call_args.kwargs["shell"], False)


class TransportAndOutputTests(unittest.TestCase):
    def test_cli_writes_only_the_original_compare_sha_and_auditable_origin(
        self,
    ) -> None:
        for sha in ("", MEASUREMENT):
            with (
                self.subTest(compare_sha=sha),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                result = {"base_sha": BASE, "compare_sha": sha, "origin": "fixture"}
                with (
                    patch.object(resolver, "resolve", return_value=result),
                    patch("sys.stdout", new_callable=io.StringIO),
                ):
                    resolver.main(
                        [
                            "--repository",
                            REPOSITORY,
                            "--base-sha",
                            BASE,
                            "--github-output",
                            str(root / "output"),
                            "--provenance",
                            str(root / "provenance.json"),
                        ]
                    )
                self.assertEqual((root / "output").read_text(), f"compare_sha={sha}\n")
                self.assertEqual(
                    json.loads((root / "provenance.json").read_text()), result
                )

    def test_http_failure_preserves_origin_and_never_writes_compare_sha(self) -> None:
        error = urllib.error.HTTPError(
            "https://coveralls.io/builds/20.json", 403, "Forbidden", {}, None
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(resolver, "resolve", side_effect=error),
                self.assertRaises(urllib.error.HTTPError) as raised,
            ):
                resolver.main(
                    [
                        "--repository",
                        REPOSITORY,
                        "--base-sha",
                        BASE,
                        "--github-output",
                        str(root / "output"),
                        "--provenance",
                        str(root / "provenance.json"),
                    ]
                )
            self.assertIs(raised.exception, error)
            self.assertFalse((root / "output").exists())
            record = json.loads((root / "provenance.json").read_text())
            self.assertEqual(record["origin"], "resolution_error")
            self.assertIn("403", record["error"])

    def test_coveralls_accepts_only_exact_https_json_endpoints(self) -> None:
        for url in (
            "http://coveralls.io/builds/20.json",
            "https://other.example/builds/20.json",
            "https://token@coveralls.io/jobs/20.json",
            "https://coveralls.io/builds/20.json?token=secret",
            "https://coveralls.io/api/v1/jobs",
            "https://coveralls.io/builds/20.json#fragment",
        ):
            with (
                self.subTest(url=url),
                patch.object(resolver.urllib.request, "build_opener") as opener,
            ):
                with self.assertRaises(ValueError):
                    resolver._coveralls_json(url)
                opener.assert_not_called()

    def test_coveralls_redirect_is_rejected_before_any_second_request(self) -> None:
        redirect = resolver._NoRedirect()
        redirect.parent = Mock()
        headers = Message()
        headers["Location"] = "https://another.example/steal"
        request = urllib.request.Request("https://coveralls.io/builds/20.json")
        with self.assertRaisesRegex(ValueError, "redirect"):
            redirect.http_error_302(request, io.BytesIO(), 302, "Found", headers)
        redirect.parent.open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
