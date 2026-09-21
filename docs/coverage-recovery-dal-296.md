# DAL-296: recover the exact main coverage measurement

This is a reviewed, one-shot data repair for PR #318. It does not grant merge
approval, replace Linux CI, or erase the failed complexity check on main.
The original Kafka branch and PR remain unchanged during recovery preparation.

## Source, control, and entry point

The measured source is exactly
`2fcc36fd224dd1c8a9f6396bddd5fbe0b60b0987` (main after #323). Its Linux run
`35640870045` never produced Python coverage: the complexity gate failed before
pytest. PR #323 run `35640307750` has the same omission. Neither is a complete
three-flag baseline, and neither old workflow should be retried.

Review branch `fix/dal-296-coverage-recovery` starts at that main commit and
reuses the one-line CalVer test split from #318 commit `858377e`. It contains
only that test repair and the recovery workflow, evidence helper, focused tests,
and this procedure. It contains no Kafka change. The test split is **not**
applied to the measured checkout; main's original tracked tree stays intact.

The control revision is the complete reviewed commit of this recovery branch.
`.github/workflows/coverage-recovery.yml` runs only on a push to
`fix/dal-296-coverage-recovery-execute`. Pushing the review branch does not start
this workflow. GitHub supports [push workflows before they are merged into the
default branch](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#push),
so this path needs neither a main merge exception nor a dispatch workflow
already installed on main. No dependency PR needs to merge to repair the data.

Each job checks out the workflow tools at `github.sha` in `control/`, and the
immutable measured main commit in `source/`. Only `source/` supplies the wheel,
tests, configuration, coverage commands, and coverage floors. The helper checks
both HEADs and tracked cleanliness before execution and again when sealing the
reports. Control files never replace measured source files.

## Measurement and publication contract

- Rust uses the existing `run_rust_coverage.py`, with all its features/targets,
  the MySQL, Kafka, PostgreSQL and ClickHouse services, and the 90% line floor.
- Python builds and inspects a wheel from the measured tree, installs that wheel,
  and runs the existing full unit/property coverage command, including
  `benchmarks/test_warm_stream.py` as correctness tests. Its 90% floor remains.
- Studio uses the same wheel and the measured editable Studio package. It runs
  existing backend coverage (85% floor) and generated-contract checks.
- This measurement-only workflow does not run the known failing historical
  complexity gate or performance sampling. That historical failure remains
  visible; the CalVer test repair must still pass ordinary CI on #318.
- The two producer jobs have bounded 90-minute budgets. Publication has 15
  minutes. These are recovery budgets; ordinary Linux CI budgets are unchanged.

Both producers must succeed before publication. Raw artifacts
`recovery-rust` and `recovery-python-studio` retain the original LCOV/Cobertura
files, preflight evidence, sealed report SHA256 values, and the Python wheel's
SHA256. On failure, `always()` artifact upload preserves whatever was produced;
an incomplete producer cannot reach publication. Artifacts expire after 30 days;
download them with logs and GitHub metadata before that deadline.

The publisher downloads only these artifacts from its own workflow run. It
requires exactly `rust`, `python`, and `studio`, all with the same source SHA,
full tree, workflow SHA, run ID and attempt; it rechecks each report digest.
`recovery-publication` retains the merged evidence. The pinned Coveralls action
uses its documented [commit, branch and build-number inputs](https://github.com/coverallsapp/github-action/blob/8d6379e14d29928660c4ba802d8e85393440b329/action.yml):

- measured commit: `2fcc36fd224dd1c8a9f6396bddd5fbe0b60b0987`;
- measured branch: `main`, regardless of the workflow control branch;
- build number: this new GitHub `run_id`, identical for all three flags and finish;
- run attempt: exactly `1`; reruns are rejected;
- no carryforward, comparison SHA override, or status API writes.

The finish request is sent only after all three uploads succeed. A Coveralls
percentage regression can remain a real failure: finish is not a synthetic
success. GitHub checks belong to the control commit; Coveralls reports belong
to the source commit they actually measured. The two identities must not be
conflated. The production resolver is unchanged.

## Review before execution

cf-reviewer must review the exact control commit and this execution path before
the same writer creates the execution ref. Confirm the source SHA, copied
coverage commands and floors, artifact layout, token scope (`contents: read`
plus the existing Coveralls GitHub integration), publication identity, and
failure behavior. Record approval against the full control SHA in DAL-296.
This is recovery-plan approval, not final #318 acceptance.

After approval, the original writer runs the following from the recovery
worktree. Set `REVIEWED_RECOVERY_SHA` to that approved full commit, not to a
moving branch name. The expected execution branch must not already exist;
if it does, stop and investigate its run history instead of forcing it.

```bash
: "${REVIEWED_RECOVERY_SHA:?set the approved full recovery commit SHA}"
test "$(git rev-parse HEAD)" = "$REVIEWED_RECOVERY_SHA"
test -z "$(git status --porcelain --untracked-files=no)"
test -z "$(git ls-remote --heads origin fix/dal-296-coverage-recovery-execute)"
git push origin "$REVIEWED_RECOVERY_SHA:refs/heads/fix/dal-296-coverage-recovery-execute"
```

Take one run-discovery snapshot using `gh run list --branch
fix/dal-296-coverage-recovery-execute --commit "$REVIEWED_RECOVERY_SHA"`.
Record the resulting run ID. Because this card explicitly requests final
recovery results, collect that run with one foreground `gh run watch
"$RECOVERY_RUN_ID" --exit-status`, then collect its run metadata, attempt-1 jobs,
raw artifacts and logs. Do not run #318 CI while recovery remains incomplete.

```bash
: "${RECOVERY_RUN_ID:?set the new recovery workflow run ID}"
mkdir -p "target/recovery-$RECOVERY_RUN_ID"
gh api "repos/wegamekinglc/calc-flow/actions/runs/$RECOVERY_RUN_ID" \
  > "target/recovery-$RECOVERY_RUN_ID/run.json"
gh api --paginate "repos/wegamekinglc/calc-flow/actions/runs/$RECOVERY_RUN_ID/attempts/1/jobs?per_page=100" \
  > "target/recovery-$RECOVERY_RUN_ID/jobs.json"
gh api --paginate "repos/wegamekinglc/calc-flow/actions/runs/$RECOVERY_RUN_ID/artifacts?per_page=100" \
  > "target/recovery-$RECOVERY_RUN_ID/artifacts.json"
gh run download "$RECOVERY_RUN_ID" --dir "target/recovery-$RECOVERY_RUN_ID/artifacts"
gh run view "$RECOVERY_RUN_ID" --log > "target/recovery-$RECOVERY_RUN_ID/run.log"
```

Verify GitHub's `head_sha` is the approved **control** SHA, event is `push`, path
is the recovery workflow, `run_attempt` is 1, and all three jobs succeeded.
Verify artifact digests/non-expiry and retain their IDs. The publication
manifest must name the exact **measured** main SHA and that same run ID.

## Resolver acceptance and downstream handoff

After the completed uploads, execute the unmodified production resolver once:

```bash
python scripts/resolve_coverage_baseline.py \
  --repository wegamekinglc/calc-flow \
  --base-sha 2fcc36fd224dd1c8a9f6396bddd5fbe0b60b0987 \
  --github-output "target/recovery-$RECOVERY_RUN_ID/resolver-output.txt" \
  --provenance "target/recovery-$RECOVERY_RUN_ID/resolver.json"
```

Require `origin == "default_complete_flags"`, an empty `compare_sha`, the exact
base/build commit above, all three flags, and `base_report.run_id` equal to the
new recovery run ID. Retain the original aggregate and flag states, Coveralls
build/job URLs and report JSON; do not turn a measured regression green. The
resolver checks every flag's `full_number` against the build's GitHub run URL,
so flags from separate runs cannot form the acceptance record. Collect current
main commit statuses as supporting evidence, not as a substitute for reports.

Only this successful provenance check establishes that the baseline condition
has changed. Hand the evidence to cf-tester to restore the necessary failed
#318 job/dependencies and collect actual-HEAD non-performance results. #318
still needs current Rust tests, SQL smoke, rustdoc, coverage, Codacy and review
resolution before cf-reviewer's separately authorized merge. Keep DAL-296 and
DAL-297 open and do not start #320.

## Failures and permissions

Stop on a source mismatch, dirty tree, failed tests/floor, missing artifact,
digest mismatch, permission error, upload failure or resolver rejection.
Preserve the run, attempt and error. Partial uploads do not authorize downstream
CI. Do not stitch old flags into a new build, carry flags forward, edit statuses,
replace the baseline, patch the measured tree, or relax the resolver.

There is no blind retry: rerunning a job/run is disabled by the original-attempt
guard. Diagnose first. A new execution requires review of the correction and
a new push/run; regenerate **all three** reports together. If Coveralls has not
yet exposed reports when the single resolver probe runs, retain that failure
and wait for explicit external-state evidence before a later probe.

Preparation requires permission to push the review branch. Execution additionally
requires creating the execution branch, GitHub Actions enabled for that push,
runner/service capacity, artifact storage, and the repository's existing
Coveralls GitHub-token integration accepting an explicit measured commit.
These runtime permissions cannot be certified by local tests. If any fails,
report its concrete error and retained artifacts to the coordinator; do not
substitute another identity or bypass a gate.
