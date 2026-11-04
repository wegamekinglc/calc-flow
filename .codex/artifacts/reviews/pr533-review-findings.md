## PR #533 corrective branch review

**Author:** wegamekinglc | **Branch:** fix/pr533-review-findings | **Files:** 13

Reviewed local verification checkpoint:
`4fbbaf3410a21b50c0faedd8a46050371e576a46`.
Code tree before adding this review artifact:
`45acd16ef519da9463e104865770acf0cf299669`.
Review base: published PR head `65dddc45`.

This is a branch review before publication. Remote publication may use a
different commit identity; compare its code tree with the reviewed tree rather
than treating the local commit as a publicly accessible GitHub revision. A
detached review worktree was used and removed after inspection.

### Summary

The corrective changes decompose the diagnostic example, fixture helpers and
Python measurement orchestration to satisfy the existing complexity gates.
They audit the intended local subprocess boundaries, resolve executables,
retain argv-only execution with `shell=False`, and add focused lifecycle,
profile and fixture tests. Neither production SUM implementation nor archived
performance evidence changes.

The known Codacy inventory contains 41 added issues: 22 Lizard complexity
findings, 13 Bandit audit findings and six Opengrep subprocess audit findings.
The reviewed changes address each affected function/call site without adding
global analyzer exclusions or weakening the complexity baseline.

### Build and Test Results

Existing evidence was inspected; the reviewer did not repeat tests, builds,
analyzers or performance measurements:

- **108 focused Python tests passed**, including collector/publication/build/
  profile/toolkit checks, complexity-gate controls and the new workflow
  dependency-order regression. The root reported 107 passing tests before that
  final configuration regression and 108 unique tests afterward.
- **Seven grouped Rust tests passed**, including the new fixture boundary/mask
  regression and the existing integer/floating state controls. The inspected
  log is `target/pr533-rust-grouped-tests.log` in the original repository.
- **Two diagnostic example functional tests passed**: fixture report/input
  round-trip with fixed expected digests, and the CLI request contract.
  The inspected log is `target/pr533-rust-example-tests.log`.
- **Fresh scoped strict Rust Clippy passed**, as recorded in
  `target/pr533-rust-clippy.log`.
- **Normal Ruff, formatting and the actual repository complexity ratchet
  passed**, as supplied in the root handoff. Lizard 1.23.0 found no targeted
  function above CCN 8 and no new complexity exemptions were introduced.
- **Bandit 1.8.6 reported zero findings and zero errors** over the four SUM
  Python modules. The reviewer inspected `target/pr533-bandit-final.json`.
  Its four narrow audit skips correspond to the reviewed subprocess imports
  and explicitly owned process boundaries; they are not a module-wide skip.
- **Semgrep 1.140.0 reported zero findings and zero errors** when applying the
  two corresponding official subprocess rules to the same four modules. The
  reviewer inspected `target/pr533-semgrep-final.json`. This is a local rule
  reproduction, not a claim that remote Codacy's Opengrep reanalysis has passed.

No full product matrix, coverage run or performance remeasurement was performed
for this diagnostic/tooling refactor. No new unresolved failure is present in
the inspected local evidence.

### Blocking Issues

None remain in the reviewed corrective branch.

One new CI blocker found during this review was corrected before approval:
the SUM tool tests had initially been appended to the package job's bare
`python -m unittest` command, before installing their `psutil` dependency.
The final workflow runs them in `lint-and-test` with
`uv run --no-sync python -m unittest`, after the existing
`uv sync --extra dev --no-install-workspace` step. The focused
`test_sum_tool_tests_run_after_installing_dev_dependencies` regression first
failed on the wrong routing and now verifies both dependency ordering and
absence of these modules from the bare package job.

### Security and Refactor Review

- `validated_argv` rejects shell-command text, non-string vectors and embedded
  NULs. The subprocess boundary selects an absolute executable regular file,
  retains literal argument values and passes `shell=False`. Benchmark workers
  remain explicitly selected local artifacts bound by the existing build
  receipts and hashes; this tool intentionally executes those artifacts.
- The fixed Cargo command resolves its executable and keeps module-owned flags.
  Source/target paths remain cwd/environment values rather than interpolated
  shell commands. Rust compiler and system-tool queries use resolved binaries.
- Profile command helpers retain their deadline and environment arguments.
  `nm` input paths are resolved before use, preventing a relative filename
  beginning with `-` from becoming a tool option. The shared toolkit helper's
  new optional environment/timeout parameters preserve existing defaults.
- The remaining inline Bandit/Semgrep audit annotations are attached to the
  specific reviewed argv-only process boundary. No `.codacy.yml` exclusion,
  rule removal, broad security bypass or complexity-baseline increase is part
  of the correction.
- Process polling, process-group termination/kill/reaping, source seals,
  cumulative budget admission, alternating sample order, complete snapshot
  validation and final status demotion remain intact through decomposition.
  Tests explicitly exercise timeout cleanup and spawn failure.
- The publication refactor retains the previously reviewed late-deadline fix:
  source verification and late publication demote durable statuses and rebuild
  dependent artifact hashes before returning failure. The existing mocked-clock
  tests continue to inspect source/published/raw/receipt records.
- Profile decoding retains exact-window selection, per-thread clock matching,
  weighted stack counts, mutually exclusive phase attribution and retained
  accepted stacks. The synthetic decode test checks all-and-only processing
  window samples without launching native tools.
- Rust fixture decomposition preserves values, null masks, schemas, key shapes
  and the greater-than-8,192 rank boundary. Diagnostic extraction preserves
  fixture generation, query declarations, primary timer placement and the
  post-timer independent snapshot oracle. Its source identity nevertheless
  changes and is correctly not described as a remeasured harness.

### Test Coverage

The added tests protect actual contracts rather than analyzer implementation:
literal shell-looking arguments stay literal; a timed-out process group is
terminated, killed when necessary and reaped; failed spawn does not kill an
absent process; invalid executable selection is rejected; profile samples are
assigned only to accepted clock windows; fixed fixture digests and CLI fields
remain stable; and CI runs dependency-requiring tests in the managed environment.
Existing negative evidence, source-compatibility, sparse/null state and late
publication tests remain in the corrected suite.

### Documentation and Historical Evidence

The reviewer inspected the complete corrective diff against `65dddc45`.
There are no changes to `incremental.rs`, `grouped_sum.rs`, `.codacy.yml`,
`scripts/complexity_baseline.json`, or either archived result directory.
The existing performance verdicts, raw observations, source seals and artifact
hashes therefore remain historical evidence for their recorded revisions.

The reproduction guide now explicitly pins measured product/harness revision
`84186554b078e25e71003a74fedc91c63b2b7f70` and harness SHA-256
`657de3e697907e473691c2fd0d1b99967d46577333135c50176801f6faa4e04e`.
It overlays that measured harness into the baseline and states that the split
diagnostic sources have not been performance remeasured. This prevents
attributing the archived timings to the newly decomposed instrumentation.

### CI State and Verdict

The prior Linux job failed at `Verify complexity gates`, before Python pytest
and coverage upload. The associated Coveralls failure does not establish a
coverage regression; the prior Rust coverage and cross-platform successes also
do not certify the new corrective head. Remote Codacy/Opengrep and CI results
must be reported from the root's single post-publication snapshot. This reviewer
made no CI status poll, remote review submission or remote mutation.

**Approve — PASS for the reviewed corrective code and local evidence.**

The known 41 findings and local CI routing regression have no remaining source
blocker. Fresh remote analysis and required CI remain unverified by this review;
pending or failed required checks still block merge. No merge authority is
granted.
