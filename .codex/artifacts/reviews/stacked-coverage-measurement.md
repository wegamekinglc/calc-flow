## Branch Review: Stacked Coverage Measurement

**Head:** `92b3d87f8787b771ec24d51d5553b99e4d580f53`
**Base:** `eccb26973811bc476f0944b977ddedf8564b0237`
**Tree:** `fe3ce6bedd71598c8bec2feb876619fb2beac079`
**Scope:** Two CI resolver scripts and their analysis; three changed files.

### Summary

**Approve.** The new resolver path accepts a complete successful Coveralls
measurement at a different commit only after proving that its entire Git tree
equals the frozen requested base and that its coverage belongs to the exact
successful Linux PR run/head/repository/attempt. The emitted comparison SHA is
the actual measurement commit. It does not replace the requested base with a
current PR head, retarget the PR, alter coverage floors or skip required jobs.

The previously failing #373 request is base
`f56cc9f1c80e47b85d2ef63d2998806b2502b121`; recorded successful run
`37508149055`, attempt 1, has that exact head. Its actual Coveralls measurement
is `6e9be8f727e8dd170872a9a7cbe8557d79d817ea`. Recorded full trees both equal
`5b81c4caf3ca0503eac6541d805c23ad97543719`. The base tree was also confirmed
from the local Git object. The synthetic measurement object is absent locally;
its tree evidence comes from the saved read-only API resolution. This review
does not represent the offline replay as a fresh API verification.

### Build and Test Results

- Rust and native builds: **Not run; production sources unchanged.**
- Python CI tooling: **Passed** in author evidence: three actual RED regressions
  before implementation, then 37 focused tests with exit 0, focused Ruff lint,
  and a whitespace-only formatter correction.
- Reviewer additions: **Passed**, 14 focused offline cases with exit 0: thirteen
  rejection cases and one historical-run acceptance case. The saved real
  provenance also replays to an exactly equal resolver result.
- Studio backend/frontend: **Not touched; not run.**
- Current generated contracts, whitespace and production/dependency/workflow
  path comparison: clean. The reviewed HEAD and tree still match the freeze.
- CI: No fresh snapshot or polling. This local unpublished head has no new
  required-check result in this review. The saved historical successful run
  validates its measurement provenance; it does not establish future #373 CI.
- New failures or regressions: None identified. Coverage floors and full CI
  verification remain required before merge.

The original passing 37 tests were not repeated. Reviewer command:

```bash
python target/issue363-stacked-coverage-review/verify_review.py > target/issue363-stacked-coverage-review/verification.log 2>&1
```

The first helper attempt omitted a mock JSON report for its newly injected
pending status and raised `KeyError`. Its raw log is retained as
`verification-initial-fixture-missing-report.log`. After providing that mock
report, the resolver rejected the pending status as intended and all focused
checks passed. This was a reviewer fixture correction, not a product failure.
No original passing suite was rerun.

### Blocking Issues

None.

`_resolve_head_measurement` requires exactly the three known flags and all
latest Coveralls statuses to succeed. `_complete_head_candidate` checks the
base and actual measurement's full trees, validates the requested repository's
name and positive integer ID, and passes the frozen base SHA into
`_successful_run`. That helper requires the precise run ID, Linux workflow,
`pull_request` event, successful completion, matching URL, positive attempt and
both repository identities. It does not read the present PR head.

The existing coverage helpers enforce all three successful jobs on that
run/attempt/head, each required successful step, and unique nonexpired
positive-byte artifacts with SHA256 digest metadata. Artifact workflow IDs,
head and repository IDs must match, and creation must fall inside the
corresponding successful job's time window. These checks validate API artifact
metadata; this review does not claim it downloaded or independently hashed
artifact archives.

Transport remains fixed read-only GitHub GET with an absolute executable and
no shell. Coveralls requests remain restricted to exact HTTPS JSON endpoints
with redirects rejected. Failed resolution preserves failure provenance and
does not emit a comparison SHA. Caller-owned response inputs are unchanged.

### Style Issues

None found in the changed source, tests or analysis. Python future annotations
and type conventions are retained. Markdown has final newlines and no trailing
whitespace; the analysis contains no pipe tables needing alignment.

### Test Coverage

The author suite covers the new success path, different full tree/head, failed
or untrusted workflow, incomplete/failed statuses, fork/repository mismatches,
coverage steps and expired/stale/unsigned artifacts. Existing tests retain
partial merged-PR alias behavior, ordinary complete same-commit default
behavior, pagination, required attempts, transport restrictions and CLI output.

Additional reviewer cases reject a newer pending complete-head flag, boolean
requested repository ID, wrong run repository ID, wrong coverage job
run/attempt/head, missing coverage job/step, wrong artifact head/repository ID,
empty/boolean artifact size and creation after job completion. A moved current
PR head leaves the historical exact-base run valid without any PR-record read.
This accepted case also checks that the input fixture is not mutated.

The same-commit complete-flags path deliberately keeps its prior default
behavior, including returning the default when a status is failed. This change
does not add an alias in that case; it preserves the existing Coveralls/CI
failure handling. The new different-measurement path is strictly successful.

### Documentation and Evidence Consistency

The [analysis](../analysis/stacked-coverage-measurement.md) accurately records the
trigger, accepted evidence, RED/GREEN results and unverified future CI. Its
complexity evidence is explicitly an AST estimate, not a Codacy result. No
public API, workflow command or normative runtime documentation changed.

Exact SHA256 bindings:

- Resolver: `74f6a995ba58fcce478cc364920cbbc3b28cf85f3e521b9ac6fea8307a9efc56`.
- Focused tests: `c677631d0207685f0d1bdfe71817bd5ae152185f534ec180afd7f754835901b9`.
- Analysis: `75de76a93ba9c20e24e2e129774f80252a7879ceba4f235ae1b4b8d72dd581b3`.
- Author evidence: `189558af01f9376e5ce02f10d4b0e6d8d384ae77b8f0a78b69198bdea48da950`.
- Saved live provenance: `00f40ecb734fc6bd5dd294f0c03bb58975b4c9f21439be76623bfbe17b80f003`.
- Actual RED log: `5f4e1c3d292295753f0e421dfa49d724be413d1fb7b87fe8d84215322e99d678`.

All match the frozen files and receipt. Reviewer helpers, raw logs and proof
are under repository `target/issue363-stacked-coverage-review`.

### Effect of Cherry-Picking the Two Scripts into #373

No native performance input rebuild is required solely for these two CI script
changes. The diff leaves Rust/Python production, dependency manifests,
toolchain and workflows identical; the scripts are CI coverage tooling and
are outside the native performance source-identity closure. The sealed A3/A4
native inputs can continue measuring their original exact approved sources.

The new #373 Git SHA/full tree will differ. Preserve the original source,
build, binary and closure seals, and record that the new head differs only in
the reviewed CI tooling before associating it with those measurements. Do not
relabel the existing binaries as fresh builds of the cherry-picked head. Any
additional production/dependency/build-setting change would require a new
assessment. Required CI must run and be green at the actual final PR head.

### Verdict

**Approve** for the frozen scoped change. Publication and merge remain separate
actions with their existing required checks. No remote mutation, CI polling,
native import/build or performance execution was performed. Reviewer-owned
processes: **0 running**.
