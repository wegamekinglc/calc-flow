# Stacked PR coverage baseline correction

## Observed failure

PR #373's Linux job `Verify coverage comparison base` failed in run
37540439308, job 112531925159. Its requested base was
`f56cc9f1c80e47b85d2ef63d2998806b2502b121`; the resolver rejected the complete
Coveralls build because GitHub's PR workflow measured a synthetic merge commit.

Read-only resolution of that actual base found successful Linux PR run
37508149055, attempt 1, with the requested base as its exact `head_sha`.
Coveralls measured `6e9be8f727e8dd170872a9a7cbe8557d79d817ea`. Both commits have
the full Git tree `5b81c4caf3ca0503eac6541d805c23ad97543719`. The Rust, Python,
and Studio coverage jobs, required coverage steps, and nonexpired artifact
metadata all match the same run, attempt, head, and repository.

## Accepted evidence

The new path accepts this measurement only with all three successful Coveralls
flags and aggregate, an identical complete Git tree, and a completed successful
same-repository Linux `pull_request` run. Each required coverage job and step
must succeed, and each artifact must have a SHA256 digest, positive byte size,
the exact run/head/repository identity, and a creation time within its successful
job. The emitted comparison SHA is the actual measurement commit, with full
provenance retained.

Failed or partial reports, forked runs, another head or workflow, another tree,
stale artifacts, and missing coverage evidence remain errors. The existing
partial merged-PR alias and ordinary complete same-commit baseline behavior
remain intact. Coverage floors and CI workflow steps are unchanged.

## Local validation

Before changing production code, three regression tests failed: one error and
two assertion failures at the original different-commit rejection. After the
change, all 37 resolver tests passed, including nine complete-head tests and
the existing transport, merged-PR, attempt, coverage-step, and artifact checks.
Focused Ruff lint passed; the formatter changed one condition's whitespace.
The actual frozen base resolved successfully through the new path.

The raw failure log, red log, local evidence receipt, and live JSON provenance
are retained under `target/issue363-integration/stacked-coverage-*`. Independent
specialist review is required before publication. This local result does not
claim that a future PR CI run has passed.
