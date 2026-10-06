# PR365 Python CI remediation

The fresh-process Polars reference test now owns a multiprocessing spawn worker
instead of constructing a Python subprocess command. The test checks the child
PID, the actual Polars thread pool size, and all seven shared reference oracles.
The worker has a 30-second completion bound; timeout and failure release its
process and pipe resources. Environment configuration stays in the owned child.

Baseline catalog parsing keeps the existing declarative, fail-closed contract.
Cap validation and backend dimension selection are separate functions; the
case-ID comprehension retains the same join, ASOF, and window cap semantics.
No Codacy rule or ignore configuration changes.

## Validation

- RED: both fresh-process cases failed because `_fresh_polars_samples` was
  missing before its implementation.
- GREEN: fresh-process 10-row and 101-row references, plus larger-pool rejection:
  3 passed.
- Existing benchmark suite script tests: 47 passed, including real-catalog
  membership, historical caps, reopened coverage, and mismatched scopes.
- Scoped Ruff check and format check passed; `git diff --check` passed.
- No Rust build, performance measurement, or remote operation was performed.
  The remote Codacy result remains a CI handoff check.
