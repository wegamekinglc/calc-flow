# Working-tree review: Rust counterparts for Python examples

## Summary

Reviewed the 36-entry Python/Rust inventory, core and connector examples,
shared source/sink and array helpers, Cargo feature gates, runner selection,
and README changes. This is a local working-tree review; no PR or CI status
was consulted.

## Findings

No unaddressed blocking issue remains in the reviewed code. The author fixed
the issues raised during review:

- Matrix exact schemas now match fixture nullability.
- The continuous-runtime collector reads the calculated `result` column.
- The connector registry registers file/format codecs once.
- Connector startup and execution share a deadline and await cleanup.
- Stock examples await finite completion, and valuation verification retains
  SQL's Int64-to-Float64 conversion boundary.
- Stateless/stateful edges declare matching exact schemas, batch projects
  describe their host input bindings, and checkpoint-free stock plans request
  BestEffort delivery explicitly.
- The matrix cache owns one static-array snapshot and borrows its values,
  matching the demonstrated 16/0 placement-byte counts.

The final focused review confirmed the composed JOIN's unique external
`left` binding and qualified `matched.right`/`settled.right` bindings. The
stock-position collector uses the distinct `collector` sink ID. The SQL JOIN
documentation now includes the same expression stage and doubled results as
the executable example.

The native array examples deliberately replace NumPy/JAX hosts with an
illustrative Rust provider. The README makes that adaptation explicit.
Financial stages, managed rolling recovery, JOIN/ASOF finality, shared window
state, late outputs, and lazy stock/quote fixtures have corresponding native
declarations and result assertions. External sink examples keep their writes
within the prepared demo destinations documented for the Python versions.

## Verification

- Author evidence: all 25 core user examples executed successfully, with zero
  exit codes recorded in `target/rust-example-results.json`.
- Six connector demos passed: file input in CSV/JSON/Parquet with ExactlyOnce
  output, late-output recovery, both offline Kafka decoder demos, local HTTP,
  and the Rust WebSocket service/source pair. Results are recorded in
  `target/rust-local-connector-results.log`; local services were stopped.
- Zero-trade positions and a partial trade/quote-round valuation also passed.
- Both Kafka example test targets passed three tests each, including offline
  managed-project compilation and connector wiring. Five runner tests passed;
  the initial missing-counterpart failures were recorded before implementation.
- Core example Clippy passed with warnings denied, followed by focused checks
  for the final JOIN binding and stock sink-ID edits. Connector example build
  and Clippy passed with all features. Existing local curl headers supplied
  through a `target/` include symlink resolved the environment's missing-header
  build failure without changing project dependencies.
- Reviewer checked Markdown table alignment, links/inventory structure,
  `git diff --check`, and absence of generated project-schema drift.
- Author reports formatting, 251 local Markdown links, whitespace, and schema
  drift checks passed. The reviewer inspected the final binding/sink-ID changes
  and documentation synchronization without duplicating the author's checks.
- Real Kafka/database service runs, full regression/coverage, CI, and
  performance measurements were not performed.

## Verdict

**Approve** for this example-porting scope. No unaddressed blocking finding
remains. Validation covers all core examples and the local connector paths
listed above; it does not establish CI success, coverage-gate success, or
merge readiness.
