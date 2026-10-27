# Working-tree review: Rust counterparts for Python examples

## Summary

Reviewed the 36-entry Python/Rust inventory, core and connector examples,
shared source/sink and array helpers, Cargo feature gates, runner selection,
and README changes. The initial review covers `ce73bb8e`; the follow-up reviews
the PR #507 working-tree corrections against that commit. The follow-up used
the author's supplied Codacy annotations and validation evidence. The reviewer
did not query or poll CI status.

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
- Real Kafka/database service runs, full regression/coverage, and performance
  measurements were not performed locally. Hosted CI results remain separate.

## PR #507 Codacy follow-up

Reviewed the extraction of graph construction, execution, and cleanup helpers,
the shared `connect_all` helper, argument decoding, stock CLI parsing, and the
corresponding SQL documentation update. Static comparison found no changes to
node IDs, operator configurations, SQL, exact schemas, edge order, or physical
bindings that would change graph fingerprints. Matrix shapes, metadata and
placement behavior, trade seed/period generation, and older-last quote ordering
remain intact. Connector startup and waiting share one deadline; failed or
timed-out startup awaits cleanup, and waiting timeouts cancel the job before
dropping ownership and awaiting cleanup.

One blocking regression was found and fixed during this follow-up: collecting
raw repeated stock options before parsing discarded earlier invalid values.
Each occurrence now validates against its u64/u32 target type before collection.
The final value still wins for valid duplicates, and final positive/count
constraints retain the prior `--accounts 0 --accounts 1` behavior. Two focused
tests first failed for invalid earlier row/account values, then passed with the
fix; a third test preserves valid duplicate handling.

Author-provided final evidence:

- The Codacy snapshot reported four argument-API findings and 31 complexity
  findings. Local Lizard reproduced those plus one pre-existing SQL example
  function above eight; the final scan reports no threshold violations.
  No rule suppression or threshold adjustment was introduced.
- Four argument entry points now use `args_os` and return a UTF-8 decoding
  error. Actual invalid-byte invocations of all four return errors without
  panicking. The original panic reproduction is retained in
  `target/pr507-argv-red.log`.
- Stock CLI tests passed all five tests, including Unicode and repeated-option
  cases. Red/green evidence is in `target/pr507-stock-cli-{red,green}.log`.
  Both Kafka example test targets passed five tests each.
- All 25 core demos again returned zero in
  `target/pr507-core-results.json`. The five affected connector demos—file,
  both Kafka offline decoders, local HTTP, and local WebSocket—passed in
  `target/pr507-local-connector-results.log`. Unchanged late-output recovery
  retains its earlier passing evidence.
- Native example builds, core example Clippy, connector example Clippy with
  all features, and the final focused persistence Clippy check passed. The
  persistence example retains its original-project immutability assertion.
  Kafka argument helpers were moved before their test modules, and the SQL
  documentation was synchronized with the self-contained executable source.

The reviewer did not duplicate Cargo checks or read another CI status snapshot.
Local evidence does not establish that the hosted Codacy/CI checks have cleared.

## Verdict

**Approve** for the example-porting scope and the reviewed PR #507 corrections.
No unaddressed blocking finding remains. Validation covers all core examples
and the local connector paths listed above; it does not establish CI success,
coverage-gate success, or merge readiness.
