# Weekly SQL implementation review

Reviewed the SQL portion of mainline interval
`ce313488017d63fc626731f9731a4bec931fc343` through
`1b91bcc00a0623d5ece37ac654bf6e2a4f045939`, following the
[weekly scope](../specs/weekly-review-20261107.md). The final corrected source
is captured in `7dbd229d23fb4060c1209f518ef08bee8c82c508`.
This is a scoped remediation handoff; its author also implemented the SQL
fixes, so independent approval is supplied by the SUM/stream reviewer.
The parent owns performance acceptance, CI and delivery.

## Confirmed and repaired findings

1. **P1 — Native SQL join admission discarded resolved relation semantics.**
   `datafusion_join::planning::derived_table` accepted nested CTEs and
   relation column-alias lists; `named_table` also ignored column-alias
   lists. The native executor subsequently selected the original registered
   tables and column names. DataFusion's successful logical planning did not
   make that replacement equivalent. An ASOF subquery containing a CTE that
   filters the right relation to zero rows incorrectly returned price 10
   instead of NULL; a matching interval query returned a row with value 22
   instead of no rows. Two valid alias-list queries returned price 10 instead
   of 20. The fix declines these unsupported forms before native admission.
   Ordinary aliases and the canonical ASOF/interval workloads stay eligible.

2. **P2 — Native joins inherited the preceding query's rolling audit.**
   `DataFusionRuntime::plan_query` read shared rolling diagnostics even when
   the native join branch bypassed `CalcFlowQueryPlanner`. Reusing a runtime
   after a rolling AVG therefore reported one candidate window on a later
   native join. The native branch now owns a default audit snapshot; ordinary
   DataFusion planning still reads the planner-produced audit. There is no
   eager mutation of shared audit state. The regression checks AVG followed
   by native ASOF and native interval SQL on the same runtime.

3. **P2 — Experimental input partitions could silently reuse a serial plan.**
   Cached row-local templates accept a single scan partition and always
   rebind one partition. Two failures were reproduced: warming a P1 plan
   before selecting P4, and keeping P4 configured while changing an input
   from one RecordBatch to eight. The second case escaped an initial
   setter-only invalidation and was identified in independent review.
   Template lookup and capture are now bypassed when experimental
   `input_partitions > 1`; the setter-only workaround was removed. Default
   P1 template reuse, including a same-value setter call, remains tested.
   Experimental P4 now replans even when one input RecordBatch limits its
   actual input partition count to one. That planning cost must be included
   in future partition experiments. The maintained probe constructs fresh
   plans, so this finding alone does not invalidate historical probe results.

## Simplification

The ASOF predicate matcher no longer maintains a duplicate equality helper
whose two expression arguments were always identical. It calls the existing
`column_pair_equality` helper with equal field names. Existing native positive
and fallback tests cover both the shared predicate handling and admission.
The cache fix enforces compatibility at lookup/capture rather than trying to
infer input layout changes from the benchmark-only setter.

## Verification evidence

Every added behavior regression failed before its corresponding fix:

- Five native-join regressions failed with the expected wrong values or stale
  audit; the corrected native-join module passed all 28 tests in 5.40 seconds.
- The configured-partition-change regression failed with physical partition
  count 1 instead of 4. Six partition tests passed after the initial fix.
- Independent review then supplied the fixed-P4 input-layout counterexample.
  Its new test failed with partition count 1 instead of 4 before the final
  cache guards. The complete focused partition module subsequently passed
  all seven tests; runtime was 0.06 seconds.
- The coordinated ASOF output-plan checks passed all 15 tests, including
  source validity accounting, compact/legacy parity, cancellation, allocation
  and tight-pool controls. The SQL reviewer independently inspected the ASOF
  source-identity fix and found no blocker.
- Scoped core Clippy passed with warnings denied. Changed Rust files were
  formatted and whitespace checks passed. No public schema was changed.

`cargo test --locked -p calc-flow --features benchmark-internals --lib
--no-run` built the unit-test executable; the native-join and ASOF output-plan
filters ran directly from that freshly rebuilt executable. Input partition
checks used `cargo test --locked -p calc-flow --features benchmark-internals
--lib` with their focused filters. Lint used `cargo clippy --locked -p calc-flow
--features benchmark-internals --lib -- -D warnings`. Source validation used
one dedicated target directory, one Cargo job, incremental compilation
disabled and development debuginfo disabled. A copied dependency cache was
reused; every changed core executable was rebuilt, and no stale executable
was relabeled as current. Build costs are separate from scientific timing.

Raw commands, outcomes and logs are under the original workspace's
`target/weekly-review-20261107/`: `sql-red`, `sql-green`, `asof-red`,
`asof-green`, `partition-cache-red`, `partition-cache-green`,
`partition-layout-red`, `partition-layout-green`, and `native-clippy`.
The initial setter-only candidate was stopped before any scientific
observation. It is retained as intermediate evidence, not the final fix.
Full Rust/Python/connector/coverage/platform matrices were not run locally;
Clippy was scoped to the core library, not every integration-test target.

## Audit coverage and limits

The review covers the changed native batch ASOF/band admission and execution,
native output chunking and ordering, rolling argmax/argmin shape admission and
sliding state, automatic partition policies and their compile-time projection
classifier, experimental input partition registration, template binding, and
the removed stream row-budget ceiling. Relevant adjacent callers and tests
were inspected. No additional concrete semantic defect was established in the
bounded Float64 extrema rewrite or AutoPlan classifier. In particular,
DataFusion 55's sliding extrema compare Float64 ScalarValues using total
ordering, consistent with the rewritten accumulator's signed-zero/NaN order.

This is not an exhaustive audit of unchanged internals in the large runtime,
SQL engine or batch lifecycle modules. Streaming SUM, streaming joins,
Python surfaces and benchmark orchestration have separate reviewers. Existing
rolling tests were inspected; unchanged rolling suites were not rerun merely
to claim broader coverage.

## Remaining SQL performance proposals

1. **Build native interval output directly into Arrow arrays.**
   `BandState::next_batch` fills three `Vec<Option<_>>` staging buffers and
   then converts each with Arrow `FromIterator`, which allocates and fills new
   value/null buffers. Typed Arrow builders could remove those staging
   allocations and conversion passes. Preserve null values, null-slot default
   values, output order/multiplicity, chunk boundaries and limits. Validate an
   allocation bound and differential nullable/duplicate/empty outputs before
   measuring canonical interval SQL and a sparse control. This is a concrete
   source-derived work reduction, not a measured end-to-end gain.

2. **Avoid unused dense-band cursor allocations.** Each partition currently
   allocates one cursor vector per key with one entry per time offset, even
   when `dense_step` makes `band_first` return without reading a cursor.
   Allocate cursors only for irregular bands, retaining the existing dense
   arithmetic and irregular fallback. The benefit is bounded setup work and
   memory, not reduced per-row join complexity. Check mixed dense/irregular
   keys, duplicate times and out-of-order probes; measure a small canonical
   case and an irregular control before accepting it.

3. **Reduce ASOF input/output materialization.** The executor concatenates
   whole input tables and gathers its left output using contiguous indices.
   Projecting required columns before concatenation and using an eligible
   left-column slice could avoid unnecessary copies. SQL output bounds count
   visible Arrow slices; retained oversized backing buffers remain a separate
   lifetime and resource risk. Test sliced inputs, nullable payloads, held
   outputs, exact schemas and tight budgets. Measure ASOF with a wide unused
   payload and its ordinary narrow control, without changing result delivery.

A separate maintenance refactor could prove native admission from resolved
logical-plan bindings instead of repeating SQL name resolution in the AST
matcher. The CTE and alias bugs demonstrate why that boundary deserves a
design, but a larger planner rewrite should not be bundled into this repair.

No new SQL profile or performance sample was collected by this reviewer.
These proposals require their own declared baseline/candidate, fixed samples,
relevant controls and resource/correctness gates within the repository's
measurement scope. Use the maintained paired regression rule; reject
confirmed regressions and do not call inconclusive results equivalent.
Previously incorrect CTE/alias execution is not a valid equal-result timing
baseline for the correct DataFusion fallback.
