# Weekly SUM and stream review

Reviewed mainline interval `ce313488017d63fc626731f9731a4bec931fc343` through
`1b91bcc00a0623d5ece37ac654bf6e2a4f045939`, corresponding to the parent
[weekly scope](../specs/weekly-review-20261107.md). This is a scoped source
review and remediation handoff, not an independent approval of changes authored
by this reviewer. There is no PR or new CI result at this stage.

## Confirmed accounting defect

**P2 — `operator/asof/output_plan.rs`,
`OutputPlanBuilder::compact_fixed_source`: the recent source cache is not the
compact prefix's source identity.** Registering another right payload changes
`SideBuilder::recent` before `push_right_source` charges the existing prefix on
promotion. The same mismatch occurs when another source is registered but not
used before `finish`.

Two batches with the same nullable Int64 schema can have different physical
validity buffers. `ColumnWorkspace::new` includes that buffer in its per-row
fixed byte charge, so using the later source undercharges or overcharges all
prior compact matches. Existing mixed-source tests use identical buffer
layouts and do not expose the error. This is an internal workspace-accounting
defect; it does not by itself prove wrong joined values or an observed process
memory limit violation.

The remediation uses the source identity already held by
`positions::CompactBuilder`. It retains one batched charge at promotion or
finish, with no new row loop, allocation, dictionary, or public contract.
Focused regressions compare exact raw and reserved output bytes against the
legacy range accounting for both validity directions, including unmatched rows,
promotion, finalization after unused registration, and reservation release.
The shared test-only RED binary, built with
`cargo test --locked -p calc-flow --features benchmark-internals --lib --no-run`,
ran `fixed_right --nocapture --test-threads=1`: one existing control passed and
both new regressions failed as expected. Promotion produced raw/reserved bytes
`(56, 21580)` instead of legacy `(58, 21588)`; finish after an unused source
registration produced `(40, 21516)` instead of `(42, 21524)`. The build took
430.22 seconds, separate from the performance acquisition budget. Raw command,
exit status, binary hash and failure logs are retained under
`target/weekly-review-20261107/asof-red.{json,log}` in the original repository.
The rebuilt GREEN binary ran
`operator::asof::output_plan::tests:: --nocapture --test-threads=1`:
all 15 tests passed in 0.027 seconds, including both validity directions in
the two new regressions and existing compact/legacy, cancellation, allocation,
reservation-release and tight-pool controls. Evidence is retained in
`target/weekly-review-20261107/asof-green.{json,log}`; the tested binary SHA-256
is `5e95a8a1be19cc42c57a3dc3d445d94421f13efc45fb8d1579e0aa7a1f0393e4`.
The SQL reviewer independently inspected the ASOF source fix and reported no
source-level blocker after these tests. Full CI and performance results remain
the parent's handoff responsibility.

## Latent internal slice issue

`operator/join/matches.rs`, `View::slice` compares a relative slice range with
absolute `RangeMatch::first` offsets. The source-level counterexample is
`selection.view().slice(10..20).slice(0..5)`: the interval branch returns an
empty selection where relative slicing should retain opposite rows 10 through
14. Existing tests only slice these interval views once.

Current production `JoinOutput` callers pass whole `Selection` objects, and
their output projection/gather paths slice those selections once. No current
production path requiring a second slice was found. This is recorded as a
latent private-contract defect, with no production fix or test execution in
this change. Address it with the interval-view proposal below before adding
callers that compose slices.

## Review observations and coverage

The SUM review covers integer-history batching, inline signed/unsigned state,
generic fallback, candidate preparation/finalization, UTF8 output builders,
checkpoint import/export adaptation, variable extrema and global coalescer
accessors. The current single-aggregate ordered-LAST_VALUE admission excludes
mixed LAST_VALUE/SUM, so the apparent double-history-seed concern is not an
admitted execution shape. No concrete SUM semantic defect was established.

The streaming review covers the changed interval match representation,
contiguous native-index proof and invalidation, fill-worker cancellation
quanta, fixed-width output preflight, eviction bitmap preparation/application,
ASOF compact output accounting and checkpoint-disabled prefix commit. It also
covers removal of the row-budget ceiling, checked queue admission arithmetic,
source-capability diagnostics and their public redaction mapping. The changed
branches and directly related callers were inspected; unchanged portions of
the large join and runtime modules were not exhaustively audited. SQL native
batch joins, rolling rewrites, Python APIs and benchmark orchestration belong
to other reviewers.

Existing SUM tests provide exact signed/unsigned NULL and wrapping behavior,
WHERE/FILTER, mixed aggregation, immutable snapshots, restore/continuation,
transaction refusal and budget rollback. These tests were inspected as
coverage evidence and were not rerun without a SUM code change. No full Rust,
Python, connector or coverage matrix was run for this scoped review.

## Remaining performance proposals

1. **Borrow bounded interval views.** Each interval `View::slice` currently
   scans every range and creates an owned Vec; `Matches::view` then clones an
   owned interval view. With R intervals and K output chunks, slice work can
   be O(R × K) even though chunk boundaries increase monotonically. Preserve
   the compact interval storage, but represent a view as borrowed ranges plus
   absolute bounds; locate boundaries by binary search or a monotonic cursor,
   and derive the visible length without rescanning. This also makes nested
   slices relative by construction. Validate empty/nested/cross-run slices,
   order, multiplicity, exact iterator lengths, cancellation and allocation
   ceilings. Measure a fragmented dense fanout case, its single-chunk control
   and a noncontiguous fallback; this is a source-derived hypothesis, not a
   measured speedup.

2. **Cache immutable SUM key-output charges.** `output_charge` walks all
   committed keys and all candidate keys before building each full snapshot.
   For integer SUM, `key_output_charge` only needs four times encoded key
   length. Cache the committed total at installation/restore and accumulate
   the candidate total while candidates are admitted. Preserve today's
   conservative double charge for touched existing keys, checked overflow,
   transaction rollback and cancellation boundaries; leave variable extrema's
   mutable result charges on their current path. This is narrower than the
   rejected candidate/rank and column-dictionary redesigns. Test exact and
   one-byte-short pool admission, new/existing groups, discard, full/delta
   restore and held snapshots. Measure sparse large-state updates plus dense
   1T/4T controls; do not change the full-snapshot API to manufacture a gain.

3. **Investigate filter materialization before another CASE rewrite.** The
   completed [CASE retest](../../../benchmarks/sum-state/sum-case-retest-20261106.md)
   attributes 49.13% of baseline samples to SUM update and 20.02% to transaction
   finishing. Inclusive `filter_record` is 15.42%; those inclusive shares
   overlap and are not elapsed-time percentages. Profile-selected filtering
   and key encoding deserve a new, narrow hypothesis, such as reusing the
   filtered key columns already produced by an upstream expression. Admission
   must preserve filtered-out error behavior, NULL selection, caller-buffer
   immutability, and UDF evaluation counts. Start with one identified duplicate
   materialization and full correctness checks. The prior CASE, ranked-store,
   and column-dictionary experiments did not establish the required dense
   benefit; they should not be reinstated on component allocation evidence.

Each proposal requires its own declared baseline/candidate, at most five
directly related cases, fixed sample count and total acquisition budget.
Use the maintained paired regression rule and retain correctness/resource
gates. Reject confirmed regressions; report inconclusive timing as inconclusive.
These are proposals, not implemented or certified performance improvements.

## Focused ASOF performance coverage

The maintained Python ASOF stream workload at 10,000 rows, 1,024-row batches,
checkpoint-off and one CPU is sufficient as the ordinary fast-path control.
Its sole terminal expression selects `left__sequence` and `right__price`.
`pipeline::stream::projection` pushes this exact column projection into ASOF;
`configure_projection` selects only the fixed-width price column on the right,
and `SideBuilder::register` builds `OutputColumns` from those selected columns.
The chunk feeder waits for each output prefix, and the initial fresh two-owner
state admits compact positions. The unit regressions cover different validity
buffers, which the ordinary benchmark does not generate.

No performance run was initiated by this reviewer. The parent owns the combined
four-case protocol and final measured verdict, so no extra native million-row
case or checkpoint matrix is needed for this accounting-only change.

## Independent review of the SQL remediation

The SQL author requested a separate review of the remediation at `c11d058d`.
The five changed Rust files were read in full, with the template capture/bind
and rolling audit callers inspected. The nested CTE and alias-column guards
correctly reject forms the native executor does not interpret; the equality
helper consolidation retains both operand orientations. Native query metrics
now use an empty local rolling audit while the regular planner retains its
own audit. No additional blocker was found in those three corrections.

**P2 — the initial partition-cache repair was incomplete.** Resetting the
cache when the requested partition count changes misses a fixed request of
four partitions followed by inputs with one and then eight RecordBatches.
The first input produces one physical partition and can seed a reusable
template; the second run then rebinds all eight batches into that template's
single partition. This affects the benchmark-only partition experiment,
not default production results. The finding was sent to the author before
performance acquisition. The actual follow-up RED returned partition count
one instead of four; `partition-layout-red.{json,log}` retains the command,
failure and 249.20-second compile-and-test receipt.

The corrected source at `7dbd229d23fb4060c1209f518ef08bee8c82c508` gates both
finite query template lookup and capture on `input_partitions == 1`, and
removes the redundant setter-level cache replacement. This matches
`QueryTemplate::bind`, whose supported scan layout is always one partition.
Changing input batch counts can no longer reuse a mismatched scan while
configured for multiple input partitions. Default P1 cache reuse is retained;
returning from P4 to P1 can safely reuse the existing immutable P1 templates.
The separate stream-expression cache is unaffected because the experimental
partition setter is reachable only from finite batch plans.

**Independent source verdict: no remaining blocker in the SQL remediation.**
The new guards, call paths and layout regression were reviewed without
rerunning tests. The earlier native join module GREEN is 28/28. The final
partition module is GREEN, 7/7, with 0.06 seconds of test execution and
228.21 seconds including compilation. Scoped
`cargo clippy --locked -p calc-flow --features benchmark-internals --lib -- -D warnings`
passed in 142.12 seconds. Commands, status and logs are retained in
`partition-layout-green.{json,log}` and `native-clippy.{json,log}`. The source
and targeted-check gates for this independent SQL review are approved. The
experimental P4 path now replans instead of incorrectly reusing a P1 scan,
so prior timings for those unequal layouts cannot establish performance
equivalence. The normal P1 performance controls still require the parent's
declared acquisition. Required CI remains outside this local source verdict.

## Performance orchestration review

The four-case draft and its local launcher, controller and worker were
independently reviewed without running a workload. The maintained paired
collector supplies two rounds of ten alternating pairs; each worker supplies
one warmup and three samples. The public Program fixture produces 1,000
Int64 values, split 640/360, and its expected Arrow table contains the complete
`value = x + 1` oracle. Output names, row boundaries and settled job tasks are
checked outside each timed sample. The existing engine adapter supplies the
other three correctness checks.

The launcher owns a new session, tracks PID plus creation time, discovers
reparented members by session, and records final cleanup. Both launcher and
controller are assigned CPU 31; workers are assigned CPU 0. The original
600-second acquisition budget includes preflight, installation, sample
processes, postflight and cleanup, with interruption at 585 seconds. Missing
or corrupt controller summaries cannot produce an accepted handoff.

The controller now binds each release manifest to the protocol's declared
revision, in addition to checking wheel/native hashes, build-source identity
and helper seals before and after collection. These changes resolve the
identified orchestration blockers. The draft's 102 helper seals were checked;
only the already-declared launcher edit differed at that checkpoint. Final
activation must use newly frozen controller/launcher seals and completed
candidate-release and local-test receipts. This is a static orchestration
approval, not a performance result or a declaration that pending builds ran.
