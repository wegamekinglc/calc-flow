# Stream Join J2a dictionary and batch admission

## Scope

The user authorizes completing issue #363's original J2a native-probe design
in an isolated worktree. Baseline: `9db4cd2563ad0ef26e52fb173cb29acf1c504fc2`.
Preserve the existing funded payload chunks, resource lifecycle, public API,
FlatV1 logical charges, exact V1 checkpoint bytes, and managed recovery.
This change does not implement J2b, J3, or ASOF changes.

## Required behavior

1. Replace the global key/time/row BTreeMap with `hashbrown::HashTable<u32>`
   dictionary IDs and per-key lists sorted by `(event_time, physical_row_id)`.
   Legal out-of-order insertion, duplicate keys, hash collisions, inclusive
   bounds, dense-state swap removal, and empty state preserve exact results.
   Prefix expiration must avoid quadratic run movement; resident-capacity
   accounting must not scan all dictionary slots for each removed row.
2. Intern canonical V1 keys once per distinct key in a native probe batch;
   repeated keys must not allocate a fresh FramedKey/encoding per row. Probe
   keys through borrowed typed values or batch scratch, keeping collision
   equality exact. Retention and dirty tracking share the interned owner.
3. Locate each window with two `partition_point` boundaries. Count matches by
   boundary difference before materialization; do not enumerate every match
   merely to count it. Preserve incoming-position then opposite-time/row-ID
   emission order and existing limit/error behavior.
4. Compute null-event-time, null-key, late, and retain masks in batches. Keep
   checked timestamp conversion, null-time/null-key/late precedence, physical
   row-ID gaps, counters, maximum lateness, cancellation/deadline checks, and
   accepted-prefix/commit-after-output semantics. A batch mask must drive both
   owned-copy and generic admission rather than hiding classify_row in a loop.
5. Prepay real requested dictionary/run/key/mask capacities and construction
   peaks before allocating, retain ownership through asynchronous release,
   and refund exactly. Do not increase public limits, disable accounting, or
   weaken allocation tests. Existing conservative fees may remain only with
   evidence that they fund the replacement representation.
6. Eligible, sufficiently funded native probes execute zero SQL equality
   queries. Preserve approved unsupported-key and denied-credit compatibility
   fallback and normal error propagation. Test fallback separately; do not
   claim standard workloads use native without path evidence.

## Verification

Start each behavior change with an observed focused RED, then GREEN. Add
repeated-key allocation/interning and window-count complexity evidence, plus
mask classification/counter/overflow controls. Reuse existing native type,
exact ordering/property, frozen V1 restore/continue, allocation-funding,
denied-credit fallback, cancellation, and checkpoint tests. Run scoped lint,
format, changed-function complexity, whitespace and generated-contract checks.
Final independent specialist review is required.

## Focused performance plan

Use maintained engine adapters, sealed baseline/candidate release builds and
unchanged output oracles. Candidate is the final tested tree and must be
identified before execution. Each case uses two rounds of ten adjacent AB/BA
pairs (20 samples per side), with the maintained paired-median regression
policy. Record machine, dependencies, thread identities, paths and native
hashes. Total measurement budget is 600 seconds, including fixture creation,
warmup, correctness, startup and cleanup; estimate builds separately.

| Case                               | Connection to change                                    |
| ---------------------------------- | ------------------------------------------------------- |
| Lookup Join, 1M rows, batch 64000  | Primary J2 end-to-end path and repeated 64-key workload |
| Lookup Join, 100k rows, batch 1024 | Batch mask and small-batch overhead                     |
| Interval Join, 100k rows           | Retained per-key time runs and inclusive windows        |
| Projection, 1M rows                | Unchanged runtime/harness control                       |

Report observed timings separately from the original J2 60/600 ms planning
targets. Only comparable environments support an absolute target verdict.
Stop at the budget or an unsuitable environment; do not expand cases, retry
for favorable timings, or substitute the old small string-materialization
microbenchmark for end-to-end evidence. Focused instrumentation tests prove
zero SQL for eligible, funded probes; release timings alone do not establish
which probe path executed.

## Recorded evidence

- Baseline RED: `test_native_probe_interns_each_distinct_repeated_key_once`
  encodes six keys instead of two distinct keys;
  `test_first_retained_batch_interns_keys_and_shares_dirty_owners` encodes
  four keys instead of one. Logs are under `target/j2a/testlogs/`.
- Baseline admission controls: four `test_batch_admission` tests and
  `test_owned_and_generic_batch_masks_keep_late_and_retention_boundaries`
  pass before replacing classification. These preserve behavior rather than
  establish the requested batch-mask implementation.
- Intermediate RED with the new dictionary: count visits 200 entries rather
  than the 100 materialized matches, and admission records zero mask blocks
  rather than two. First-retention interning and collision/ordering/reused-ID
  controls already pass at this stage.
- Interner allocation RED: a single UInt8 key requests a 340-byte peak with
  only 264 bytes prepaid by the old per-row estimate. Fixed interner controls
  require independent prepaid overhead; peak/live allocation tests retain
  their original inequalities.
- Hot-run movement RED records 916,736 moved entries against a linear bound
  of 1,536; a head cursor and geometric in-place compaction replace repeated
  prefix removal. The 61-key admission control records the oversized Quantum
  step assertion before fixed per-row metadata accounting.
- One parallel 185-test Join run aborts after bounded native-process
  infrastructure is exhausted. Subsequent owned native-process checks run
  serially. The exact owned executable path has no orphan workers after the
  abort. Empty append must return without requesting construction peaks.
- Dictionary owner lifetime RED: after the first batch expires, the pool
  retains 3,364 bytes instead of 2,340, pinning 1,024 bytes of obsolete key
  credit. Refreshing the dictionary owner from a surviving row must also
  handle legal out-of-order batches.
- Serial Join module GREEN: 186 passed, zero failed or ignored, in 5.02
  seconds. This includes both dictionary-owner arrival orders, frozen V1
  restore/continue, exact allocation funding, collision/order controls and
  both admission paths. Changed Rust functions remain at CCN 8 or below.
- Join property suite GREEN: four passed, zero failed or ignored, in 0.82
  seconds. Package-scoped Clippy with warnings denied, workspace formatting,
  whitespace, spec references and unchanged generated-contract checks pass.
  The final three lint-only adjustments preserve the tested behavior.
- Independent specialist source review: approved with no unresolved
  blockers. Performance and required CI/coverage acceptance remain separate.
- Baseline sealed release: clean `9db4cd2563ad0ef26e52fb173cb29acf1c504fc2`,
  native SHA-256
  `8a1d4f9caf96b26a7bd9bf5878d13a9498b1ed0c61841c305549dbbf7dc59840`.
  Build took 12 minutes 46 seconds with Rust 1.88.0, CPython 3.13 and four
  Cargo build jobs.
- Sealed measurements and CPU profiles are complete in the
  [measurement report](../analysis/stream-join-j2a-dictionary-measurements.md)
  and its linked lossless evidence package. The user explicitly expanded the
  measurement budget to 60 minutes, with builds separate, to add ASOF,
  compaction/steady, capture and restore. The four initial cases were retained
  without resampling; all nine cases have two rounds of ten adjacent AB/BA
  pairs. Actual measurement/profile supervision totals 1,957.469 seconds;
  including the 90-second reserve totals 2,047.469 seconds, within that budget.
- The maintained rule confirms Lookup regressions of +19.85% at 1M rows and
  +37.81% at 100k rows. Steady improves by 18.54%; retained Join, projection,
  ASOF, compaction, capture and restore remain inconclusive. These comparisons
  do not establish historical A4/J1.6 gains or the original compaction
  single-handler tail gate. Required performance acceptance remains blocked,
  so PR #393 remains draft and must not merge in this state.
- Current CPU evidence prioritizes generic admission/yield scheduling and
  remaining row slices/output materialization. Native probe sample shares
  decrease, while baseline checkpoint profiles identify row IPC/checksum and
  restore parsing work separately. Shared-host interference, retained lost
  events, process-median timer scope and unmeasured allocations are explicit
  limitations in the report; CPU shares are not causal wall-time fractions.
- Independent raw evidence review passes: fixed samples, output/probe checks,
  source and binary seals, maintained statistics, resource scope, budget and
  owned-process cleanup. Production source stayed frozen throughout this
  measurement-only phase; earlier source tests were not rerun.

## PR #394 selective integration follow-up

The review and integration use main
`25435770d41d3ab8e5cbc1367ed8c791400ed6fa` with its V2 checkpoint reader.
Keep this work item's dictionary, borrowed interner, capacity/peak funding,
key-owner refresh, head-cursor expiry, admission masks and physical-row error
order. Add parent-backed generic rows and one gather plan per output side.
Use UInt64 offsets and independently compact flat types; nested Dictionary,
view/run-end and mixed-parent outputs retain the existing fallback. Owned-copy
construction refusal keeps its original Legacy representation.

Focused RED observes parent column length one instead of three, then 36 output
column views instead of zero. Final GREEN: 203 serial Join tests and four
property tests. Scoped Clippy, formatting, whitespace and generated-contract
checks pass; actual changed-function maximum CCN is eight. Independent source
review approves local candidate `eff2fcad17b8922de6aa83955cc14ef602611a0b`
and source-equivalent published commit
`bfbe6fbf68e8715e5b0e5c800396fc80e24dfd6e`, both at tree
`ba005b744e10c3bf183aa64f646390f6b07fc1da`.

The fixed four-case measurement uses two rounds of ten adjacent AB/BA pairs
per case and a separate 600-second budget. All reference checks pass; three
Join cases improve under the maintained paired rule: Lookup 1M -72.51%,
Lookup 100k/batch 1024 -32.68%, retained interval Join 100k -62.97%.
Projection is inconclusive. Supervised elapsed time is 143.774 seconds,
including setup and cleanup; the final owned process group is empty.
Independent raw evidence review passes. Builds are separate, including a
rejected shared-cache candidate and its correctly rebuilt replacement.

The earlier confirmed regressions above describe the previous frozen
candidate. The new comparison measures cumulative #393 against current main;
it does not isolate individual changes or compare PR #394's head. It does
not establish the complete six-case acceptance aggregate, absolute J2 targets,
projection equivalence or this integration's checkpoint performance. Required
CI and coverage remain separate merge gates. The original evidence is retained
unchanged; see the [integration review and sealed results](../analysis/stream-join-pr394-integration.md)
for findings, source/binary seals, resource scope, cache rejection and limits.
