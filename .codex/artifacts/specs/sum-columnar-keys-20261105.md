# Columnar composite grouping-key study

The user authorized continuation from the latest main, with the established
test-first, independent-review, fixed-measurement and green-CI merge process.
This work starts at `22f4b9a562bdd087a75d810aeeb2161f29a3d6b7`.
Preserve unrelated stock-comparison work in the original dirty checkout.

## Hypothesis and scope

For two plain UTF-8 column keys and exactly one admitted trusted built-in
Int64/UInt64 SUM, investigate DataFusion 55's columnar GroupValues implementation.
Generate local group IDs from columns, then canonical persistent keys for
distinct groups. Retain per-row hashing; reduce composite-key serialization
and copying. The existing primitive-key path and all fallback shapes remain
available. Do not combine candidate-map or expression-filter optimizations.

This is a design-gated prototype, not an approved guard relaxation. First
establish whether chunk-local or transaction-wide grouping can preserve admission
and materially reduce encoding on the maintained real workload. State the
tradeoff before production edits.

The accepted initial design uses transaction-wide scalar
`GroupValuesColumn<true>`, never the factory's vectorized collision path. After
each original chunk, canonicalize newly assigned IDs from their first source
rows and reuse existing admission in original order before the next chunk.
Keep the old primitive-key guard intact. Drop the temporary dictionary before
history merge and output construction.

Admission requires the exact default unbounded engine pool, default compact
byte allowance, configured group cap G, no SQL WHERE/eager input checks, and
N >= 2G (checked as G <= N/2). The default byte allowance is a finite u64
sentinel, not Option::None. Unsupported shapes, pools and budgets remain on the
old path before dictionary work. The half-encoding floor is algorithmic, not
chosen from timing results. Seed, sparse and all-distinct controls decline it.

For R <= 8192 and N selected input rows, fund dictionary cardinality
K = min(N, G+R), covering same-chunk rejection overshoot. Check per-column UTF-8
spans against i32::MAX and the full growth/rehash/take/encoding overlap against
isize::MAX before construction. Header census checks cancellation per record
and yields at least every 128 records. Record exact constants and source-based
capacity derivation in the review; a size() read after work is only a check.

## Required invariants

- Preserve canonical key equality, NULL versus empty strings, Unicode and
  delimiter bytes, first-appearance order, WHERE versus FILTER group creation,
  integer wrapping/null behavior, and exactly-once history merging.
- Preserve actual group/byte limits and source error ordering. Hints cannot
  replace actual key or budget validation. Never simply remove the current
  explicit-grouping/compact-budget gate.
- Prepay complete transient capacities before allocations, including native
  column builders, hash/collision buffers, indices, emitted arrays, canonical
  keys and simultaneous coordinator state. A post-allocation size check alone
  is insufficient. Prove full owned peaks as well as retained allocations.
- Optional acceleration yields to the existing path before work when its
  envelope cannot be funded. Keep immutable caller inputs, bounded cancellation,
  output refusal/discard atomicity and reservation cleanup.
- Preserve public APIs, project schema, checkpoint policy and wire encoding.
  Native local IDs are ephemeral and never replace persisted identities.

Observe meaningful focused RED before implementation. Require encoded-row/byte
and total-allocation evidence, whole-container capacity/credit controls, exact
single-SUM semantic and recovery tests, and scoped formatting/strict lint.
Full workspace, connector and cross-platform regression belongs to CI.

## Frozen measurement requirements

Prepare five representative cases: dense one CPU, dense four CPUs, sparse one
CPU, all-distinct-within-batch keys on one CPU, and first eligible N=2G on one
CPU. The fourth is a regression
control for local grouping overhead; use the smallest relevant input and the
maintained schema, oracle and harness where possible. Resolve fixture details
before execution and do not create a general measurement framework.

The fifth is an unfavorable admission-edge control, added before any measurement
because the new duplication selector can admit a less favorable ratio than the
dense target. Below-edge decline and cancellation are focused unit controls;
near-below latency remains unmeasured. Do not add another threshold or tune this
one after timings. Derived controls keep 10k zero seeds and twenty business
batches: one full distinct 10k-key cycle per batch, or two cycles for N=2G.

Use two rounds of ten alternating pairs per case: 200 formal observations and
twenty warmups. Both dense-one-CPU profiles run six jobs with the first excluded;
five valid processing windows and exact clock/loss checks are required per arm.
One 600-second acquisition includes preparation, startup, warmup, independent
complete-snapshot correctness, resources, capture/decoding, analysis, publication,
post-seals and cleanup. Estimate and record builds separately.

Freeze exact public source/build/helper/protocol identities before acquisition.
Use current-main baseline source; neither the old P74 binary nor unused time from
its campaign is a substitute. Stop on unsuitable conditions, failed evidence
or budget exhaustion. Do not add cases, samples or retries for a favorable result.

Runtime acceptance requires all evidence complete, at least one dense target's
exact maintained improved verdict and no confirmed regression on any selected
case. Inconclusive is uncertainty, not equivalence or a speedup. Reject an
unproven/regressing prototype and retain evidence honestly. Final specialist
review and green required checks on the final head precede an authorized merge.
