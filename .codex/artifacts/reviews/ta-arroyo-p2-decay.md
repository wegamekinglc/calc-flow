# Branch review: P2 incremental rolling decay

**Branch:** `feature/ta-arroyo-p2-decay` → `main`
**Base:** `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`
**Status:** initial source review; final coverage/diagnostic refinement pending.

## Summary

The implementation admits Float64 Rows decay to the typed kernel and maintains
an optional private S/Q cache. A conservative numerical certificate compares
the recurrence with the old ordered fold. Uncertified rows retain that fold;
cache-credit refusal selects reference execution for the individual Decay group.
Neighboring numeric state and the existing persistent snapshot contract remain.

The reviewer read the P2 specification and critic/resource follow-up, the full
new Decay module and focused tests, typed preparation/readout integration,
ordered/general entry routes, checkpoint identity/validation, original scan,
generator fixtures, and Python manifest/cost-model changes. Review occurs in
the author's isolated worktree; no extra review worktree or remote mutation
was created. Product code is still being refined, so this is not final approval.

## Numerical review

- `Summary::remove` updates Q with the old S before subtracting the departing
  value. Valid ordinal weights and raw-position expiry implement the existing
  right-aligned missing-value semantics. The neighboring Numeric add/remove
  implementation and M2 repair state are unchanged.
- `rounding_radius` uses the largest adjacent spacing with an eta floor and a
  finite MAX case. Radius propagation includes the product, addition and
  subtraction operations and never subtracts historical error radii.
- `exact_denominator` uses checked u128 arithmetic, even-factor division and
  the 2^53 bound. Admitted weights and all old positive denominator partials
  are exact; other shapes use the old conversion/fold behavior.
- `certified` combines the recurrence envelope with the specified old-fold
  gamma/underflow/division envelope. One-sign Aw uses the weighted numerator
  and its radius; mixed-sign Aw uses W times the non-cancelling absolute sum.
  Non-finite intermediates/bounds fail closed.
- The sign guard and outward/lower-rounded quarter-budget comparison implement
  the reviewed candidate-to-reference and candidate-to-candidate proof. No
  wider tolerance or FMA replacement was introduced.
- Exact all-zero frames return the old fold's +0. Infinite, ambiguous-zero,
  subnormal and weighted-overflow cases retain the original ordered fold.
  Scheduled or failed-certificate rebuilds are real scan work, not free repair.

No arithmetic defect was found in this source snapshot. This judgment assumes
the specification's ordinary IEEE round-to-nearest, gradual-underflow and
no-fast-math execution conditions; it is not a timing result.

## Resource and transaction review

- `OwnedDecay` carries a real MemoryReservation and its pool Arc. New storage
  reserves before queue allocation, adjusts to retained capacity, and includes
  owner/Arc/registration allowance. A shallow Arc shares one lease; a deep COW
  clone obtains a distinct lease. Credit lives until the final allocation owner
  drops, independently of the current operator slot.
- The 64 MiB pool is for the optional cache only. The logical history/buffer
  charge code and public StateBudget meaning are unchanged. Cache refusal does
  not consume logical history headroom or turn allowed history growth into a
  new resource error.
- `prepare_inputs` changes only the relevant Decay cache slot. Reference mode
  reads authoritative history plus the canonical batch and owns only callback
  scratch/readout arrays. It does not clear/rebuild the whole typed plan or
  reset Mean/EWMA/cumulative state. Each row advances neighboring groups once.
- Both ordered and general-buffer production entry routes pass retained history
  to the new preparation seam. The SQL physically sorted route cannot construct
  Decay: its DataFusion window adapter admits AVG only, so it is not an omitted
  public Decay entry route.
- Candidate owner allocation and scan readouts remain private until the existing
  successful output boundary. The initial refusal/cancellation tests check old
  state, output sequence and old credit survive while candidate credit retires.

## Checkpoint and surface review

`decay::legacy_fallback_reason` preserves the old group-before-scan order and
old reason strings; the existing cumulative legacy classifier retains priority.
The durable accessor normalizes the old General fingerprint while resident
typed state keeps its physical fingerprint. Writer fields and recurrence seeds
do not contain the new cache, radius, work counters or local reseed phase.
Configuration, profile, schema, layout and inventory checks remain strict.

Five pinned-old-source snapshots cover actual writer layouts 3 and 4, including
both decay profiles, mixed Mean, cumulative state and an earlier unsupported
group. The fixture generator-patch SHA was independently checked against its
header. Tests continue several batches after restore and reject foreign identity,
profile/schema and logical budget failures atomically. These are not invented
layout-1/2 golden fixtures.

Manifest generation and the Python cost-model group-kind change are internal;
the supplied same-old-native export comparison supports unchanged Python
Project JSON/fingerprint only. It is not new Rust runtime verification.

## Required follow-up before final review

1. The initial tests do not cover stable_v2 cache refusal/reacquisition, the
   general-buffer route, or multiple entities with mixed cached/reference mode.
   The author confirmed this gap and is adding one focused dual-profile buffered
   regression across the 64-transition phase, checking Mean bits, counters and
   actual reservation ownership. This closes explicit critic resource criteria;
   it does not require a benchmark or combinatorial matrix.
2. Add the minimal real Rust work diagnostics needed by FR4/performance evidence.
   Initial `cfg(test).scanned` combines reseed work and cannot report actual
   cache-refusal/reference rates in the measured runtime. Certified/reference,
   seed/reseed/scan and copied-sample work must be distinguishable. Observe
   attempted work in batches, including errors, without per-row mutex traffic,
   persistent fields or changes to the Mean fast loop.
3. Preserve and disclose the COW cost. `try_clone_for_batch` copies the entire
   resident queue per touched entity/group/batch. Normal total work includes
   O(N + sum of those queue lengths + repair/reference samples). A one-row
   append can still perform O(W) memory copying even with zero guard scans.
   The spec explicitly permits this transaction cost; it does not permit
   describing the whole stream as O(N). Actual copied samples, including
   reference-history cloning/materialization, belong in the evidence.
4. Finish final formatting/lint evidence, align the changed D13 table, and
   re-review the frozen delta. Performance acquisition and CI/Codacy remain
   separate required merge gates.

## Verification evidence and limits

The author reports the initial ten new tests, 25 typed regressions, three
generator tests and eleven Python optimizer tests passed. Final product Clippy
and the new buffered/diagnostic tests were pending at this snapshot. The reviewer
did not repeat those tests, build native code or run performance measurements.
No speedup, final CI success, coverage floor or merge readiness is claimed.

## Verdict

**Comment Only — no source correctness blocker found in the initial snapshot;
the explicit coverage/work-evidence follow-ups and final freeze remain.**
Do not merge until actual performance improvement, the relevant tests/checks,
CI/Codacy and final specialist review all satisfy the user's gate.
