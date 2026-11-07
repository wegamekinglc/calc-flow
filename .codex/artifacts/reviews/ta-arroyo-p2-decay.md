## PR #565 Review: perf: add certified incremental row-window decay

**Author:** wegamekinglc (Cheng Li)
**Branch:** `feature/ta-arroyo-p2-decay` → `main` | **Files:** 30
**Base:** `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`
**Reviewed head:** `77db7583ded5c0d017edf0fa8c9b88eb91ed0dc2`
**Reviewed tree:** `ec15ae7199bfac32bc22b1eaf277abf35190cc7b`
**Additional reviewed delta:** root's uncommitted specification paragraph
qualifying the direct-operator replay test. No product change is in that delta.
Future Codacy helper extraction is outside this reviewed snapshot.

### Summary

Float64 Rows Decay gains an optional private S/Q recurrence cache. A conservative
certificate compares its result with the original ordered floating-point fold;
uncertified rows retain that fold. Cache-credit refusal selects Reference for
the individual Decay group without resetting neighboring Mean/M2 state.

The reviewer read the P2 specification, critic/resource follow-up, Decay module
and tests, typed integration, ordered/general routes, codec identity/validation,
original fold, old-source fixtures, manifest generation, optimizer/explain,
private metrics/binding delta, and new four-case adapter/protocol. Review used
the author's isolated worktree and changed only this report. No extra worktree
or remote review was created. The PR remains a draft.

### Build and Test Results

- Rust: **Partially verified; final retry test remains unresolved.** Existing
  author logs establish the results below. The reviewer did not repeat tests,
  compile native code or acquire timings.
- Python: **Focused author checks passed; candidate runtime measurement pending.**
  Old-native tests establish harness/lowering behavior only.
- New failure: the final expanded work-counter test failed with 13 transitions
  versus an expected 18. Its corrected test has not been compiled or run.
- Performance regressions: **Unmeasured.** No P2 timing verdict exists.

| Evidence                                        | Actual result and scope                                                                                    |
|-------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `target/p2-decay-work-green.log`                | 13 passed: 12 P2 tests and one existing diagnostic, before the expanded retry assertions                   |
| `target/p2-decay-kernel-regression.log`         | 25 existing typed-kernel regressions passed                                                                |
| `target/p2-decay-buffered-regression.log`       | One dual-profile, general-buffer, mixed-quota regression passed                                            |
| `target/p2-decay-binding-green-final.log`       | One private work-mapping test passed                                                                       |
| `target/p2-decay-binding-status-regression.log` | One existing full-integer-range/nanosecond/snapshot status test passed                                     |
| `target/p2-decay-final-clippy.log`              | Scoped core/binding library Clippy completed; not a workspace test-target Clippy result                    |
| `target/p2-decay-work-final-green.log`          | **Failed:** 0 passed, 1 failed; 13m20s build, actual 13 versus expected 18. Filename is not a green result |
| Author's Python/generator evidence              | Generator 3 and optimizer 11 passed; later focused explanation and harness checks reported passed          |
| Reviewer contract/whitespace checks             | Baseline schema diff and whitespace diff checks returned success                                           |

Logs reside under the repository root `target/`. The reviewer ran
`git diff --exit-code 8d8be7d9 -- schemas/project-v3.schema.json` and
`git diff --check 8d8be7d9`; neither changed source. Passing evidence from the
parent commit does not prove the corrected final test passes.

The handoff reuses the parent's non-blocking CI snapshot rather than polling:
Codacy run `128506874897` reports action required with seven complexity issues;
other required CI is unresolved. No workspace coverage or cross-platform green
claim is made. The combined 90% line gate remains required.

### Blocking Issues

Must resolve before performance acceptance and merge:

1. **`decay_tests.rs` —
   `p2_work_counts_actual_copy_reference_seed_and_certified_attempts`:** the
   corrected final test has not run. After the coordinated quiet window, run
   only the affected focused check, preserve the failed log, and record the
   actual result. Static validity does not close this verification blocker.
2. **`scripts/benchmark_suite/decay_warm.py` — `valid_counters` / `total_work`:**
   missing native fields are silently filled with zero. For example, removing
   `cache_copied_samples` passes the current validator and can hide COW work.
   Require the exact complete 12-field set for each present `_decay_work`;
   keep an entirely absent baseline witness unavailable. Add a focused metadata
   regression for a missing field before acquisition.
3. **`scripts/benchmark_suite/decay_warm.py` — `total_work`:** the enclosing
   `RollingMetrics.overflowed=true` is ignored. The core contract excludes such
   observations from evidence even if partial counts happen to satisfy row
   classification. Reject overflowed native observations and add a focused
   metadata regression before acquisition.
4. **New Decay helpers/tests — Codacy:** address the seven reported complexity
   failures without exclusions or weaker checks: `OwnedDecay::advance` (10),
   `exact_denominator` (9), `legacy_fallback_reason` (10), `prepare_inputs` (19),
   `history_values` (13),
   `p2_extra_decay_normal_work_and_recovered_candidates_remain_certified` (9),
   and `assert_equivalent_columns` (12). Re-review factoring at the actual new
   head and verify directly affected behavior.
5. **Performance/CI acceptance:** no candidate measurement or final required
   CI/Codacy success exists. Freeze the corrected common harness and candidate,
   then execute only the preregistered four workloads within their budget.
   Actual benefit, complete controls, numerical/resource/work evidence and final
   specialist review remain independent gates. Typed admission or zero scan
   counts do not establish benefit.

No additional runtime correctness defect was found in the reviewed production
snapshot. These are unresolved evidence, harness and quality gates; they do not
justify changing numerical tolerances or resetting Mean state.

### Numerical and Transaction Review

- `Summary::remove` updates Q with the old S before subtracting the departing
  sample. Valid ordinal weights and raw-position expiry preserve right-aligned
  null/NaN semantics. Numeric Mean/M2 update and repair state are unchanged;
  a row advances neighboring groups once.
- `rounding_radius` covers adjacent binary64 spacing, eta and finite MAX.
  Outward radius propagation covers products/additions/subtractions without
  subtracting old error radii. `exact_denominator` uses checked u128 arithmetic,
  even-factor division and the 2^53 bound; other shapes retain the old fold.
- `certified` bounds the recurrence against the original ordered fold, including
  gamma/underflow/division error. One-sign and mixed-sign bounds fail closed
  on non-finite intermediates. The sign guard and quarter-budget comparison
  retain the reviewed 1e-12 D13 pairwise guarantee. No wider tolerance or FMA
  substitution was introduced.
- Empty/all-zero frames retain null/+0 behavior. Infinity, ambiguous zero,
  subnormal and weighted-overflow cases retain the ordered reference fold.
  Scheduled and failed-certificate rebuilds perform real counted scans.

The retry correction was independently checked against `on_watermark`,
`take_buffered`, `emit_rows`, `projected_rows_charge`, and the existing
refusal/replay test. On this **general-buffer, first-chunk rejection** path,
`take_buffered` removes the five ready rows. Only a projected-charge failure
reinserts them; `output.emit` refusal returns without reinserting them. Installed
history, typed state, output sequence and successful input watermark have not
advanced. The charge has removed those buffered rows, so resubmitting them does
not charge twice. Context watermark 8 still admits times 9–13. Resubmitting the
identical batch before watermark 13 therefore constructs a second five-row
attempt; watermark-only repetition performs none.

This establishes the test construction statically, with its retained expected
18 transitions, two guard repairs and eight copied slots. It does not establish
a passing test result or promise a live StreamingRunner retry: a managed send
failure terminates the job and recovery uses the last compatible durable cut.
The reviewed specification paragraph now states that boundary explicitly.

The arithmetic judgment assumes ordinary IEEE round-to-nearest, gradual
underflow and no fast-math execution. It is not a timing result.

### Resource and Work Evidence

- `OwnedDecay` owns a real MemoryReservation and pool Arc. Reservation precedes
  allocation, adjusts to retained capacity and includes owner/Arc/registration
  allowance. Shallow Arc clones share a lease; deep COW candidates reserve
  independently. Held old owners retain credit until their final drop.
- The private 64 MiB pool funds only the optional cache. Logical history/buffer
  charges and public StateBudget meaning remain. Refusal selects Reference per
  Decay group without clearing the typed plan or rebuilding Mean. Reference
  scratch/readouts retain callback lifetime.
- Both ordered and general-buffer production routes pass authoritative history.
  The SQL sorted adapter admits AVG only; it cannot construct an omitted Decay
  route. Candidate state remains private until the original output boundary.
- The new dual-profile buffered test crosses the 64-transition phase with two
  entities and retained owners, forces one cached/one Reference entity, then
  reacquires both caches. It checks Mean bits, logical charges, actual owner
  credit and frozen identity. Its passing log closes the initial coverage gap.
- The 12 private counters use fixed local slots. Preparation and generic-fill
  completion/error paths batch merges into the existing recorder. No per-row
  recorder mutex or callback map allocation was added. Deep COW owners begin
  with fresh pending counts. Attempted work remains visible on ordinary output
  refusal while installed logical state and cache ownership stay transactional.
- Row classes partition transitions. Seed insertion/count/fold, scheduled and
  guard fused folds, quota-reference count/fold, history materialization and
  actual COW copies are distinguished. Counts describe slot visits, not equal
  CPU costs. The binding bridge preserves integers, omits `_decay_work` for
  ordinary zero-work callbacks and adds no durable field.

`try_clone_for_batch` copies the resident queue for each touched entity/group
per batch. Total work includes O(N + sum of those queue lengths + repair and
reference samples). A one-row append can still perform O(W) memory work with
zero guard scans; frequent guard failure can retain O(NW) arithmetic. FR4,
D13, the explanation supplement and protocol disclose these limits. The generic
`amortized_constant` label must be read with the adjacent Decay qualification,
not as unconditional whole-stream complexity.

### Test Coverage

Coverage includes mixed Mean/Decay, both profiles, missing values, magnitude
eviction, finite weighted overflow, infinities, subnormal/signed-zero behavior,
repair, cached/Reference transitions, allocation-owner/COW credit, cancellation
and refusal, buffered multi-entity recovery and future batches after restore.
The expanded failed-attempt/replay assertion remains the unverified delta.

Five pinned-old-source snapshots cover real writer layouts 3 and 4: both
profiles, mixed Mean, cumulative state and an earlier unsupported group. The
generator-patch SHA was independently checked against its header. Identity,
profile, schema and budget rejection remain atomic. No layout-1/2 golden or
new runtime result from the old native is invented.

### Documentation Consistency

`legacy_fallback_reason` preserves old group-before-scan reason priority and
strings; the cumulative classifier still takes priority. Durable identity
normalizes the old General fingerprint while resident identity stays physical.
The writer contains no summary, radius, work count or repair phase. Strict
configuration/profile/schema/layout/inventory validation remains.

Generated manifests and the optimizer admit the shape without changing Project
JSON/fingerprint. Same-old-native comparisons prove that Python export boundary
only. D13 now assigns weighted Decay rtol=atol1e-12; its table pipe positions
align. The clarified replay paragraph agrees with the streaming-guide failure
boundary. The PR title/body follow repository format and accurately retain the
failed/unrun final test.

The protocol keeps four fixed cases, a 600-second inclusive budget, 30 advancing
AB/BA pairs per case, a COW control, an original-fold oracle and native work
witnesses. The two witness validation gaps must close before acquisition. No
speedup, final CI success, coverage floor or merge readiness is claimed.

### Verdict

**Request Changes — no new runtime correctness defect found; final test,
native-witness guards, Codacy and performance/CI gates remain unresolved.**

Finish the focused corrections after the coordinated quiet window, then review
the actual new head. Future helper extraction is not approved by this report.
No repeated test, native build or performance run was started by the reviewer.
