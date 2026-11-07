## PR #565 Review: perf: add certified incremental row-window decay

**Author:** wegamekinglc (Cheng Li)
**Branch:** `feature/ta-arroyo-p2-decay` → `main` | **Files:** 30
**Base:** `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`
**Reviewed published head:** `b9171fbf44aa86c9d799961b658039175085c38d`
**Reviewed published tree:** `4cf524386e662c0f181f3f3e558bb79a3eb13e55`
**Additional reviewed dirty scope:** Rust history-context arity repair;
worker exit receipts/owned close and their tests in `profile_warm_stream`;
receipt enforcement/remaining-phase budget and their tests in the Decay adapter.
The results document and archived evidence were read for provenance boundaries;
this is not the final independent performance verdict.

The reviewed five source/test file SHA-256 values are:

```text
scripts/benchmark_suite/decay_warm.py
2db78fbde4168ae0297f6878e877332929f706ba41025d1ad98ef885553c4546
benchmarks/test_decay_warm.py
7574375699334b9b6336bcb2f47628783799b02a60eb9f1f519c00ef39e39e98
crates/calc-flow/src/operator/rolling/kernel/decay.rs
30010dc67de00b6397c2e838f7e14807a6eac1e56197dcaf184c82a67d35aa60
scripts/profile_warm_stream.py
5fe67d33c6722948e32df10df010ec466fd7a351442ed65bf1c5eeea24c12f34
scripts/test_profile_warm_stream.py
5f7845e256a7dd58da3734a7f224944ba02347d22e6966ea04dbb009653c91ae
```

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

- Rust: **Focused checks passed, including the final test-style delta.** The
  reviewer read terminal author logs without repeating tests, compiling native
  code or acquiring timings. This is not a full-workspace green claim.
- Python: **Focused checks passed; new cancellation coverage is required below.**
  Mock harness checks are not actual OS process evidence from an acquisition.
- Resolved failure: the work-counter replay test's historical 13-versus-18
  failure is fixed and passed at `37e7d7d`, including 18/guard2/COW8 assertions.
- Resolved check: the scoped library/test Clippy's 13 new-test issues were fixed
  without weakening assertions. Final scoped Clippy and all four directly
  affected tests passed against the unchanged reviewed file hashes.
- Performance: the earlier four-case positive acquisition belongs exclusively
  to `b9171fbf`. The repaired final source has no release wheel or timing yet.
  This handoff does not independently approve those earlier timing results or
  transfer them to different native bytes.

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

New handoff evidence, all under the repository root `target/`:

- `p2-decay-codacy-green.log`: **13 passed, 0 failed**, comprising 12 P2
  tests and one existing intentional-worker-panic diagnostic; 3m09s build,
  0.09s execution. This includes the corrected replay and both-profile buffered
  mixed-quota/legacy/numerical checks.
- `p2-counter-guards-red.log`: two focused tests failed because the missing
  field/overflow conditions did not raise. `p2-counter-guards-green.log`:
  **3 passed, 5 deselected**, including the existing unavailable-baseline control.
- `p2-decay-codacy-clippy-tests.log`: **Failed**, 13 new-test style issues;
  no reported production lint. Superseded by the final scoped check below.
- `p2-decay-tests-clippy-green.log`: **Passed**, terminal core library/test
  Clippy, 2m24s. The parent confirmed exit success and the reviewed log ends at
  `Finished dev profile`; no pending process is used as passing evidence.
- `p2-decay-tests-focused-green.log`: **4 passed, 0 failed**, 4m06s build and
  0.01s execution. The final radius-bits, refusal/cancellation replay,
  actual-work 18/guard2/COW8, and dual-profile buffered resource/Mean tests all
  passed. These are the four tests affected by the last style-only extraction.
- `p2-counter-complexity-green.log`: **3 passed, 5 deselected**, 0.45s,
  after the final adapter-only `_phase_work` extraction. The reviewer read the
  terminal log; no new pytest/native/build execution was started for review.
- `p2-history-context-focused-green.log`: **5 passed**, 4m18s build/0.03s run;
  history/reference/quota, buffered mixed profiles, old snapshot continuation
  and actual-work paths. `p2-history-context-clippy-green.log`: terminal core
  library/test Clippy **passed**, 2m28s.
- `p2-complexity-ratchet-red.log`: actual 0→1 `too_many_arguments` allowance
  failure; `p2-complexity-ratchet-green.log` and `-final.log`: ratchet holds.
  No baseline or numerical precision-proof allowance was weakened.
- `p2-worker-exits-final-green.log`: **11 passed, 33 deselected**, 0.46s;
  normal/nonzero/forced close, stdin-error settling, two-worker ordinary-error
  cleanup/receipts and adapter receipt rejection. It does not cover cancellation
  while installing or repeated cancellation during owned pair cleanup.
- `p2-phase-budget-red.log`: 9 failed/1 passed before implementation;
  `p2-phase-budget-green.log`: **10 passed, 11 deselected**, 0.51s.
- `p2-installer-recancel-red.log`: installer Task.cancel lacked terminate/wait,
  **1 failed, 1 passed**. `p2-installer-recancel-green.log`: **2 passed,
  33 deselected**, 0.38s. The second test repeatedly cancels pair cleanup,
  checks both children are reaped before return and propagates the first cancel.
- `p2-final-controller-mock.log`: **3 pure mock tests passed**. These exercise
  normal/deadline/unsettled outcomes, not an actual final-source acquisition.

Historical logs above remain historical; their filenames are not verdicts.
The reviewer ran
`git diff --exit-code 8d8be7d9 -- schemas/project-v3.schema.json` and
whitespace diff checks; neither changed source. The new terminal logs verify
the final test-style source separately from the earlier published-product run.
Closed prior Rust/binding/test-style checks were not repeated. The current
five changed files were read and hashed; the reviewer ran no native build,
pytest or timing during this follow-up.

The handoff reuses the parent's non-blocking CI snapshot rather than polling.
The earlier seven complexity findings and adapter complexity were resolved;
canonical Codacy for `b9171fbf` succeeded. All 30 checks for that published head
are terminal: 28 succeeded, while the unit complexity-ratchet job and its Linux
aggregate gate failed. The genuine cause is the newly added arity allowance,
not a pre-existing failure. The reviewed Rust fix removes that allowance, but
local focused evidence does not prove a new head's required CI green. The
combined 90% workspace line gate remains required.

### Blocking Issues

Must resolve before performance acceptance and merge:

1. **Target `collect-final.py` — final seals before success:** controller
   `child` validates source/builds/helpers before `exec`, but there is no
   post-acquisition source/build/native/helper verification gate before outer
   success. Execute the final seal checks under the same supervised absolute
   phase deadline, before acceptance. Do not move mandatory verification to an
   uncounted manual post-run step; the helper map must include the warm-profile
   module as well as the maintained benchmark identity map.
2. **Target `collect-final.py` — exception cleanup:** `main`'s monitor-error
   handler kills the tree, waits only for the direct collector, and immediately
   scans descendants once. A descendant still exiting can outlive that return
   even while phase time remains. Reuse the ownership monitor or a small loop
   to wait for the known owned tree to become empty within the original absolute
   deadline; do not reset its budget. Unsettled ownership or timeout must remain
   a failure. Keep the original controller error as the primary failure.
3. **Final identity/performance/CI:** after these focused fixes, freeze the
   actual source/helper identities, obtain that head's Codacy/required checks,
   and build its real native. Any authorized final-source acquisition uses
   only the unchanged four cases/30 pairs and remaining 574.654709 seconds as
   a separate evidence set. Do not pool, overwrite or relabel old `b917` data.

The previous missing-field, overflow, replay and helper-review findings are
closed by the reviewed fixes and evidence below. Final test-style verification
is also closed. Installer ownership and repeated-cancellation coverage are now
closed: `Worker.start` catches BaseException around installation, waits for the
existing owned stop helper and preserves the original error through
`_raise_command_error`. The real Task.cancel red/green and repeated-cancel test
verify the missing paths. No additional Rust correctness defect was found;
the remaining target-controller evidence/cleanup findings are separate from
algorithm tuning or a request for more timing cases.

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
18 transitions, two guard repairs and eight copied slots. The published-product
run now verifies those assertions. It does not promise a live StreamingRunner
retry: a managed send failure terminates the job and recovery uses the last
compatible durable cut. The specification states that boundary explicitly.

The seven Codacy helper extractions preserve operation and validation order.
`refresh_latest` moves the original branch chain after the same updates;
`exact_weight_total` retains checked integer operations and the final Float64
conversion stays after the same admission checks. The legacy iterator keeps
group-before-scan and declared-output order. `DecayInputPreparation` retains
the previous allocation owner through each cache preparation, the same stage
drop boundary, scalar-before-columnar history order and pool admission/release
points. The outer wrapper merges attempted work after both successful and
ordinary-error preparation results; numeric-fill error draining is unchanged.

The dirty test-style delta changes the six positive finite exact comparisons
to bit equality, preserving strictness. `decay_job` creates the identical fresh
context per invocation. The replay helper borrows operator/job/store/the pool
Arc; it does not move allocation owners or change pressure/old-owner drop
points. Existing assertions, D13 tolerance and the 18/2/8 counts remain intact.

The arithmetic judgment assumes ordinary IEEE round-to-nearest, gradual
underflow and no fast-math execution. It is not a timing result.

The CI arity repair moves the old free history reader into the existing
`DecayInputPreparation` context. Every former per-call argument maps to that
same invocation's plan/input/histories/node/observer/work field; first row,
input index, window and transition count remain explicit. History absence and
entity-key errors retain priority, followed by scalar then columnar visits into
the same local queue. No new record, allocation, reference replay, credit owner
or stage lifetime is introduced. `prepare_inputs` still merges work after an
ordinary error. The removed `too_many_arguments` allowance is the real CI fix;
the precision-loss proof annotations and numerical contract are unchanged.

The new worker close path records actual wait results and rejects nonzero or
forced termination, including a forced child returning zero. `_fresh_case`
closes both known workers concurrently using gather with exception collection;
its shield/owned helper waits before rethrowing a primary measurement failure.
Receipts are written outside temporary roots and kept out of environment
identity comparison. The added installer stop and cancellation tests now close
the worker lifecycle findings; final controller gates remain as listed above.

The reviewer read the complete target controller and independently checked its
SHA-256 as `ffb2ea0b3dd54a8d3ea5177c3207c3afc8231704859a28aa827164090a35fe4c`.
Its single start precedes child source/build/quiet preflight; the child shares
the collector PID across exec. SIGINT at 504.654709 seconds cancels acquisition;
TERM at total-minus-5 and KILL at total-minus-3 operate on the owned group and
tracked escaped descendants. PID/create-time tracking, polling/reaping the
collector, and rejecting nonzero exits, leftovers, cancellation or elapsed
overrun are appropriate. Raw case success separately requires two normal,
reaped exit-zero worker receipts per case. These correct paths do not replace
the missing post-seal and exception-tree-wait gates.

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

### Style Issues

The seven published Codacy complexity findings were resolved through reviewed
helper extraction. The subsequent 13 library/test Clippy findings were resolved
with strict bit comparisons, declaration placement, the original job fixture
and a borrowing test helper; no tolerance, assertion, pool-owner lifetime or
drop point was weakened. Final scoped Clippy passed. No open style finding
remains in the reviewed files; future-head required checks remain independent.

The latest adapter-only extraction was also reviewed independently. `_phase_work`
first returns None for an absent phase witness, then rejects node overflow,
then applies the unchanged exact-12-field/integer/nonnegative validator. Missing
baseline metrics remain unavailable, and overflow still takes priority over
malformed-present-counter validation. `total_work` retains node/phase traversal,
summation, row-class coverage and its final available/None result. Moving
validation before the local `available=True` assignment changes no observable
error path. Three existing guard/witness tests passed; canonical Codacy for
the eventual new commit remains pending.

### Test Coverage

Coverage includes mixed Mean/Decay, both profiles, missing values, magnitude
eviction, finite weighted overflow, infinities, subnormal/signed-zero behavior,
repair, cached/Reference transitions, allocation-owner/COW credit, cancellation
and refusal, buffered multi-entity recovery and future batches after restore.
The corrected failed-attempt/replay assertion passed with the published product
and again after final style factoring in the four directly affected tests.

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
boundary. The PR title/body follow repository format. Evidence summaries must
distinguish the historical failed/unrun snapshot, the passing published product,
and the now-completed four-test/final-Clippy style verification. Final source
identity and performance results must be added when they actually exist.

The protocol keeps four fixed cases, a 600-second inclusive budget, 30 advancing
AB/BA pairs per case, a COW control, an original-fold oracle and native work
witnesses. `valid_counters` now requires exactly all 12 fields, preserving real
zeros while rejecting absent copy counters. `total_work` rejects an overflowed
node whenever it contributes a witness. Entirely absent old-baseline metrics
still produce None/unavailable rather than zero work. The focused red/green
checks close both earlier evidence blockers. No speedup, final CI success,
coverage floor or merge readiness is claimed.

The initial archived acquisition reports positive medians of about
2.49/7.80/8.26/1.13 for the four cases, bound to `b9171fbf`. The reviewer checked
the archived gzip SHA-256 equals
`d57c682db5d656a5663d3594097b578a0652e3d5db798457722ad2de658569aa`;
this hash check is not the final paired-statistics review. The document correctly
states that the original harness preserved eight native terminal statuses and
awaited closure, but did not persist eight OS PID/exit-code receipts. The new
receipt instrumentation cannot retroactively provide those missing facts.
The first phase consumed 25.345291 seconds; any authorized final-source phase
retains the separate 574.654709-second remainder and original workload contract.

### Verdict

**Request Changes — Rust and worker lifecycle fixes are clear; outer-controller
post-seal and exception-tree-wait gates remain before final collection signoff.
Final-source performance and required CI/Codacy remain merge gates.**

This is a source/harness review handoff, not final PR approval. Preserve the
fixed acquisition contract and review the eventual candidate identity/results.
No repeated test, native build or performance run was started by the reviewer.
