## PR #565 Review: perf: add certified incremental row-window decay

**Author:** wegamekinglc (Cheng Li)
**Branch:** `feature/ta-arroyo-p2-decay` → `main` | **Files:** 30
**Base:** `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`
**Reviewed published head:** `37e7d7d136f5f07da94e6a69d50a60df934cf0da`
**Reviewed published tree:** `f700a4099073c80ac8255a335040a24d606974fb`
**Additional reviewed dirty scope:** `scripts/benchmark_suite/decay_warm.py`
and `benchmarks/test_decay_warm.py` counter guards; test-only style changes in
`kernel/decay.rs` and `tests/decay_tests.rs`. Production helpers remain those
of the published head. Final scoped Clippy and affected-test checks completed
successfully; all four hashes below still match the prior static review.

The reviewed dirty file SHA-256 values, in the order above, are:

```text
19643f270e5d6ceef691a53c67587e1ffd98f59ab79c84fb4fe5fc3d22fa8876
bc3821f55237413d6461ff8d6519a5d49e53c9ba468e800c2b050edbedb58453
da7fc50409d87e79b101c5adabc43e7d85fc717809cb244aab7595d466f5fed2
04a822a71e5abf7b8cbfe9ec85961555aeb35f0b1f375ed5a1f6f37ec20eaddd
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
- Python: **Focused author checks passed; candidate runtime measurement pending.**
  Old-native tests establish harness/lowering behavior only.
- Resolved failure: the work-counter replay test's historical 13-versus-18
  failure is fixed and passed at `37e7d7d`, including 18/guard2/COW8 assertions.
- Resolved check: the scoped library/test Clippy's 13 new-test issues were fixed
  without weakening assertions. Final scoped Clippy and all four directly
  affected tests passed against the unchanged reviewed file hashes.
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

Historical logs above remain historical; their filenames are not verdicts.
The reviewer ran
`git diff --exit-code 8d8be7d9 -- schemas/project-v3.schema.json` and
whitespace diff checks; neither changed source. The new terminal logs verify
the final test-style source separately from the earlier published-product run.
SHA-256 of all four reviewed dirty files was independently rechecked and matches
the recorded values exactly; no additional source review or test rerun was needed.

The handoff reuses the parent's non-blocking CI snapshot rather than polling:
Codacy run `128510602794` is **SUCCESS, 0 issues** for `37e7d7d`, superseding
the earlier seven findings. It does not verify the dirty test-style/harness
delta. Other required CI remains unresolved. No workspace coverage or
cross-platform green claim is made; the combined 90% line gate remains required.

### Blocking Issues

Must resolve before performance acceptance and merge:

1. **Final source/build identity:** commit and freeze the reviewed four-file
   delta together with the evidence/documentation updates before building the
   candidate. Record the resulting actual SHA and recheck Codacy for that head;
   `37e7d7d`'s success cannot be generalized to a future commit. Production,
   harness and test source have no remaining review blocker for this fixed-source
   build. Do not build from a moving/dirty checkout or relabel an older wheel.
2. **Performance/CI acceptance:** no candidate measurement or final required
   CI success exists. After build authorization, execute only the fixed four
   workloads within budget. Actual benefit, complete controls, numerical and
   native-work evidence, current-head required CI/Codacy and final specialist
   review remain merge gates. Typed admission or zero scans do not prove benefit.

The previous missing-field, overflow, replay and helper-review findings are
closed by the reviewed fixes and evidence below. Final test-style verification
is also closed. No source, harness or test correctness/style defect remains
open in this reviewed scope. The remaining entries are provenance and later
performance/merge gates, not requests for additional implementation.

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

### Verdict

**Comment Only — production, harness and test source clear for a frozen-source
candidate-wheel build. Final commit/build provenance, actual performance and
current-head required CI/Codacy remain pending; this is not merge approval.**

This is a source/harness review handoff, not final PR approval. Preserve the
fixed acquisition contract and review the eventual candidate identity/results.
No repeated test, native build or performance run was started by the reviewer.
