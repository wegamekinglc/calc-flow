# P2 incremental rolling decay — critic critique

Date: 2026-11-08. Source baseline: `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`.
This report includes the second-pass review of the revised specification and
its Implementation design section, beginning at line 285. Production code was
not changed for this review.

## Target

- Spec: `.codex/artifacts/specs/ta-arroyo-p2-decay.md`.
- No public API note is needed for this internal optimization.
- Scope: numerical certification, pairwise recovery, legacy identity and
  resource/restart requirements before test-first implementation.

## Verdict

**Proceed with caveats.** The revised concrete design closes the first-pass
numerical and identity blockers sufficiently to begin test-first implementation.
This is approval of the specified guarded algorithm, not of an unguarded S/Q
recurrence, a future implementation, or a merge. Required implementation tests,
resource accounting (including the cache-only follow-up below), real measured
improvement, final review and CI remain gates.

## Findings

### Blocking issues

None remaining in the reviewed design. The proof depends on implementing the
operation order, outward rounding, admission checks and fallback exactly as
specified; replacing these with heuristic thresholds reopens the numerical gate.

### Closed first-pass findings

1. **Normative tolerance is explicit.** The specification now conservatively
   assigns weighted decay `rtol=atol=1e-12`, requires the corresponding D13
   update, and retains stricter assertions. The generic finance helper at
   `python/tests/test_symbolic_finance_reference.py:70` is no longer treated as
   permission for a `1e-10` normative tier. The D13 edit remains a P2 deliverable.
2. **The two-candidate error gap has a concrete budget.** Every finite
   certified result is bounded against the same old ordered readout r by
   `T(r)/4`, using exact internal constants `a=t=2^-40`, both smaller than
   `1e-12`. A fallback has zero distance from r. Thus two candidates differ by
   at most `T(r)/2`, while their internal additive D13 threshold is at least
   `T(r)*(1-t/4)`, which is larger. Outward/lower rounding of the acceptance
   inequality makes this sufficient even when reconstruction resets the local
   reseed phase. The earlier counterexample—two results each within T of r
   but almost 2T apart—does not pass this quarter-budget design.
3. **The certificate now covers the old operation sequence.** The design
   accounts for update error, weighted products, the old ordered numerator,
   the old sequential denominator, final division, underflow and classification.
   Its normal-path usefulness is not restricted to exact-integer samples.
4. **Legacy identity precedence is specified.** The new classifier first
   preserves `cumulative::legacy_fallback_reason`, then reproduces the old
   group-before-scan validation and exact first reason. It normalizes the
   durable fingerprint through the existing general-kernel identity mechanism,
   rather than accepting arbitrary additional hashes.
5. **Transactional and attempted-work counters are distinguished.** FR5 now
   protects committed state/counters while retaining diagnostic attempted-work
   accounting. This matches `kernel.rs:1532–1573` and
   `kernel/mean.rs:92–106`, which observe work before propagating errors.

### Numerical argument checked

- **S/Q invariant and radii:** expire Q using the old S before subtracting the
  outgoing value from S; append with its new valid ordinal. These identities
  hold over real arithmetic, including missing raw positions. The specified
  upper-rounded radius propagation encloses each stored operation's error;
  departing values do not subtract old error radii. The rho formula uses the
  largest adjacent spacing, with an eta floor for zero/subnormals and an
  explicit largest-finite case. Non-finite stored results/radii fail closed.
- **Weights and denominator:** checked integer admission with total D at most
  `2^53` makes the old weight conversions and every positive partial weight
  sum exact. Since D is at least `n*(n+1)/2`, an admitted n also makes `n+1`
  and `2*n` exactly representable for the displayed error formulas. A shape
  outside this proof executes the old loop, including its rounded denominator.
- **Reference envelope:** the old product-plus-add path has at most `n+1`
  rounding factors. The specified gamma bound and at most `2n` additive eta
  residues conservatively enclose its numerator error. One-sign Aw derives
  from the weighted result and radius; mixed-sign `W*A` avoids cancellation.
  The finite Bsum guard bounds each prefix as well as the final magnitude,
  supporting a first-overflow exclusion argument rather than assuming a finite
  reference result. Its conservatism can cause fallback and is disclosed.
- **Division and classification:** exact positive D and the outward-rounded
  quotient envelope cover normal/subnormal division. Finite checks and
  `B < abs(y)` prevent certification of a mismatched nonzero sign or a zero
  reference. Ambiguous zeros, infinities and uncertifiable intermediate
  overflow use the exact old fold. The all-valid-values-zero special case
  preserves the old fold's +0 result under the stated rounding mode.
- **Pairwise/restart:** the quarter-budget argument above applies to different
  certified accumulation/reseed histories around the same canonical window.
  It permits ephemeral error/radius state rebuilt from history without adding
  a lifetime phase field to layout-3 stable_v1. It does not excuse missing rows,
  altered row order, changed validity or lost neighboring recurrence state.

### Significant implementation caveats

- **A diagnostic estimate is not a resource reservation.** The current
  `kernel.rs:683–691` estimate is exposed by a getter and metadata; it does not
  itself enforce `StateBudget`. Actual retained charges are checked through
  `rolling/budget.rs` and the owning operator's budget paths. Implement the
  specified queue/summary/COW/scratch accounting in real admission/refusal
  paths, not only by increasing `estimated_state_bytes_per_entity`. Check the
  shared enum size and avoid changing unrelated plans' charges. Focused budget
  refusal and failed-restore tests must prove this before merge.
- **Freeze the old classifier rather than deriving it from newly broadened
  logic.** Preserve reason strings, validation order and cumulative priority.
  `state_codec.rs:893–940` must continue validating configuration, schema,
  numerical profile and current-or-declared-legacy layout. The specified
  pinned-old-source fixtures are necessary: a new-code round trip alone cannot
  prove backward compatibility, especially for multiple fallback causes.
- **Match the proof at helper boundaries.** An upper operation that overflows
  must stay invalid; a lower bound must not accidentally become an upper one.
  An optimization for a known exact zero must not mean “the rounded result was
  zero.” Include power-of-two spacing boundaries, min-normal/subnormal values,
  MAX, signed zeros and invalid/stale summaries in the focused helper tests.
- **Repair belongs to owned candidate state.** Readout fallback must be able
  to reseed/cache the row result without mutating installed or caller-owned
  state. Keep min_periods as a readout gate while queues and counts still
  advance. Test output refusal/cancellation followed by retry across a repair.

### Resource-gate follow-up: cache-only credit and per-Decay fallback

**Proceed only with the cache-only revision agreed after this review.** The
proposal to reserve history/buffer logical bytes and optional cache bytes in
one operator-private pool capped by `StateBudget.max_bytes` is not approved.
`RollingOperator::set_state_budget` documents a logical history/buffer limit
and explicitly excludes checkpoint copies and process resident memory
(`rolling.rs:1284–1305`). Old owners can keep a cache reservation alive after
it has been removed from the current state. Consequently “drop the current
cache, then retry logical growth” does not guarantee room: this scheme can
turn a formerly accepted history growth or restore into a new pool error.
Charging multiple current/candidate/checkpoint history copies against the same
logical maximum would also change its existing meaning.

Use a real `MemoryPool`/`MemoryConsumer`/`MemoryReservation` domain for the
**additional optional Decay cache only**, with a finite documented internal
cap. Reuse an appropriate existing runtime pool if one is actually available;
the current native RollingOperator has no DataFusion runtime/pool field, so a
small operator-owned `GreedyMemoryPool` is a viable local implementation.
A cap may use an existing limit as policy input, but must not subtract cache
credit from logical history headroom or imply that `StateBudget` now caps RSS.
Keep all existing history/buffer budget checks and checkpoint-copy treatment.
No new public budget framework is required.

The implementation must meet these ownership and fallback conditions:

1. Reserve before allocating a new Decay queue, summary or cache-owned
   container. Account for actual retained capacities and control/container
   bytes, including a transactional candidate's allocations. Attach the RAII
   reservation to the allocation owner, not merely the current operator slot
   or a temporary preparation scope. Retained owners keep their credit until
   the owned storage actually drops.
2. A shallow Arc clone can share one reservation because it shares the same
   allocation. A deep queue/COW copy needs its own successful reservation
   before allocation. The ordinary serial stream has a resident state and a
   touched candidate (`kernel.rs:1432–1488`); cold reconstruction/seed also
   makes temporary states. “Two copies” is a useful ordinary-case estimate,
   not an upper bound for arbitrarily retained old owners. The pool must
   aggregate every still-live allocation and refuse a further optional copy
   when credit is unavailable. Never reset its usage while old leases live.
3. On refusal, preserve the typed plan and every active Numeric/Pair/EWMA/
   cumulative state. Use a per-Decay reference mode from authoritative history
   plus the current canonical batch, feeding the same kernel a reference
   value/valid-count input or equivalent. Do not return whole-plan None:
   `TypedStreamTransition::replace(None, ...)` clears the resident typed state,
   changing Mean/reconstruction/rebase behavior. Do not mutate the frozen
   fingerprint or numerical profile to select resource fallback.
4. Reference mode must not allocate an uncredited persistent replacement Decay
   queue or precomputed output cache. Its transient reference readout buffers
   remain part of the existing output/work accounting. If credit later becomes
   available, reconstruct only Decay's optional summary; do not reconstruct or
   rephase neighboring accumulators. Each row advances those groups once.
5. Cover both typed preparation routes: ordered execution before
   `ensure_ordered_kernel_state` allocates/reconstructs, and the general-buffer
   `build_typed_stream_output`/`typed_stream_prior` reconstruction path. Restore
   may keep the cache absent, then attempt optional admission at first use.
   Installed history, output sequence and transactional state remain unchanged
   on preparation/emission failure; cancellation drops uncommitted owners and
   preserves any legitimately retained old owners.
6. Checkpoint serialization currently reads logical histories and buffer rows,
   not the typed cache (`rolling.rs:1628–1658`). Do not charge their copies to
   this new cache-only pool or add cache fields to the snapshot. If any owned
   operation retains an actual cache allocation, its allocation lease follows
   that owner. Existing checkpoint/restore validation stays unchanged.

The current `entity_work::try_reserve` is not a substitute for this domain:
its slot limits numeric pair launches and records scratch bytes, but it is not
persistent cache byte admission (`entity_work.rs:301–344`). Likewise increasing
the kernel estimate does not own credit.

Focused implementation evidence must demonstrate: insufficient optional credit
still permits a previously valid logical-history query; old retained cache
owners prevent only new cache allocation; deep copies have distinct credit;
failed/cancelled candidates release only their own allocations; repeated
cache/reference/recovery transitions preserve neighboring Mean state and
both profiles; and final owner drop returns pool usage to its expected value.
A constant “resident plus candidate” assertion alone does not prove these
properties. No build or test was run for this resource follow-up.

### Minor notes

- The earlier Open design decisions/status text still describes gates that
  this section now resolves. Reconcile it when preparing the implementation
  handoff so there is one authoritative current status.
- No new public profile, unsafe arithmetic, persistence field or general
  interval framework is needed by this design. Keep the proof in the artifact
  and leave short implementation comments referring to it.

## Axis check

- **Correctness:** no remaining design blocker; certification and exact
  fallback preserve the required weight/validity/classification contract.
- **Hidden assumptions:** the explicit IEEE round-to-nearest, gradual-underflow,
  no-fast-math, exact-denominator and canonical-order assumptions are load-bearing.
  Shapes not covered by the numerical proof retain the existing reference path.
- **Missing edge cases:** the required cases cover null/current-invalid rows,
  magnitude eviction, weighted overflow, signed/subnormal zeros, segmentation,
  repair, refusal/retry and recovery. Add the helper-boundary checks above.
- **Backward compatibility:** the explicit legacy classifier and ephemeral
  reconstruction are viable; actual old-source fixtures and atomic failures
  remain required implementation evidence.
- **Performance:** scheduled W-scale rebuilds give linear normal-path total
  work, while repeated guard failures may retain O(NW). Short diagnostics are
  evidence of usefulness only; guard cost and actual speedup remain unmeasured.
- **Surface and ergonomics:** OK; no public API change is proposed.
- **Test plan:** concrete enough for red/green implementation; no additional
  broad matrix or standalone measurement framework is required.
- **Risk and scope:** one P2 change can cover recurrence plus typed mixed-output
  admission. Numerical/resource/checkpoint deviations must return to review
  rather than weakening the certificate or reducing the goal to a narrow subset.

## Verification and handoff

Read the revised spec and concrete design, the short diagnostic at
`target/p2_decay_certificate_diagnostic.py`, the old scan, typed bootstrap,
legacy cumulative identity and actual budget/estimate code. The diagnostic's
reported certification counts were not treated as proof; no performance or
Cargo command ran. The first pass ran only a short binary64 demonstration of
the former pairwise-gap issue. Document whitespace and cited-path checks pass.

Proceed to focused failing tests and implementation of this guarded design.
After implementation, independently verify its arithmetic, actual resource
accounting and old-source recovery evidence, then run the preregistered
performance comparison. This critique does not establish implementation
correctness, speedup, CI success or merge readiness.
