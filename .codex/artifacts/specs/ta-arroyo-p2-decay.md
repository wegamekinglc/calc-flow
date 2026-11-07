# P2 incremental rolling decay — specification

Date: 2026-11-08. Status: numerical/resource design reviewed; implementation
and focused verification in progress. Real performance, final review and CI
remain required before merge.

## Source and scope

The user authorized the TA-Lib/Arroyo development plan, one PR per implementation
step, autonomous CI/Codacy remediation, and merging only with real performance
improvement and passing review/checks. This artifact specifies plan P2, not the
separate P1 fusion or P3 shared-history changes.

Source baseline: Calc Flow `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`.
Rebase and revalidate affected requirements against latest main before implementation.
TA-Lib comparison: `58160778ecaab21e90eafd6e5462f6e833100af9`,
`ta_codegen/input/wma/wma.c:146–202`. Its periodic rebuild is evidence of a
technique, not a proof for Calc Flow inputs, missing values, or recovery.

Normative vocabulary and boundaries come from
[introduction](../../../docs/introduction.md),
[symbolic API](../../../docs/symbolic-api.md),
[Rust rolling reference](../../../docs/rust-api.md), and
[floating equivalence D13](symbolic-computation-contract.md#15-d13--floating-point-equivalence).
Apply [AGENTS verification](../../../AGENTS.md#verification) and `code-style`.

P2 is an internal Rust table-runtime optimization. It covers Float64 inputs,
Rows frames, both existing numerical profiles, batch and owned continuous
execution, including a stage containing decay and otherwise supported numeric
outputs. It adds no public API, provider, engine, configuration, or numerical
profile. Duration frames and other numeric input types retain their current
implementation; their presence must not be misreported as newly optimized.

## Problem and current evidence

Every decay output currently collects valid frame samples and folds weighted
products for every row:
[rolling.rs:4166](../../../crates/calc-flow/src/operator/rolling.rs:4166).
This creates per-output-window work and allocation proportional to the retained
frame. Typed admission rejects decay:
[kernel.rs:1609](../../../crates/calc-flow/src/operator/rolling/kernel.rs:1609).
Consequently decay also prevents a shared stage from using the typed path for
its otherwise admitted numeric outputs.

Current scan semantics are independent of `stable_v1` versus `stable_v2`.
The profiles affect existing numeric/pair transitions and must continue to do
so; optimizing decay must not change a neighboring mean/variance algorithm.

| Concern                     | Authoritative baseline evidence                                                        |
|-----------------------------|----------------------------------------------------------------------------------------|
| Missing/non-finite samples  | `rolling.rs:3058`: null/NaN excluded; infinities valid                                 |
| Window membership           | `rolling.rs:3894`: last W original row positions, including invalid positions          |
| Weight and fold order       | `rolling.rs:4222`: valid samples oldest first; weights W-n+1 through W                 |
| Python regression tolerance | `python/tests/test_symbolic_finance_reference.py:70`: rel 1e-10, abs 1e-12             |
| Checkpoint identity         | `kernel.rs:694`, `kernel.rs:2076`, `state_codec.rs:893`: exact kernel identity checked |
| Reconstructed typed state   | `state_codec.rs:1249`: replay retained history, then seed supported summaries          |
| v1/v2 cadence metadata      | `state_v3.rs:93`, `state_v3.rs:588`: v3 v1 omits/rejects entity transition count       |

D13 does not explicitly list decay in its primitive-family tolerance table.
For this implementation, classify decay conservatively as a weighted mean:
rtol=1e-12 and atol=1e-12, with stricter existing assertions still binding.
The P2 PR must make that classification explicit in D13; the generic finance
helper is regression evidence and cannot justify a wider normative tier.

## Functional and compatibility requirements

- **FR1 — Results.** Preserve output schema, metadata, row identity/order,
  null positions, valid-count/min-periods gates, warm-up, W=1, mixed outputs,
  entity isolation, late-row behavior, and first-error ordering. Invalid
  current input may still produce a non-null result from valid earlier rows.
- **FR2 — Weights.** Let n be the number of valid values in the last W
  original rows. Valid values receive contiguous ascending weights W-n+1…W;
  missing rows do not keep a zero-valued weight slot. For W=3,
  `[1, null, 4]` produces `(2*1+3*4)/5 = 14/5`. All-invalid frames remain null.
- **FR3 — Determinism.** Batch and stream use the same canonical transition
  rules. Entity interleaving, RecordBatch boundaries, append segmentation,
  watermark release sizes, and checkpoint/recovery cannot change validity,
  IEEE result classification, or values beyond the existing acceptance bound.
  They cannot excuse nondeterministic iteration or different error ordering.
- **FR4 — Incremental work.** Admit decay into the typed plan for the covered
  shape, including mixed mean/decay output. A certified transition uses bounded
  scalar arithmetic without a weighted frame fold. Scheduled rebuilds,
  bootstrap, guard/reference scans, authoritative-history materialization and
  transactional queue COW are counted separately and included in total work
  and end-to-end measurement. Queue COW remains O(W) per touched entity/batch;
  large windows or ill-conditioned values may frequently reject certification
  and use the original O(W) fold. Documentation and explain estimates must not
  promise unconditional O(N) total execution for all frames or inputs. Existing
  general evaluation remains a reference and supported fallback.
- **FR5 — Transaction.** Compute candidate state from an uncommitted view.
  Failed checks, cancellation, output refusal and retry leave installed
  history, summaries, committed counters and output sequence unchanged.
  Preserve the existing distinction between transactional counters and
  attempted-work instrumentation; do not retroactively roll back the latter. A fallback must
  not replay already emitted rows or apply a sample twice.
- **FR6 — Resources.** Cover every added queue, summary, candidate copy and
  scratch allocation in the appropriate established state/work-memory
  accounting. Keep checked size/count conversion and oversized-message
  rejection. Do not remove existing logical charges to offset new storage or
  describe reduced allocation as reduced state budget without evidence.
- **FR7 — Existing declaration.** Preserve project-v3 canonical JSON,
  primitive/configuration versions, numerical-profile selection,
  configuration/pipeline identity, and supported checkpoint compatibility.
  No silent reuse of an old identity for a changed persistent contract.

## Numerical design gates

The proposed algebraic identity, with valid values x1…xn, is:

```text
S = sum(xi)
Q = sum(i * xi)
weighted numerator = Q + (W - n) * S
weight denominator = n * (2W - n + 1) / 2
```

It is valid over real arithmetic. It does not prove equivalence of IEEE
product/add/subtract/divide order, and does not by itself authorize a runtime
recurrence. The design/critic must supply a finite-error admission and repair
argument before implementation, with these requirements:

1. A certified output has a conservative bound on its distance from the
   **existing ordered scan result**, not merely from an ideal real weighted
   mean. Account for weight conversion, product rounding, denominator
   rounding, update/subtract error, cancellation, division, and subnormals.
   A sum-absolute-values bound cannot rely on a cancelling ordinary sum.
   The arithmetic that maintains an upper bound must not round it downward
   unnoticed. A test sweep alone is not a proof of admission.
2. The certificate must fail closed on a non-finite/invalid bound or on any
   output whose old/new IEEE classification cannot be established. An
   uncertified row uses the old oldest-first weighted fold on the retained
   values before emission; repair/reseed may prepare future incremental work.
   A compensated answer must not replace the old fold on fallback rows.
3. The designer must show that well-conditioned financial finite inputs can
   normally be certified without scanning W elements. A design that scans to
   certify every row, or admits only a contrived exact-integer subset, does
   not meet FR4. The extra bound/guard work is part of performance cost.
4. A fixed `8*W` rebuild cadence is insufficient alone. The guard must also
   cover the loss of low-magnitude contributions and the first row after a
   large/cancelling value departs. Include repair work in amortized analysis;
   adversarial fallback may retain O(NW) behavior and must be disclosed.
5. Null/NaN do not poison summaries; infinities remain valid samples. Preserve
   reference sign-of-infinity/NaN classifications, including finite inputs
   whose weighted products or ordered sums overflow. Check every operation
   used by the certificate, including `0 * infinity`, not only input validity.
6. Handle large W and conversion limits using the actual old float weight
   and denominator behavior. Do not assume W or triangular weights fit i32,
   or that every admitted usize is exactly representable in f64. If a shape
   cannot be certified, keep the reference path without changing validation.

For a finite candidate y, a designer may provide a conservative output-unit
distance D to the old fold. A candidate bound alone is insufficient:
uninterrupted and restored candidates can lie on opposite sides of the old
fold, producing a distance up to the sum of their errors. The design must
reserve a conservative paired error budget within rtol=atol=1e-12, or prove
identical reconstruction state. Supply the actual conservatively evaluated
inequalities for both reference comparison and candidate-to-candidate
segmentation/recovery; do not assume two individually passing comparisons
imply a passing pair. This gate is not a completed proof of D.

### Required counterexamples

The following were verified using a short numerical diagnostic, not a build
or benchmark:

- **Magnitude eviction:** W=2 and `[1e20, 1, 1]`. A naive S/Q add/remove
  recurrence yields `2/3` on the third row; the old fold yields `1`. Periodic
  reseeding after 16 rows cannot repair an already emitted third-row error.
- **Finite weighted overflow:** W=2 and first input `1e308`. The old scan
  computes weight 2 times that value and emits positive infinity. Returning
  the finite input value as a seemingly more stable weighted mean is a
  compatibility break.
- **Non-finite algebra:** when a summary is infinite and n=W, evaluating
  `(W-n)*S` introduces `0*Inf = NaN`; this cannot silently replace a positive
  infinity returned by the original ordered fold.

These examples block an unguarded recurrence, not the whole P2 objective.

## Checkpoint and restart gates

The initial implementation must continue reading supported layout 1/2/3
snapshots for the same declaration, including old snapshots made while decay
selected the general kernel. For stages legitimately using layout 4 because
of another output, retain that supported compatibility as well.

Adding typed groups changes the kernel fingerprint inputs: selection,
complexity, group/output descriptors and fallback reason. A bare typed
admission edit therefore rejects old snapshots. The designer must define and
test an exact legacy-identity mapping, following the existing separation of
runtime versus checkpoint fingerprint where appropriate. Reproduce the old
fallback classifier and its precedence exactly, including mixed outputs with
more than one fallback cause; matching only decay-only cases is insufficient.
It must not accept
unrelated/profile/schema/configuration mismatches or broad arbitrary hashes.

Prefer ephemeral summaries reconstructed from existing retained history and
unchanged writer layout. On restore, rebuild a certified baseline and verify
future outputs against both uninterrupted candidate and old reference runs.
Testing only the value immediately after restore is insufficient. Layout-3
stable_v1 does not encode a lifetime entity transition count and rejects one
in its existing entity field; do not insert a cadence counter into that field.
The design must explain how rebuild/guard cadence works with existing
metadata and tolerances. A checkpoint boundary must not be used to hide
otherwise unbounded drift.

If required summaries or cadence cannot be recovered within the existing
contract, stop the default-path implementation and propose an explicit state
version/identity migration. Do not smuggle private summaries into unused
fields or unknown metadata. Failed restore remains atomic.

## Acceptance criteria and focused red tests

Each behavior change begins with a targeted failing test, then implementation.
Test names below describe required behavior; implementer selects exact names
consistent with neighboring tests.

- [ ] Typed admission succeeds for Float64 Rows decay alone and decay+mean;
  Duration and unsupported types retain correct fallback.
- [ ] W=1, W larger than available history, partial/full frames, min_periods
  below/equal W, `[1,null,4]`, all-invalid rows, NaN gaps, invalid current row,
  and alternating valid/invalid expiry match the old fold and independent
  direct-window oracle. Null positions and finite/non-finite classes are exact.
- [ ] Magnitude-eviction counterexample, strong cancellation, alternating
  signs, offset plus tiny variation, subnormal/zero-sign cases, signed
  infinity entry/exit, both infinity signs, and finite weighted overflow pass.
  Do not replace old infinite outputs with finite mathematically nicer ones.
- [ ] A long deterministic finite sequence crossing multiple proposed rebuild
  intervals stays within existing bounds. A deliberately uncertifiable
  sequence demonstrably falls back and later returns to certified operation
  when the design permits; it does not silently stay wrong or grow history.
- [ ] Whole batch versus one-row/mixed-size appends, entity interleaving and
  multiple Arrow RecordBatches give equivalent full output under both
  numerical profiles. Neighboring mean/variance keeps its existing behavior.
- [ ] Old-baseline checkpoint with live history and pending rows restores;
  several subsequent batches cross warm-up/expiry/rebuild points correctly.
  Candidate checkpoint round trip, legacy identity mapping, wrong-profile
  rejection, malformed state and restore-budget rejection remain atomic.
- [ ] Output refusal/cancellation/budget error before commit followed by retry
  matches a clean run, including summaries, output sequence and metrics.
- [ ] Test work counters show O(N) certified arithmetic and scheduled reseed
  sample work. Report cache COW O(sum of resident W per touched entity/batch),
  guard/reference scans and history materialization separately. Do not call
  one-row appends globally O(N) by excluding copies. This is complexity
  evidence, not a substitute for timing.
- [ ] Relevant rolling inline tests and the finance-reference decay scenario
  pass; focused format/lint/module checks pass. Full regression, coverage and
  cross-platform checks run in CI; required unresolved checks block merge.

## Performance acceptance

Use the maintained warm-stream harness with four preregistered cases, fixed
before the first timing. These are direct targets and the demonstrated COW
risk control, not a Cartesian matrix or a new measurement framework:

1. Decay-only Float64 Rows W=20, append=1024: recurrence versus scan target.
2. Decay-only W=1000, append=1024: frame-width/reseed/guard target.
3. Decay plus existing Mean W=1000, append=1024: mixed-stage target, including
   all Mean work and its unchanged algorithm in timing.
4. Decay-only W=1000, append=1: direct risk control for actual O(W) cache COW
   per touched entity/batch. Count it separately from the three targets.

All cases use one entity, H=history segment=1000, stable_v1, normal GC and
four Tokio workers. Use one fresh worker pair per case with exactly 30 AB/BA
warm pairs under the maintained paired-statistics/correctness/provenance
contract. Record exact source revisions/hashes and builds separately before
execution. The total measurement budget is 600 seconds: 530 seconds for
acquisition and 70 seconds reserved for settlement/cleanup; setup, restore,
preflight, warmup, correctness, process startup and cleanup are included.
Do not retry, expand cases or increase samples for a favorable verdict.

At least one of the first three end-to-end targets must show real improvement
under that contract. None of the four may show a confirmed regression above
5%. Report total time, real allocation/resource cost, all private scan,
certification, rebuild, COW and history counters, and uncertainty. Complexity
reduction alone does not authorize merge. If the environment/budget cannot
establish improvement, keep the PR unmerged.

## Resolved design decisions and next role

The numerical tier, conservative paired certificate and exact legacy identity
mapping below passed the second critic review. The resource follow-up requires
a cache-only pool and per-Decay Reference mode; the former proposal to mix
history logical charges into the pool is not the implementation design.

Next: finish focused implementation verification, then independent source and
resource review, selected real performance comparison, and CI remediation.
The initial specification was static research; subsequent implementation and
local test evidence are recorded separately below.

## Implementation design

Status: numerical design approved with implementation caveats; cache-only
resource follow-up incorporated. It uses ordinary binary64 operations under the same IEEE round-to-nearest,
gradual-underflow and no-fast-math assumptions as the existing runtime. No FMA,
second engine, public configuration or durable-state field is introduced.

### State and canonical transition

Reuse the private `TypedGroupPlan::Scan` descriptor with kind Decay, appended
after existing compiled window groups, and add a separate Float64 Decay state.
Do not put Float64 readouts into `TypedScanState.latest: Option<u64>` or modify
neighboring Numeric/Pair transitions. Deduplicate identical input/window
Decay groups; per-output `min_periods` stays in the output plan.

An admitted cache allocation owns the last W original row positions as
`Option<f64>` (null and NaN map to None), counts, S/Q, absolute error radii,
absolute-sum bound and rows-since-reseed. Its Arc shares the same allocation
and lease; a deep COW copy must acquire a separate lease before allocating.
The new enum payload is only an optional Arc and does not enlarge the existing
largest state variant or unrelated logical charges. None is Reference mode;
it owns no replacement persistent queue or summary.

Use a private 64 MiB DataFusion GreedyMemoryPool per Decay kernel plan, with
MemoryConsumer/MemoryReservation RAII credit for each actual cache allocation.
This independent optional-cache cap is not a public configuration, a global
RSS guarantee, or a reduction of StateBudget logical headroom. History/buffer
budget checks remain in their original domain. A held old Arc can prevent a
new acceleration cache, but cannot turn permitted logical growth into an error.
Include the owner/Arc header, registration metadata and queue capacity; verify
capacity and keep credit until the final allocation owner drops. Static
resident-plus-one-copy reasoning cannot replace the per-allocation leases.

Admission occurs on private touched-entity state before numeric transitions.
Reserve the incoming-batch capacity before fill, including a deep copy when an
old owner still exists. On refusal, only that Decay group uses Reference mode;
precompute its original ordered scan readouts/valid counts from retained history
and this batch, using transient work/output arrays with the existing lifetime
and bounds. Numeric/Pair/EWMA/cumulative state advances once, without changing
its phase or kernel identity. Refused mode stores no persistent fallback tape.
On later admission, reconstruct only Decay from authoritative retained rows.
Both ordered and general stream entry points supply that history. Restore may
install no cache; first use still performs admission. The reference work is
bounded by the former scan's transient-history/derived-output work; it is not
a way to retain an uncredited cache across callbacks.

Use this explicit operation order, without reassociation:

```text
expire a valid oldest x, with old S and old n:
    Q := fl(Q - S)
    S := fl(S - x)
    n := n - 1
append a valid x:
    n := n + 1
    t := fl(float(n) * x)
    Q := fl(Q + t)
    S := fl(S + x)
```

The Q expiry identity is sum((i-1)*xi); the departing first value has zero
coefficient. Missing raw positions affect expiry but not valid ordinals.
Decay alone expires before adding; this is private to the new group and must
not reorder the established add-before-expire updates of neighboring groups.
Queue/count updates occur even when readout is null or a summary is invalid.

A retained infinity disables the numeric certificate. Keep its valid membership
and sign counts, use the original ordered fold, and reseed when a finite frame
can be represented again. A non-finite intermediate or radius also disables the
summary. Never subtract from an infinite/stale summary and then trust the result.
All-valid-values-zero is an exact special case: the old fold starts at +0 and
returns +0 for any mixture of signed zeros; reset its summaries to zero without
scanning. Subnormal nonzero values are covered by the error rules below, and
ambiguous zero/sign results use the original fold.

### Conservative arithmetic primitives

The following are private helpers, not a general interval framework. Bounds
are nonnegative binary64 values; NaN, a negative radius, or infinity makes a
certificate fail. Use `next_up` after every rounded positive addition,
multiplication and division when forming an upper bound. Use `next_down` for
lower-bound subtraction and positive lower-bound expressions. Write the
upper helpers as Uadd/Umul/Udiv and lower helpers as Ladd/Lmul/Lsub/Ldiv; each
helper outward-rounds that one operation. Known exact zero bound operations
can stay zero. Clamp a lower bound for an absolute magnitude
to zero; do not clamp a failed upper bound into a valid value.

Let eta=2^-1074. For a finite rounded result z, define rho(z) to be a
representable upper bound on half the largest adjacent spacing:

```text
z normal with unbiased exponent e: rho(z) = 2^max(e-53, -1074)
z zero or subnormal:                rho(z) = eta
```

Construct the exact powers of two with safe integer/`from_bits` operations.
At the largest finite exponent, use the same finite spacing formula rather
than subtracting `next_up(MAX)`, which is infinity. For round-to-nearest,
`abs(z - exact_operation_on_stored_operands) <= rho(z)`. The eta floor covers
ties and underflow where half an ulp is not itself representable.

For tracked a/ea, b/eb and an exactly representable nonnegative integer c:

```text
v = fl(a +/- b):    ev = Uadd(Uadd(ea,eb),rho(v))
v = fl(c * a):     ev = Uadd(Umul(c,ea),rho(v))
```

An incoming stored Float64 has zero input error. Apply these rules to every
operation in the canonical S/Q transition and to the readout. This inductively
bounds S/Q against the exact real sums of the current window, independently of
how many earlier appends or reseeds occurred. No error radius is reduced by
subtracting the error of a departing value.

Maintain A without a cancelling signed sum: on expiry set
`A := max(0, up(A - abs(x)))`, and on entry set `A := up(A + abs(x))`.
Since the prior A bounds the actual nonnegative total, subtraction preserves
an upper bound. If A overflows or loses usefulness after a large eviction,
fail the certificate and reseed from the queue.

### Exact weights and denominator admission

The typed group accepts the full Float64 Rows declaration. The fast numerical
certificate additionally requires W and every weight to be exactly representable
in Float64 and the integer weight total
`D = n * (2*W - n + 1) / 2` to be at most 2^53. Compute this integer expression
with checked u128 arithmetic and an even-factor division before multiplication.
Do not use an unchecked usize intermediate.

These conditions cover practical full financial windows far beyond the planned
cases. They ensure each old weight conversion, each positive integer partial
weight total, and the final total are exact. Thus the old sequential denominator
is exactly float(D), without silently substituting a differently rounded formula.
If this shape proof fails, execute the old loop, including its original rounded
weight conversions and sequential total; do not reject an otherwise valid
configuration or truncate W.

For admitted n>0, set c=W-n. If c=0, use `P=Q, Ep=Eq` without evaluating a
redundant product. Otherwise evaluate t=fl(c*S), P=fl(Q+t), and propagate its
radius Ep by the rules above. Let y=fl(P/float(D)). Its distance from the exact
real weighted mean M is bounded by `Ey = Uadd(Udiv(Ep,D), rho(y))`.

### Bound against the actual old ordered fold

The certificate must also enclose the old scan, not just M. Let Aw bound the
exact sum of absolute weighted inputs. For a one-sign frame, use
`Aw=up(abs(P)+Ep)`; no cancellation is possible in the exact weighted sum.
For a mixed-sign frame use `Aw=up(W*A)`. The latter is conservative and may
force fallback on badly conditioned windows. Evaluate these as Uadd(abs(P),Ep)
and Umul(float(W),A), respectively.

Let u=2^-53, m=n+1, mu=Umul(float(m),u), dlow=Lsub(1,mu)>0,
and g=Udiv(mu,dlow). Each old product-to-output path has at most n+1 rounded
operations. The standard product expansion, including an additive eta error
at each of the at most 2n product/add sites, gives:

```text
Eold_sum = Uadd(Umul(g,Aw), Udiv(Umul(float(2*n),eta),dlow))
Bsum     = Uadd(Aw,Eold_sum)
```

This follows by expanding each rounded operation into an exact operation times
(1+delta), abs(delta)<=u, plus an underflow residue of magnitude <=eta.
Products of at most m such factors differ from 1 by at most m*u/(1-m*u);
the additive residues are amplified by at most 1/dlow. Evaluate all displayed
upper expressions outward, including the residue term. The argument applies
to arbitrary signs; Aw prevents a cancelling bound.

Require Bsum finite. It bounds every partial absolute product/sum as well as
the final numerator, so the same inductive argument proves that the reference
products and additions do not overflow. This is a sufficient condition, not
a claim that every finite reference result can be certified.

For reference r, after the unchanged positive exact denominator, use:

```text
qmax = Udiv(Bsum,D)
Eold = Uadd(Udiv(Eold_sum,D), Uadd(Umul(2*u,qmax),eta))
B    = Uadd(Ey,Eold)
```

The universal division bound `2*u*abs(exact quotient)+eta` is conservative for
normal and subnormal quotients. Thus `abs(y-r)<=B`. A non-finite P, y, radius,
Bsum, or qmax fails closed. Require `B < abs(y)` for certified nonzero output;
this proves r is finite, nonzero and has the same sign. Otherwise the old fold
supplies the exact finite/non-finite/zero classification. In particular the
finite-input weighted-overflow example cannot pass this certificate.

### Pairwise recovery and segmentation budget

Use exact internal constants a=t=2^-40, both strictly below the normative
1e-12 constants. Compute `l=max(0,Lsub(abs(y),B))` and the conservative lower
budget `L=Ladd(a,Lmul(t,l))`. Certify only if `B <= Ldiv(L,4)`.
For any old ordered result r, l<=abs(r), hence every emitted certified value
satisfies `abs(y-r) <= T(r)/4`, where `T(r)=a+t*abs(r)`.
An old-fold fallback has zero error and satisfies the same statement.

For any two outputs p and q of uninterrupted, differently segmented or restored
runs with the same canonical window, the triangle inequality gives
`abs(p-q)<=T(r)/2`. Also
`max(abs(p),abs(q)) >= abs(r)-T(r)/4`, so their internal D13 threshold is at
least `T(r)*(1-t/4)`, which exceeds T(r)/2 because t<2. The normative threshold
is larger still. This closes the two-candidate gap without serializing error
radii or a lifetime reseed counter. Non-finite outputs and ambiguous zeros are
always taken from the same old fold, so the tolerance proof cannot excuse a
classification difference.

### Repair policy, normal work and known limit

Reseed after W raw transitions since the last seed, independent of batch
boundaries; guard failure may reseed earlier. A fresh seed folds valid inputs
oldest first to reconstruct S/Q and their radii with the same helper rules.
Do not reset this counter at ordinary append boundaries. On restore a zero
counter and fresh seed are allowed by the pairwise proof above; layout-3 v1
continues omitting the lifetime transition counter.

When certification fails, produce that row's result using the exact old
ordered weight/product/add/divide sequence. In the same queue traversal, rebuild
S/Q/A for future rows if possible. Count the fallback fold and summary work,
even when fused into one traversal. No extra `Vec` of valid samples is needed:
n and original-position membership are already in state.

The normal arithmetic transition and certificate perform constant work;
scheduled reseeds cost O(W) every W raw transitions, hence O(N) arithmetic
work. Transactional queue copying remains O(W) per touched entity/batch, as in
the existing typed queues; include it in allocation and end-to-end evidence.
Do not call a one-row-append execution globally O(N) by omitting that cost. Additional failed
certificates can retain O(NW) worst-case behavior. In particular the reference
gamma bound grows with n: very large otherwise well-conditioned windows can
fail this conservative certificate even when the true rounding error is small.
Do not advertise unconditional O(N) for every finite sequence or hide those
fallbacks in measurement. This is a known design cost, not an arbitrary small-W
admission cap. Ordinary positive prices avoid the mixed-sign W*A overestimate;
with a W-scale reseed interval the accumulated relative bound is O(u*W), far
below the internal budget for conventional short/moderate financial windows.

A short numerical diagnostic (not a timing benchmark) used smooth prices near
100 with nulls every 23 rows, four windows of history plus 17 rows, and a restore
partway through each case. Every asserted certificate enclosed the actual old
fold and every restored/uninterrupted pair passed 1e-12 D13. W=20 certified
96 of 97 positions, W=256 certified 1040 of 1041, and W=1000 certified 4008 of
4017; the unclassified first positions include all-null readouts. Magnitude
expiry `[1e20,1,1]` emitted the reference 1 on the third row, and W=2 with
`[1e308]` emitted reference +Inf. These diagnostics support expected usefulness;
they do not replace the proof or establish speedup. The implementation must
add focused work-count tests and include fallback/repair work in the selected
real performance cases. If normal workload guard cost or fallback frequency
defeats the requested acceleration, stop rather than relax the certificate.

### Checkpoint identity and bootstrap

Persist no new Decay summary/radius/counter fields. Keep declaration identity,
state layout/schema and authoritative retained rows unchanged. A Decay group
is added after existing window groups and contributes no durable recurrence
seed; `recurrence_seeds` continues carrying only the established EWMA and
cumulative entries. Reconstruct the Decay queue and scalar seed from retained
history using `reconstruct_typed_state`/`rebuild_history`; its local reseed phase
is ephemeral. Existing EWMA/cumulative restoration and the restored entity
transition count remain untouched.

Compute the checkpoint identity using an explicit frozen pre-P2 classifier:

1. Apply the existing `cumulative::legacy_fallback_reason` first, unchanged.
   Its earliest unsupported window-group/cumulative reason retains priority.
2. For a declaration containing Decay, reproduce the old `compile_typed_plan`
   order: compile all window groups first; then scan outputs in declaration
   order, validating Float64, Rows, usize conversion, and finally the old
   allowed-kind set Argmax/Argmin/UniqueCount. Return the first old reason,
   including a different earlier unsupported scan. Freeze the old strings.
3. When this classifier returns an old general reason, use the existing
   `checkpoint_kernel_fingerprint` mechanism: General selection/complexity,
   empty typed groups/outputs, exact old reason, and all original input schema,
   ordering, profile and layout fields. Do this even if the new runtime later
   falls back for a different output; its physical reason is not the durable
   identity. Without a new Decay group, retain the current identity behavior.

Do not change unrelated fallback messages as a side effect of broadening the
allowed-kind set. Preserve `state_codec` configuration/schema/profile/inventory
checks and current-or-declared-legacy layout admission. Do not accept arbitrary
layout numbers or an arbitrary extra hash. The public metadata fingerprint is
the old durable identity; internal physical descriptors can describe Decay.

Required old-source fixtures cover Decay alone, Decay+mean, Decay+cumulative,
and an earlier unsupported group/scan. Generate them with pinned baseline
runtime source and record their generator patch/commit and hashes; do not
claim that a new-code round trip proves backward compatibility. Restore each
supported fixture, run several future appends through expiry/reseed, compare
with baseline and uninterrupted candidate, and verify malformed/wrong-profile/
wrong-schema/budget failures remain atomic. If the existing retained tail is
insufficient for any mixed declaration, treat that as an implementation blocker
rather than silently dropping the corresponding recovery assertion.

### Implementation order after critic approval

1. Add focused failing tests for the scalar certificate, the supplied magnitude
   and overflow counterexamples, and the pairwise budget; implement the private
   arithmetic/state helpers and direct-window oracle comparisons.
2. Add typed admission/readout and mixed-output red tests; append the new group
   without editing existing numeric transitions. Test null gates and ordering.
3. Add frozen legacy-identity and old-fixture restore red tests; wire ephemeral
   reconstruction and retain writer/decoder format and validation.
4. Add transactional failure/retry and attempted-work accounting tests; expose
   only the minimal diagnostic work counters required by the maintained tools.
5. Run the focused Rust/Python checks and selected paired performance protocol.
   No merge claim precedes real measured improvement and final review/CI.

### Implementation evidence so far

- Typed admission first failed with General versus OrderedPrimitive, then the
  integrated Float64 Decay/mean path passed its focused test.
- Numerical/owner helpers first failed for absent symbols, then passed the
  magnitude-eviction, overflow, missing-value, infinity, radius-boundary and
  real MemoryPool owner/copy/release checks.
- Five old-runtime snapshots were captured before production edits using
  baseline `8d8be7d9` plus a test-only generator. Their actual writer layouts
  are 3 and 4; do not describe them as layout-1/2 golden evidence. The checked-in
  fixture contains the baseline/capture-input/reproducible-generator hashes,
  metadata and segment bytes. The generator patch applies to the pinned source.
- Each fixture restored into the new implementation and continued through
  several batches against uninterrupted execution and the unchanged general
  scan. Cache refusal/recovery retained the neighboring Mean outputs bitwise
  and left logical StateCharge unchanged.
- Further focused tests passed long normal sequences and reconstructed-pair
  comparisons at W=20/256/1000, subnormal/zero signs, identity/profile/budget
  rejection, and refused/cancelled repair followed by replay. These are local
  correctness evidence; they establish neither CI success nor speedup.
- A pure Python explain test first failed on the missing Decay group and then
  passed after the cost-model kind update. The baseline and candidate Python
  exports have identical Project JSON, logical fingerprint and native plan
  fingerprint when using the same sealed baseline native module. That native
  module is only evidence for unchanged Python lowering, not new Rust behavior.

### Private attempted-work witness

`RollingMetrics` carries optional serialized `_decay_work` diagnostics through
its existing callback snapshot and Python status mapping. Ordinary callbacks
without Decay work retain their existing keys. These counters record attempted
execution, including rejected output and retries; they are not committed state,
checkpoint fields, or delivery counts. Owned caches collect fixed-size local
counters and the existing recorder receives bounded preparation/numeric batch merges, including before
ordinary error returns. Deep copies start fresh counters. No per-row recorder
lock is added.

Rows partition into `shortcut_rows`, `certified_rows`,
`scheduled_reseed_rows`, `guard_fallback_rows`, and `quota_reference_rows`;
their sum is `transitions`. Sample counts below do not overlap:

- `initial_seed_samples`: slots copied into a new cache, visited for valid
  counts, and visited in its initial combined summary/reference fold.
- `scheduled_reseed_samples` and `guard_reseed_samples`: slots visited by the
  corresponding single combined summary/reference fold.
- `reference_scan_samples`: slots visited in quota-reference mode, including
  both its valid-count traversal and its old weighted-fold traversal.
- `cache_copied_samples`: slots actually copied when an existing cache becomes
  a private candidate for a touched entity in a batch.
- `reference_history_samples`: authoritative retained-history slots visited to
  materialize a reference/recovered seed, or existing queue slots copied for a
  quota-reference candidate.

The counters describe slot visits and row classifications; they do not equate
visits to equal CPU cost. Normal certified arithmetic is O(N), scheduled reseeds
add O(N), cache COW adds O(sum of resident W per touched entity per batch), and
guard/reference work is exposed separately. A zero reference-scan count does
not imply constant total work for a one-row batch.

The added buffered regression runs both numerical profiles with two entities,
forces general buffered preparation, retains old allocation owners, admits one
new cache while the other entity uses Reference, and then restores both caches.
It passed with neighboring Mean Arrow values bitwise equal, unchanged logical
StateCharge and unchanged frozen identity. The initial private-work runtime
test failed because `_decay_work` was absent, then passed its real copy,
certification, scheduled reseed, quota-reference and recovery counts. The
binding mapping test independently failed for the missing helper and then
passed; the existing full-integer-range/snapshot status test also passed.
These focused checks do not establish performance acceptance or CI success.
The reproducible old-source fixture patch uses zero-context hunks and is
validated against its pinned baseline with `git apply --check --unidiff-zero`.

The final expanded attempted-work test initially failed after a rejected output
because it sent only another watermark. This direct-operator unit test rejects
the first chunk before delivery, then explicitly replays the same input batch
and watermark to exercise a second attempted transition. It does not promise
live StreamingRunner callback retries: a managed send failure still terminates
the job, and recovery uses the last compatible durable cut. The failure log is
retained as
`target/p2-decay-work-final-green.log`: 13m20s build, one test failed with 13
actual transitions versus an incorrectly constructed retry expectation of 18.
The first failed-attempt assertions already passed: guard count 1, guard slots
3, and unchanged installed state, cache credit and output sequence. The
subsequent test-only correction replays the same five-row Arrow batch and
keeps all expected assertions: transitions 18, guard count 2 and COW slots 8.
No runtime code or threshold changed. This corrected test has not yet been
recompiled or executed; do not describe the final focused check as green.
