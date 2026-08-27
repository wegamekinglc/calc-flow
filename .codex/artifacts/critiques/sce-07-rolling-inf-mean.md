# SCE-07 Rolling Mean over ±inf Windows — Critic Adjudication

| Field         | Value                                                                          |
| ------------- | ------------------------------------------------------------------------------ |
| Status        | **Defect confirmed (blocking); frozen readout semantics supplied**             |
| Baseline      | `feature/rolling-aggregates@778a4c5` (PR #202)                                 |
| Review target | cf-tester Defect 1 on DAL-139 / GitHub #174 (SCE-07)                           |
| Review style  | semantic adjudication against the frozen SCE-00 contract; no code, no redesign |

## Target

- Spec: `.codex/artifacts/specs/symbolic-computation-contract.md` (SCE-00, frozen
  decisions D3.2 §5.2, D5 §7, D13 §15, acceptance matrix §16.2)
- Implementation under adjudication: `crates/calc-flow/src/operator/rolling.rs`
  (PR #202), specifically the West add/remove update at lines 1991-1993 and
  2028-2030, the repair refold at 2269-2298, and the mean readout at 2371
- Evidence: cf-tester report on DAL-139 (2026-08-27) with repro
  `sce07_inf_mean_repro.py`, run-confirmed

## Verdict

**Revise — this is a defect, not a caveat.** The behavior violates the frozen
D3.2 text, and the frozen acceptance matrix (§16.2) makes "infinity follows D3"
a testable acceptance item, so shipping it is an acceptance failure. The fix
stays inside the D5-frozen algorithm properties; no frozen decision needs to be
re-opened, only the previously unadjudicated D3.2 × D5 interaction needs to be
pinned, which this document does.

## Findings

### Blocking Issues

- **Mean over a window containing +inf is misclassified as NaN, violating frozen
  D3.2 IEEE semantics.** D3.2 (contract §5.2): "Positive and negative infinity
  are numeric sample values and follow IEEE arithmetic; an undefined result is
  NaN, not null." For the window {+inf, 1.0, 2.0}, the IEEE sum is +inf under
  every fold order (finite + +inf = +inf; no NaN can arise), so the mean is
  +inf/3 = +inf — a *defined* result. The engine emits NaN when the +inf sample
  is the oldest in the window: the West update at `rolling.rs:1991-1993`
  computes `delta = 1.0 - inf = -inf`, then `mean = inf + (-inf)/n = NaN`.
  NaN is licensed by D3.2 only for *undefined* results; this result is defined.
  That is a direct contradiction of the frozen text, not a tolerance issue —
  D13 (§15) states tolerances "MUST NOT excuse … different missing-value
  classification".
  - **Suggested fix:** freeze the readout semantics in the "Frozen readout
    semantics" section below and implement to it.

### Why the "inevitable consequence of the frozen algorithm family" defense fails

1. D5 (§7) freezes algorithm *properties* — "a deterministic ordered algorithm
   with reversible removal; segmentation may not select a different algorithm" —
   not the West recurrence by name. An accumulator that also counts ±inf
   samples, with a classification-aware readout, satisfies every frozen
   property: counts reverse exactly on removal, the readout is a pure function
   of accumulator state, the fold order is unchanged, and checkpoint rebuild
   derives identical counts from the same retained rows.
2. The implementation already deviates from raw West twice to protect
   classification consistency — the repair refold (`rolling.rs:2269-2271`) and
   the readout-side M2 clamp (`rolling.rs:2379-2384`). The frozen family
   demonstrably admits such repair layers; the current NaN is not "forced".
3. The same-multiset position dependence ({+inf,1,2}→NaN vs {1,2,+inf}→+inf) is
   diagnostic evidence, not the charge itself: no frozen text requires
   cross-dataset order invariance, but the dependence proves the NaN is an
   artifact of update order rather than of the frozen semantics.
4. "All paths agree" is necessary but not sufficient: batch, stream, refold,
   and checkpoint rebuild currently agree on a value that violates D3.2.
   Consistency across paths does not launder a frozen-semantics violation.

## Frozen Readout Semantics (semantics and acceptance only — not implementation)

Each shared window accumulator tracks the count of sampled +inf and −inf values
currently in the window (`pos_inf`, `neg_inf`): incremented on add, decremented
on remove (reversible, D5-compliant), derived by the same ordered fold on
checkpoint restore. Null/NaN exclusion is unchanged (D3.2); infinities are
valid samples and count toward `valid_count` and `min_periods` (unchanged).

**Mean readout** — applied only after the `min_periods` null gate
(`valid_count ≥ min_periods`):

1. `pos_inf > 0` and `neg_inf > 0` → **NaN** (undefined: ∞ − ∞)
2. `pos_inf > 0` and `neg_inf = 0` → **+inf**
3. `neg_inf > 0` and `pos_inf = 0` → **−inf**
4. `pos_inf = neg_inf = 0` → the frozen finite-path West mean, **unchanged**

Rule 1, not net-count: over any multiset containing both signs, *every*
sequential IEEE fold yields NaN (once the fold state is +inf, adding −inf
produces NaN; NaN absorbs; symmetric for the other order). NaN is the unique,
order-independent IEEE answer; a net-count rule (pos>neg→+inf) contradicts IEEE
for e.g. {+inf, +inf, −inf} and invents non-IEEE semantics.

Rule 4 keeps West, not naive sum/count (tester's warning confirmed): a naive
f64 sum/count readout misclassifies {1e308, 1e308} — true mean 1e308, finite —
as +inf through overflow of the sum. The finite path is the D5-frozen
algorithm's behavior under D13 tolerances; this adjudication changes nothing
for finite-only windows.

**Variance / stddev readout:** any inf sample in the window
(`pos_inf + neg_inf > 0`) → **NaN** (undefined: deviations involve ∞ − ∞).
Null gates keep precedence: `min_periods` gate first (null), then divisor
positivity (`valid_count - ddof ≤ 0` → null), then the NaN classification.
Current West behavior already yields NaN on all paths here (tester verified);
this pins the status quo as contract so future changes cannot regress it
silently.

**Sum / count / min_periods:** unchanged. The f64 sum fold is already IEEE and
order-independent in classification (both signs → NaN; single sign → that
sign's inf). Integer columns cannot contain inf. NaN produced by rule 1 remains
observably distinct from null produced by the gates (D3.2).

**Invariants the fix must preserve:**

- Zero public-surface change: `RollingOutputSpec` JSON shape, JSON Schema,
  OpenAPI, generated TypeScript, and `_native.pyi` are all untouched.
- Zero checkpoint segment-format change: counts are derived from retained rows
  during the rebuild fold. Explicitly accepted consequence: restoring a
  pre-fix checkpoint with post-fix code yields post-fix (corrected) semantics
  for inf-containing windows — that is the purpose of the fix; D13's
  checkpoint/recovery equivalence promise is scoped to one code version.
- Batch/stream/checkpoint classification consistency: the counts are multiset
  properties folded in the same canonical order everywhere, so live, refold,
  and rebuild paths produce identical readouts by construction.

## Testable Acceptance Expectations

Rust unit (`operator::rolling` / `rolling_state`):

- A1 — same multiset with the inf at oldest / middle / newest position gives
  the same mean: +inf for {+inf, 1, 2} in all positions, −inf mirror (kills
  the position dependence; this is the tester repro's two entities).
- A2 — {+inf, −inf} and {+inf, +inf, −inf} in any order → NaN.
- A3 — after an inf slides out, subsequent windows read finite values with no
  NaN stickiness; the slide path and the refold path read out exactly equal.
- A4 — {1e308, 1e308} (no inf samples) → finite mean within D13 of 1e308
  (regression guard against a naive sum/count readout).
- A5 — variance/stddev over inf-containing windows: positive divisor → NaN;
  non-positive divisor → null (precedence lock); finite again after the inf
  slides out.
- A6 — infs count toward `count` and `min_periods` (e.g. window {+inf, NaN,
  null} with `min_periods = 1` produces +inf, not null).

Integration / parity:

- B1 — batch vs stream across 1- / small- / large-batch segmentations:
  inf-window classifications identical point-for-point (extends the existing
  parity suites).
- B2 — checkpoint taken while a window contains inf → restore → identical
  classification and sign; an inf expiring across a checkpoint boundary yields
  post-restore values equal to the live path (classification exact, finite
  values under D13).
- B3 — the Python independent reference model implements the frozen rule above
  and matches the engine with zero divergence on randomized null/NaN/±inf
  mixes (the existing probe shape; inf windows must now pass).

Explicitly out of scope for the fix (guardrails against over-reach):

- Do not switch the mean readout to naive sum/count (breaks {1e308, 1e308}).
- Do not "fix" finite-path West overflow (e.g. {−1e308, +1e308} → +inf through
  delta overflow): that is a D5/D13 frozen algorithm-family property,
  consistent across all paths, and outside this adjudication.
- No public API change, no checkpoint format change, no review of Defects 2/3
  (routing already set).

## Minor / Caveat Notes (recorded, non-blocking)

- Finite-extreme corner: West's mean can overflow to ±inf for finite
  opposite-sign extremes ({−1e308, +1e308} → +inf, true value 0); likewise the
  f64 sum fold classifies {−1e308, −1e308, +inf} as NaN (finite partial-sum
  overflow manufactures a spurious −inf) while rule 2 classifies the mean as
  +inf. Both are pre-existing frozen algorithm-family properties, consistent
  across every path. Recommend the doc phase records one sentence in the
  rolling documentation so these corners are not later mistaken for fresh
  unfrozen ambiguities.
- The variance/stddev "any inf → NaN" rule pins current behavior; changing it
  later requires a semantic adjudication first.

## Counter-Proposals

None. The tester-proposed ±inf counting scheme is the right shape; this
adjudication refines its rule 1 (both-signs → NaN unconditionally, rather than
net-count comparison) to match IEEE exactly, and scopes rule 4 to leave the
finite path untouched.

## Questions for the Author

None blocking. The doc-phase team should decide where the one-sentence
finite-extreme caveat above lives (rolling docs vs. SCE-00 addendum).
