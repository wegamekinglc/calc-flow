# P2 Rolling kernel v2 - Critic Critique

## Target

- Spec: `.codex/artifacts/specs/rolling-kernel-v2.md`, revision 3 (`fab246cc`):
  §4, §6.5 and §9.
- Plan: `docs/plans/2026-11-15-streaming-core-rearchitecture.md` §2 and §5.
  These are fixed inputs.

## Verdict

**Proceed with caveats.** The arithmetic is sound. The §4.4 constants are
derived correctly, apart from the small fixes in items 5–7.

B1–B4 are spec-text fixes:

- B1–B3 before P2.2's red tests;
- B4 before P2.6's session arm.

None changes a plan decision, phase or gate.

## Blocking

**B1. The oracle cannot apply its own acceptance rule** (lines 288, 324-326).

- (a) **Irrational terms.** Exact rational comparison is impossible for
  `√(Q_K/n)` (variance), σ (stddev), `√(Q_Kx·Q_Ky)` (covariance) and r
  (correlation).
  - Fix: enclose each root in outward-rounded rationals, for example
    `math.isqrt` on a value scaled by 2^2k.
  - Round bound terms up. Use two-sided enclosures for σ and r.
- (b) **Correlation null vs finite.** Classification follows the computed
  `Mx`, not the exact one.
  - After non-constant samples are evicted, `fl(s + e)` need not return to
    exactly 0.
  - `max(0, ·)` can clamp a small positive exact `M` to 0.
  - A correct implementation therefore fails "exact classification" exactly
    where corpus 6 aims.
  - Fix: use a three-valued rule.
    - Null is required iff every x (or y) operand folded, inserted or evicted
      since the family's last rebuild/reset equals `k`. The oracle can check
      this exactly.
    - Finite is required iff `ΔMx < Mx` and `ΔMy < My`.
    - Otherwise either is accepted and the value is unchecked.
  - Corpus 6 needs stretches of at least `W − 1 + R` positions, so that the
    *window* is constant for R positions (§4.3 line 269). "At least R
    positions" (line 336; R2, line 1094) is insufficient for W > 1.
- (c) **Out-of-domain classification.** Outside the domain, the expected
  class is whatever binary64 produces.
  - Example: `[1e308, 1e308]` has a finite exact mean but computes +inf, as
    plan §5.3 item 2 requires.
  - `Fraction` cannot predict this.
  - Fix: classify out-of-domain positions by a binary64 replay of §4.3 in
    Python floats, which are IEEE binary64. Alternatively, exclude them and
    cover overflow with fixed unit cases.

**B2. The non-finite rebuild trigger couples families in a group** (line 245,
against line 219, §4.5 line 351 and §12 line 1124).

- "Rebuild when any float sum of the group is non-finite" resets every
  family and `since_rebuild`.
- So when Σq overflows (|x − k| > 1.34e154) but Σx does not, mean's bits
  differ between "mean alone" and "mean + variance".
- Corpus 3 (magnitudes up to 1e300, line 333) is inside the domain only for
  sum and mean. For dispersion, `A_Q ≤ 2¹⁰⁰⁰` caps |d| at about 2⁵⁰⁰ ≈
  3e150.
- Fix: a non-finite-triggered rebuild refolds only the non-finite family and
  leaves the shared row-count cadence alone. Give corpus 3 a per-family
  magnitude cap.

**B3. Overflowed dispersion windows read 0** (line 262).

- Once Σq overflows, TwoSum's error term is `inf − inf = NaN`, so
  `Q = fl(qs + qe)` is NaN.
- Even with a guard on Q, `Q = T = +inf` still gives NaN.
- `f64::max(0.0, NaN)` returns 0.0. Variance and stddev then read 0, and
  correlation reads null.
- This holds for the whole overflowed stretch, because each rebuild refolds
  into the same overflow.
- It contradicts plan §5.3 item 2, under which overflow reads ±inf.
- Fix:
  - give the D and Q readouts the same non-finite guard as F;
  - clamp only non-NaN negatives;
  - state the overflowed variance value (+inf or NaN).

**B4. G5a is unattainable for a computed column that stays inside the graph**
(lines 597-604, 647-650).

- Reusing an Arrow value buffer frees one block and allocates another:
  - `Buffer::into_vec` frees the `Arc<Bytes>` control block through
    `Arc::try_unwrap` (`arrow-buffer-59.3.2/src/buffer/immutable.rs:421-441`);
  - `Buffer::from_vec` allocates a new one (`mutable.rs:537-545`);
  - unwrapping a typed `Arc<PrimitiveArray>` frees its block too.
- Without `unsafe` (decision 4), every recycled value buffer therefore costs
  one free and one allocation per push, and `count_total == count_current`
  fails.
- G5a is reachable only when every computed column goes to the caller. That
  is P1's harness shape (`benchmarks/warm_stream.py:484-493` and `:663-676`
  on `feature/stream-session`):
  - one rolling output, `difference(mean(5), mean(20))`;
  - a column `select` projection.
- In that shape only batch shells need recycling:
  - the `RecordBatch` column `Vec`;
  - `Arc<[RecordBatch]>`, via `Arc::get_mut` and a swap with a zero-column
    placeholder;
  - the cost cache, via `OnceLock::take`.
- Fix: pin the G4/G5 graph to that shape, and limit the recycling text to
  batch shells. A graph that feeds mean(5) and mean(20) into a projection that
  computes the spread makes G5a impossible.

## Checklist

### Numeric contract (§4)

1. **Duration cadence.** `R = max(n_prev, 64)` (line 250).
   - Define `n_prev` as the group's own frame row count right after the
     rebuild, not the shared ring length. Otherwise mean(1h) bits depend on a
     co-resident mean(1d).
   - Persist `n_prev` (or R) in layout 7. Restore's `since_rebuild < R` check
     (line 708) needs it.
2. **Reset position.** The per-row order (line 242) omits Reset.
   - State that Reset runs after evictions and before both the insert and
     `since_rebuild += 1`, so that the insert can set `k`.
   - The oracle's r, m and A depend on this order.
3. **Group definition.** A group is (input column or pair, frame), so all its
   families share validity. Pair counters count pairwise-valid positions.
   Otherwise Reset couples families the same way as B2.
4. **Integer inputs.** For non-sum kinds, the oracle's exact quantities are
   over the converted `x as f64` values (line 213), not the raw integers.
   Otherwise variance at i64 extremes fails by conversion error that the
   bound does not carry.
5. **Sum bound.** The bound holds and the factor 2 is unnecessary.
   - With pairing, `e` sums N ≤ m error terms in a tree, so its error is at
     most γ_{N−1}Σ|q|.
   - Σ|q| ≤ (1+u)(u + γ_{m'−1})A ≤ γ_m A.
   - So `u|S| + γ_m²A` holds by the Ogita–Rump–Oishi Sum2 argument. Keeping
     or dropping the 2 is sound either way.
6. **Mean and dispersion constants.**
   - Add `+ η` to the float mean bound: `fl(F/n)` can underflow with absolute
     error η/2.
   - Variance and covariance already absorb underflow in `4m·η`, since
     `(n+2)/2 + 1/2 ≤ 4m` for `m ≥ n`.
   - First-order constants: 11u is needed and 12u is declared. The 11u is 4u
     from d, q and Sum2, 6u from D²/n, and u from the final subtraction.
   - γ² coefficients: 2 and 4 are needed; 3 and 5 are declared.
7. **Correlation constant.** The bound is formally short by a factor
   (1+γ₄) on the ΔC term at the corner `ΔM = −M/2`, where r ≈ 0 leaves no
   slack. This is unreachable in practice. Writing `(2 + 10u)·ΔC/√(Mx·My)`
   keeps "a corpus violation is an implementation defect" literally true.
8. **Corpus size.** Corpora 1 and 8 alone fill the "≤ 20,000 rows" cap
   (line 329). Raise the cap or shrink them.
9. **Oracle operands.** A_D, A_Q and A_P are over the computed operands
   `fl(x−k)`, `fl(d·d)` and `fl(dx·dy)`. Compute them in Python floats (exact
   binary64) and the sums with `Fraction`.

### G5 (§6.5)

10. **G5b strictness.** Compare with `==`, not `≤`, and build the probe with
    the engine's shared metadata.
    - The current probe over-counts. The derived `BatchMetadata::default()`
      allocates an `Arc<JsonMap>` (`batch.rs:51-65`), and
      `Batch::table(Vec<…>)` (`batch.rs:710`) allocates and frees a `Vec`.
    - With `≤`, that slack admits exactly the per-batch metadata allocation
      the gate forbids.
    - If the engine count is lower, tighten the probe.
11. **G5a warm-up.** Run at least R unmeasured 1-row pushes after the
    4,096-row warm-up segment, which runs in the run tier.
    - Otherwise first use of the scalar path breaks "identical across 1,000
      pushes".
    - The allocation counter is thread-local
      (`allocation-counter-0.8.1/src/allocator.rs`). Assert zero `block_on`
      calls and polls in the same test.
12. **G5c flake.** Process-global lazy statics are allocated by whichever test
    thread touches them first and are never freed.
    - Example: DataFusion's per-function `LazyLock<Arc<ScalarUDF>>`
      (`datafusion-functions-55.2.0/src/macros.rs:119-121`).
    - Whether they land in this test's `bytes_current` depends on test
      scheduling.
    - Construct and drop one identical session on the same thread before
      measuring.
13. **G5c footprint scope.**
    - Build the warm-up inputs inside the measured scope, and drop the
      harness's handles before reading `bytes_current`. Input buffers pinned
      by a pending slice then count.
    - Define "pending bytes" logically, as rows × width.
    - The keep-handles-alive rule (line 654) applies only to per-push G5a.
14. **Retained runtime.** "No `SessionContext`, Tokio runtime or worker
    pool" (line 684) is undetectable by `bytes_current`: a current-thread
    runtime (`session.rs:272`) is far below 256 KiB. Add a structural
    assertion, such as the runtime field being `None`.
15. **Debug vs release.** The 256 KiB figure is absolute and differs between
    debug CI and release builds. Gate it in CI. The release arm should report
    "the same quantities", not "the same numbers" (line 690).

### PR ordering (§9, §10)

16. **Cross-contract digests.** G4 compares are cross-contract if the P2.6
    registration (line 906) predates P2.2.
    - `_validate_pair_digest` then raises on unequal `dual_sma_spread`
      digests (`scripts/profile_warm_stream.py:839-843` on
      `feature/stream-session`; the spec cites `:630-634`).
    - Add `--numeric-contract-change` to G4 (line 1042) and to every session
      compare whose arms straddle P2.2.
    - The `native` subcommand needs the same rule for P2.2's controls.
17. **G5 baseline.** The G5 baseline "at this PR's parent" (line 906) needs
    the session arm, which P2.6 itself adds. Either measure it on the parent
    plus the arm in a temporary worktree, or mark it informational. G5 is
    absolute, so no gate is affected.
18. **Other gates.** Every other gate stays measurable:
    - G1's sync boundary exists from P2.7;
    - G2's stage attribution exists from P2.8 (F5);
    - the G5 CI tests can run from #618 onward.
