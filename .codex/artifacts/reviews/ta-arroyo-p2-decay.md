## PR #565 Review: perf: add certified incremental row-window decay

**Author:** wegamekinglc (Cheng Li)
**Branch:** `feature/ta-arroyo-p2-decay` → `main` | **Files:** 50
**Current main:** `a359d632b3854d2da979886b39c77c583be88d8c`
**Original development base:** `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`
**Measured candidate:** `cc6e33e00acd6a8f1feba7a7776f208c29e71638`
**Measured tree:** `9d109d9f9d85acee548406c1f37d47bad6d2a1e5`
**Reviewed local integration:** `896b63bfa0aeff09f5f63d7d10cdcff7b21c0143`
**Integration tree before this report update:** `bb5f26b6879a2bdf4b2059c3f3b14f8c409c2d08`
**Handoff:** source, final performance/archive and main release equivalence clear;
required CI on the eventual published integration head remains pending.

### Summary

Float64 Rows Decay gains an optional S/Q recurrence cache. A conservative
certificate compares its result with the original ordered floating-point fold;
uncertified rows retain that fold. Cache-credit refusal selects Reference for
that Decay group while neighboring Mean/M2 state and durable identity remain.

The final cc6 experiment passes the preregistered four-case acceptance rule.
The reviewer independently reproduced statistics from its 240 raw observations
and checked numerical, native-work, resource, process, budget and archive
records. New main changes only test code and publishing metadata; the actual
integration preserves all 23 P2 code/test/helper paths exactly. Reusing cc6
performance evidence for this release-equivalent integration is justified.
This is not yet merge approval because final-head required CI is unresolved.

The reviewer used the existing isolated worktree and modified only this report.
No runtime, harness, raw result or frozen clone was changed. No native code was
compiled or loaded, no tests or new timings were run, and no remote action or
CI polling was performed. Local Python/NumPy work only analyzed saved data.

### Build and Test Results

- Rust: **Scoped checks passed.** Reviewed logs cover 12 P2 tests plus one
  existing lifecycle diagnostic; 25 existing typed-kernel regressions; final
  four affected tests; and five history/reference/checkpoint tests following the
  CI arity fix. These counts overlap and are not a count of distinct tests.
- Rust lint: final core library/test Clippy passed, including the history-context
  revision. Formatting and complexity-ratchet evidence passed without relaxing
  the baseline, numerical proof annotations or required checks.
- Binding/Python: private status mapping and ordinary full-u64/nanosecond/status
  regression passed. Generator, optimizer, harness guard, worker cancellation
  and phase-budget checks have focused passing evidence. Old-native checks are
  lowering/harness evidence, not verification of the new Rust runtime.
- Release build: actual cc6 builder exited zero in 958.197177 seconds. Exactly
  one core and one binding compiler artifact have `fresh=false`, release
  optimization and `test=false`, both naming the cc6 frozen source paths.
  The independently owned cache/build target did not substitute an older native.
- Historical failures were genuine and repaired: the 13-versus-18 replay fixture,
  test-target Clippy findings, Codacy complexity findings, and CI's arity-allow
  ratchet. Their failed logs remain preserved; filenames do not determine result.
- Final integrated-head CI: **Pending.** The earlier b917 CI failure and earlier
  cc6 snapshots are not generalized to the new integration head. The full
  required coverage/cross-platform/package gates remain independent.

Reviewed logs include `p2-decay-codacy-green.log`,
`p2-decay-tests-focused-green.log`, `p2-decay-tests-clippy-green.log`,
`p2-history-context-focused-green.log`, `p2-history-context-clippy-green.log`,
`p2-complexity-ratchet-{red,green,final}.log`,
`p2-counter-guards-{red,green}.log`, `p2-counter-complexity-green.log`,
`p2-worker-exits-final-green.log`, `p2-installer-recancel-{red,green}.log`,
`p2-phase-budget-{red,green}.log`, and
`p2-final-controller-postseal-{red,green}.log`, all under root `target/`.
Existing passing checks were not repeated for review.

### Blocking Issues

No implementation, numerical, resource, harness, performance or main-equivalence
finding remains open in the reviewed scope.

Before merging, publish the integration/review result and obtain green required
CI/Codacy for that exact head. Review the actual final head for any additional
runtime or harness changes beyond the equivalence proof below. Pending or absent
checks are not green, and the workspace 90% line floor is not proved by skipped
local coverage. Final approval is deliberately deferred to that follow-up.

### Numerical, State and Resource Review

- `Summary::remove` updates Q using the old S before removing the leaving value.
  Valid ordinals and raw-position expiry preserve right-aligned missing-value
  weights. Null/NaN treatment, minimum periods and neighboring Mean/M2 updates
  are unchanged; each row advances neighboring groups once.
- Outward error radii cover binary64 products, additions and subtractions,
  including subnormal eta and finite MAX. Checked u128 denominator arithmetic,
  even-factor division and the 2^53 bound preserve exact admitted denominators.
- The certificate bounds both recurrence and original ordered fold, including
  gamma, underflow and division error. The sign guard and quarter internal
  tolerance retain D13 rtol=atol1e-12 pairwise equivalence without FMA replacement
  or a wider tolerance. It assumes normal IEEE rounding/gradual underflow.
- Empty/all-zero frames retain null/+0 behavior. Infinity, ambiguous zero,
  subnormal and weighted overflow use the old ordered fold. Scheduled and guard
  repairs are real scans, fully represented in native counters.
- The optional 64 MiB pool funds actual allocation owners. Reservation precedes
  allocation, covers owner/Arc/registration allowance and retained capacity,
  and remains with the last owner. Deep COW reserves separately; shallow Arc
  clones share the original lease. Public logical history/buffer budgets retain
  their meaning. Cache refusal does not rebuild the whole plan or reset Mean.
- Ordered/general entry routes pass authoritative history. Per-Decay Reference
  scratch has callback lifetime. Candidate state commits at the original output
  boundary. The direct-operator replay test rejects the first chunk before
  delivery and explicitly resubmits input; it grants no live StreamingRunner
  retry after a failed send. Managed failure still requires durable recovery.
- Legacy classifier order/reason strings and cumulative priority remain. Durable
  identity normalizes the old General fingerprint while resident cache identity
  is physical. No S/Q/radius/work/repair-phase field enters the writer. Strict
  profile/schema/layout/inventory validation and atomic rejection remain.
- Five pinned old-source snapshots cover real layouts 3/4, both profiles, mixed
  Mean/cumulative state and earlier unsupported groups, followed by future
  batches. General-buffer/two-entity mixed-quota tests preserve Mean bits,
  logical charge and actual retained-owner credit across the 64-step phase.
- The final arity fix reuses `DecayInputPreparation` fields with exactly the old
  per-call values. History absence/key errors, scalar-before-columnar order,
  work merging on ordinary errors, and stage/owner/drop boundaries are retained.
  No new allocation or record was introduced; the real arity allowance was removed.

### Independent Final Performance and Archive Verification

The measured identity stays baseline actual `7a2463a8` (product-equivalent to
8d main) versus cc6. It is not relabeled as a build of a359 or the integration.
Final native SHA-256 is
`d72ce8a5444c4bbc42bd4251261b3ecd9f40b074aacc47ead3be2a65ed78c2ce`;
wheel SHA-256 is
`c870c22adef996e6e31014eee904e70381c8806c7373ea02da6f60b8df49e4fe`.
The reviewer hashed the actual wheel and its sole native member, validated
both build receipts and their raw-byte seals, and parsed actual Cargo JSON.
Worker fingerprints differ only in native hash. The build is separate from
measurement and retains actual source/build identities.

Only the four fixed finite-price workloads were measured: one entity, 1,000
history rows, Float64 Rows, default stable_v1, normal GC and four Tokio workers.
Each case has one fresh process pair with persistent warmed native jobs and
30 advancing-state AB/BA append pairs. No Cartesian expansion, pooling or
favorable retry was added.

| Case              | Window | Append | Paired median | Independently reproduced 95% interval |
|-------------------|--------|--------|---------------|---------------------------------------|
| Decay             | 20     | 1024   | 2.507374×     | [2.349689, 2.670110]                  |
| Decay             | 1000   | 1024   | 8.706189×     | [8.381238, 8.998500]                  |
| Decay + Mean      | 1000   | 1024   | 7.900375×     | [7.550449, 8.391012]                  |
| Decay COW control | 1000   | 1      | 1.154980×     | [1.115009, 1.267761]                  |

Using NumPy 2.5.1 and the contract's seed 771031, the reviewer independently
recomputed all P50/P95/CV/throughput values and the 20,000-resample medians of
paired ratios, obtaining exact equality with saved statistics. This is the
paired estimator, not a ratio of independent P50s. All three targets exceed
1.05 with CI lower bound above 1; no selected case has a confirmed slowdown.
The fourth case is present. Performance acceptance is therefore satisfied.

All 240 saved oracles pass with zero mismatches, exact advancing start rows,
row/identity/order checks and disjoint phase timing accounting. The frozen
oracle traverses finite frames oldest-first using old-fold weights; Decay keeps
rtol=atol1e-12. Maximum candidate errors are respectively
1.5632e-13, 7.2475e-13, 7.2475e-13 and 5.6843e-13. Neighboring Mean is checked
separately through its unchanged maintained oracle. Stored Arrow outputs are
not retained, so this is verification of the frozen oracle and recorded full
checks, not a new execution or fabricated independent output replay.

The reviewer checked all eight completed/natural-end native statuses: no tasks,
errors or overflow, all actors ended, zero resident edge rows/bytes/envelopes,
31 source/delivery batches and exact expected totals. History plus appends
validate 192,380 rows. Eight distinct real worker PIDs have normal/reaped/exit0
receipts; four journals match the raw cases and have no close error. RSS figures
are terminal process snapshots, not peak or operator-only memory.

All candidate maps contain exactly 12 nonnegative native fields with no
node overflow, monotonic before/after counts and row classes partitioning
transitions. Baseline instrumentation is unavailable, not zero. The measured
append counts reproduce directly from Rust status:

| Case | Transitions | Certified | Scheduled rows | Guard rows | Reseed visits   | COW copies |
|------|-------------|-----------|----------------|------------|-----------------|------------|
| 1    | 30720       | 29184     | 1536           | 0          | 30720 scheduled | 600        |
| 2    | 30720       | 30630     | 0              | 90         | 90000 guard     | 30000      |
| 3    | 30720       | 30630     | 0              | 90         | 90000 guard     | 30000      |
| 4    | 30          | 30        | 0              | 0          | 0               | 30000      |

Other append work fields are zero. Arithmetic scans and memory/history work
stay separate. Case 4 copies 1,000 retained samples per single-row append;
this does not become O(1) total work because certified arithmetic has no scan.
Frequent guard rejection can still retain O(NW) arithmetic. Case 3 candidate
CV 0.702 and P95 2.837 ms remain disclosed; the median gain is no tail-latency
or cross-host guarantee. Stable_v2 performance, other types/frames/entities,
nonfinite/adversarial data, larger scales, restore/checkpoints and multicore
scaling remain unmeasured; their correctness checks are separate.

The complete final archive is 564,443 bytes, SHA-256
`1bd46cee5fd4c91811b556196a2514e7bfd150134cc11f70da2e6f4317743c33`,
Git blob `c343a27f90836962ee8ed069937434628291187d`. The reviewer decompressed
it, checked all eight manifest-listed payload seals, all 105 helper source
texts/hashes, target auditor text hashes, raw-report/postseal equality and
statistics/resource views. All 240 CSV records round-trip to raw numeric
fields exactly; CSV pair labels are one-based.
The reviewed narrative SHA-256 is
`0a73882407858f92ed6f0a155f8c03b6ac84b09e017ae2ad02c1a1ac365cebe3`.

The original b917 archive/report are unchanged through cc6→integration; the
old gzip still hashes to
`d57c682db5d656a5663d3594097b578a0652e3d5db798457722ad2de658569aa`.
Its missing individual OS-exit receipts remain explicitly disclosed and are
not invented retrospectively. For clarity, final bundle
`first_phase_reference_only.report_sha256=83f74db7…` refers to the historical
Markdown `benchmarks/runtime-fusion/p2-20261108.md`; its readable JSON
`p2-20261108/report.json` separately hashes to `0ffd0aca…` under its own
manifest. These are different named representations, not a raw JSON seal mismatch.

### Absolute Budget and Owned Processes

The independently reviewed controller remains SHA-256
`0258656e7514e565cb41109ba523e1bae241dbaa139bfa0ee9db5d82830e1583`.
Its one phase start precedes source/build/quiet preflight. The collector and
postseals remain in the supervised owned group. Pre/post source identity, both
build/native/wheel manifests, receipt-byte hashes and full common-helper union
are equal; successful postseals precede the original absolute deadline.
Actual outer exit is zero, no cancellation/escalation occurred, every observed
descendant was gone and final ownership is empty.

The final outer phase took 25.354389399988577 seconds of its 574.654709-second
remainder. The first phase took 25.345291340025142 seconds; cumulative measured
wall time is 50.69968074001372 seconds of the original 600. Postseals and cleanup
are included, the 70-second cleanup reserve is retained, and statistics use
only final cc6 samples. No source verification was moved outside the budget.
Installer cancellation and repeated parent cancellation have focused coverage;
exception cleanup waits for the whole observed owned tree under the original
deadline. The actual successful receipts now close the prior lifecycle evidence gap.

### Main Integration and Release Equivalence

The reviewer inspected the complete changed-path inventory and relevant source,
module boundaries, manifest/workflow and build/environment consumers for
8d→a359, then independently checked the actual 896b integration:

- All 28 main changes are docs/release tooling, publishing metadata or tests.
  Core Cargo.toml adds only readme, documentation, keywords, categories and
  publish destination. Version, dependencies/features/profiles, Cargo.lock,
  toolchain, binding manifest and binding build.rs are unchanged.
- ASOF `prefix.rs` changes are wholly inside its `#[cfg(test)] mod tests`;
  the production prefix is byte-identical. In `gather_work/process.rs`, the
  only addition is `hold_worker_permit` inside `#[cfg(test)] impl TestService`.
  Removing that exact added method makes the file byte-identical to old main.
  TestService itself and its Drop implementation are also test-gated.
- The remaining Rust changes are in join/SQL test modules. Their parent module
  chain is `cfg(test)`, including the nested compact tests and restore fixture.
  They settle test executor generations before comparing reservations; native
  release execution does not compile these helpers or calls.
- Production's package environment macro consumes unchanged CARGO_PKG_VERSION.
  No source or build-script consumer of the five new publishing fields exists;
  binding build.rs still only invokes PyO3 configuration. New release workflow
  and external-package verification scripts do not inject flags into the
  measured Maturin/runtime build. No executable dependency or backend changed.
- Both a359 and measured cc6 are ancestors of actual local merge 896b. Every
  one of the 23 P2-changed crate/Python/script code, test, manifest and helper
  paths is byte-identical from cc6 to that integration; no manual runtime
  conflict resolution was introduced.

This proves release behavior equivalence for the changed main inputs and
retains the measured P2 implementation/harness. It supports reuse of cc6's
measurement without a third build/acquisition solely for test/metadata changes.
It does not claim equal complete SOURCE_PATHS hashes or identical artifact
bytes, and neither actual measured native is relabeled as the integrated head.
Any later runtime/harness change requires a new focused review of that delta.

### Style Issues and Documentation Consistency

All reviewed implementation/harness findings are addressed without excluded
files, weakened tolerances, relaxed complexity baseline or test suppression.
Public Project JSON/fingerprint and persistent schema remain; generated Rust/
Python manifests, D13 tolerance, explanation cost caveat and performance plan
are synchronized. Tables are aligned and this report's whitespace check passes.
The final report correctly keeps resource limits, variation, unmeasured paths,
actual build identity and independent datasets visible. Root's integration
handoff should replace its pending-equivalence prose with this conclusion while
preserving raw artifacts. Required CI on the published integrated head is still
necessary, including the new packaging workflow if required by repository policy.

### Verdict

**Comment Only — source, final performance/archive and main release equivalence
clear. Required CI/Codacy on the exact final integration head remains pending;
final merge approval is reserved for that green-head follow-up.**
