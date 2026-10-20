# Streaming Join single-thread performance

The bounded current-main comparison completed in **220.075 seconds**. Single-CPU
Interval Join meets the maintained improvement gate: median time falls from
799.04 to 672.51 ms (-15.83%). The other four cases have lower observed medians
and the verdict `no-confirmed-regression`; they do not establish a greater-than-5%
improvement under the maintained rule. No selected case has a confirmed
regression. This is a five-case result, not a whole-suite or universal guarantee.

The historical PR #482 profile percentages and timings are diagnostic context,
not measurements of this implementation. No new profile or external-engine
comparison was acquired.

## Exact sources

| Arm       | Commit                                     | Tree                                       | Sealed build source                                    |
|-----------|--------------------------------------------|--------------------------------------------|--------------------------------------------------------|
| Baseline  | `78b75724f408f9d46a479488b4ecc76ea17aec68` | `2cde42df5d7ef9412b3871ee4a4142d08c95371e` | `target/worktrees/join-single-thread-baseline`         |
| Candidate | `98012d8a91bb3d0847a50cb77d21025f13629864` | `1ae0a8e48818c7464fbf51425c9931347cafa928` | `target/worktrees/join-single-thread-candidate-sealed` |

The baseline is the selected main commit after PR #482. Candidate runtime evidence belongs
to the sealed code commit above. Later report/documentation commits on
`feature/join-single-thread-performance` do not change that build identity.
Both source checkouts remain available so acquisition can verify their actual
HEAD/tree against each release manifest.

The evidence-only follow-up commit (Y) adds this report, archive and review
evidence on top of code commit X (`98012d8a`). Its runtime and maintained-harness
files must remain unchanged from X. The measured native build remains X;
Y must not be presented as an independently measured build.

The baseline and candidate default-feature release builds completed in 612.734
and 599.604 seconds respectively, budgeted separately from measurement. Both use
CPython 3.13.9, rustc 1.94.0, the same Cargo.lock, default features, release
profile, and no RUSTFLAGS override. Baseline wheel SHA-256 is
`4e01b17b1813aaeb9d721a1200ca92ced10c036394942743f8b43c4baf12cb11`;
native SHA-256 is
`e95cee127d74c7052109c367c2ab8cfdd3d669c25071f7feb0a208d90e266836`.
Candidate wheel SHA-256 is
`5f07d5d94fd1f580971e9fbff6124718e2404f19adc4e74d38c6fedd82c32e85`;
native SHA-256 is
`2740ff5037253d672f52c77193cf7de45afbd29abf49779f42a5b85fea08cc60`.
The task's local manifest directory is
`target/performance/join-single-thread-performance/releases/`. Each manifest
records its actual build command, source/tree, recompiled core, compiler,
dependency lock, environment, wheel/native paths, and artifact hashes.

## What changed

The implementation removes repeated bookkeeping while preserving one atomic
input transaction, full-input match-limit checks, explicit budget refusal,
canonical keys/checkpoint identities, immutable inputs, cancellation ownership,
and output ordering guarantees. The default source/transport batch remains
1,000,000 rows, implicit memory remains unlimited, and production CPU capacity
continues to derive from available CPUs.

| Review item            | Disposition and precise scope                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Source width        | Each side caches its previous borrowed source/projection identity. Stable one-source-per-side output performs at most two column-width inspections; alternating owners may inspect again at source transitions. No uncharged distinct-owner map is allocated.                                                                                                                                                                                                                                                                                    |
| 2. Retained lookup     | A borrowed payload cursor caches the current retained chunk. Consecutive rows in that chunk avoid repeated BTreeMap lookup. Sparse stable coordinates, removed chunks, backwards revisits, and existing storage ownership remain supported. This is a transition cache, not a replacement of the entire sparse index.                                                                                                                                                                                                                            |
| 3. Internal work units | Existing 64-row hash/position scratch and transactional count/gather morsels are retained. New short-string work also uses bounded stack scratch. The complete-group local deduplication table and final identity arrays still scale with the input; this patch does not claim a reduced whole-group preparation working set.                                                                                                                                                                                                                    |
| 4. Short keys          | UTF-8 values of at most 15 bytes use two stack-local packed words with the same length mixing and generic word-hash result. Longer/mixed strings retain the fallback. Canonical framed keys, collision equality, IDs, and checkpoint bytes do not change.                                                                                                                                                                                                                                                                                        |
| 5. Eviction            | Existing expiration-plan summaries supply aggregate dirty/metric updates before mutation; captured raw key owners move into tombstones. Prefix partitioning and dead-chunk reclamation already existed. Canonical binding resolution remains per expired row where restore/lease semantics require it; no unconditional per-key lookup reduction is claimed.                                                                                                                                                                                     |
| 6. Interval runs       | Worker fill and ordered owner assembly append existing match runs in bulk while retaining required opposite-index writes. Existing forward-window cursors, reused count windows, disorder fallback, global match caps, and bounded worker stop checks remain in place.                                                                                                                                                                                                                                                                           |
| 7. Output indices      | Eligible single-source ASOF output stores a UInt64 index plus optional validity instead of a source/row tuple per result. Compact positions remain consumable through range slicing and Arrow take. Multi-source output keeps one-pass interleave. Owner eligibility requires at most two live owners and at most four backing buckets, proving at most one potential right source; larger/sparse pools stay legacy. Native worker promotion checks stop conditions every 64 positions with funded live overlap. No u32 narrowing is introduced. |
| 8. ASOF grouping       | General admission already groups typed key rows and bulk-merges reserved buckets. That existing implementation is retained rather than counted as a newly added optimization.                                                                                                                                                                                                                                                                                                                                                                    |
| 9. Cooperation         | Short-string offset/value work is prepaid by bounded blocks. Existing 64-visit and 4,096-byte limits, including long values, remain enforced; the worker probe cadence remains its existing contract.                                                                                                                                                                                                                                                                                                                                            |
| 10. CPU capacity       | Native CPU-derived pool sizes and Tokio's automatic production defaults already satisfy the requirement. Historical benchmark settings are preserved. Both measurement arms explicitly use the same worker count and affinity so worker-count changes cannot be credited as algorithmic gains.                                                                                                                                                                                                                                                   |

Supported flat nullable ASOF payloads retain their current behavior. Existing
ASOF rejection of nested, dictionary, run-end, and view payloads is preserved.
Ordinary Join fallback and ownership tests cover its broader payload surface.

## Focused verification

There are 487 unique focused Rust checks: 387 ASOF, 98 Join, and two runtime
checks. The 98 Join checks comprise 41 earlier checks in subsequently unchanged
modules, 17 final local-hash checks, and 40 eviction/dictionary checks.
`verification.json` preserves 446 durably logged passes plus the explicit
41-check earlier inventory; it does not claim all 487 were rerun from the final
binary. Scoped Clippy passed against the collaboration worktree's implementation
in 91 seconds; formatting, generated-contract drift, and whitespace checks passed.
These are focused local checks, not a full workspace/connector coverage result
or CI status.

The installed candidate wheel also passed 32 focused Python tests from
`test_asof_stream_results.py`, `test_symbolic_stream_join_acceptance.py`, and
`test_stream_budget_defaults.py`. The installed-native proof names the sealed
candidate commit and verifies the loaded native SHA-256 above. These functional
tests ran before acquisition and did not add performance cases.

The acquisition helper passed 11 Arrow/provenance-only tests after switching
to the sealed source checkout. They verify exact schema/nullability/metadata,
all row identities and values, pair multiplicity, sequence order, source geometry,
completed interval progress/resource counters, fixed samples/budget/inventory,
and exclusion of competing compilers/benchmark workers. The tests did not import
`calc_flow`, load its native module, start a worker, or acquire timings. Scoped
Ruff lint and format checks passed.

The implementation work-count and failure regressions exercise stable and
alternating owners; nullable slices and projections; sparse chunks and backwards
revisits; short/mixed/long strings, collisions, and cooperative bounds; bulk run
assembly and slices; compact ASOF unmatched/row-zero/promotion/wide indices;
and ordered eviction, restored leases, metric/dirty failures, and tombstone
funding. The final specialist review is **Approve**. See the linked
[specification](../specs/join-single-thread-performance.md),
[critique](../critiques/join-single-thread-performance.md), and
[review](../reviews/join-single-thread-performance.md).

Selected operation counts from the regression fixtures substantiate removed
work; they are not wall-time measurements or general workload ratios:

| Fixture operation                | Old observed | Verified expected |
|----------------------------------|--------------|-------------------|
| Stable-source width inspections  | 1,000,001    | 2                 |
| Retained chunk lookups           | 11           | 6                 |
| Empty-string block debit calls   | 64           | 2                 |
| Ordered owner per-pair pushes    | 100,000      | 0                 |
| Worker/serial per-pair pushes    | 173          | 0                 |
| Owner prefix-promotion visits    | 64           | 0                 |
| Uncaptured raw-key clones        | 1            | 0                 |
| Captured raw-key clones          | 2            | 1                 |
| Mixed captured/uncaptured clones | 8            | 3                 |

Bulk append still writes every required opposite index; zero per-pair pushes
does not mean zero output work. Compact/multi-source allocation regressions also
pass, but their fixture bounds are not process-RSS measurements.

## Predeclared measurement contract

Exactly five maintained EngineCase configurations use the 1,000,000-row workload
parameter. ASOF and interval replay that input on each side; ordinary Join has
1,000,000 quote rows and 64 static dimension rows. Interval produces 10,998,080
exact output pairs. The immutable input is coalesced before event construction;
the helper records actual physical Arrow record-batch sizes in both arms.

| Case          | Logical CPUs | Source batch | Connection to changed code                          |
|---------------|--------------|--------------|-----------------------------------------------------|
| Join          | 1            | 1,000,000    | Source-width cache and local short-string work      |
| ASOF          | 1            | 1,000,000    | Compact right-side output positions                 |
| Interval Join | 1            | 1,000,000    | Retained-chunk reuse, match runs, and eviction work |
| Interval Join | 32           | 1,000,000    | Multicore regression control for those same paths   |
| Join          | 1            | 64,000       | Transport segmentation control                      |

Each case has two rounds of ten adjacent alternating AB/BA pairs, reversing
the leading arm in round two: 200 formal observations and exactly 20 excluded
warmups. The single 600-second acquisition budget includes startup, manifests,
fixture construction, preflight, warmup, complete correctness oracles, EOF,
progress observation, and cleanup. Requests stop at 570 seconds, reserving
30 seconds for owned-process cleanup. Builds are separate. There is no fresh
profiling budget, sample expansion, favorable retry, or automatic budget extension.

Timing uses the maintained ready-source enqueue through Arrow concatenation
boundary. Source/runner lifecycle, explicit state/edge budgets, and the sealed
static-dimension preload remain those of the maintained adapters. The 64k Join
control uses the maintained exact-cursor adapter; 1M Join/ASOF use maintained
bounded feeds. Version comparisons within a case are paired and compatible;
cross-adapter batch-size ratios are descriptive only.

Both arms explicitly set Tokio workers to their selected affinity count (1 or
32), with BLAS/OpenMP workers at one. The helper verifies matching machine,
dependency and workload fingerprints, clean exact source/tree, release/default
features, recompiled core, sealed wheel/native hashes, extracted Python package,
loaded native identity, actual batch geometry, successful full oracle, and owned
completion. Root must launch with host `/proc` visibility after all native work
settles. WSL host power and host-wide quiet remain unobservable.

Interval's maintained two-input oracle validates the exact pair multiset,
multiplicity, values, and schema. Its normalized identity-order hashes are
**descriptive only**, never a global acceptance gate: legitimate cross-input
scheduling can change global emission order. Focused Rust tests validate the
guaranteed per-probe/run order on both sides. Join/ASOF sequence emission order
is checked directly.

Per-case verdicts use maintained `report.comparison`: both round paired-median
confidence lower bounds above +5% confirm regression; any upper bound above
+5% is inconclusive; both upper bounds below -5% establish improvement;
otherwise the result is no confirmed regression. No standalone minima rule or
extra confidence requirement replaces that contract.

## Results

All five cases completed with 20 observations per arm: 200 formal observations
and 20 excluded warmups. The entire acquisition, including owned cleanup,
finished in 220.075 seconds of the fixed 600-second budget. There were no
missing cases, visible competing compiler/benchmark detections, cleanup errors,
sample extensions, or retries.

Times below are milliseconds. P95 uses the maintained linear-percentile helper.
P50 change is the ratio of aggregate arm medians; round confidence intervals
describe median per-pair percentage changes, a different statistic.

| Case / CPUs / source batch | Base P50 | Candidate P50 | P50 change | Base P95 | Candidate P95 | Maintained verdict      |
|----------------------------|----------|---------------|------------|----------|---------------|-------------------------|
| Join / 1 / 1M              | 90.81    | 78.51         | -13.54%    | 99.48    | 90.32         | no-confirmed-regression |
| ASOF / 1 / 1M              | 86.75    | 84.02         | -3.15%     | 91.55    | 86.52         | no-confirmed-regression |
| Interval / 1 / 1M          | 799.04   | 672.51        | -15.83%    | 830.64   | 697.25        | improved                |
| Interval / 32 / 1M         | 578.57   | 487.17        | -15.80%    | 697.10   | 653.77        | no-confirmed-regression |
| Join / 1 / 64k             | 63.20    | 59.17         | -6.37%     | 74.51    | 61.54         | no-confirmed-regression |

| Case / CPUs / source batch | Round 1 paired median | Round 1 interval   | Round 2 paired median | Round 2 interval   |
|----------------------------|-----------------------|--------------------|-----------------------|--------------------|
| Join / 1 / 1M              | -9.93%                | [-18.30%, -3.26%]  | -15.39%               | [-22.31%, -8.21%]  |
| ASOF / 1 / 1M              | -3.02%                | [-10.95%, +3.37%]  | -3.17%                | [-7.57%, +3.88%]   |
| Interval / 1 / 1M          | -17.24%               | [-20.80%, -16.82%] | -12.99%               | [-16.43%, -12.48%] |
| Interval / 32 / 1M         | -20.13%               | [-27.02%, +1.46%]  | -18.30%               | [-25.37%, -11.12%] |
| Join / 1 / 64k             | -8.24%                | [-14.84%, -3.56%]  | -6.28%                | [-9.44%, -2.04%]   |

The exact order-statistic intervals have coverage 0.978515625 for ten pairs
(the maintained method selects at least 95% coverage). Independent/common pair
distributions remain assumptions; alternating execution does not prove them.
Only single-CPU Interval has both upper bounds below -5%. ASOF's roughly 3.2%
median decrease is unconfirmed; its intervals cross zero. Join and the 32-CPU
Interval result also do not meet the maintained greater-than-5% improvement
criterion despite lower aggregate medians. No interval upper bound exceeds +5%,
so none of these cases has the maintained `inconclusive` verdict.

Interval's absolute median falls about 15.8% at both CPU settings. Its descriptive
1-to-32-CPU ratio remains about **1.38x in both revisions**. This is not a paired
scaling verdict: affinity changes the machine fingerprint, and these were version
comparisons within each case. The patch removes bookkeeping work; this result
does not establish improved multicore scaling or explain all remaining ASOF cost.

## Correctness, lifecycle and resources

All 220 warmup/formal invocations passed exact all-row schema, identity, value,
and multiplicity checks. Join/ASOF sequence order matched the input. Interval
produced 10,998,080 exact pairs per invocation, with no state/match-limit failures
and the expected accepted watermark/eviction counters. The 32-CPU Interval
samples contain two descriptive identity-order hashes in **both** arms, consistent
with legitimate cross-input scheduling; exact multiset and focused per-probe
order checks remain the contract.

Every formal invocation returned through maintained `run_with_completion`.
All 20 directly recorded warmup terminal jobs ended naturally with zero live
tasks, zero task errors, ended sources/sinks, the expected delivered row count,
and drained edges. Each worker's finish response is completed, with no cleanup
errors. The offline audit reproduced all five maintained statistics dictionaries
exactly and verified the frozen harness, manifests, wheel/native hashes, paired
fingerprints, sample order/counts, and terminal evidence without importing
`calc_flow` or collecting new timings.

| Case / CPUs / source batch | Base peak RSS MiB | Candidate peak RSS MiB | Base whole-sample CPU s | Candidate whole-sample CPU s |
|----------------------------|-------------------|------------------------|-------------------------|------------------------------|
| Join / 1 / 1M              | 671.93            | 659.95                 | 0.1957                  | 0.1856                       |
| ASOF / 1 / 1M              | 403.92            | 388.77                 | 0.1993                  | 0.1970                       |
| Interval / 1 / 1M          | 2,039.61          | 2,023.91               | 1.7259                  | 1.5892                       |
| Interval / 32 / 1M         | 3,749.93          | 3,593.57               | 2.3770                  | 2.2298                       |
| Join / 1 / 64k             | 301.21            | 294.42                 | 0.1815                  | 0.1775                       |

RSS is the maximum worker-process lifetime `ru_maxrss`, including fixtures,
warmup, retained allocator pages, outputs and full oracles. It is neither active
operator allocation nor state-only memory and has no separate confidence verdict.
Whole-sample CPU is a median including untimed oracle/lifecycle work; dividing
it by ready-window wall time would give a misleading utilization estimate.
Excluded warmups retain separate ready-window process/thread CPU diagnostics,
but this task does not establish sustained worker concurrency.

The machine is WSL2 `chengli-i9`, exposing **32 logical CPUs**, not an asserted
32-physical-core system. Single-CPU affinity was `[0]`; the parallel control used
`[0..31]`, with explicit Tokio overrides 1/32 respectively in both arms. The guard
saw no competing known builds/benchmarks. Host power mode and activity outside
WSL are unobservable, and the 32-CPU interval is visibly noisier.

Matching local dependencies were CPython 3.13.9, NumPy 2.5.1, PyArrow 24.0.0 and
Polars 1.44.1, plus the identical recorded Cargo.lock/compiler/default release
settings. Polars is recorded as environment identity only: **Polars 2.0.0 was not
measured or compared in this task**. Both rounds preserve matching machine,
dependency and workload fingerprints within every version comparison.

## Evidence and coverage limits

The [evidence archive](join-single-thread-performance-evidence.tar.gz) contains
99 members in 266,719 bytes: raw formal report and worker transcripts, common
maintained harness, task helpers, exact build manifests/logs, test evidence,
and the offline audit. SHA-256:
`a3dda02639851c8d3f43674480bac9660a1b6e2121ad9ea9323a696ec2075f87`.
The [member index](join-single-thread-performance-evidence-index.json) and
[verification record](join-single-thread-performance-evidence-verification.json)
preserve per-member hashes, exact source/tree identities, and successful archive
verification. Wheel/native binaries remain in the sealed release directories;
their hashes and manifests are archived. Runtime checkpoint state directories
are excluded; raw correctness/lifecycle observations remain included.

For portable statistics reproduction, extract the archive, put its `harness/`
directory on `PYTHONPATH`, load `task/reports/measurement.json`, and pass each
case's two saved A/B sample arrays to the archived
`scripts.benchmark_suite.report.comparison` with `comparison="interleaved"`.
This recomputes the paired verdict without native code or new measurement.
`task/audit_evidence.py` additionally validates the original workspace's sealed
artifact paths; absolute paths in manifests are historical provenance.

These cases cover ordinary flat/short-string Join, flat ASOF compact output,
and fresh-state Interval probe/gather/eviction. The remaining hottest coverage
advisories are sparse long-lived or restored retained state and captured
tombstone eviction in the maintained lifecycle/Join harness, then skewed keys
and multi-source/wide ASOF in the existing ASOF settlement/materialization
targets. These are follow-up coverage suggestions, not extra cases run here.

Unmeasured paths include ASOF/ordinary Join multicore, other input scales,
skewed/wide/nested payload timing, checkpoint/recovery timing, and external-engine
comparisons. Full CI and local combined coverage were not run for this new branch;
focused passes do not establish the coverage floor or green required checks.
