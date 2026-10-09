# Stream Join: cooperative scheduling and PR #394 key buffering

## Outcome

All three descriptive P50 values are lower than PR #394: Lookup 1M -6.14%, Lookup 100k -5.97%, and retained interval -9.30%. The prior confirmed roughly +52% Lookup gap is absent in this fixed comparison. Lookup verdicts remain inconclusive, and interval is no-confirmed-regression: this does not prove equivalence or a stable speedup. All 120 timed observations passed the full-row oracle with zero maximum absolute error. The deliberately incorrect no-op diagnostic remains separate evidence, not a shipped speedup.

This frozen candidate directly adopts #394's reusable row key buffer: encode one frame, immediately lookup/intern, then clear and reuse the buffer. It removes the whole-batch key arena, N+1 offsets and third traversal. Typed bindings, D canonical owners, distinct opposite-ID caching and the original P reservation remain. It also replaces forced yield_now at every Quantum with Tokio 1.52.3 consume_budget. All existing work accounting and cancellation/deadline checks remain; output and retained indexing are unchanged. These combined timings do not isolate each optimization.

| Case                   | PR #394 P50 ms | Candidate P50 ms | P50 change | Round paired median and interval %              | Verdict                 |
|------------------------|----------------|------------------|------------|-------------------------------------------------|-------------------------|
| Lookup 1M              | 132.429        | 124.301          | -6.14%     | -9.79% [-15.86, -2.62]; -5.02% [-13.53, +6.85]  | inconclusive            |
| Lookup 100k            | 16.432         | 15.451           | -5.97%     | -12.29% [-19.22, +6.05]; -2.12% [-8.42, +5.22]  | inconclusive            |
| Retained interval 100k | 271.828        | 246.543          | -9.30%     | -9.28% [-12.85, -1.69]; -10.70% [-11.82, -6.79] | no-confirmed-regression |

The [first-candidate failure](stream-join-strict-profile.md) and [second-candidate failure](stream-join-strict-profile-bulk.md) remain preserved with all raw data. Neither failed source was resampled. The second candidate had confirmed Lookup regressions of +50.78% (1M) and +40.01% (100k). The [third, reusable-buffer candidate](stream-join-strict-profile-reusable.md) also failed (+52.46% and +52.32%).

## Fixed comparison

Exactly three existing Join workloads at the catalog default 64,000-row batches, two rounds of ten adjacent alternating AB/BA pairs: 120 timed observations in this new phase. Each timed observation runs the maintained full-row oracle. Workers persist within a round; each sample builds a fresh plan/runner with empty state. Warmup is excluded from timing. Setup, installs, fixtures, correctness, profiles, statistics and cleanup count toward the same 600-second cumulative budget. No extra workload, changed sample count or result-dependent resampling.

Pooled P50 is descriptive. Maintained per-round paired rule: both CI lower bounds above +5% mean regression; any upper bound above +5% is inconclusive; both upper bounds below -5% mean improved; otherwise no-confirmed-regression. Ten-pair distribution-free median intervals have nominal 95% coverage (97.85% under iid). Shared-host interference and dependence are not excluded. Inconclusive results do not prove equivalence or a speedup.

## CPU profile

One warmup and two fixed 1M samples under the same userspace 4,000 Hz CPU sampler. CRC markers and CLOCK_MONOTONIC map exact timed windows across every thread of the driver process; warmup and marker bursts are excluded. Profiles diagnose CPU sample distribution, not elapsed-time attribution or allocation counts. The #394 profile is reused from the frozen diagnostic phase. The new symbol-bearing native module matches all 26 allocated ELF sections of its sealed wheel except build-id. Module hashes, ELF codeId and symbol-sidecar debugId are retained.

| Version   | Timed CPU samples | Missing stacks | Unresolved leaf/native | Recording warnings |
|-----------|-------------------|----------------|------------------------|--------------------|
| pr394     | 1091              | 0              | 0/0                    | none               |
| optimized | 1162              | 0              | 0/0                    | none               |

Any lost-event warnings apply to the complete recording, cannot be assigned to timed windows, and do not establish a timed-window loss rate.


Disjoint leaf/self groups use the preserved first-match rules in cpu_summary.py; percentages are of covered CPU samples, never of wall time.

| Leaf group        | PR #394     | Candidate   |
|-------------------|-------------|-------------|
| scheduler         | 3 (0.3%)    | 8 (0.7%)    |
| quantum           | 0 (0.0%)    | 3 (0.3%)    |
| key_arena         | 0 (0.0%)    | 124 (10.7%) |
| borrowed_keys     | 0 (0.0%)    | 0 (0.0%)    |
| siphash           | 0 (0.0%)    | 0 (0.0%)    |
| native_dictionary | 0 (0.0%)    | 34 (2.9%)   |
| probe_interner    | 114 (10.4%) | 163 (14.0%) |
| arrow_take        | 92 (8.4%)   | 142 (12.2%) |
| clock             | 2 (0.2%)    | 5 (0.4%)    |
| cancellation      | 1 (0.1%)    | 12 (1.0%)   |
| other             | 879 (80.6%) | 671 (57.7%) |

## Correctness and compatibility

Focused RED: 129 repeated keys retained 3,301 bytes of transient scratch, above the fixture's 1,024-byte bound. The reusable-buffer transplant passed this bound and the existing composite/type V1-byte oracle, actual allocation/refund and forced-collision checks. The scheduling correction has separate focused RED/GREEN evidence: small work must not force a peer to run before its Tokio budget is exhausted; sufficiently large work must yield for peer cancellation without committing state/metrics. Existing row-ID/error and copy controls remain. Twelve focused checks passed (five optimization entrypoints and seven directly affected copy/home-close/sparse controls), without a full Join/property rerun. Warning-denying scoped Clippy, formatting, generated-contract/lock checks and independent specialist review are recorded with their exact scope in the evidence.

The original reservation remains P = 1024 + 12*N + 4*sum(logical_cell_charge + timezone_length + 64). The row-buffer upper bound uses each parent's per-column maximum frame sum and the maximum across parents; it covers every row, including composite keys. Peak funding covers canonical bytes, scratch, typed writers and old/new interner capacity. Scratch drops before opposite-slot allocation. Original borrowed fallback and allocation/error ordering remain.

#394's synchronous generic admission is not copied: it omits bounded Quantum cancellation and reserves all row IDs before timestamp validation, reintroducing documented correctness bugs. This candidate retains bounded grants and first-error order. Checks remain at 64 visits / 4096 bytes; actual suspension now follows the shared Tokio cooperative budget, up to about 128 quanta in a normal task poll, and other Tokio operations may exhaust it earlier. It does not promise every-Quantum suspension or a wall-time/peer realtime deadline. The [single-file no-op diagnostic](stream-join-strict-profile-quantum.md) attributes a joint effect to deleting the complete Quantum mechanism, not a pure yield/cancellation cost, and is deliberately unshippable.

## Test-only CI follow-up

[Windows Rust CI on the initial published head](https://github.com/wegamekinglc/calc-flow/actions/runs/37974396930/job/113968941964) found one remaining old scheduling assumption in `test_empty_metadata_history_allocations_are_actually_funded`: its synchronous manual-poll helper required a wake outside a budgeted Tokio task. Cooperative budget is unconstrained there. The follow-up expects zero forced wakes and preserves every actual allocation, peak funding, refund, schema and legacy IPC assertion. Real suspension under a budgeted task remains covered by the existing large-work controls.

This correction changes only a `cfg(test)` fixture and this report, leaving the measured production implementation unchanged. It does not add performance observations or change the frozen archive. The maintained full source fingerprint includes test files, so the final CI follow-up has a different full source fingerprint; it is not represented as the original measured Git tree. The corrected fixture passed its focused test (1 passed, 2,047 filtered); actual peak 1,965 bytes and retained 1,941 bytes were covered by a 3,130-byte guard, with complete refund and exact schema/IPC assertions. Formatting and whitespace checks passed. Corrected full source fingerprint: `2c33300d12f1dd850b9019d52cfe8e5a0eb5783945154b72509686564e6455ce`. Final required CI applies to the corrected published head.

## Build and resource provenance

Frozen measured source: `13a2578acaa3a0c07aa150422458e0950d11f27c`. The lossless package records its exact Git tree. The initial report/evidence commit preserved the measured source hash; the subsequent test-only CI correction is described above.

Source-equivalent GitHub commit: `12915453eea2a93300194fab0edf408515791267`, with the same measured Git tree. Failed runtime-source patches relative to published base `0596258984d6c6650a82e579b3959f22ca9f62cf` are recoverable from the evidence package.

- pr394: Git `8b9154103614bd3608d13dd30a45b2657af7ea73`; runtime source SHA-256 `6a608be53d7e01fbb1f18efb77e49c4ef6c252983b4516ed4eae34d5dcdee633`; Cargo.lock `b218cb59ba8e028f716c098af1ab65902f49b4158ee2267943028e1b8ff66ab9`.
  Wheel `202cc4b4b63cbd74b72ea54a16d110631174ceff1f45df0e72695853e565c111`; sealed native `2b3a53a6b025dc4250245132dbefd588b5cbd07a078f5a59ecd8f3ff3313867b`; diagnostic native `f823b993b16347ec9e3d5c3c25072e2438102c52f9057bf53f86e2b06a004c07`.
- optimized: Git `13a2578acaa3a0c07aa150422458e0950d11f27c`; runtime source SHA-256 `85e6992312b069e335a900b03d0dce29a504eaba18b897c9e796acd98d51727f`; Cargo.lock `361783d6f8e4797c744059eec61c97df2c75d85cc8aaa06b412eaf708f9b3892`.
  Wheel `d0a516f91ee7b06b8ea0f3a76f795dfc1ae08a62a925fd69755a8a4fccc250a9`; sealed native `f4b30f0d66ea1188f92ac27b768100fc46fde466aae742e5b5a1bb0ea293f7a0`; diagnostic native `c62e95ff03ce78abee850a4fe21b98341de3b58fd30b0434d746e64c40f972f5`.

New release build: 243.248s; symbol module: 51.452s, separate from 366.656/600s cumulative measurement. Prior phases: 314.096475s; new supervised phase: 52.559s. The owned-process supervisor finished exit 0 with an empty process group. Before this cross-revision build the three owned workspace packages were cleaned, and their actual compile paths verified. Reference artifacts are reused; prior build attempts and invalid-artifact rejections remain in their original failure reports.

RSS is a worker-lifetime high-water mark including fixtures, warmup, native/Python threads and repeated samples. The new comparison has 12 worker receipts (two sides × two rounds × three cases), not independent per-sample allocation counts. Funding behavior is verified by focused actual-allocation tests.

| Case                   | PR #394 peak MiB (two workers) | Candidate peak MiB (two workers) |
|------------------------|--------------------------------|----------------------------------|
| Lookup 1M              | 579.7–591.8                    | 582.2–589.7                      |
| Lookup 100k            | 406.0–426.9                    | 405.8–406.5                      |
| Retained interval 100k | 842.4–857.9                    | 790.8–847.3                      |

## Limits and delivery gates

This compares the three affected #394 Join cases, not every #394 microbenchmark, ASOF, capture/restore, compaction, all types or low-credit SQL fallback. It does not establish the absolute 60ms J2 target or the full issue acceptance aggregate. Final delivery requires matching #394 performance except necessary bug fixes, green required CI/Codacy and resolved review on the published #393 head, a verified merge, then closing #394. These gates are not inferred from local tests or old-head checks.

## Recoverable evidence

[Lossless package](stream-join-strict-profile-cooperative-evidence.json.gz) includes exact raw timings and oracles, full CPU profiles/sidecars, window/identity audits, fixed scripts and invocations, build seals/logs, RSS receipts, supervisor cleanup and scoped RED/GREEN/lint logs. Wheel/native binaries remain local, identified by portable seals.

Compressed bytes: 1,221,973; SHA-256 `06c686e04a504dcb2fff251cf3c4cc0c66ff9659801cc2897f8e8d78901cec67`. JSON bytes: 4,464,667; SHA-256 `baddf7878ca6c640c28f03b47e8705a5ca90a2fa3b5279104e0267e7e1ef5238`. Recoverable files: 204, each with size/hash and UTF-8/base64 content.
