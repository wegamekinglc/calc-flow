# Stream Join: final main integration versus PR #394

## Outcome

The final main-integrated candidate has lower descriptive P50 values than PR #394 in all three selected cases: 8.71% for Lookup 1M, 10.85% for Lookup 100k and 7.61% for retained interval 100k. All three maintained verdicts are no-confirmed-regression, not a formal improved or equivalence claim. All 120 timed observations passed the complete oracle. These conclusions belong only to this frozen production source; the deliberately incorrect no-op diagnostic is separate evidence, not a shipped speedup.

This frozen candidate directly adopts #394's reusable row key buffer: encode one frame, immediately lookup/intern, then clear and reuse the buffer. It removes the whole-batch key arena, N+1 offsets and third traversal. Typed bindings, D canonical owners, distinct opposite-ID caching and the original P reservation remain. It also replaces forced yield_now at every Quantum with Tokio 1.52.3 consume_budget. All existing work accounting and cancellation/deadline checks remain; output and retained indexing are unchanged. These combined timings do not isolate each optimization. Main `5999eee330daf738d19a1c1344945cd0852b9f94` adds V2 writer changed tracking in process_data/on_ingress_progress; this is a new frozen production source, so the previous cooperative comparison is retained separately and not presented as the result of this source.

| Case                   | PR #394 P50 ms | Candidate P50 ms | P50 change | Round paired median and interval %              | Verdict                 |
|------------------------|----------------|------------------|------------|-------------------------------------------------|-------------------------|
| Lookup 1M              | 106.768        | 97.463           | -8.71%     | -8.84% [-13.47, -4.23]; -7.69% [-11.81, -5.92]  | no-confirmed-regression |
| Lookup 100k            | 12.566         | 11.203           | -10.85%    | -9.86% [-13.15, +2.60]; -13.16% [-19.12, -3.99] | no-confirmed-regression |
| Retained interval 100k | 201.353        | 186.031          | -7.61%     | -9.26% [-17.15, -0.40]; -6.40% [-13.33, -3.24]  | no-confirmed-regression |

The [previous cooperative comparison](stream-join-strict-profile-cooperative.md) reported negative P50 changes but inconclusive Lookup verdicts; it belongs to the earlier source before main integration. This new comparison does not resample that source.

The [first-candidate failure](stream-join-strict-profile.md), [second-candidate failure](stream-join-strict-profile-bulk.md), and [third-candidate failure](stream-join-strict-profile-reusable.md) remain preserved with all raw data. None of these failed sources was resampled. The second candidate had confirmed Lookup regressions of +50.78% (1M) and +40.01% (100k).

## Fixed comparison

Exactly three existing Join workloads at the catalog default 64,000-row batches, two rounds of ten adjacent alternating AB/BA pairs: 120 timed observations in this new phase. Each timed observation runs the maintained full-row oracle. Workers persist within a round; each sample builds a fresh plan/runner with empty state. Warmup is excluded from timing. Setup, installs, fixtures, correctness, profiles, statistics and cleanup count toward the same 600-second cumulative budget. No extra workload, changed sample count or result-dependent resampling.

Pooled P50 is descriptive. Maintained per-round paired rule: both CI lower bounds above +5% mean regression; any upper bound above +5% is inconclusive; both upper bounds below -5% mean improved; otherwise no-confirmed-regression. Ten-pair distribution-free median intervals have nominal 95% coverage (97.85% under iid). Shared-host interference and dependence are not excluded. Inconclusive results do not prove equivalence or a speedup.

## Harness migration

Both sides use the same current maintained harness. Its only changed file is scripts/benchmark_suite/process.py (owned command creation/cancellation settlement and error journaling). The complete old/new file manifest and exact diff are sealed in harness-migration.json/source-patches/harness-process.patch. Worker, child_environment, install and stop ASTs are unchanged; timer, oracle, fixtures and AB/BA implementations are byte-identical. The old #394 CPU reference remains explicitly from the old harness; this new candidate profile uses the new harness. Guards validate the exact migration and every current file rather than suppressing provenance differences.

## CPU profile

One warmup and two fixed 1M samples under the same userspace 4,000 Hz CPU sampler. CRC markers and CLOCK_MONOTONIC map exact timed windows across every thread of the driver process; warmup and marker bursts are excluded. Profiles diagnose CPU sample distribution, not elapsed-time attribution or allocation counts. The #394 profile is reused from the frozen diagnostic phase. The new symbol-bearing native module matches all allocated ELF sections of its sealed wheel except build-id. Module hashes, ELF codeId and symbol-sidecar debugId are retained.

| Version   | Timed CPU samples | Missing stacks | Unresolved leaf/native | Recording warnings |
|-----------|-------------------|----------------|------------------------|--------------------|
| pr394     | 1091              | 0              | 0/0                    | none               |
| optimized | 807               | 0              | 0/0                    | Lost 1 events.     |

Any lost-event warnings apply to the complete recording, cannot be assigned to timed windows, and do not establish a timed-window loss rate.


Disjoint leaf/self groups use the preserved first-match rules in cpu_summary.py; percentages are of covered CPU samples, never of wall time.

| Leaf group        | PR #394     | Candidate   |
|-------------------|-------------|-------------|
| scheduler         | 3 (0.3%)    | 3 (0.4%)    |
| quantum           | 0 (0.0%)    | 3 (0.4%)    |
| key_arena         | 0 (0.0%)    | 34 (4.2%)   |
| borrowed_keys     | 0 (0.0%)    | 0 (0.0%)    |
| siphash           | 0 (0.0%)    | 0 (0.0%)    |
| native_dictionary | 0 (0.0%)    | 29 (3.6%)   |
| probe_interner    | 114 (10.4%) | 130 (16.1%) |
| arrow_take        | 92 (8.4%)   | 100 (12.4%) |
| clock             | 2 (0.2%)    | 0 (0.0%)    |
| cancellation      | 1 (0.1%)    | 6 (0.7%)    |
| other             | 879 (80.6%) | 502 (62.2%) |

## Correctness and compatibility

Focused RED: 129 repeated keys retained 3,301 bytes of transient scratch, above the fixture's 1,024-byte bound. The reusable-buffer transplant passed this bound and the existing composite/type V1-byte oracle, actual allocation/refund and forced-collision checks. The scheduling correction has separate focused RED/GREEN evidence: small work must not force a peer to run before its Tokio budget is exhausted; sufficiently large work must yield for peer cancellation without committing state/metrics. Existing row-ID/error and copy controls remain. The main integration uses the focused writer/optimization/output checks and compile/lint scope recorded in main-integration-followup/verification.json; no new full local suite was requested. Warning-denying scoped Clippy, formatting, generated-contract/lock checks and independent specialist review are recorded with their exact scope in the evidence.

The original reservation remains P = 1024 + 12*N + 4*sum(logical_cell_charge + timezone_length + 64). The row-buffer upper bound uses each parent's per-column maximum frame sum and the maximum across parents; it covers every row, including composite keys. Peak funding covers canonical bytes, scratch, typed writers and old/new interner capacity. Scratch drops before opposite-slot allocation. Original borrowed fallback and allocation/error ordering remain.

#394's synchronous generic admission is not copied: it omits bounded Quantum cancellation and reserves all row IDs before timestamp validation, reintroducing documented correctness bugs. This candidate retains bounded grants and first-error order. Checks remain at 64 visits / 4096 bytes; actual suspension now follows the shared Tokio cooperative budget, up to about 128 quanta in a normal task poll, and other Tokio operations may exhaust it earlier. It does not promise every-Quantum suspension or a wall-time/peer realtime deadline. The [single-file no-op diagnostic](stream-join-strict-profile-quantum.md) attributes a joint effect to deleting the complete Quantum mechanism, not a pure yield/cancellation cost, and is deliberately unshippable.

## Build and resource provenance

Frozen measured source: `e7e37ad1da2ca0c4b92f6b72029074892c3acec5`. The lossless package records its exact Git tree. A later report/evidence commit must preserve the same runtime source hash.

Source-equivalent GitHub commit: `98e407ae409268af40f7a753da4ec5258cea1d13`, with the same measured Git tree. Failed runtime-source patches relative to published base `0596258984d6c6650a82e579b3959f22ca9f62cf` are recoverable from the evidence package.

- pr394: Git `8b9154103614bd3608d13dd30a45b2657af7ea73`; runtime source SHA-256 `6a608be53d7e01fbb1f18efb77e49c4ef6c252983b4516ed4eae34d5dcdee633`; Cargo.lock `b218cb59ba8e028f716c098af1ab65902f49b4158ee2267943028e1b8ff66ab9`.
  Wheel `202cc4b4b63cbd74b72ea54a16d110631174ceff1f45df0e72695853e565c111`; sealed native `2b3a53a6b025dc4250245132dbefd588b5cbd07a078f5a59ecd8f3ff3313867b`; diagnostic native `f823b993b16347ec9e3d5c3c25072e2438102c52f9057bf53f86e2b06a004c07`.
- optimized: Git `e7e37ad1da2ca0c4b92f6b72029074892c3acec5`; runtime source SHA-256 `8c7ef5e46abb75313de06a5926c264215202b4aa58c9e163d1805a3a626aa5bf`; Cargo.lock `361783d6f8e4797c744059eec61c97df2c75d85cc8aaa06b412eaf708f9b3892`.
  Wheel `7da2b409f07aca355545b6a43146feec59fb32b0c2ba51e3a33426dba4de57ab`; sealed native `2d3e15aadff8665f55ec649fd3608543a7c61aa45cc064e5696f42b156489d12`; diagnostic native `c5c38a4b5853148f6ac4c6786a771fcd6b26cd56bddf430219791f5a24403763`.

New release build: 224.438s; symbol module: 42.808s, separate from 416.214/600s cumulative measurement. Prior phases: 366.655847s; new supervised phase: 49.558s. The owned-process supervisor finished exit 0 with an empty process group. Before this cross-revision build the three owned workspace packages were cleaned, and their actual compile paths verified. Reference artifacts are reused; prior build attempts and invalid-artifact rejections remain in their original failure reports.

RSS is a worker-lifetime high-water mark including fixtures, warmup, native/Python threads and repeated samples. The new comparison has 12 worker receipts (two sides × two rounds × three cases), not independent per-sample allocation counts. Funding behavior is verified by focused actual-allocation tests.

| Case                   | PR #394 peak MiB (two workers) | Candidate peak MiB (two workers) |
|------------------------|--------------------------------|----------------------------------|
| Lookup 1M              | 578.2–603.8                    | 590.4–599.4                      |
| Lookup 100k            | 408.4–421.3                    | 398.4–418.8                      |
| Retained interval 100k | 801.9–824.8                    | 786.7–816.2                      |

## Limits and delivery gates

This compares the three affected #394 Join cases, not every #394 microbenchmark, ASOF, capture/restore, compaction, all types or low-credit SQL fallback. It does not establish the absolute 60ms J2 target or the full issue acceptance aggregate. Final delivery requires matching #394 performance except necessary bug fixes, green required CI/Codacy and resolved review on the published #393 head, a verified merge, then closing #394. These gates are not inferred from local tests or old-head checks.

## Recoverable evidence

[Lossless package](stream-join-strict-profile-main-integration-evidence.json.gz) includes exact raw timings and oracles, full CPU profiles/sidecars, window/identity audits, fixed scripts and invocations, build seals/logs, RSS receipts, supervisor cleanup and scoped RED/GREEN/lint logs. Wheel/native binaries remain local, identified by portable seals.

Compressed bytes: 1,326,536; SHA-256 `6e847b0a0c6542193b8deef938c825101b26a5adb53916dc04862acea54b837f`. JSON bytes: 5,178,616; SHA-256 `2ada754264fe8adb0bfa497559725a5e93a3a3d634a89715b59ea0f5634a2697`. Recoverable files: 246, each with size/hash and UTF-8/base64 content.
