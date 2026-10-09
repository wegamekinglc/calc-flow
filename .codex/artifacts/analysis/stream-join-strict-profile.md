# Stream Join: strict comparison and fresh CPU profiles

## Outcome and scope

**The first candidate fails the performance objective.** It remains about twice as slow as #394 on both Lookup workloads, with confirmed paired regressions against that reference. Its comparisons with original #393 do not establish a greater-than-5% improvement in both rounds. It must not be merged as a completed performance fix.

This experiment directly compares frozen PR #394 with original and optimized PR #393 on three fixed workloads. The initial comparison confirmed a real gap: original #393 took 315.595 ms versus #394's 119.975 ms on 1M Lookup Join. Earlier historical reports used different sampling environments, and the 100k report also used a different batch size. They could not explain this gap. The measurements below use one maintained harness and the catalog's default 64,000-row batches.

The candidate implements four changes: resolve each distinct opposite key ID once for counting and collection; use a common stack-backed canonical hash stream with hashbrown's default hasher; keep one canonical Arc owner per distinct probe key plus physical-row u32 IDs; and append proven admission blocks through exact-budget synchronous grants. This preserves collision equality, serialized key bytes, actual-capacity funding, cancellation boundaries and first-error order. These are combined measurements; they do not isolate the benefit of each change.

## Fixed method

The plan was recorded before execution: three workloads, two rounds of ten adjacent alternating AB/BA pairs per comparison. The diagnostic phase compares original #393 with #394; the final phase independently compares the optimized version with each reference. That is nine case comparisons and 360 timed observations. Every timed observation runs the maintained full-row correctness oracle. Workers persist within a round; each sample constructs a fresh plan/runner and starts with empty state. Warmup is outside the timed window. Fixture creation, installs, validation, profiles, statistics and owned-process cleanup are charged to the measurement budget.

Pooled P50 change is descriptive. Verdicts use the maintained per-round paired rule: both lower bounds above +5% establish regression; any upper bound above +5% is inconclusive; both upper bounds below -5% establish improvement; otherwise the verdict is no-confirmed-regression. The ten-pair distribution-free median interval has nominal 95% coverage (97.85% under the iid assumption). Shared-host interference and dependence are not ruled out. No case expansion or result-dependent resampling occurred.

### Original #393 versus #394

| Case                   | Base P50 ms | Candidate P50 ms | P50 change | Round paired median and interval %                 | Verdict  |
|------------------------|-------------|------------------|------------|----------------------------------------------------|----------|
| Lookup 1M              | 315.595     | 119.975          | -61.98%    | -62.07% [-62.95, -59.39]; -62.14% [-63.94, -60.52] | improved |
| Lookup 100k            | 35.491      | 13.376           | -62.31%    | -59.40% [-68.40, -57.41]; -65.58% [-67.71, -60.38] | improved |
| Retained interval 100k | 251.730     | 226.436          | -10.05%    | -12.37% [-13.78, -5.50]; -11.07% [-13.62, -5.09]   | improved |

### pr393-vs-optimized

| Case                   | Base P50 ms | Candidate P50 ms | P50 change | Round paired median and interval %               | Verdict                 |
|------------------------|-------------|------------------|------------|--------------------------------------------------|-------------------------|
| Lookup 1M              | 387.535     | 322.230          | -16.85%    | -14.62% [-26.56, -1.49]; -17.05% [-25.68, -8.19] | no-confirmed-regression |
| Lookup 100k            | 44.706      | 36.915           | -17.43%    | -16.09% [-28.11, -7.06]; -11.57% [-21.23, +3.38] | no-confirmed-regression |
| Retained interval 100k | 312.757     | 291.926          | -6.66%     | -5.00% [-13.00, +2.26]; -6.53% [-11.60, +0.69]   | no-confirmed-regression |

### pr394-vs-optimized

| Case                   | Base P50 ms | Candidate P50 ms | P50 change | Round paired median and interval %                     | Verdict      |
|------------------------|-------------|------------------|------------|--------------------------------------------------------|--------------|
| Lookup 1M              | 142.510     | 297.838          | +109.00%   | +116.41% [+95.01, +126.52]; +104.84% [+90.25, +113.59] | regression   |
| Lookup 100k            | 15.916      | 34.862           | +119.04%   | +95.58% [+56.12, +142.69]; +116.65% [+82.74, +142.58]  | regression   |
| Retained interval 100k | 252.500     | 262.484          | +3.95%     | +9.97% [+0.96, +15.54]; +4.05% [-3.41, +7.23]          | inconclusive |

## Fresh CPU diagnostics

Each version has one warmup and two fixed 1M timed samples under the same userspace CPU sampler at 4,000 Hz. CRC markers and monotonic clock mapping delimit the exact timed windows across all driver threads. Warmup and markers are excluded. Module hashes, debug identities and symbol sidecars bind stacks to their native module. Symbol-bearing diagnostic modules are code-equivalent to sealed wheels: all 26 SHF_ALLOC section metadata/layout/bytes match except build-id. The diagnostic and stripped module hashes differ. Profiles are directional CPU evidence, not wall-time attribution, allocation counts or additional statistical timing samples.

| Version   | Timed CPU samples | Missing stacks | Unresolved leaf/native | Recording warnings |
|-----------|-------------------|----------------|------------------------|--------------------|
| pr393     | 2696              | 0              | 0/0                    | Lost 18 events.    |
| pr394     | 1091              | 0              | 0/0                    | none               |
| optimized | 2580              | 0              | 0/0                    | Lost 29 events.    |

Original #393 reports 18 lost events across its complete recording. They cannot be assigned to the timed windows, so 18/2,696 is not a valid timed-window loss rate. Optimized #393 likewise reports 29 lost events over the complete recording, which cannot be assigned to its timed windows or turned into a timed-window loss rate. Native inclusive hotspots in original #393 included native_matches (38.13%), native_probe_keys (25.26%) and BorrowedKey::hash (8.83%). These overlap and must not be added.

The following disjoint leaf/self groups use frozen first-match symbol rules preserved in cpu_summary.py. They sum to covered timed CPU samples; their percentages are not percentages of elapsed time.

| Leaf group        | Original #393 | PR #394     | Optimized    |
|-------------------|---------------|-------------|--------------|
| scheduler         | 397 (14.7%)   | 3 (0.3%)    | 376 (14.6%)  |
| quantum           | 38 (1.4%)     | 0 (0.0%)    | 21 (0.8%)    |
| borrowed_keys     | 329 (12.2%)   | 0 (0.0%)    | 486 (18.8%)  |
| siphash           | 103 (3.8%)    | 0 (0.0%)    | 1 (0.0%)     |
| native_dictionary | 176 (6.5%)    | 0 (0.0%)    | 38 (1.5%)    |
| probe_interner    | 131 (4.9%)    | 114 (10.4%) | 93 (3.6%)    |
| arrow_take        | 124 (4.6%)    | 92 (8.4%)   | 146 (5.7%)   |
| clock             | 128 (4.7%)    | 2 (0.2%)    | 117 (4.5%)   |
| cancellation      | 45 (1.7%)     | 1 (0.1%)    | 52 (2.0%)    |
| other             | 1225 (45.4%)  | 879 (80.6%) | 1250 (48.4%) |

## Resource observations

RSS is a worker-lifetime high-water mark including fixtures, warmup, Python/native threads and repeated samples. There are four worker receipts per case comparison (two sides × two rounds), 36 in total; these are not 20 independent per-sample memory observations. No allocation counts were measured. Native credit and actual-allocation behavior are covered by focused tests, not inferred from RSS.

| Comparison         | Case                   | Base peak MiB (two workers) | Candidate peak MiB (two workers) |
|--------------------|------------------------|-----------------------------|----------------------------------|
| pr393-vs-pr394     | Lookup 1M              | 581.8–591.6                 | 589.3–590.4                      |
| pr393-vs-pr394     | Lookup 100k            | 415.4–418.8                 | 400.5–434.0                      |
| pr393-vs-pr394     | Retained interval 100k | 767.9–829.7                 | 816.4–858.0                      |
| pr393-vs-optimized | Lookup 1M              | 562.0–578.4                 | 573.1–605.8                      |
| pr393-vs-optimized | Lookup 100k            | 397.1–403.3                 | 400.6–411.1                      |
| pr393-vs-optimized | Retained interval 100k | 796.5–817.8                 | 798.5–799.2                      |
| pr394-vs-optimized | Lookup 1M              | 573.9–579.8                 | 569.0–581.5                      |
| pr394-vs-optimized | Lookup 100k            | 406.7–409.9                 | 403.3–404.8                      |
| pr394-vs-optimized | Retained interval 100k | 851.0–859.9                 | 804.2–824.3                      |

## Source and build seals

Frozen optimized commit: `1728cab907ba9eee6913ba7d5de163db2e5ef01d`. The evidence package records its exact Git tree. The subsequent report/evidence commit preserves the measured runtime source hash.

- pr393: source commit `eff2fcad17b8922de6aa83955cc14ef602611a0b`; runtime source SHA-256 `840064b5c55aebc29eb4b5eafb5a3ec49668e78accb3a62f923e815c7d10f520`.
  Wheel `12f5761cb65b167c5fbe26b4e1ba9ca33784553c98a67f9d1a1fca56dfed8e72`; sealed native `3c4b6d31a2c41d74abe155e98e99e02d5b1de0c28be5d4036986d7c108141428`; diagnostic native `56c8af1b96ce6925d874323977cb6c26b73d5402173244b168189797b299ba91`; Cargo.lock `361783d6f8e4797c744059eec61c97df2c75d85cc8aaa06b412eaf708f9b3892`.

- pr394: source commit `8b9154103614bd3608d13dd30a45b2657af7ea73`; runtime source SHA-256 `6a608be53d7e01fbb1f18efb77e49c4ef6c252983b4516ed4eae34d5dcdee633`.
  Wheel `202cc4b4b63cbd74b72ea54a16d110631174ceff1f45df0e72695853e565c111`; sealed native `2b3a53a6b025dc4250245132dbefd588b5cbd07a078f5a59ecd8f3ff3313867b`; diagnostic native `f823b993b16347ec9e3d5c3c25072e2438102c52f9057bf53f86e2b06a004c07`; Cargo.lock `b218cb59ba8e028f716c098af1ab65902f49b4158ee2267943028e1b8ff66ab9`.

- optimized: source commit `1728cab907ba9eee6913ba7d5de163db2e5ef01d`; runtime source SHA-256 `3322f6ab146bd25ee4802d3a433e856a145076d084cb5e1ad8ad739cfcd58567`.
  Wheel `77353631fbc3705cefbe7434780064c1951641bc50d2ac926f4b7576d7c41456`; sealed native `e26ea7fc775100d13de397aa4051979a7476a577f0262dafc1db09ce66062e67`; diagnostic native `f790f5665524d11b1283147d6f2280e168cfa0cdd1ad21132f930a37caf682e5`; Cargo.lock `361783d6f8e4797c744059eec61c97df2c75d85cc8aaa06b412eaf708f9b3892`.

Logged build attempts in this experiment total 733.323 seconds, separate from 180.186/600 seconds of supervised measurement (55.047 diagnostic + 125.139 final). Workspace cleaning/setup has separate receipts. Original #393 reuses its previously sealed release; its prior build is not counted again.

Before cross-revision release builds, the three owned workspace packages were explicitly cleaned in the owned target cache; actual compile paths and clean source identities were checked. The first original-393 diagnostic-module attempt omitted the extension-module environment, linked libpython and changed allocated sections. It was rejected before sampling and is preserved with its rejection/build receipt. The corrected build has the required extension-module environment and allocated-section equivalence. No invalid binary contributed timing or profile samples. The two references have equivalent dependency graphs; their Cargo.lock difference is package ordering only.

## Correctness and review

Observed RED evidence precedes implementation: opposite lookups were 12 rather than 3, repeated-key strong owners were 2 rather than 1, and the all-admitted fixture still used 130 scalar admissions. GREEN verifies three opposite lookups for three distinct keys, a single canonical owner, and 12 grants for 130 rows with the original 14 yields. Canonical hash tests cover fragmented and stored bytes, supported type/timezone framing and block boundaries. The final serial Join suite passes 209 tests, with four property tests; scoped Clippy denies warnings, formatting and generated-contract/lock/whitespace checks pass. Actual changed-function Lizard maximum CCN is 8. Independent specialist source review approved the frozen implementation.

The explored CPU-lane approach is deferred. Its minimal hook could leave cold shared controls charged on rejection, or consume scratch budget needed by later pair/output work. Correct integration requires a complete preparation/rollback ownership design. Its real failing tests and earlier fixture failures are retained as investigation evidence, not counted as passing acceptance. There are no production CPU-lane hooks in this candidate.

The full-row oracles cover these selected workloads; they do not resolve the separate source-review blockers in #394. #393 retains capacity-aware funding, surviving-owner refresh, bounded prefix eviction, nested/Dictionary fallback, wide output offsets and error precedence.

## Limits and further work

This is three selected #394-related workloads, not all of #394's microbenchmarks or the full issue acceptance set. It does not measure current ASOF, generic capture/restore, compaction, low-credit SQL fallback or every data type. The prior retained/checkpoint reports describe their own frozen revisions. The absolute 60 ms J2 target and issue aggregate require separate evidence; neither is inferred from a relative speedup. Required CI/coverage and cross-platform checks remain merge gates.

The same representation strategy still has source-grounded opportunities in rolling checkpoint scalar expansion/rebuilding, cross-section generic payloads, rolling out-of-order fallback, ASOF alternating accepted/late input, and late-output preflight. Details and preservation constraints remain in [the integration analysis](stream-join-pr394-integration.md#similar-opportunities-elsewhere). These are hypotheses, not measured improvements of this candidate. Future prioritization should use profiles of those current paths rather than extrapolating this Join profile.

## Recoverable evidence

[Lossless evidence package](stream-join-strict-profile-evidence.json.gz) preserves raw observations, all full CPU profile recordings and symbol sidecars, exact timing/identity audits, driver and sampler invocations, releases/build attempts, RSS receipts, supervisor cleanup, fixed scripts, RED/GREEN/lint logs and investigation fixtures. Wheel/native binaries remain local; their seals are portable.

Compressed bytes: 1,198,759; SHA-256 `3c6c64fe3b91397c8341c8bb425c9909219a3beb00e77978d2a94787b82bbe8d`. Decompressed JSON bytes: 4,452,796; SHA-256 `40094d0b8aa392bc2c6be089c23cf9e1658d7f4bbb58bffdec732662e0511da3`. Files: 172 with per-file size/hash and recoverable utf-8/base64 content.
