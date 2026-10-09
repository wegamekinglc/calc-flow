# Stream Join: batch keys and vectorized admission

## Outcome and scope

Performance outcome: **FAILED against PR #394**. Lookup 1M is 190.737 ms versus 126.500 ms (+50.78%); Lookup 100k is 21.747 ms versus 15.532 ms (+40.01%), both confirmed regressions. Retained interval is inconclusive. This frozen source is investigation evidence, not the final performance implementation.

This experiment directly compares frozen PR #394 with original and optimized PR #393 on three fixed workloads. The initial comparison confirmed a real gap: original #393 took 315.595 ms versus #394's 119.975 ms on 1M Lookup Join. Earlier historical reports used different sampling environments, and the 100k report also used a different batch size. They could not explain this gap. The measurements below use one maintained harness and the catalog's default 64,000-row batches.

The candidate resolves each distinct opposite key ID once, binds typed columns per parent, writes N canonical frames into one funded contiguous arena, and keeps only D canonical Arc owners. It uses hashbrown's default hasher on contiguous bytes. Microsecond copies and constant temporal masks remove scalar scans; proven admission grants charge the remaining metadata appends. This preserves collision equality, serialized key bytes, actual-capacity funding, cancellation boundaries and first-error order. These are combined measurements; they do not isolate the benefit of each change.

## Fixed method

The plan was recorded before execution: three workloads, two rounds of ten adjacent alternating AB/BA pairs per comparison. The diagnostic phase compares original #393 with #394; the final phase compares the optimized version directly with #394, following the narrowed scope. These stages contribute six case comparisons and 240 timed observations. The earlier failed candidate adds six case comparisons and 240 observations, preserved separately, for 480 timed observations across both source candidates. Every timed observation runs the maintained full-row correctness oracle. Workers persist within a round; each sample constructs a fresh plan/runner and starts with empty state. Warmup is outside the timed window. Fixture creation, installs, validation, profiles, statistics and owned-process cleanup are charged to the measurement budget.

Pooled P50 change is descriptive. Verdicts use the maintained per-round paired rule: both lower bounds above +5% establish regression; any upper bound above +5% is inconclusive; both upper bounds below -5% establish improvement; otherwise the verdict is no-confirmed-regression. The ten-pair distribution-free median interval has nominal 95% coverage (97.85% under the iid assumption). Shared-host interference and dependence are not ruled out. No case expansion or result-dependent resampling occurred.

### Original #393 versus #394

| Case                   | Base P50 ms | Candidate P50 ms | P50 change | Round paired median and interval %                 | Verdict  |
|------------------------|-------------|------------------|------------|----------------------------------------------------|----------|
| Lookup 1M              | 315.595     | 119.975          | -61.98%    | -62.07% [-62.95, -59.39]; -62.14% [-63.94, -60.52] | improved |
| Lookup 100k            | 35.491      | 13.376           | -62.31%    | -59.40% [-68.40, -57.41]; -65.58% [-67.71, -60.38] | improved |
| Retained interval 100k | 251.730     | 226.436          | -10.05%    | -12.37% [-13.78, -5.50]; -11.07% [-13.62, -5.09]   | improved |

### pr394-vs-optimized

| Case                   | Base P50 ms | Candidate P50 ms | P50 change | Round paired median and interval %                 | Verdict      |
|------------------------|-------------|------------------|------------|----------------------------------------------------|--------------|
| Lookup 1M              | 126.500     | 190.737          | +50.78%    | +47.85% [+39.13, +80.94]; +48.24% [+39.08, +59.20] | regression   |
| Lookup 100k            | 15.532      | 21.747           | +40.01%    | +31.24% [+22.40, +60.38]; +43.33% [+5.78, +64.37]  | regression   |
| Retained interval 100k | 237.428     | 222.348          | -6.35%     | -7.40% [-14.75, -3.10]; -3.62% [-13.51, +8.47]     | inconclusive |

## Fresh CPU diagnostics

Each version has one warmup and two fixed 1M timed samples under the same userspace CPU sampler at 4,000 Hz. CRC markers and monotonic clock mapping delimit the exact timed windows across all driver threads. Warmup and markers are excluded. Module hashes, debug identities and symbol sidecars bind stacks to their native module. Symbol-bearing diagnostic modules are code-equivalent to sealed wheels: all 26 SHF_ALLOC section metadata/layout/bytes match except build-id. The diagnostic and stripped module hashes differ. Profiles are directional CPU evidence, not wall-time attribution, allocation counts or additional statistical timing samples.

| Version   | Timed CPU samples | Missing stacks | Unresolved leaf/native | Recording warnings |
|-----------|-------------------|----------------|------------------------|--------------------|
| pr393     | 2696              | 0              | 0/0                    | Lost 18 events.    |
| pr394     | 1091              | 0              | 0/0                    | none               |
| optimized | 1473              | 0              | 0/0                    | none               |

The [first-candidate failure report](stream-join-strict-profile.md) preserves its two confirmed Lookup regressions against #394, fresh profile and independent review. That source was not resampled. Original #393 reports 18 lost events across its complete recording. They cannot be assigned to the timed windows, so 18/2,696 is not a valid timed-window loss rate. Native inclusive hotspots included native_matches (38.13%), native_probe_keys (25.26%) and BorrowedKey::hash (8.83%). These overlap and must not be added.

The following disjoint leaf/self groups use frozen first-match symbol rules preserved in cpu_summary.py. They sum to covered timed CPU samples; their percentages are not percentages of elapsed time.

| Leaf group        | Original #393 | PR #394     | Optimized   |
|-------------------|---------------|-------------|-------------|
| key_arena         | 0 (0.0%)      | 0 (0.0%)    | 165 (11.2%) |
| scheduler         | 397 (14.7%)   | 3 (0.3%)    | 200 (13.6%) |
| quantum           | 38 (1.4%)     | 0 (0.0%)    | 11 (0.7%)   |
| borrowed_keys     | 329 (12.2%)   | 0 (0.0%)    | 0 (0.0%)    |
| siphash           | 103 (3.8%)    | 0 (0.0%)    | 0 (0.0%)    |
| native_dictionary | 176 (6.5%)    | 0 (0.0%)    | 31 (2.1%)   |
| probe_interner    | 131 (4.9%)    | 114 (10.4%) | 0 (0.0%)    |
| arrow_take        | 124 (4.6%)    | 92 (8.4%)   | 130 (8.8%)  |
| clock             | 128 (4.7%)    | 2 (0.2%)    | 61 (4.1%)   |
| cancellation      | 45 (1.7%)     | 1 (0.1%)    | 14 (1.0%)   |
| other             | 1225 (45.4%)  | 879 (80.6%) | 861 (58.5%) |

## Resource observations

RSS is a worker-lifetime high-water mark including fixtures, warmup, Python/native threads and repeated samples. There are four worker receipts per case comparison (two sides × two rounds), 24 in this package, plus 24 in the separate first-candidate final phase; these are not 20 independent per-sample memory observations. No allocation counts were measured. Native credit and actual-allocation behavior are covered by focused tests, not inferred from RSS.

| Comparison         | Case                   | Base peak MiB (two workers) | Candidate peak MiB (two workers) |
|--------------------|------------------------|-----------------------------|----------------------------------|
| pr393-vs-pr394     | Lookup 1M              | 581.8–591.6                 | 589.3–590.4                      |
| pr393-vs-pr394     | Lookup 100k            | 415.4–418.8                 | 400.5–434.0                      |
| pr393-vs-pr394     | Retained interval 100k | 767.9–829.7                 | 816.4–858.0                      |
| pr394-vs-optimized | Lookup 1M              | 578.1–585.6                 | 580.7–587.5                      |
| pr394-vs-optimized | Lookup 100k            | 413.5–435.1                 | 403.9–406.2                      |
| pr394-vs-optimized | Retained interval 100k | 812.4–835.3                 | 798.3–807.7                      |

## Source and build seals

Frozen optimized commit: `118e6212b80bc1933bc400cc871916dacf81ff7b`. The evidence package records its exact Git tree. The subsequent report/evidence commit preserves the measured runtime source hash.

- pr393: source commit `eff2fcad17b8922de6aa83955cc14ef602611a0b`; runtime source SHA-256 `840064b5c55aebc29eb4b5eafb5a3ec49668e78accb3a62f923e815c7d10f520`.
  Wheel `12f5761cb65b167c5fbe26b4e1ba9ca33784553c98a67f9d1a1fca56dfed8e72`; sealed native `3c4b6d31a2c41d74abe155e98e99e02d5b1de0c28be5d4036986d7c108141428`; diagnostic native `56c8af1b96ce6925d874323977cb6c26b73d5402173244b168189797b299ba91`; Cargo.lock `361783d6f8e4797c744059eec61c97df2c75d85cc8aaa06b412eaf708f9b3892`.

- pr394: source commit `8b9154103614bd3608d13dd30a45b2657af7ea73`; runtime source SHA-256 `6a608be53d7e01fbb1f18efb77e49c4ef6c252983b4516ed4eae34d5dcdee633`.
  Wheel `202cc4b4b63cbd74b72ea54a16d110631174ceff1f45df0e72695853e565c111`; sealed native `2b3a53a6b025dc4250245132dbefd588b5cbd07a078f5a59ecd8f3ff3313867b`; diagnostic native `f823b993b16347ec9e3d5c3c25072e2438102c52f9057bf53f86e2b06a004c07`; Cargo.lock `b218cb59ba8e028f716c098af1ab65902f49b4158ee2267943028e1b8ff66ab9`.

- optimized: source commit `118e6212b80bc1933bc400cc871916dacf81ff7b`; runtime source SHA-256 `9527d9e5fd23e97c5612a6fdfe7469d45f0f137a7b59d52da2044ec3a745cb9d`.
  Wheel `062a9e2859d4eb6da42363f34ab2ff9736e41f50529e9918468d1b297cbce0f8`; sealed native `e62c41f96a8428fd05a68d977d14d0de611cd620712d10054022ecb68abef2ad`; diagnostic native `a80f6d18f0721241d63151e72bd99780c91876e54ccdf3107927e6bac13c7fa3`; Cargo.lock `361783d6f8e4797c744059eec61c97df2c75d85cc8aaa06b412eaf708f9b3892`.

Logged release/diagnostic module build attempts across both candidates total 1080.091 seconds, separate from 232.209/600 seconds of supervised measurement (55.047 diagnostic + 125.139 failed first candidate + 52.023 final bulk candidate). Workspace cleaning/setup has separate receipts. Original #393 reuses its previously sealed release; its prior build is not counted again.

Before cross-revision release builds, the three owned workspace packages were explicitly cleaned in the owned target cache; actual compile paths and clean source identities were checked. The first original-393 diagnostic-module attempt omitted the extension-module environment, linked libpython and changed allocated sections. It was rejected before sampling and is preserved with its rejection/build receipt. The corrected build has the required extension-module environment and allocated-section equivalence. No invalid binary contributed timing or profile samples. The two references have equivalent dependency graphs; their Cargo.lock difference is package ordering only.

## Correctness and review

Focused RED/GREEN evidence covers distinct opposite lookups, canonical ownership, contiguous arena encoding, typed column binding, actual allocation peaks and refusal, vectorized masks and bounded admission grants. The arena writes N physical rows; D is the distinct canonical-owner count, not the number of encoded rows. The earlier candidate kept 14 yields for its 130-row fixture; the final candidate lowers yields only by eliminating documented scalar work, while retaining the same Quantum thresholds. Canonical hash tests cover fragmented and stored bytes, supported type/timezone framing and block boundaries. The serial Join run passed 216 cases; the one corrected fixture then passed separately. After consolidating new test entrypoints into existing tests, all eight focused checks passed with their new assertions retained. Four property tests and scoped Clippy (warnings denied) passed; formatting and generated-contract/lock/whitespace checks pass. Actual changed-function Lizard maximum CCN is 8. Independent specialist source review approved the frozen implementation.

The full-row oracles cover these selected workloads; they do not resolve the separate source-review blockers in #394. #393 retains capacity-aware funding, surviving-owner refresh, bounded prefix eviction, nested/Dictionary fallback, wide output offsets and error precedence.

## Limits and further work

This is three selected #394-related workloads, not all of #394's microbenchmarks or the full issue acceptance set. It does not measure current ASOF, generic capture/restore, compaction, low-credit SQL fallback or every data type. The prior retained/checkpoint reports describe their own frozen revisions. The absolute 60 ms J2 target and issue aggregate require separate evidence; neither is inferred from a relative speedup. Required CI/coverage and cross-platform checks remain merge gates.

## Recoverable evidence

[Lossless evidence package](stream-join-strict-profile-bulk-evidence.json.gz) preserves raw observations, all full CPU profile recordings and symbol sidecars, exact timing/identity audits, driver and sampler invocations, releases/build attempts, RSS receipts, supervisor cleanup, fixed scripts, RED/GREEN/lint logs and investigation fixtures. Wheel/native binaries remain local; their seals are portable.

Compressed bytes: 1,165,156; SHA-256 `35db99ab36dc00a733bf7d7ebe69ef6df9c98bbcc7ef6201346de127d79308ad`. Decompressed JSON bytes: 4,213,780; SHA-256 `0fd51ba96bbe1537fc887fbbe83edcef02044e9e8c8dfcf23315e7d49aafd30b`. Files: 172 with per-file size/hash and recoverable utf-8/base64 content.
