# Stream Join: isolated Quantum no-op diagnostic

## Result and restriction

**Diagnostic only; this source must never be shipped or merged.** Relative to production control de1c020, the only runtime-source change deletes the body of Quantum::step, retaining its signature and Ok(()). It deliberately removes work accounting, cancellation/deadline checks and yields. It can also change admission grant subdivision and optimizer decisions. This experiment estimates the joint effect of deleting that mechanism; it cannot isolate cancellation-only or yield-only costs.

| Comparison           | Baseline P50 ms | No-op P50 ms | P50 change | Round paired median and interval %                 | Verdict  |
|----------------------|-----------------|--------------|------------|----------------------------------------------------|----------|
| control-vs-optimized | 166.201         | 91.552       | -44.91%    | -44.64% [-48.80, -42.16]; -45.53% [-47.69, -42.79] | improved |
| pr394-vs-optimized   | 111.209         | 89.990       | -19.08%    | -17.62% [-20.80, -16.22]; -17.34% [-21.61, -13.72] | improved |

The control comparison provides evidence that Quantum is a major source of the observed performance gap. The #394 comparison shows what this otherwise identical but incorrect runtime can do. It does not establish a shippable speedup or prove that all of the control's overhead is necessary for correctness. A correct candidate must retain cancellation/deadline checks, bounded fairness, accounting, first-error order and funding; only its own frozen paired results can meet production acceptance.

## Fixed method

One existing 1M Join workload, default 64,000-row batches, exactly two comparisons. Each comparison has two rounds of ten adjacent alternating AB/BA pairs: 80 new timed observations, each with the maintained full-row oracle. No new CPU profile or symbols build. Workers persist within each round; every sample creates a fresh plan/runner with empty state. The existing maintained regression rule and distribution-free paired-median intervals are unchanged; pooled P50 remains descriptive. Shared-host interference and iid dependence limits remain. No resampling or added workload.

Supervised phase 27.289s; cumulative 314.096/600s, including installs, fixtures, warmup, oracle, statistics and cleanup. Supervisor exit 0 with an empty owned group. Release build 253.673s is separate. All three owned workspace packages were cleaned before the cross-revision build and their actual canonical compile paths verified.

## Frozen identities

- control: Git `de1c020de2d45fa7b82cc27d452c036d092216fd`; runtime SHA-256 `f4f4482c86847e5ac1a6521917c2af51eaa2dd8525c6e8b859338d7643177c91`; native `188c08103dfc9ef5afab4e9335b9580b66a3b7c209c884bbd20bdca726810bcd`; wheel `4c4940cdb2e80047d4d0c0caa4de93f2210d4e644046f4c602ad7b567795cfb3`.
- pr394: Git `8b9154103614bd3608d13dd30a45b2657af7ea73`; runtime SHA-256 `6a608be53d7e01fbb1f18efb77e49c4ef6c252983b4516ed4eae34d5dcdee633`; native `2b3a53a6b025dc4250245132dbefd588b5cbd07a078f5a59ecd8f3ff3313867b`; wheel `202cc4b4b63cbd74b72ea54a16d110631174ceff1f45df0e72695853e565c111`.
- optimized: Git `5dbc74df5232ccc55cbdfd0343a0c5b1c25c327c`; runtime SHA-256 `500441a0a2db58883506e5ef2f592161170c57588d23fb869f0a5c68b4ce86d8`; native `fd75b83d4222e46e881cdd6b2374b2381ecbd5bc138a91f1e4cb44994c9b0c30`; wheel `0e6f04bf1bf74930ce20a77f04659df7f17c5def3ca705ebf2bf4edf46da04b3`.

The single-file diff and worker receipt verify 0 inserted/14 deleted lines relative to de1c020 and an unchanged Cargo.lock. The evidence package contains control-runtime.patch relative to published base 0596258984d6c6650a82e579b3959f22ca9f62cf, then patch.diff relative to that control, with explicit application order. Those recover the exact runtime sources independently of unpublished local Git commits. This diagnostic was not tested for correctness of cancellation and was not linted as a production proposal; its unused-argument build warnings are retained. Row-output oracles do not validate the intentionally removed lifecycle behavior.

RSS receipts are worker-lifetime high-water marks, not per-sample allocation counts. This phase has eight worker receipts (two sides × two rounds × two comparisons). Prior failed source timing/profile evidence remains in the [reusable-buffer report](stream-join-strict-profile-reusable.md) and its linked earlier reports.

## Recoverable evidence

[Lossless package](stream-join-strict-profile-quantum-evidence.json.gz) retains all raw paired observations/oracles, fixed scripts, release seals/build warnings, source patches/identity receipts, RSS and supervisor cleanup. No profile was collected in this phase. Reused original diagnostic evidence is identified as reused.

Compressed 819,378 bytes, SHA-256 `54602f860d401356dc93e3d9720711fb40fc0edc450465250c6cb19def1d349d`; JSON 3,007,802 bytes, SHA-256 `9949e13dcca166015fd75f191d7af66512efd60baa3677cff0469b7d31900e0a`; 175 recoverable files with per-file size/hash.
