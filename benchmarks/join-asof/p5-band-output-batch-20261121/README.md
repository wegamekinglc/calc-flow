# P5 larger BandJoin output batches

Candidate `1bd3b3ce` sets the native BandJoin DataFusion batch size to 65,536 rows instead of the default 8,192. The 140,000-row regression now produces three batches with a 65,536-row first batch. All hot-path tests and diagnostic-feature Clippy passed before the release build.

The baseline is sealed candidate `5e7c22db` from `../p5-fused-projection-20261121/result.json`. Only interval was remeasured; static Join is unchanged and retains its passing evidence. The case used one CPU, two warmups and ten measured samples, with the maintained complete-output oracle on every sample. The release build took 587.980 seconds separately; acquisition took 23.171 seconds within the 600-second budget. No samples were added after seeing the result.

| Case | Previous P50 | Candidate P50 | Candidate P95 | Target | Verdict | Complete oracle |
|------|-------------:|--------------:|--------------:|-------:|---------|-----------------|
| Interval 1M, one CPU | 355.067 ms | **282.396 ms** | 290.990 ms | ≤350 ms | **Passed** | 10/10; 10,998,080 rows |

The larger batches reduced median time by approximately 20.4% from the fused-projection candidate. Interval is approximately 3.81× faster than the original 1,076.113 ms baseline. These are descriptive medians, not paired significance estimates.

**Both primary Join gates are now met**: static Join 17.047 ms (≤30 ms) and interval Join 282.396 ms (≤350 ms). The next performance work moves to keyed SUM and the P4/P6 targets. Multi-batch hot-path inputs, durable hot-path restore, ASOF and the observed high-duplicate-right-side BandJoin pathology remain outside the completed acceptance scope and require separate work.
