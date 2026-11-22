# P5 native temporal index and projected BandJoin

Candidate `7ba44180` replaces the rejected temporal SQL with a direct per-key hash index, sorted right-side temporal entries and checked inclusive-bound binary searches, followed by projected Arrow `take`. Interval uses native BandJoin with only the four columns required downstream. Production output and retained-layout pruning remain enabled.

The baseline is sealed candidate `705adc7f` from `../p5-key-run-cache-20261120/result.json`. The failed activation and temporal SQL candidates are retained in the plan; this is a new implementation, not a timing retry. Both cases used one CPU, two warmups and ten measured samples, with the maintained complete-output oracle on every sample. The release build took 986.940 seconds separately; acquisition took 37.375 seconds within the 600-second budget. No samples were added after seeing the result.

| Case | Baseline P50 | Candidate P50 | Candidate P95 | Target | Complete oracle |
|------|-------------:|--------------:|--------------:|-------:|-----------------|
| Join 1M, one CPU | 399.845 ms | **39.336 ms** | 73.729 ms | ≤30 ms | 10/10; 1,000,000 rows |
| Interval 1M, one CPU | 1,076.113 ms | **363.002 ms** | 451.309 ms | ≤350 ms | 10/10; 10,998,080 rows |

The descriptive medians are approximately **10.17×** and **2.96×** faster, with 90.2% and 66.3% less time. These are not paired significance estimates. **Both absolute gates remain narrowly unmet**: Join is 9.336 ms above its target and interval is 13.002 ms above its target.

This is the first validated substantial end-to-end improvement from the architecture change. The next experiment is Join+projection fusion: static Join should directly produce `sequence/value`, and interval should directly produce `sequence/right_sequence/value`, avoiding the remaining physical columns and downstream arithmetic without bypassing the expression contract. Multi-batch hot-path inputs and durable hot-path restore remain unsupported and unmeasured.
