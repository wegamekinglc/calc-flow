# P5 fused Join projection

Candidate `5e7c22db` fuses exact single-consumer Join projections into native final outputs. Static Join emits `sequence/value` directly from the matched rows. Interval uses the native BandJoin `sequence/right_sequence/value` output. Downstream Expressions become identity projections; no intermediate price/factor columns or second multiplication remains.

The baseline is sealed candidate `7ba44180` from `../p5-native-temporal-index-20261121/result.json`. Both cases used one CPU, two warmups and ten measured samples, with the maintained complete-output oracle on every sample. The release build took 739.080 seconds separately; acquisition took 28.786 seconds within the 600-second budget. No samples were added after seeing the result.

| Case | Previous P50 | Candidate P50 | Candidate P95 | Target | Verdict | Complete oracle |
|------|-------------:|--------------:|--------------:|-------:|---------|-----------------|
| Join 1M, one CPU | 39.336 ms | **17.047 ms** | 17.616 ms | ≤30 ms | **Passed** | 10/10; 1,000,000 rows |
| Interval 1M, one CPU | 363.002 ms | **355.067 ms** | 398.374 ms | ≤350 ms | Failed by 5.067 ms | 10/10; 10,998,080 rows |

Static Join is approximately 23.46× faster than the original 399.845 ms baseline; interval is approximately 3.03× faster than its 1,076.113 ms baseline. These are descriptive medians, not paired significance estimates. **The static absolute gate is now met; interval remains unmet.**

The next experiment raises the native BandJoin output batch size from DataFusion's default 8,192 rows to 65,536 rows, matching the existing native executor cap and the static hot-path chunking. This targets per-batch downstream and Arrow conversion overhead without changing match semantics. Multi-batch hot-path inputs and durable hot-path restore remain unsupported and unmeasured.
