# P5 multi-batch hot Join recovery and terminal acceptance

Runtime `bf26a02f` added live multi-batch continuation and native recovery, but its first sealed wheel failed the first static warmup after emitting 1,000,000 rows. The EOF/checkpoint failure produced zero valid samples and interval was not started. That plan, failure and build receipt remain in `bf26a02f-plan.json`, `bf26a02f-failure.json` and `bf26a02f-build.json`; they are not overwritten by the replacement result.

Runtime `d6e49ddc` releases hot history for terminal checkpoints instead of rebuilding obsolete run state. Live and non-terminal recovery retain atomic migration. Six focused tests cover same-side and post-completion batches, pending-side and completed native disk recovery, static/BandJoin fused output, static match caps and million-row terminal cleanup. All 599 Join tests and all-feature Clippy pass.

The replacement sealed wheel was built from clean source `0c7b4ee7` in 611.470 seconds. Its wheel SHA-256 is `150967d22ab53e9b1ac384dc54aafaf73b475a8a8d1e5e73f96f48d7be80324f`; the loaded native SHA-256 is recorded in `result.json`. Acquisition used one CPU, two warmups and ten measured samples per case with complete output oracles, and finished in 27.406 seconds within the 600-second budget.

| Case | Previous P50 | Candidate P50 | Candidate P95 | Target | Verdict | Complete oracle |
|------|-------------:|--------------:|--------------:|-------:|---------|-----------------|
| Join 1M, one CPU | 17.047 ms | **16.417 ms** | 16.675 ms | ≤30 ms | **Passed** | 10/10; 1,000,000 rows; zero error |
| Interval 1M, one CPU | 282.396 ms | **307.755 ms** | 327.228 ms | ≤350 ms | **Passed** | 10/10; 10,998,080 rows; zero error |

The static median is about 3.70% lower and the interval median about 8.98% higher than their previous unpaired observations. These are descriptive cross-run differences, not paired significance or equivalence results. No samples were added to seek a better interval median. Multi-batch throughput, ASOF and the remaining acceptance matrix are not measured here.
