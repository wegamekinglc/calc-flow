# P5 projected hot Join experiment

Candidate `2983e787` fixed the production activation defect: output and retained-layout pruning no longer disable `HotJoinCore`. Full and projected hot-path regressions, 28 shared DataFusion Join tests, 589 production Join tests and diagnostic-feature Clippy passed.

The sealed default-feature release build took 1,181.720 seconds separately. The formal acquisition used the same one-CPU, two-warmup/ten-sample, full-oracle protocol as the previous candidates, with a 600-second total budget and a 570-second acquisition deadline.

**No valid performance sample exists.** The first static Join warmup did not complete before the 570-second deadline. The harness stopped with `TimeoutError`, wrote no case result and never started interval acquisition. The result is within the external budget; there is no P50/P95 and no oracle claim for this candidate. Raw provenance is in `result.json`.

This rejects the temporal SQL physical plan for the primary static workload. The next implementation removes that SQL and uses a direct per-key hash index, sorted temporal entries and binary-search bounds followed by Arrow `take`. This is a new candidate, not a timing retry of `2983e787`.
