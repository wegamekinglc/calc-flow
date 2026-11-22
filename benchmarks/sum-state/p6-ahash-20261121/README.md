# P6 AHash routing — correctness failure, no performance verdict

Candidate `7f61738b` was sealed at source HEAD `026f8cc5`, with binary SHA-256 `58dacbf8d76aa7c84f79a94bd55c9aeeb1ddec940f82a9fecac60c2bc5400810`; the release build took 1,009.139 seconds. The P6 driver stopped after 5.611 seconds when the first candidate warmup failed in `positions`. **No valid timing sample, P50, speedup or complete oracle exists.** The other three cases and P4 were not measured. Retained files are `summary.json`, the header-only `samples.csv`, `environment.json`, `warmup-command.json` and `warmup-stderr.log`; the oversized acquisition dump was removed during PR cleanup.

A focused regression reproduced the identity defect directly: after 10,000 initial stock groups, a 100,000-row continuation produced 10,024 output groups. Comparing two batches also found different hashes for identical account/stock text. Replacing fixed seeds with `AHasher::default()` did not fix the mismatch. The candidate was therefore withdrawn rather than retimed or patched speculatively. The regression remains with the restored FNV implementation and requires exactly 10,000 groups.

The earlier small shared-key, SUM, SQL and Join suites passed but did not cover this scale and continuation shape. Passing those tests is not evidence of production correctness or performance for this candidate.
