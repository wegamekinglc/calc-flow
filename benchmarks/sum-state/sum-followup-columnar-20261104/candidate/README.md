# sum-followup-columnar-20261104: candidate

Status: **completed**.
Comparison: merged4f2 versus new profile-selected candidate.

| Case               | Baseline P50 ms | Candidate P50 ms | Change  | Verdict                 |
|--------------------|-----------------|------------------|---------|-------------------------|
| stock-stream-1cpu  | 280.088         | 266.919          | -4.70%  | no-confirmed-regression |
| stock-stream-8cpu  | 278.658         | 269.054          | -3.45%  | inconclusive            |
| sparse-stream-1cpu | 34.302          | 28.346           | -17.36% | improved                |
| stock-batch-8cpu   | 57.158          | 54.550           | -4.56%  | inconclusive            |

Two rounds of ten adjacent alternating pairs use the maintained paired-median
confidence intervals and 5% rule. All raw observations and full per-snapshot
oracles, schemas, hashes, native affinity/thread environment, lifecycle and
source/binary seals are retained. An inconclusive result proves no equivalence.

The common native diagnostic harness is 657de3e6 from public 84186554, overlaid
only in permanently frozen build trees. Product revisions are distinct from
instrumentation provenance; a historical 841 product is never used as 4f2.

Accepted current-stage profiles: 1.
Profiles use user-mode CPU-clock 999Hz and exact monotonic processing windows,
excluding the first of six jobs. Every imported per-thread timestamp is
verified against perf before attribution. Inclusive shares overlap; unresolved
or inlined symbols do not imply zero cost. No hardware bandwidth/caches or
off-CPU wait attribution is claimed. Profiles are never mixed into timings.

[summary.json](summary.json) contains all per-case intervals, CPU/RSS and
build/environment identities. [samples.csv](samples.csv) contains every formal
row. [complete-evidence.json.gz](complete-evidence.json.gz) retains every raw
warmup/formal/profile row, command, oracle digest and full source seal.
[Profile summary](profile-summary.json) and [targets](profile-targets.json)
preserve accepted stacks/counts and provenance.
[evidence-manifest.json](evidence-manifest.json) records the exact shared 600s
budget debit, hashes and stop status. Native builds are separately timed.

Stopped reason: None
