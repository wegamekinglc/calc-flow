# Grouped integer SUM state: measured results

This is the complete fixed four-case before/after collection on one WSL2 host.
All 160 formal observations passed the independent full per-snapshot oracle,
schema/hash checks, native affinity/thread checks and stream lifecycle checks.

## Elapsed time

| Case               | Baseline P50 ms | Candidate P50 ms | P50 change | Paired verdict          |
|--------------------|-----------------|------------------|------------|-------------------------|
| stock-stream-1cpu  | 362.127         | 340.849          | -5.88%     | no-confirmed-regression |
| stock-stream-8cpu  | 373.363         | 325.202          | -12.90%    | no-confirmed-regression |
| sparse-stream-1cpu | 37.558          | 36.025           | -4.08%     | no-confirmed-regression |
| stock-batch-8cpu   | 55.288          | 55.869           | +1.05%     | inconclusive            |

The P50 ratio and paired-difference median are different statistics. Verdicts
use the maintained two-round paired method and its existing 5% rule.

| Case               | Round | Paired median | Median interval   | Coverage  |
|--------------------|-------|---------------|-------------------|-----------|
| stock-stream-1cpu  | 1     | -6.74%        | [-15.15%, -0.67%] | 97.85156% |
| stock-stream-1cpu  | 2     | -5.63%        | [-7.41%, -0.58%]  | 97.85156% |
| stock-stream-8cpu  | 1     | -10.68%       | [-13.87%, -4.81%] | 97.85156% |
| stock-stream-8cpu  | 2     | -13.72%       | [-16.22%, -9.87%] | 97.85156% |
| sparse-stream-1cpu | 1     | -4.33%        | [-9.13%, +0.38%]  | 97.85156% |
| sparse-stream-1cpu | 2     | -4.08%        | [-7.32%, +0.57%]  | 97.85156% |
| stock-batch-8cpu   | 1     | +5.48%        | [-7.30%, +10.01%] | 97.85156% |
| stock-batch-8cpu   | 2     | -4.68%        | [-14.42%, +7.68%] | 97.85156% |

No extra cases, pairs, or timing-driven retries were added.

## CPU and whole-process memory

| Case               | Side      | Wall P95 ms | CPU P50 ms | Average cores P50 | Peak RSS P50 MiB |
|--------------------|-----------|-------------|------------|-------------------|------------------|
| stock-stream-1cpu  | baseline  | 385.024     | 362.105    | 1.000             | 231.54           |
| stock-stream-1cpu  | candidate | 368.208     | 334.407    | 1.000             | 232.94           |
| stock-stream-8cpu  | baseline  | 402.928     | 411.441    | 1.103             | 244.79           |
| stock-stream-8cpu  | candidate | 355.514     | 374.694    | 1.147             | 248.51           |
| sparse-stream-1cpu | baseline  | 40.239      | 37.556     | 1.000             | 70.50            |
| sparse-stream-1cpu | candidate | 39.639      | 36.023     | 1.000             | 71.73            |
| stock-batch-8cpu   | baseline  | 61.579      | 154.328    | 2.812             | 234.38           |
| stock-batch-8cpu   | candidate | 60.326      | 153.411    | 2.771             | 234.61           |

CPU time covers the primary processing interval; sampled peak RSS includes
IPC preload, all retained outputs and validation. It is not operator-only memory.

## Independent CPU profiles

| Side      | Samples | Unresolved leaf | Update | Finish | Commit | Expression | Other |
|-----------|---------|-----------------|--------|--------|--------|------------|-------|
| baseline  | 1785    | 23.75%          | 41.18% | 34.68% | 10.59% | 11.09%     | 2.46% |
| candidate | 1625    | 28.68%          | 44.00% | 26.65% | 14.15% | 12.49%     | 2.71% |

Profiles sample user-mode on-CPU work at 999 Hz with DWARF stacks. The first
of six jobs per side is excluded; five exact monotonic processing windows are
accepted after every imported per-thread timestamp matches original perf time.
The phase columns are mutually exclusive. Inclusive paths in profile-summary.json
overlap and must not be added. Missing stack symbols do not prove zero execution.
No off-CPU, memory-bandwidth, cache-miss or source-line attribution was measured.

## Workload and provenance

Dense input: 2,000,000 business rows (1,600,000 trades) plus 10,000 seed trades.
Sparse input: 10,000 seed groups and 20 batches of 100 rows; each update batch
contains 80 trades and 20 non-trades, with 100 distinct groups touched across
the full fixture and 80 trade groups per update batch. Stream outputs contain
21 complete 10,000-group snapshots; finite output contains one final snapshot.
The common harness retains every snapshot for post-timer correctness validation,
so these absolute times must not be compared to the older final-only diagnostic.

- baseline: `199d70b4156591153eb6b15cc2c8ffc238b3348f`, binary SHA256 `65c13e041bc26f49492fde6dbda8860ac2e6d07866c1ff3040e786895bf1efda`.
  Accepted Release build: 692.276 seconds.
- candidate: `0a728e8e2e001a4dcf18c23f64b08f525b3d140b`, binary SHA256 `baf20dfbe20c822f132fe4eb398a675e0ad8ad1833043342fe01c251f1380aec`.
  Accepted Release build: 584.620 seconds.
- Common harness SHA256: `657de3e697907e473691c2fd0d1b99967d46577333135c50176801f6faa4e04e`.
- Acquisition including preflight, fixtures, warmups, oracle, profiles and cleanup: 147.067 seconds.
- Superseded baseline build: 726.020 seconds; excluded from comparisons.

Builds are outside the 600-second measurement/analysis budget. Eight logical
CPUs correspond to four reported core IDs; WSL host load and power state are
unobservable. No native build or benchmark conflict was accepted by the guard.
Results cover this grouped integer SUM path and finite control, not the whole project.

## Complete outputs

- [All 160 formal observations](samples.csv), readable without decompression.
- [Statistics, resources, fixtures, environment and build identities](summary.json).
- [Profile phase/self/inclusive tables](profile-summary.json).
- [Every accepted baseline stack](baseline-stacks.json.gz) and [candidate stack](candidate-stacks.json.gz).
- [Complete evidence](complete-evidence.json.gz): every raw formal/warmup/profile row,
  invocation command/environment, resource receipt, all intermediate snapshot digests,
  plan, full source seals, compiler-log hashes and stderr.
- [Artifact hashes and measurement receipt](evidence-manifest.json).

Large raw perf captures remain under the recorded local collection paths, with
SHA256 and accepted windows in the committed evidence. Reproduce with the
[common sealed build and collection commands](../README.md#reproduction).

Read the full evidence using Python:

```python
import gzip, json
from pathlib import Path

data = json.loads(gzip.decompress(Path("complete-evidence.json.gz").read_bytes()))
print(data["acquisition"]["status"])
```

## Remaining costs and next implementation decision

The complete accepted stacks, rather than only the top-40 listing, give these
inclusive counts. A sample is counted once for each matching category, and
categories overlap. These percentages are CPU-sample shares, not wall time.

| Observed symbol path   | Baseline inclusive | Candidate inclusive |
|------------------------|--------------------|---------------------|
| candidate              | 291 / 16.30%       | 281 / 17.29%        |
| candidate_accumulators | 0 / 0.00%          | 0 / 0.00%           |
| scalar_drop            | 73 / 4.09%         | 77 / 4.74%          |
| group_drop             | 193 / 10.81%       | 235 / 14.46%        |
| row_converter          | 129 / 7.23%        | 129 / 7.94%         |
| row_encode             | 129 / 7.23%        | 129 / 7.94%         |
| hashing                | 55 / 3.08%         | 49 / 3.02%          |
| hash_map_any           | 113 / 6.33%        | 99 / 6.09%          |
| memory_reservation     | 0 / 0.00%          | 0 / 0.00%           |
| new_empty              | 0 / 0.00%          | 1 / 0.06%           |
| output_chunk           | 146 / 8.18%        | 130 / 8.00%         |
| merge_integer_history  | 0 / 0.00%          | 11 / 0.68%          |
| merge_rank             | 98 / 5.49%         | 93 / 5.72%          |
| finalize_candidate     | 217 / 12.16%       | 7 / 0.43%           |

`candidate_accumulators` and `MemoryReservation` have no separately observed
symbol in either capture; inlining prevents attributing their independent cost.
The candidate `new_empty` match is one Arrow `new_empty_array` sample, not proof
about memory-reservation construction. Detailed matching symbols and self counts
are retained in [profile-targets.json](profile-targets.json).

The targeted finalization path falls from 217/1785 (12.16%) to 7/1625 (0.43%)
of samples, consistent with using batch state instead of finalizing a scalar
accumulator per group. Overall candidate construction, old Group destruction,
row encoding and hashing remain visible. Their relative shares cannot establish
that their absolute cost increased after this change.

The next narrow implementation to evaluate is removing the retained integer
state copy in each candidate: `candidate()` still clones `previous.states[index]`,
`merge_integer_history` reads that copy, and `merge_rank` replaces it. Seeding
history from the immutable previous groups by stable slot could eliminate this
clone/drop work while preserving the existing DataFusion arithmetic. That needs
ownership/container-order, rollback, byte-budget and recovery proof plus a new
candidate measurement; these profiles do not quantify or promise its gain.
Composite-key encoding is a subsequent candidate, with 7.94% of candidate
samples in row conversion/encoding; hash and hash-table paths overlap with each
other and cannot be added as an independent total.

The current run does not establish a sustained improvement of at least 5% under
the maintained classification rule: streaming labels remain
`no-confirmed-regression`, and finite control is `inconclusive`. The dense
streaming median decreases are observations; both rounds' intervals are below
zero, but the 8-CPU first-round upper endpoint (-4.81%) narrowly misses the
stronger `improved` rule. Sparse timing does not establish a speedup. No new
samples or retries were added to change these outcomes.

Average processing use remains about 1.15 cores for the 8-worker stream case.
Group sharding and the finite filter/projection path were not implemented or
measured as new candidates in this collection. The evidence supports progressing
through the planned state-copy step before making a larger key/sharding change.
