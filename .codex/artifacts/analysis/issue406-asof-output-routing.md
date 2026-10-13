# Issue 406: route ASOF output matching by interned keys

The eight-target objective remains active. This increment follows PR #420's
prefix-counting work and the [routing specification](../specs/issue406-asof-output-routing.md).
The paired release comparison establishes a scoped single-CPU OFF improvement;
the parallel comparison is inconclusive and the eight-target objective is unmet.

## Profile and implementation

The ordinary-release `b77cd8bf` profile uses ASOF 1M rows per input, 64k batches,
true checkpoint OFF, and one / 32 CPUs. Its one warmup and five processing
windows per case complete in 11.765 seconds of the fixed 180-second inclusive
budget. All 12 observations pass the full oracle, lifecycle and cleanup checks;
there are no checkpoint files or surviving process groups. All 42 output seals
were independently checked. Lost lifetime events are 473 / 2, with unknown
window membership; WSL2 host isolation is unverified.

Admission accounts for 51.69% / 40.76% inclusive samples and output matching
27.77% / 32.17%. These shares overlap and are diagnostic evidence only. The
monotonic output matcher hashes an encoded key per row, despite left chunks
already storing dictionary IDs. A focused failing test records 1,026 hashes for
1,024 rows with one key and an empty right bucket.

Small chunk runs now resolve each dictionary key to a borrowed right bucket and
local monotonic cursor, then match rows through their integer IDs. Scratch is a
fixed 64-slot array. Routing requires at least four selected rows per dictionary
slot; legacy state, larger dictionaries and short/interleaved runs retain the
generic matcher. The existing generic cursor map and its funding are unchanged.
Each local cursor starts at its run's first time; a later generic run can catch
up across rows processed through local routes. Output collection still checks
cancellation and source funding before selecting a candidate, and keeps the
same per-1,024-row yield boundaries. No stored state, checkpoint format,
ownership accounting, caller buffer or default runtime memory cap changes.

## Verification

The counter regression passes after the implementation. Differential checks
compare real payload and identity-only matches with binary search, complete
output plans, canonical rowwise ownership, projected inventory and journals.
They cover 1/63/64/65-key dictionaries, nonzero prefix consumption, retired
slots, missing matches, canonical string sequences, overlapping chunks and
legacy state. An explicit 201/1/1/1/318 run sequence verifies routed → generic
→ routed matching against the same oracle.

The new scenarios are named async helpers called by the existing matching
ownership regression. Rust 1.88 synthesizes an array of test pointers in its
test entry point; the core's existing 2,048 registrations reach Clippy's 16 KiB
array threshold exactly on this platform. A minimal standalone reproduction
passes at 2,048 empty tests and fails at 2,049 with the same generated-span
diagnostic. Consolidation retains every scenario and assertion without changing
the pinned toolchain, lint levels or threshold. This increment adds no test
registrations. Logs and the isolated reproduction are retained under
`target/issue406-asof-output-routing/`.

The final affected run passes all 341 registered ASOF tests in 26.27 seconds.
Scoped core/library/test Clippy with all features passes in 86 seconds.
Formatting, the changed modules' complexity limit of eight, generated-contract
drift and whitespace checks also pass. Final specialist source review is Approve
at implementation commit `5aa03e54`. Performance and corrected-base CI remain
pending; the source review does not certify a latency improvement.

PR #420 merged at `ac5d7dffabe098d33ee9c07c912d3c6c63898552` after all 32
final checks passed and no unresolved review threads remained. The routing
branch now uses that exact main, including its independent version-release and
Kafka-test updates. Range-diff confirms all five reviewed patches are unchanged;
the implementation is now commit `c7b964bc`. Earlier test timings retain their
original implementation identity, while new ordinary release builds must use
the aligned revisions. Required CI on the routing branch remains pending.

## Paired release measurement

The pre-acquisition plan freezes exact merged main including PR #420, the
reviewed routing candidate, ordinary production wheels and maintained execution
harness. Earlier measured binaries keep their original commit identities.
Both arms must match compiled dependency features/profiles, environment,
compiler and normalized lock identity; benchmark internals remain absent.
Estimated release builds are 6–14 minutes per arm, recorded separately.

| Case                        | Connection to this increment                         |
|-----------------------------|------------------------------------------------------|
| ASOF 1M/input, 64k, OFF, 1  | Direct monotonic match path                          |
| ASOF 1M/input, 64k, OFF, 32 | Parallel matching and runtime regression control     |
| ASOF 1M/input, 1024, OFF, 1 | Repeated small-run routing and fallback overhead     |
| ASOF 1M/input, 64k, LOW, 1  | Same matching path with checkpoint interval 24 hours |
| ASOF 1M/input, 64k, ON, 1   | Same path with real advancing processing checkpoints |

The plan fixes two rounds of ten adjacent AB/BA pairs and one warmup per arm per
round, with 600 seconds total inclusive of audit, preflight, fixtures, warmup, full
oracles, process startup, lifecycle settlement and sealing. Stop at the budget
or competing work; do not expand counts or retry for a favorable verdict.
Keep OFF zero-file checks, LOW's actual checkpoint evidence and ON's recomputed
windows backed by advancing nonterminal manifests. Preserve the maintained
paired regression rule and report inconclusive results accurately.

### Identities and completed cases

The 2026-10-13 acquisition uses exact main `ac5d7dff` and candidate
`6739fa2b5573f4e184266793fd9ee4d4281a5b14`, whose product change is
`c7b964bc`. Baseline reuses its attested ordinary wheel, native SHA-256
`ac9686cd24c69199c969ff61ed4ed8af6d3d37c849f42d786801e4c7f122f6b6`;
candidate is freshly rebuilt, native SHA-256
`a131c569adaa5fdb629e868be6af42ca0e5a3e31359694a37a39116930340011`.
Recorded builds take 330.173 / 425.812 seconds, separately from acquisition.
Both products pass the 294-test ordinary artifact audit and compiled
feature/profile identity checks. Benchmark-internal features are absent.

The sealed execution-plan SHA-256 is
`89f1292710be0722a102fe811f764cebfe714cc4c82ad24cd68a55df0f6df394`.
Artifacts live under `target/issue406-routing-current-main-20261013/`, separate
from earlier profiles and comparisons. The maintained harness comes from main
`ac5d7dff`, with identical sealed CPU-accounting instrumentation in both arms.
It brackets the existing processing clock with process CPU and per-thread
`/proc` reads; the thread grouping/window helpers match issue #406 Appendix A.
These reads have boundary uncertainty and cannot recover retired threads.
They do not change native product code or the reported processing-clock bounds.

Four cases complete both rounds of ten adjacent AB/BA pairs: 160 measured
observations and 16 warmups. The acquisition stops during ON qualification,
after two ON warmups and one accepted baseline measured observation. Its first
candidate observation passes the full output oracle, lifecycle and cleanup but
lacks retained processing-period manifest proof; it is preserved and rejected.
All 180 raw observations remain sealed, without filtering, retries or increased
sample counts. Inclusive acquisition and sealing take 96.049 / 600 seconds;
all process groups settle, with no forced kills, survivors or cleanup errors.
An independent audit verifies all 400 acquisition-output seals and 3,100 sealed
harness/product inputs, and recomputes the four completed comparisons.

| ASOF case         | Main p50 ms | Candidate p50 ms | Change  | Maintained paired verdict |
|-------------------|-------------|------------------|---------|---------------------------|
| 64k, OFF, 1 CPU   | 126.390     | 112.488          | -11.00% | improved                  |
| 64k, OFF, 32 CPUs | 165.433     | 169.663          | +2.56%  | inconclusive              |
| 1024, OFF, 1 CPU  | 653.423     | 632.759          | -3.16%  | no-confirmed-regression   |
| 64k, LOW, 1 CPU   | 146.811     | 126.314          | -13.96% | no-confirmed-regression   |
| 64k, ON, 1 CPU    | —           | —                | —       | incomplete                |

The two single-CPU OFF round changes are -9.80% and -10.33%, with intervals
[-12.75%, -5.39%] and [-15.82%, -8.06%]. Parallel changes are +3.60% and
+3.45%, with intervals [+0.79%, +29.51%] and [-1.00%, +5.49%]; this does not
establish equivalence or a confirmed regression. Small-batch intervals are
[-6.36%, +3.50%] and [-8.82%, +4.64%]; LOW intervals are [-19.25%, -5.07%]
and [-19.22%, -2.38%]. The maintained order-statistic intervals have 97.85%
coverage for ten pairs. The rule requires both rounds' upper bounds below -5%
for `improved`, so LOW retains its original `no-confirmed-regression` result.

Processing CPU medians, in milliseconds, provide context for further diagnosis:

| Case             | Main process CPU | Candidate process CPU | Main busiest thread | Candidate busiest thread |
|------------------|------------------|-----------------------|---------------------|--------------------------|
| 64k, OFF, 1 CPU  | 126.200          | 112.333               | 116.401             | 102.895                  |
| 64k, OFF, 32 CPU | 190.132          | 193.803               | 53.176              | 60.642                   |
| 1024, OFF, 1 CPU | 651.543          | 631.666               | 477.867             | 462.846                  |
| 64k, LOW, 1 CPU  | 142.270          | 125.509               | 129.353             | 113.562                  |

Per-group run and runqueue-wait times remain in each raw observation and the
independent audit. Process CPU and the sum of surviving-thread run times have
slightly different read boundaries; they are diagnostic rather than an exact
thread-accounting identity. WSL2 host isolation remains unverified.

### ON evidence retention and harness correction

The rejected candidate's processing window lasts 250.055 ms. Epoch 1 completes
wholly inside it, but post-EOF inspection retains only epochs 2 and 3; epoch 2
completes after the processing timer and epoch 3 is terminal. Default runtime
retention is two epochs. Status alone cannot reconstruct epoch 1's deleted
manifest or source progress, so the stopped acquisition has no ON comparison.

The subsequent harness correction copies actual committed manifest bytes after
the unchanged processing timer and before EOF, using an owned `to_thread`
reader. Its separate `processing_checkpoint_snapshot` records exact JSON,
SHA-256, source cursors, captured committed-epoch bound and capture timestamps.
Final `checkpoint_evidence` continues to describe only files retained after
settlement. Advancing proof still requires a nonterminal manifest, positive
source rows and completion wholly within the original processing window.
Reading a manifest after that window never promotes its completion time.
Concurrent deletion records missing evidence without retry. Cancellation waits
for the reader before job settlement and directory cleanup. Default retention,
checkpoint interval, source scheduling during processing and processing-clock
boundaries remain unchanged; snapshot I/O counts toward full-cycle and
acquisition time.

The focused regression first fails with the absent snapshot helpers; the final
affected Python module passes 30 tests. Tests cover terminal retention, no rows,
outside-window completion, uncommitted/terminal manifests, tampered proof,
capture boundaries, concurrent deletion, EOF ordering and owned cancellation.
This correction has no new timing acquisition and does not repair or relabel the
stopped raw evidence. A future ON acquisition requires a fresh frozen harness
identity and a separately fixed case/sample/budget plan.

## Original acceptance inventory

| Workload                       | Single-thread target         | Parallel target | This increment            |
|--------------------------------|------------------------------|-----------------|---------------------------|
| Sealed Join 1M                 | ≤60 ms, OFF                  | ≤15 ms, OFF     | Not measured              |
| Sealed Join 10M                | ≤600 ms, OFF                 | ≤150 ms, OFF    | Not measured              |
| Retained Join 4M / 1M          | ≤4.4×, OFF                   | —               | Not measured              |
| Join capture / restore per row | ≤0.1 / ≤0.2 µs, enabled      | —               | Not measured              |
| Join compaction handler ratio  | ≤2× steady, enabled          | —               | Not measured              |
| ASOF 1M per input              | ≤100 ms, OFF                 | ≤30 ms, OFF     | 112.49 / 169.66 ms; unmet |
| ASOF 1024-row harness overhead | ≤2× matched projection floor | —               | Not measured              |
| Full Join 1024-row callback    | ≤50 µs                       | —               | Not measured              |

LOW/ON throughput targets remain ≥90%/75% of the corresponding OFF **goal**,
equivalent to ASOF latency budgets of approximately ≤111.11/33.33 ms and
≤133.33/40 ms at one/32 CPUs. These are targets, not results. Row-6 small-batch
timing cannot establish row-7 matched-harness overhead or a full Join callback.
