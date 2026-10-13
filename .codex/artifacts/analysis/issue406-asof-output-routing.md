# Issue 406: route ASOF output matching by interned keys

The eight-target objective remains active. This increment follows PR #420's
prefix-counting work and the [routing specification](../specs/issue406-asof-output-routing.md).
It does not establish a latency gain before paired release measurement.

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
drift and whitespace checks also pass. Final specialist review is pending.

## Planned paired measurement

Before acquisition, freeze exact merged main including PR #420, the reviewed
routing candidate, ordinary production wheels and the maintained execution
harness in the sealed execution plan. Do not relabel earlier measured binaries
as these commits. Both arms must match compiled dependency features/profiles,
environment, compiler and normalized lock identity; benchmark internals remain
absent. Estimated release builds are 6–14 minutes per arm, recorded separately.

| Case                        | Connection to this increment                         |
|-----------------------------|------------------------------------------------------|
| ASOF 1M/input, 64k, OFF, 1  | Direct monotonic match path                          |
| ASOF 1M/input, 64k, OFF, 32 | Parallel matching and runtime regression control     |
| ASOF 1M/input, 1024, OFF, 1 | Repeated small-run routing and fallback overhead     |
| ASOF 1M/input, 64k, LOW, 1  | Same matching path with checkpoint interval 24 hours |
| ASOF 1M/input, 64k, ON, 1   | Same path with real advancing processing checkpoints |

Fix two rounds of ten adjacent AB/BA pairs and one warmup per arm per round,
with 600 seconds total inclusive of audit, preflight, fixtures, warmup, full
oracles, process startup, lifecycle settlement and sealing. Stop at the budget
or competing work; do not expand counts or retry for a favorable verdict.
Keep OFF zero-file checks, LOW's actual checkpoint evidence and ON's recomputed
windows backed by advancing nonterminal manifests. Preserve the maintained
paired regression rule and report inconclusive results accurately.

## Original acceptance inventory

| Workload                       | Single-thread target         | Parallel target | This increment |
|--------------------------------|------------------------------|-----------------|----------------|
| Sealed Join 1M                 | ≤60 ms, OFF                  | ≤15 ms, OFF     | Not measured   |
| Sealed Join 10M                | ≤600 ms, OFF                 | ≤150 ms, OFF    | Not measured   |
| Retained Join 4M / 1M          | ≤4.4×, OFF                   | —               | Not measured   |
| Join capture / restore per row | ≤0.1 / ≤0.2 µs, enabled      | —               | Not measured   |
| Join compaction handler ratio  | ≤2× steady, enabled          | —               | Not measured   |
| ASOF 1M per input              | ≤100 ms, OFF                 | ≤30 ms, OFF     | Planned        |
| ASOF 1024-row harness overhead | ≤2× matched projection floor | —               | Not measured   |
| Full Join 1024-row callback    | ≤50 µs                       | —               | Not measured   |

LOW/ON throughput targets remain ≥90%/75% of the corresponding OFF **goal**,
equivalent to ASOF latency budgets of approximately ≤111.11/33.33 ms and
≤133.33/40 ms at one/32 CPUs. These are targets, not results. Row-6 small-batch
timing cannot establish row-7 matched-harness overhead or a full Join callback.
