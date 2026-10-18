# DAL-383: columnar state admission

Implementation base: `main@2ed52443222781023b7d371fc29bb7b1172ad7c9`.
The issue's revised plan and sealed G0 evidence determine the order of work.
The 40 ms Join target remains open; neither that target nor the output-path
non-goal is changed by this implementation.

## Implemented paths

Window accumulator rows and closed-window checkpoint marks share immutable
group values. Existing groups reuse their owner across batches, including
long strings, composite groups and null values. Aggregate scratch remains
owned by the current batch. Checkpoint encoding still writes the same values.

Join binds incoming segment metadata once per group or sequential traversal.
Dense admission appends bounded blocks of times and retention bits. Key hashing
dispatches each typed column once per bounded block, reuses stack hash scratch,
and specializes single-column equality. Full typed equality resolves collisions;
canonical framing and persistent identity resolution remain per distinct key.
Flat admission fees sum fixed widths once and then add variable offsets.

Local hashing charges each fixed-width column block before reading values;
Utf8/LargeUtf8 hashing and equality charge at most 4 KiB before processing
each chunk. Composite columns share the same 64-visit/4-KiB quantum. Integer
slots make every collision probe resumable and chargeable before its work,
including probes whose full hash differs. Small equality retains its typed
dispatch only when the entire comparison fits one quantum. The slot array
fits the existing conservative interning allowance; refusal/fallback and the
scratch lifetime are unchanged. Long-key cancellation and deadline regressions
observe processed bytes, with an end-to-end check for unchanged committed state.

Short-key probes and comparisons debit that prepaid allowance synchronously;
only exhausted credit or wide comparisons enter the async continuation. Hash
loops likewise await only when more credit is needed. Failed debits leave both
counters untouched, and the same shared byte/visit limits cover fast and slow
work. Forced collision tests check real cooperative suspension and continuation
through this transition, preserving first-appearance IDs and retained counts.

The funded native retained path prepares selection and logical charges, borrows
the existing `NativeKeys` identity column, and writes owned chunks directly.
There is no batch-sized `Vec<StoredRow>` on this path. The temporary selection
credit drops before output preflight. Local hash scratch still drops at the end
of dictionary preparation, before matching. Persistent chunks use their own
necessary row-to-key slots and one owner per distinct key in each chunk; these
slots are never serialized or reused as admission identities.

Native index append resolves each distinct chunk key once. Surviving stable
coordinates do not move. Expiration planning caches its current chunk when
reading time, ID, charge and key metadata. Capture continues to derive dirty
rows from the allocation frontier and existing lazy delta state. Native v2/v3
restore validates immutable histories and fills chunks directly; layout-1 and
unprojected-to-projected migration keep their existing compatibility path.

## Boundaries

- Input buffers and `Batch` values remain read-only. Retained payload copying
  and projection finish before chunk installation; state does not acquire a
  source-buffer ownership exception.
- Logical state charges, sparse physical row IDs, output order, match limits,
  capture frontier, lineage and fingerprint checks keep their existing rules.
- Chunk containers, key tables, selection buffers and overlapping owners are
  funded before allocation. Required chunk funding still fails closed. The
  legacy optional funding qualification is unchanged.
- Last-key removal releases the chunk's owner. Dictionary namespace leases
  remain authoritative, including after restore and slot recycling.
- Join layouts 1/2/3 and Window layouts 1/2 keep their wire formats. No public
  Rust, PyO3, Python, configuration or Studio interface changes are needed.
- Join output preflight, gather, materialization and matching algorithms remain
  outside this change. A faster incoming accessor is not an output redesign.

## Reuse outside Join

| Path          | Existing implementation                                               | Decision                                                                                  |
|---------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Window        | Existing-slot scratch copied group scalars every batch                 | Share values now; independently required by the issue                                      |
| ASOF          | `asof/admission.rs` already caches dictionary IDs and batches key hashes | Keep its identity/sequence ordering and workspace limits; no measured additional gain here |
| Rolling       | `rolling/ordered_stream.rs` buffers proven ordered Arrow records        | Keep its ordered fast path and row fallback; Join chunks do not match window history       |
| Cross-section | Complete groups own duplicate evidence and pending columnar mean input | Keep group finality and ownership; no evidence that the Join interner is its bottleneck    |

The Join hash implementation remains private until another operator has both a
compatible identity contract and evidence of a repeated cost. Cross-record
deduplication is also deferred: the acceptance workloads contain one record per
event, so their measured benefit cannot justify changing that scope.

## Verification and measurement

Focused validation covers Window sharing/close/checkpoints and Join admission,
typed identity equivalence, sparse IDs, budget refusal, cancellation, eviction
and recovery. Existing assertions remain authoritative; the direct preparation
path additionally checks that it constructs no retained descriptor views.

Candidate measurements must use a clean matching commit, the sealed G0 wheel,
the locked dependency environment and the original two 1M catalog cases. Keep
64,000-row batches, 64 keys, source modes, watermarks, output oracles and timing
scope. Build time is recorded separately. The maintained two-round, ten-sample
AB/BA rule governs comparison; descriptive latency is not a new interpretation
of the 40 ms gate. Raw results, actual commands and remaining costs belong in
the PR and its evidence attachment. Final full profiling follows focused review
under the coordinator's existing handoff.

The sealed `9440f561f1b11b8c598639212c053c935683a49a` wheel and its profiles
precede the cooperative-work fix above. They remain reusable historical
evidence, not measurements of subsequent commits. The repair receives only
targeted interning/stop checks before blocker review; final-commit measurement
remains with the coordinator.
