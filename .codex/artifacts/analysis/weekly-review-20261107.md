# Weekly implementation review and performance controls

The [PR 556 follow-up](pr556-repair.md) records integration with newer main,
cleared CI findings, exact-wheel verification and two later resource preflight
stops. The material below preserves the original review checkpoint.

The review covers 36 mainline commits from 2026-10-31 00:00 Asia/Shanghai
through main `1b91bcc00a0623d5ece37ac654bf6e2a4f045939`. The preceding mainline
commit is `ce313488017d63fc626731f9731a4bec931fc343`. Scope and fixed controls
are recorded in the [specification](../specs/weekly-review-20261107.md);
the [independent review](../reviews/weekly-review-20261107.md) records source
coverage and approval limits. The original dirty stock-comparison worktree
was preserved.

## Repairs

- Native ASOF/interval SQL admitted nested CTE shadowing and column aliases
  without implementing their bindings, returning incorrect rows or values.
  Those forms now use DataFusion. Canonical admitted queries retain the native
  path, and a duplicate equality matcher was removed.
- Native joins could report the preceding query's rolling rewrite audit.
  Each native query now starts with its own empty audit.
- Compact ASOF memory accounting could charge the most recently registered
  source instead of the prefix's actual source. It now uses the source held
  by the compact positions, preserving batched charging.
- Materialized Python sequences could lose zero-column rows or supplied empty
  RecordBatch boundaries; later list mutation could replace captured inputs.
  The exceptional Arrow shapes now use a reader, ordinary nonempty tables
  retain concatenation, and sequence items are captured at construction.
- Validation errors could not survive copy/deepcopy/pickle. Reconstruction
  retains structured error data, notes and custom diagnostic state.
- Benchmark cleanup could skip the second worker when closing the first
  raised. Every worker is now settled while retaining the primary failure.
- Experimental input partitions could reuse a P1 physical template for P4,
  both after changing the setting and after changing input batch count.
  Template lookup and capture now apply only to P1, which is the layout that
  rebinding supports. Experimental P4 replans; default P1 reuse remains.

Every behavior repair has an observed failing regression before its fix.
SQL joins pass 28 focused tests, ASOF output planning passes 15, the final
partition module passes seven, and benchmark cleanup passes eight. Scoped
core Clippy, Rust formatting, focused Python lint and schema/whitespace checks
pass. Both complete changed Python modules pass against the exact candidate
release wheel: 93 tests in 1.32 seconds. Full regression, connector coverage
and cross-platform checks belong
to CI; local omissions do not establish the 90% line floor.

## Performance acquisition

Baseline source is the captured main above. The candidate prepared for comparison
is `7dbd229d23fb4060c1209f518ef08bee8c82c508`, tree
`93ac3f2ea6b3090b9e1627452fe6255b78912d83`. Both wheels are built in release
mode from clean isolated sources, with identical Cargo.lock and build options.
The initial candidate `240611ff` was withdrawn before any observation after
review exposed the changing-input P4 defect; its first cache-copy/build
preparation also overlapped and was discarded. Its source and failed
preparation receipts are retained, not used as measurements.

The fixed selection is canonical finite ASOF SQL (1,000 rows), canonical
finite interval SQL (1,000), projected fixed-width stream ASOF (10,000,
1,024-row batches, checkpoint off), and the public list-fed Program projection
lifecycle (1,000 rows split 640/360). The last case checks every output value,
name and boundary, plus terminal task cleanup.

Each case uses the maintained collector and two rounds of ten alternating
AB/BA pairs. Each observation is a fresh process with one warmup and three
timed samples reduced to their median. The full campaign has 160 observations,
160 warmups and 480 timed samples. CPU affinity is 0 for workers and 31 for
the controller/supervisor; Tokio, Arrow and external pools use one thread.
The 600-second acquisition includes startup, fixture/oracle work, installation,
validation, statistics and cleanup. Native builds are reported separately.
No retries or expanded samples are permitted.

**Performance was not measured; this change is not merge-ready.** After both
owned builds and all local tests completed, an unrelated `calc-flow-mt-impl`
release-test compilation was still active. It finished one cycle and started
another. Following the repository's unsuitable-environment stop rule, no
worker, warmup or timed sample was started, and no external process was
modified. The frozen protocol is retained for a separately resumed acquisition
in a suitable environment. There was no scientific retry and no timing verdict.

Baseline and candidate builds took 1,127.10 and 1,028.57 seconds respectively,
outside the acquisition budget. Candidate wheel SHA-256 is
`2cc87049517cd08bb51013bf384b94af2277017dff3b13efe2d1faa88e65550d`;
its native module SHA-256 is
`97955f68501bdcdcd31f9d96e7f46ed6240e8e1d8bda8377f07b9017a54d75fe`.
The exact-wheel tests verify that imported Python and native modules come
from that installed artifact. Build identities, test logs, the frozen
protocol, local adapters and environment-stop receipt are retained in the
[evidence archive](weekly-review-20261107-evidence.tar.gz), with a
[per-file digest index](weekly-review-20261107-evidence-index.json).

The maintained joint two-round 5% regression rule applies unchanged.
Inconclusive timings do not establish equivalence or a speedup. Corrected
CTE/alias results cannot use their previous wrong-result latency as an
equal-semantics baseline. Nullable accounting has exact byte/budget tests;
the normal ASOF benchmark uses nonnull values. Experimental P4 replanning is
outside these ordinary production controls, and future partition studies
must distinguish planning from execution cost. WSL host load/power cannot
be fully observed from the guest.

## Next optimization plan

These are proposals, not changes included in the prepared candidate. Each
requires a separate fixed, bounded comparison before adoption; none has a
promised percentage gain.

1. **Cache immutable SUM key-output charges.** First profile the exact charge
   subphase. Then cache the committed total and accumulate candidate charges
   during admission, preserving the conservative double charge for touched
   keys, overflow checks, rollback and cancellation. Derive the cache during
   restore without changing the checkpoint format. Controls: sparse updates
   against large state, dense 1T and dense 4T, plus exact/one-byte-short budget,
   restore and held-snapshot correctness. Historical transaction-finishing
   samples include other work and do not predict this proposal's gain.
2. **Construct interval SQL output with typed Arrow builders.** Eliminate
   three output-sized optional-value vectors and their subsequent conversion
   passes in `BandState::next_batch`. Check nulls, order, duplicates and chunk
   boundaries. Measure dense/high-output, non-dense, small and sparse/short
   final batches. A full-capacity builder can retain excess Arrow buffers;
   acceptance must include retained bytes and RSS, not just allocation count.
3. **Borrow interval ranges for output views.** Replace repeated full-range
   scans and owned copies across output chunks with borrowed ranges, explicit
   bounds and a cursor or binary search. Include nested-slice correctness:
   a private double-slice counterexample was found, with no current production
   caller identified. Controls: fragmented multi-chunk fanout, one chunk and
   noncontiguous fallback, with multiplicity, cancellation and allocation gates.

Dense-band cursor suppression, contiguous ASOF slice reuse, early projection
and single-pass materialized sequence preparation are secondary candidates.
Buffer retention and preparation overhead can offset local savings. The
previous ranked SUM, column-dictionary and CASE experiments should not be
reintroduced from component-only evidence.

Adopt a proposal only after its correctness/resource gates pass and its
declared target demonstrates the required benefit without a confirmed
maintained regression in its controls. Retain inconclusive outcomes as such;
do not increase sampling until a favorable classification appears.
