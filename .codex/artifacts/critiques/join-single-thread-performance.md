# Join Single-Thread Performance - Critic Critique

## Target

- Spec: `.codex/artifacts/specs/join-single-thread-performance.md`.
- Original review: the original worktree's
  `.codex/artifacts/reviews/pr482-single-thread-performance.md`.
- Source examined: current-main implementation based on `78b75724`, particularly
  join preparation, eviction, selection assembly, and ASOF admission/output.
- Governing contracts: `docs/introduction.md`, sections **The basic vocabulary**
  and **Supported boundaries**; `docs/asof-join-guide.md`, identity, payload,
  ownership/accounting, and finalization contracts.

## Verdict

**Proceed with caveats** for the bounded residual changes below. Preserve the
existing transaction, state representation, and grouped/forward-window paths.
Correct the scope statements identified here before claiming complete acceptance;
the historical review does not justify replacing already-optimized machinery.

## Findings

### Blocking Issues

None for the narrower implementation described in this critique. A proposal to
split one input into independently committed admissions, replace sparse stable
coordinates with a dense historical-span allocation, or change checkpoint
identity would require a new review and is not approved here.

### Significant Concerns

- **FR1 demands an unbounded owner cache without identifying its funding.** The
  reviewed source-level inspection can use a bounded previous-source cache, but
  FR1 currently requires reusing every source/projection after arbitrarily many
  intervening sources. That implies storage growing with distinct sources; it is
  unnecessary for the demonstrated single-owner hotspot. The original review
  explicitly permits caching by segment. A per-side last-source cache is
  approved: the one-source-per-side workload performs at most two inspections;
  alternating owners may inspect again at source transitions. Borrowed immutable
  owner identity must distinguish slices and the selected projection.
  - **Suggested fix:** State the bounded-cache transition guarantee and test
    stable owners, alternating owners, nulls, slices, and fallback equivalence.
    Do not introduce an uncharged distinct-source map merely to satisfy FR1.

- **FR3 conflates already-bounded hash scratch with the whole local table.**
  `join/key_dictionary/local.rs::intern_blocks` already reuses fixed
  `COPY_VISITS`-sized stack arrays for positions and hashes. Count/gather morsels
  already exist in `join/data_work/morsel.rs`, with first-output/state-commit
  refusal regressions in `join/tests/morsel_tests/boundaries.rs`. In contrast,
  `Dictionary::prepare`, `LocalIntern::new`, and `LocalIntern::begin_group` still
  fund, allocate, and clear local-table storage using the complete input size.
  Replacing that table with independently classified subgroups can change
  first-retained birth ordering, unresolved IDs before birth, frequencies,
  optional-refusal fallback, and parallel remapping.
  - **Suggested fix:** Name the selected scratch precisely. Preserve existing
    64-row stack work and improve its short-string/accounting work without
    inventing a second admission transaction. Record complete-group local-table
    capacity as a remaining limitation unless it is actually changed and
    validated. Existing fixed hash scratch is evidence of an existing primitive,
    not a new measured reduction of the whole preparation working set. The
    user's request to start implementing this review does not require a risky
    all-or-nothing preparation rewrite.

- **FR5 cannot replace expiration-plan order with native-key order.**
  `ExpirationIndex::expired_cuts` already performs one `partition_point` per
  sorted expiration run, and `apply_cuts` advances the heap once per cut run.
  Those runs are not a declaration that physical chunks or native key buckets
  share one order. `join.rs::EvictionPlan::plan` visits expiration run IDs and
  their entries; `apply_eviction_plan` preserves the first appearance of each
  canonical key in that plan. `StateRows::remove_bulk` already removes emptied
  chunks. Replacing these with key-sorted iteration would alter tombstone and
  slot-recycling order while duplicating existing prefix/reclamation work.
  - **Suggested fix:** Reuse summaries prepared in the existing plan order:
    counts/charges, tombstone eligibility, first-seen canonical retirement groups,
    and chunk-coordinate groups. Preserve the existing tombstone pre-funding
    boundary. A raw `Arc<FramedKey>` is not automatically a canonical dictionary
    identity: `SideBindings::canonical_key` can resolve a restored row through
    its lease. Avoiding per-row resolution requires an explicit valid grouping
    proof; otherwise retain that resolution while removing redundant passes.

- **FR7 needs to distinguish supported payloads from unchanged rejection.**
  `docs/asof-join-guide.md` explicitly rejects nested, dictionary, run-end, and
  view payload types. FR7 currently lists these alongside positive ASOF output
  equivalence fixtures. They are useful ordinary-join tests, but making them
  accepted ASOF payloads would expand schema and checkpoint behavior far beyond
  this optimization.
  - **Suggested fix:** Preserve rejection for unsupported ASOF schemas. Positive
    ASOF tests cover its supported flat nullable payloads, unmatched rows, row
    zero, source promotion, projections, slices, and wide-index fallbacks.

- **Run append must remove repeated owner assembly work as well as worker work.**
  `join/data_work/worker.rs::fill_range` and `fill_all` call `Selection::push` per
  pair. `join/data_work/pairs.rs::ordered_output` repeats the same work through
  `pairs.extend(fragment.iter())`. Optimizing only worker construction leaves
  that repeated per-pair transition/projection work at fragment assembly.
  - **Suggested fix:** Cover both seams with equivalent bulk/run append, retaining
    the necessary opposite-index writes. Preserve selection identity-to-runs
    transitions, sliced views, funded destination/fragment overlap, and worker
    abandonment checks. The worker's current cadence is 4,096 probe/match items;
    do not silently treat dictionary's 64-visit bound as a new global probe
    contract or postpone all checks until after a huge window copy.

- **ASOF compaction requires all position consumers, not only the builder.**
  `OutputPlanBuilder::push_right_source` currently stores a tuple per right row.
  Consumers include `output.rs::GatherPlan`, `output/segments.rs` range slicing,
  and `finalize/segments.rs` shared-right detection. A compact builder followed
  by unconditional conversion back to tuple positions before gather would
  recreate the allocation/copy being removed. Late single-source-to-multiple-
  source promotion must preserve leading unmatched rows and its live old/new
  allocation overlap must stay funded.
  - **Suggested fix:** Keep compact positions consumable through range slicing
    and single-owner Arrow take. Preserve the existing multiple-source fallback
    and visible-slice/backing-ownership checks. Test promotion after unmatched
    rows, matching physical row zero, sparse indices, and wide-domain decisions
    with synthetic metadata rather than billions of allocated rows.

### Minor / Style Notes

- FR8's general typed-sequence grouping is already present in
  `asof/admission/batch/duplicates.rs::group_keys` and
  `asof/admission/batch.rs::Columns::install`, which calls `merge_admitted` once
  per reserved key. The ordered direct path in `admission/right.rs` is separate:
  it visits physical rows and uses stack-borrowed reserved buckets for small key
  counts. Do not report adding grouped general installation as a new result.
- Native and Python runtime CPU defaults already derive capacity automatically;
  the concrete remaining default issue is benchmark worker selection. Identical
  explicit worker settings must be used for baseline/candidate timings so that
  oversubscription correction is not misreported as an algorithmic gain.
- Preserve the user's 1M transport default and unlimited implicit memory budget.
  The old review's default rollback and PR-splitting suggestions are historical
  and are not part of this authorized implementation.

## Axis Review

- **Correctness:** OK for immutable borrowed caches, local-only key packing,
  equivalent selection append, and output-index compaction. The introduction's
  immutable `Batch` and consistent checkpoint requirements remain applicable.
  Whole-input match-limit, mandatory budget, and cancellation gates must still
  precede first output/state publication. The original proposal explicitly
  protects those boundaries.
- **Hidden assumptions:** Source/chunk locality and monotonic probe order are
  optional, not universal. Existing `WindowCursor` and disorder fallback in
  `join/data_work/inputs.rs` and `native_lookup` must remain; sparse chunks cannot
  be assumed to form a dense deque. Direct ASOF ordering is proved separately
  from general grouped admission.
- **Missing edge cases:** Add explicit compact-position promotion after nulls;
  retain removed-middle-chunk/backwards revisit tests; exercise restored canonical
  bindings and interleaved key retirement. Unsupported ASOF payloads must remain
  rejected rather than become new success cases.
- **Backwards compatibility:** OK if project defaults/fingerprints, canonical
  framed keys, Arrow/state identity, and checkpoint bytes stay unchanged. No
  Python API or public project contract change is necessary.
- **Performance:** The bounded source/chunk caches and block/run paths directly
  remove work identified by the review. Whole-group scratch replacement and
  re-grouping an already grouped path are not justified without new evidence.
  Keep allocation/funding effects visible in operation counters and the paired
  current-main comparison; historical profile percentages are not speedups.
- **Surface and ergonomics:** OK; these changes need no public tuning parameters,
  new error surface, or new memory default. Existing explicit limits remain
  observable and must retain their refusal semantics.
- **Test plan:** Strong semantic constraints, but narrow their scope as above.
  Existing `native_window_probe_tests` already check cursor equivalence and
  search reuse; `state/right/merge_tests` check bulk merge, allocation-free
  reserved installation, retirement history, and cancellation preserving the
  original bucket. Reuse those oracles rather than replace them with structural
  existence tests. Do not claim per-key work bounds using a counter that omits
  fallback canonical binding resolutions.
- **Risk and scope:** Approve bounded residual optimization while preserving
  storage and transaction boundaries. Eviction summaries and compact ASOF
  positions need the strongest semantic/ownership review. Replacing the join
  transaction, native monotonic probe engine, checkpoint encoding, or complete
  admission grouping is unnecessary for this work item.

## Counter-Proposals

Prefer bounded previous-source and borrowed chunk caches over unbounded owner
maps or historical-span deques. Prefer appending existing run metadata and
compact single-source output indices over replacing the native probe engine.
For eviction, consolidate the existing plan's summaries while keeping its
ordered commit, tombstone ownership, and canonical fallback; do not manufacture
a new native-key expiration order. Demonstrate current general ASOF grouping
with its work-count/equivalence evidence and only change a residual ordered path
when an actual repeated operation is identified.

## Questions for the Author

- Which named scratch buffer is newly bounded or reduced, and which remains
  proportional to the complete input? The final report must distinguish them.
- If canonical lookup is reused during eviction, what invariant proves every
  summarized row has that same canonical lease after restore/rebuild?
- Does compact ASOF position storage remain compact in both ordinary and
  segmented gather, including range slices and late source promotion?

These are internal evidence questions, not requests for user approval. No builds,
tests, benchmarks, source edits, or remote operations were run for this critique.
