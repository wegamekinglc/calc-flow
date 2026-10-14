# Issue 406 Join batch admission - Critic Critique

## Target

- Spec: [issue406-join-batch-admission.md](../specs/issue406-join-batch-admission.md).
- Source base: `bb142320062995aeb318ad406c901288537fd766`.
- This verdict includes the author's review clarifications: bounded inline
  metadata for at most 64 physical rows and four record segments; existing
  default runtime policy; checked funding for larger metadata; and detachment
  of selected parents before any owned worker transfer. Incorporate these
  clarifications into the main spec before implementation.

## Verdict

**Proceed with caveats.** No remaining design blocker for the clarified proposal.
The original mandatory-allocation rule alone did not settle finite-pool
compatibility or background ownership. The author's bounded-inline and
detach-or-decline clarification resolves those concerns without permitting an
unfunded heap fallback or a disguised incoming row container.

## Findings

### Blocking Issues

None for the clarified proposal. This is a design verdict, not approval of an
implementation, recovery proof, or performance result.

### Significant Concerns

- **Preserve the small zero-headroom path explicitly.** Base
  `join/native_lookup.rs::optional_credit` and
  `join/columnar/owned_copy.rs::CopySelection::reserve` may decline, while
  `join/tests/sql_key_scratch_tests.rs::paid_key_pressure_case` verifies accepted
  legacy SQL with zero remaining headroom. An unconditional heap reservation
  for a one-row probe would break that control. The clarified inline path can
  represent its incoming metadata without allocation: `batch.rs::TableBatch`
  stores `Arc<[RecordBatch]>`, and `Batch` and its metadata clone shared owners.
  Carry the exact inline bounds, larger-batch refusal policy, and unchanged
  default runtime policy into the spec. Define mandatory workspace-refusal
  priority separately from semantic row-error priority. Include inline/spill
  boundary and vector-capacity overlap controls; never satisfy a failed
  reservation with an unfunded heap allocation. Owned SQL/probe input boxing
  still needs its existing admission accounting, including the enlarged inline
  owner's actual size.

- **A shared table owner is broader than selected record ownership.** Keeping
  one `Batch` is cheap but retains every record, including zero-admission records
  and full parents whose selected payloads were already copied. The outer
  `process_data` call can drop during cancellation while a worker survives.
  The clarified detachment rule must apply before native probe, owned SQL, and
  selected-copy transfer; decline to the vectorized serial route if detachment
  cannot be funded. The temporary table owner must also stay out of retained
  state. Preserve the existing `StoredRow` ownership boundary, rather than
  storing the whole input table there. Use mixed dropped/copied/borrowed
  records and actual input/result worker gates to test omitted-array release.
  Existing `state_projection/tests/worker.rs::stopped` checks that a paused
  selected-copy worker keeps selected input and releases omitted payload after
  the caller drops; a single-record Arc-count assertion is insufficient for
  the new mixed-table case.

- **Keep the position namespaces distinct.** Durable row ID, original record
  position, global admitted ordinal, and dense copied-parent position are
  different values. Base `join.rs::probe_key_batch` writes dense admitted
  ordinals to `__cf_join_pos`; `filter_and_order_pairs` uses them to index and
  order output. A dropped physical position still consumes a durable ID.
  Explicitly test a sliced multi-record input with null/late gaps on a genuinely
  generic SQL key path and after projection/recovery. Copying original position
  1023 into dense payload position zero must preserve its original durable ID.
  Existing cross-record/word-boundary controls in `columnar_state_tests.rs`
  and the sparse projection control provide useful analogues.

- **Make structural RED controls observe the actual path.** Counting the old
  constructor is a useful RED witness, but removing that counter can make a
  renamed row adapter appear eliminated. Pair the counter with independent
  segment/parent ownership checks before retained-row construction, nonempty
  output parity, and a SQL execution witness for the fallback case. Exercise
  sparse admission and declined owned transfer; neither may recreate a vector
  of owning row views. Check first-error precedence in both directions using
  `columnar_state_tests.rs::test_batch_admission_row_id_overflow_precedes_timestamp_conversion_and_null_key`
  and `output_gather_tests.rs::test_earlier_timestamp_error_precedes_later_row_id_overflow`.
  Add the metric-overflow/earlier-row counterpart. Existing recovery controls
  must actually process a post-restore batch with ID gaps, rather than only
  comparing restored metadata.

### Minor / Style Notes

- Issue #406 phase 1.1 also names charge sums from offsets. This proposal keeps
  full-fee cache selection but does not explicitly complete that requirement;
  do not close the whole phase solely because incoming row descriptors vanish.
- Random access needs a bounded lookup into segments and sparse selections.
  Record its traversal work so repeated probe/materialization access does not
  hide a scan from the start of the batch. This establishes complexity only.

## Axis Check

- **Correctness:** OK with the position and ownership controls above. Immutable
  Arrow input and DataFusion equality execution agree with
  [the introduction](../../../docs/introduction.md#the-basic-vocabulary).
- **Hidden assumptions:** Covered by explicit multiple-record, nullable,
  unsupported-native-key, timestamp-unit, and frontier cases; finite-pool
  behavior needs the author's clarification in the main artifact.
- **Missing edge cases:** Add inline/spill boundaries and mixed-parent worker
  release; empty/all-dropped input must not transfer unnecessary parents.
- **Backwards compatibility:** OK by requirement; keep project schemas,
  fingerprint, full V1 fees, layout 1/2 migration, and strict layout 3 controls.
- **Performance:** No claim established. Retained rows, matches, key-ID
  persistence, and checkpoint row logs remain later work.
- **Surface and ergonomics:** OK; this is private runtime representation with
  no proposed Python or crate-root API change.
- **Test plan:** Feasible with focused tests; strengthen the structural and
  ownership witnesses as described above. No build was run for this critique.
- **Risk and scope:** The consumer migration is load-bearing, but excluding
  persistent state/key/checkpoint redesign keeps this increment coherent.

## Counter-Proposals

The author's bounded-inline, funded-spill, and detach-or-decline correction is
preferable to either mandatory allocation for every tiny input or an unfunded
heap fallback. No further architectural alternative is necessary for this step.

## Questions for the Author

- State the precise first-error rule when mandatory workspace admission fails
  on an input that also contains a semantic row error.
- Identify the focused generic SQL and mixed-owner tests that witness the
  clarified policy, and record their actual RED/GREEN evidence.
