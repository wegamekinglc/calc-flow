# Weekly Python and Arrow boundary review

Review range: `ce313488017d63fc626731f9731a4bec931fc343` through
`1b91bcc00a0623d5ece37ac654bf6e2a4f045939`. Worktree:
`fix/weekly-review-20261107`. No PR or new CI result was available for this
local implementation handoff.

## Confirmed findings and repairs

1. **High: materialized sequence row loss.**
   `_materialized_sequence_binding` used `pyarrow.concat_tables` for all
   sequences. With PyArrow 24, two zero-column tables holding two and three
   rows concatenate to zero rows. A public stream projecting a literal
   therefore silently discarded all five input rows. Sequences containing
   trailing empty RecordBatch values also lost those source boundaries,
   changing observable per-batch output.

   The fix imports an Arrow RecordBatchReader into the existing native Batch
   only for zero-column or zero-row sequences. It retains the ordinary
   nonempty-column concatenation path. Reader schema normalization matches
   the existing convenience-stream normalization; caller schemas and buffers
   remain untouched. Supplied empty RecordBatch items are retained. A Table
   still uses its Arrow-exported boundaries: the API does not reconstruct
   empty chunks already omitted by that export. Explicit
   `SourceBinding.from_arrow` behavior is unchanged.

2. **Medium: deferred materialized-list mutation.**
   `_stream_table` and `_stream_program` copied their input mapping but retained
   a caller-owned list. Replacing its contents after `stream(...)`, before
   context entry, replaced the supposedly captured input. Both entry points
   now capture materialized Sequence item references into a tuple. Generic
   async iterables remain unconsumed at construction. Public tests cover
   both TableExpr and Program entry points.

3. **Medium: validation errors cannot be copied or transported.**
   The Pydantic replacement `ProjectValidationError` initializes ValueError
   with empty arguments despite requiring title and error records for
   reconstruction. `copy.copy`, `copy.deepcopy` and pickle round trips
   therefore raised TypeError. Its reduction now reconstructs from copied
   structured records and includes exception dictionary state. The existing
   empty `args`, title, message, error details, notes, diagnostic attributes
   and independent top-level error dictionaries remain intact. This changes
   error serialization only, not calculation paths.

## Verification

- Sequence regressions: **four actual RED failures**, then **four GREEN**.
  These cover zero-column rows, trailing empty records and both list-capture
  entry points. Existing materialized-input controls: **33 passed**.
- Initial exception round trips: **three actual RED failures**, then three
  GREEN. Independent review found the first fix omitted exception dictionary
  state; expanded assertions produced **three actual RED failures**, then
  **three final GREEN** including notes and custom diagnostics.
- Changed Python source/tests and the fixed performance adapter pass focused
  Ruff check/format. `git diff --check` passes.
- Early stream checks deliberately use the adjacent #532 wheel native module
  at `target/configurable-stream-budgets/site/calc_flow/`, SHA-256
  `3ceea95d25f2b694379d1fdc9924c1d7b740727c3a94f9ae2ceafdbce0d9a30c`.
  Its Arrow source/result implementations have no source diff from current
  main, but this is **not an exact-main native build**. Exact candidate-wheel
  verification is assigned to the parent and remains pending at this handoff.
  The exception checks call no native operation.
- Raw early logs live under
  `target/weekly-review-20261107/python/`: `red.txt`, `green.txt`,
  `materialized-controls.txt`, `config-red.txt`, `config-green.txt`,
  `config-state-red.txt` and `config-state-green.txt`.
- No local performance measurements, full regression matrix or native build
  were started by this specialist.

## Other reviewed surfaces

Reviewed the complete changed Python config validation implementation and
relevant tests; array reshape validation helpers and evaluation context;
DataFusion configuration argument validation; streaming input binding and
ownership paths; Python Batch metadata conversion; Rust Batch active-entity
metadata helper; and the Arrow run-end validation regression tests and pinned
dependency change. No additional confirmed correctness defect was found in
these changes. This review is not an audit of every unchanged function in
the large array, runtime or native Batch modules.

Independently reviewed the parent's `measure._close_workers` fix and its
eight-test GREEN evidence. **Approve:** after a successful measurement,
failure to close the first worker no longer skips later workers. The first
failure remains primary; subsequent failures become notes. Existing primary
measurement failures retain their identity. No repeated test run was needed.

## Performance plan and remaining opportunities

The sequence fix adds call-time reference capture and an empty-input check to
ordinary inputs. Its focused control should use the maintained Program engine
fixture: 1,000 Int64 rows split 640/360, a list input, SQL projection, and the
complete public stream lifecycle. Validate every output value, both output
boundaries and completed/task-zero status. Compare fresh baseline and
candidate wheel processes with the maintained two-round, ten-pair gate;
retain the original three samples per observation and environment identity.
An inconclusive result establishes neither a speedup nor equivalence.

There is no new fast-path claim for the fixed zero-column/empty cases: the old
result was incorrect. Error serialization needs correctness checks rather
than an unrelated throughput benchmark. The parent coordinates the four-case
measurement budget and exact native builds.

For subsequent work, sequence preparation still scans items for type
validation, Arrow conversion, schema validation and empty-input detection.
Large numbers of small prebuilt Batch inputs also pass through PyArrow export
and reimport. A candidate simplification is one preparation pass; a larger
candidate is a private native sequence descriptor retaining Batch references.
Both need a many-small-batch preparation profile first, plus a two-batch
ordinary-input control and exact schema/metadata/empty-boundary tests. No
speedup is asserted from static inspection, and neither candidate is included
in this repair.

## Handoff verdict

**Implementation ready for exact-wheel checks and performance comparison.**
The specialist does not approve its own changes for merge. Final independent
review, performance evidence and required CI remain parent-owned gates.
