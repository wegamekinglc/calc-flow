# ASOF output source ranges (A3)

## Approved scope and remaining gap

FR15 extends the existing `Span`, `SourceRun`, selected-range workspace,
and safe shared-column paths. Left materialization already copies spans or
shares complete funded arrays; it does not rebuild an owned row table.
The remaining cost is output planning: each selected row repeats left source
lookup and `SourceRun` bookkeeping, writes an unused left positions vector,
and reserves a row-sized left span vector even for one physical source range.

Reuse the existing range workspace estimates and gather behavior. Preserve
the 256-byte per-row planning reservation, source-registration order, full
backing-owner charges, sink acceptance boundary, and layout 10 state bytes.
Keep matching/cancellation and right candidate references per selected row.
Use each canonical output run to begin a left source once, then coalesce its
physical row ordinals; lexsort permutations and overlap retain exact spans.
All A2 state/canonical encoding files remain outside this change.

The public ASOF schema currently rejects nested and dictionary payloads.
The parent confirmed their matrix means preserving those explicit rejections,
not extending the codec/accounting surface. Supported flat narrow, wide and
variable-width payloads retain existing full/projected/zero-column behavior.

## TDD checkpoint

Prepared a production-matching test and test-only counters for actual left
source lookup and range bookkeeping sites. A 1,024-row contiguous canonical
run must visit each site once, retain no redundant left positions, and allocate
left span capacity by ranges. Right null-candidate references and exact row
count/span order are checked. Existing code performs 1,024 visits of each site.

A second test measures actual constructor allocations with `allocation_counter`.
It begins at capacity 1,024 so an unwanted row-sized allocation must fail the
byte check before vector-capacity assertions. It also covers empty, tiny, and
64,000-row constructors. Only right candidate positions may allocate by rows;
left vectors begin empty. The unchanged `capacity * 256 + 16 KiB` reservation
is asserted independently of actual allocation reduction.

Both tests are migrated to isolated `stream-asof-output-ranges-main` at main
`eccb26973811bc476f0944b977ddedf8564b0237`; the earlier WIP and critic files
remain intact and read-only. Gather, Join, admission, public contracts, and
checkpoint encoding are outside this change.

After the explicit shared-cache handoff, the first check is:

```bash
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 \
  cargo test --locked -p calc-flow --lib \
  operator::asof::finalize::output_ranges:: -- --test-threads=1
```

The actual RED run completed after the explicit exclusive-cache handoff. Both
tests failed for the intended reasons: the matching test observed
`(1,024, 1,024)` left source/range visits versus `(1, 1)`, and the constructor
allocated 57,344 bytes across three vectors at capacity 1,024 versus its
16,896-byte bound. The independent reservation assertion passed at 278,528
bytes. An earlier attempt stopped at a test-only unused-qualification lint;
that fixture spelling was corrected before this behavioral RED.

The implementation keeps one inline pending physical span and registers the
left source at the first selected row of each canonical output run. Matching,
right positions, and cancellation checks remain per row. Closing each span
records the existing `SourceRun` estimate once and emits the existing `Span`
representation. Left positions stay empty; span capacity grows only as needed.
No production A2 state, gather, source, checkpoint, or admission code changes.

The two cost tests passed on the minimal GREEN implementation. All binary,
monotonic, and supplied parallel-candidate modes satisfy the one-source,
one-range check. The constructor allocation bound holds at capacities
0, 1, 17, 1,024, and 64,000 while preserving the original reservation.

## Funding and compatibility checks

Only right positions retain a row-sized constructor allocation. One pending
`Span` is inline; emitted spans allocate by physical discontinuities. At most
one span is emitted per selected row. A doubling vector holds at most two
row-counts of spans; old/new backing during growth needs at most three
row-counts, or 72 bytes per row on the tested 64-bit target. Adding the
16-byte right position keeps this within the unchanged 256 bytes per row;
inline state and tiny-vector minimum capacity fit the unchanged 16 KiB base.
Existing per-source descriptor/header/backing reservations remain separate
and unchanged, and buffer credit is still obtained before gather allocation.

An actual allocation-counter test covers reversed physical rows at cuts
0, 1, 2, 3, 7, 17, 1,024 and 64,000. Its peak scratch allocations fit the
original planning charge, including vector growth. Cropped, fragmented,
repeated left and right rows are compared with independent rowwise Arrow
materialization and row-slice byte sums. The matrix includes 7/19-column
sources, nullable UTF-8/large UTF-8/binary/large binary payloads, full output,
two-sided projection, each one-sided projection and zero columns.

The first serial ASOF run passed 305 of 306 tests. Its sole failure was the
new fixture's immediate pool-zero assertion while the managed gather owner
was still live (32,768 bytes). Output, byte and credit comparisons had passed.
Draining the live default owner alone left a 16,384-byte service control lease,
so the fixture now uses an isolated test service. It awaits each job owner,
drops its Tokio runtime, and shuts down that service before asserting every
pool's final refund, as the existing probe lifecycle fixture does. Production
lifecycle/accounting code is unchanged. The failed matrix's exact rerun passed
all 130 cuts after that fixture correction. No existing passing test was
rerun merely for the fixture correction.

Selected integration checks passed: `stream_asof_join_boundaries` (3),
`stream_asof_join_properties` (8, including generated watermark/restore
oracles), and `stream_asof_join_resources` (5, including 50k-row-per-side
stalled/hot-key/wide traces, checkpoint allocation bounds and atomic refusal).
The explicit dictionary/list/struct rejection check passed on both sides,
including the existing structured schema path/reason and constructor error.
`cargo clippy --locked -p calc-flow --lib --tests -- -D warnings` passed after
naming the right row's exclusive range end; this spelling keeps the same
half-open range and adds no allow/ignore. The final five-test focused recheck
passed after that syntax-only production adjustment. No measured throughput
gain is claimed.

## Local verification commands

Cargo commands use the explicitly handed-off root `CARGO_TARGET_DIR` above,
`CARGO_BUILD_JOBS=2`, unchanged default debug settings, and serial Rust tests.

- RED/GREEN: `cargo test --locked -p calc-flow --lib
  operator::asof::finalize::output_ranges:: -- --test-threads=1` (final 5 passed,
  exit 0).
- Compatibility: `cargo test --locked -p calc-flow --lib
  operator::asof:: -- --test-threads=1` (305 passed, new fixture refund check
  failed; 302 existing tests passed and were not repeated for fixture changes).
- Isolated matrix and rejection exact tests each passed after their respective
  fixture additions/corrections.
- Integration: `cargo test --locked -p calc-flow
  --test stream_asof_join_properties --test stream_asof_join_resources
  --test stream_asof_join_boundaries -- --test-threads=1` (16 passed).
- Core lint: `cargo clippy --locked -p calc-flow --lib --tests -- -D warnings`.
- `cargo fmt --all --check`, generated-contract no-drift checks and
  `git diff --check` passed.

Only the scoped local checks were run. Full workspace coverage and remote CI
remain CI gates; publishing, final specialist review and sealed performance
measurement belong to the parent task.

## Static complexity gate

Before publication, the parent added the observed Rust Codacy threshold of
8. Lizard found the three modified candidate consumers at 9/9/10. Their
shared first-row source selection is now one helper, and monotonic cursors
extend the same pre-sized owned map with the original borrowed
`RightState::IntoIterator`. A compile check caught an initial `.iter()`
spelling, which that state type does not expose; it now uses exactly the
original borrowed iterator. These changes preserve registration and
cancellation order, key/cursor ownership, allocation capacity and row work.

`lizard -l rust -C 8 -w` on `finalize.rs`, `output_plan.rs` and the new
`finalize/output_ranges.rs` exits 0. The candidate functions are 8/8/8,
`left_run_source` is 2, and the three new workspace counters are 1 each.
The broader five-file scan still reports only unchanged `input_workspace`
(16), `identity_row_workspace` (10), and `GatherPlan::shared_column` (9).
They are outside the new/modified-function gate and were not changed or
suppressed. The post-refactor focused five passed (exit 0), and the scoped
core Clippy rerun passed (exit 0). Unchanged passing module/integration checks
were not repeated for this refactor.
