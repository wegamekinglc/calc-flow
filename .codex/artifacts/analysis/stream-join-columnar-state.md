# Stream Join columnar state preparation

## Approved scope and parent

J2a implements FR11 and the shared FR1–FR4 contracts in the
[acceleration specification](../specs/stream-join-asof-acceleration.md), subject
to the [approved critic gate](../critiques/stream-join-columnar-state.md).
The isolated `feature/stream-join-columnar-state-main` worktree was safely
fast-forwarded from merged main `eccb2697` to reviewed J1.6
`583d749ad4dcee0caf6edd7748f3d576e3dbe127`. The original untracked critique and
older WIP worktrees remain intact. The critique's former unsealed-J1.6 caveat
is resolved by that reviewed parent, including its private funded submission,
caller-owned cleanup observer, true refund wait, and synchronous installation.

The initial static preparation contained only focused tests and test-only
instrumentation and ran no native commands. Its first authorized RED is
recorded below. The current source also includes constructor-owned shared payload
retention, paid SQL scratch and sparse backing replacement. Targeted checks are
green; complete source review, the authorized focused measurement and CI remain
separate gates.
Prepared expectations and actual observations are kept distinct. No performance
measurement, failing-test commit, or remote operation was made.

## Smallest implementation slices

1. Replace eligible equality probing with canonical V1 key equality and an
   incremental per-key time/physical-ID index. Keep legacy row retention,
   materialization, dirty capture, and V1 codecs in this first slice. Prepare
   matches without mutating committed state; time qualification precedes the
   match cap, and ordering remains incoming position then opposite `(time, ID)`.
2. Replace proven flat retained/dirty payload records with immutable payload
   chunks and compact row locators. Store identity, converted event time,
   canonical key, and exact frozen row charge separately. An ephemeral single
   row view is allowed at output/capture; retaining one payload RecordBatch
   and another dirty-record clone per live row is not the columnar regime.
   Preserve V1 row IPC, metadata, dirty order, and canonical base bytes.
3. Add funded reclamation and sparse replacement for proven backing ownership.
   Reuse the reviewed J1.6 bounded gather/guard/observer lifecycle. Old inputs,
   complete known backing, index/sort scratch, and replacements stay funded
   through actual attempt/output refund. Reset/restore and dropped futures
   keep tracking and cannot install stale state. Denied copy funding keeps
   valid original ownership and the identified legacy fallback.

Each slice receives its own observed RED before production changes, then the
smallest relevant GREEN. Native lookup alone is not reported as completed
columnar retention. J2b's private layout-2 writer/dual decoder and authentic
old managed-checkpoint migration remain a separate stage. Semantic capability
1, public `stream_join@1`, fingerprint, schema, and exact recovery identity
checks remain unchanged throughout this preparation and J2a.

## Eligibility and fallback

Native key equality starts with Boolean, Int16/32/64, UInt8/16/32/64,
Utf8/LargeUtf8, every Timestamp unit/timezone, and composites entirely drawn
from those exact paired types. Ordered lookup compares complete framed bytes.
Timestamp keys use raw declared-unit values; event time alone uses
the existing checked microsecond conversion. Key bytes do not define signed
or chronological sort order.

Int8, Date32/64, and composites containing them retain the existing SQL and
row-state path. Negative Int8 and ordinary Date arrays have pre-existing
retained-key codec failures; they are not silently repaired here. Formerly
successful non-retained probes must remain successful. Unsupported key shapes
remain construction errors. Dictionary/nested payload and unknown external
ownership retain the original retention/checkpoint path unless their exact
V1 byte and ownership proof is established. Eligible keys can still use native
probing over such generic payload. Output dictionary GC and nested trial
preflight remain effective independently of retention representation.

The backing policy is deliberately distinct from V1 logical charges. The
known large-slice fixture requires one retained flat row to own at most 4 KiB
after its copy completes, while reproducing the original row IPC. The broader
sparse threshold/bound must be fixed and proven in the third slice using full
known capacities, aliases, live rows, and the largest funded active chunk.
Visible capacity alone does not prove hidden external ownership. Legacy or
denied-funding shapes cannot be counted toward a claimed bounded regime.

## Prepared focused tests

The new module is
`crates/calc-flow/src/operator/join/tests/columnar_state_tests.rs`.

- `test_native_lookup_avoids_legacy_sql_probe_tables` verifies the full six
  output columns, incoming order, equal-time physical-ID tie order, physical
  IDs and emitted count, then requires zero legacy SQL scratch-table builds
  and no retained SQL key cache. Current valid matching uses the SQL path, so
  the zero-work assertion is expected to fail. The existing `JoinWork` gains
  only a `cfg(test)` table-build counter at `equality_tables`; it measures that
  exact boundary, not an invented public query/allocator metric. Source review
  must also confirm eligible dispatch does not invoke SQL. Runtime initialization
  is not used as a SQL oracle because funded copies may legitimately initialize
  the same runtime.
- `test_retained_and_dirty_record_containers_scale_with_batches` admits 96
  rows in two source batches and checks actual live RecordBatch column-container
  addresses in retained state and pending upserts. At most two payload containers
  are permitted; the existing per-row slices and dirty clones are expected to
  exceed that bound. When locators replace records, the inspection helper must
  enumerate actual chunk containers without manufacturing per-row views. This
  counts persistent payload containers, not total process allocations.
- `test_all_frozen_v1_captures_restore_and_continue_with_native_lookup`
  compares all five frozen capture byte inventories, restores every capture,
  matches all retained rows, checks the full ordered six-column output, frozen
  left-byte charge, IDs and sequence, captures V1 again, restores into another
  operator, and continues with two more incoming rows. Every parity/recovery
  assertion executes before the aggregate zero-SQL assertion. Existing byte
  parity is coverage; expected nonzero SQL work after recovery supplies the
  new-behavior RED. This is operator recovery, not the J2b managed migration.
- `test_known_large_slice_bounds_backing_and_keeps_v1_row_ipc` constructs a
  known owned 8,192-row flat input and admits only its final one-row slice. It
  checks the exact original standalone row IPC in the V1 dirty segment before
  requiring retained backing at most 4 KiB. Existing slices are expected to
  retain more backing. Caller aliasing and successful IPC encoding alone do
  not imply global allocation reclamation or an RSS ceiling.
- `test_legacy_int8_and_date_codec_behavior_stays_on_fallback` checks the
  existing non-retained success and retained Internal failure for negative
  Int8 and Date32/64. It is preservation coverage and is expected to be already
  green; no independent RED will be claimed if it passes unchanged.

## Subsequent test-first gates in the same scope

The native-key matrix must cover Boolean; every eligible integer width with
negative/extreme values where applicable; empty/non-ASCII Utf8/LargeUtf8;
all Timestamp units/timezones; composite tuples; and ns keys that differ
inside one microsecond. Compare with the original SQL reference, including
both arrival directions, batch partitions, null-time before null-key before
late classification, dropped physical-ID gaps, watermark equality, duplicates,
inclusive extremes, and legal out-of-order insertion. Unsupported constructors
and the Int8/Date composite fallback must stay unchanged.

The retained/dirty chunk gates cover exact same-input row IPC with schema/field
metadata, null buffers and slice offsets before eligibility expands. Preserve
prospective FlatV1 row/byte limits, match/counter/output-sequence limits,
single-row/range preflight, dictionary GC, nested output, accepted-prefix
cancellation, and commit-after-all-output acceptance. Run the existing exact
schedule/random-cut property suite after the representation slice.

Resource fixtures must use actual chunk Weak owners, known backing capacities
and aliases, pending-upsert coalescing, carried captures and native snapshots.
After live eviction, pending/snapshot/worker ownership may legitimately keep
the chunk alive. Gate real bounded workers through cancellation/reset/restore,
drop observer futures, fill the actual memory pool, and require retry/managed
drain Pending until payload and original reservation truly drop. Reuse the
reviewed J1.6 released-snapshot-before-credit, pre-ticket-publication, escaped
output and repeated-drain proofs; adapt their owner assertions to real chunks.
Do not equate input-release notification, empty live rows, or slot removal
with actual credit refund. New descriptors/control allocations must have exact
live funding assertions, including normal home/generation fees separately.
No new shared-gather changes or raw spawn_blocking are authorized by this slice.

## Static checks and first actual RED

Before the cache grant, only rustfmt, whitespace/contracts inspection, and
Lizard source analysis ran.
Every new test/helper is at or below cyclomatic complexity eight; the only
existing function with a test-instrumentation change, `equality_tables`, stays
at three. No suppression or limit change is added. The original critique
remains untracked and unchanged.

After explicit debug-cache authorization, the following first run used the
reviewed `583d749a` parent plus only the prepared tests/test instrumentation:

```bash
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib \
  operator::join::tests::columnar_state_tests:: -- --test-threads=1
# Session 10170, exit 101; 2m32s build, 0.09s runtime.
# 5 tests: 4 expected behavioral failures, 1 existing-behavior pass.
```

No fixture compilation correction was needed. Actual observations:

- Native lookup: actual SQL probe table-build count **1**, expected **0**.
  Full ordered output, physical-ID and emitted-count checks passed first.
- Full frozen recovery: all five original byte inventories and every capture's
  first continuation, V1 recapture, second restore, six-column ordered output,
  left charges, IDs and sequence checks passed. The final aggregate actual
  table-build counts were **[2, 2, 2, 2, 2]**, expected **[0, 0, 0, 0, 0]**.
  Byte parity/recovery are existing passing coverage; native dispatch is RED.
- Persistent payload containers: the actual retained/dirty container count
  exceeded the bound **2** for two batches and 96 live rows. The failed boolean
  assertion did not print the exact count; none is claimed here.
- Known sliced backing: the original standalone row IPC matched in the V1
  dirty segment, with one retained row, before actual retained memory exceeded
  the bound **4,096 bytes**. The assertion did not print its exact byte count.
- Negative Int8 and Date32/64 fallback: all three existing non-retained successes
  and retained Internal failures passed unchanged. This is coverage, not RED.

The actual tool-output excerpt is preserved at
`target/issue363-j2a-red/initial-module.txt` under the root target tree. It is
an excerpt from the tool result, not an original redirected process log.
Cargo and the test binary exited; no owned background process remained. The
author explicitly released the debug cache and stopped new native/build work
at this boundary for the parent's quiet measurement window.

The first production candidate implements the native probe slice only;
retained records and dirty upserts still own the legacy per-row RecordBatch.
Shared locator and backing-copy drafts are kept outside source under the root
`target/issue363-j2a-wip` tree for subsequent cycles. No backing-memory or
columnar-retention benefit is claimed for the native-only candidate.

The native index compares the complete framed V1 key and stores one ordered
tree entry `(key, EventTime, physical row ID) -> dense index`. It clamps inclusive
i128 time ranges to representable EventTime bounds and emits each range in
incoming admitted-position order. Append and dense swap-removal update the
index incrementally. Exact paired key schema validation remains the existing
compiler's responsibility; negative Int8, Date32/64 and composites containing
them remain SQL and row-state fallback shapes.

Index construction reserves `1,024 + 128 * retained rows` before allocating
entries. The conservative node allowance is anchored to the repository's
pinned Rust 1.88.0 source: `alloc/collections/btree/node.rs` uses 11 key/value
slots and 12 child pointers; `map.rs` requires five entries in every non-root
node. On 64-bit targets this index's maximum internal-node layout is 464 bytes;
128 bytes per entry plus the fixed allowance covers node occupancy, root,
controls and insertion split slack. The reservation consumer has a fixed
private label, so arbitrary caller operator-name lengths cannot exceed this
control allowance. This is funded requested allocation and
bookkeeping, not an allocator/RSS ceiling. Append prepayment uses an RAII guard
that refunds unused growth on output rejection or a dropped preparation future;
only committed index entries retain that growth. Erased nodes/keys release
before the matching allowance is refunded. A denied prepayment uses the
reviewed SQL path and invalidates a stale index rather than allocating unpaid
new entries. Construction/key/pair prepayment refusal selects SQL fallback;
incoming-side append refusal invalidates that optional index while keeping the
already prepared opposite-side outputs and original row state valid.

Probe key vectors and framed byte allocations are prepaid from per-cell
logical widths, timezone bytes and control allowances. A shared real
MemoryReservation travels with each native key, including retained and dirty
aliases; encoded bytes drop before that credit. A credit-last NativeKeys bundle
also keeps its real reservation until the key-vector backing drops, including
zero-match results after all encoded key owners have been cleared. Matching first counts only
through the existing limit plus one, applies the unchanged match-limit error,
then prepays the compact pair vector before allocating it. V1 restore still
decodes the original row state; its cold native index is rebuilt on the next
eligible probe or append. Fresh eligible retention prepays index growth before
emission and installs entries only at the existing commit boundary. Existing captured metadata, IPC and public capabilities do
not change.

The parent granted the next exclusive debug-cache window after completing all
32 J1 preflight cases. The first native-only candidate actually passed:

- Native lookup, all five frozen captures with both recovery continuations,
  and negative Int8/Date preservation: three targeted tests, each 1/1.
- Native reference/funding module: initially 3/3, then 5/5 after composite
  framing, actual denied SQL fallback and key-vector backing evidence.
  Its key matrix contains 22 concrete key shapes in both directions: 44
  comparisons against the original SQL path, including all eligible integer
  widths/extremes, Boolean, UTF-8 strings, and four timestamp units with three
  timezone forms.
- Scoped Join module: 71/71, explicitly excluding the two future
  container/backing tests. Ordered/random-cut Join properties: 4/4.

The first candidate still built its index at the first query. A separate actual
RED, `test_native_index_is_ready_before_the_first_probe`, failed because fresh
eligible retention left the index absent. Session 39649 exited 101 after a
44.68s compile; its assertion read "fresh eligible retention must prepare its
index before the first query". Only after that RED was observed was the eager
prepay/commit path implemented.

With eager indexing, session 36069 ran the same scoped Join module and observed
**67 passes / five failures**, not a completed GREEN. The readiness test,
reference types/order/fallback checks and both frozen wire oracles passed.
Five existing checkpoint fixtures expected no live state reservation after
worker retirement even though their operator still held a native index:

| Fixture                                                     | Actual pool bytes | Previous expected bytes |
| ----------------------------------------------------------- | ----------------- | ----------------------- |
| checkpoint worker admission failure                         | 1,664             | 0                       |
| dropped checkpoint and mutation futures                     | 1,664             | 0                       |
| reset then new retained row                                 | 1,152             | 0                       |
| failed checkpoint worker while home/generation remain alive  | 34,432            | 32,768                  |
| released snapshot / real credit retry                        | 1,536             | 0                       |

The narrow fixture adaptation reads the actual NativeIndex MemoryReservation
through a test-only accessor and independently asserts its amount equals
`1,024 + 128 * committed live rows`; it does not infer an owner by subtracting
worker totals. Old worker Pending checks, 1 GiB budgets, pressure/spare values,
home/generation funding and exact refund equalities remain. While the operator
still owns its index, only that verified funding is added. Each affected
fixture additionally drops the operator and requires the original exact final
pool-zero boundary. The existing managed-close fixture already drops its
operator before drain and keeps its exact-zero assertion unchanged. Their
focused rerun, session 23235, exited zero: the full checkpoint compaction
module passed **14/14** (1m10s build, 0.09s runtime). No worker-credit assertion,
Pending boundary, budget or final-zero assertion was removed.

The parent's additional shrink-funding review requested actual allocation
evidence beyond the pinned standard-library node bound. The new
`native_index_allocation_tests` measures the real NativeIndex consumer and
credit-Arc controls, tree append/removal and final Drop with thread-local
`allocation_counter`. Row/key owners and the common memory pool are prepared
outside those measurements and remain alive separately. For 4,096 rows, it
combines sorted/disordered key insertion with dense/random swap-removal,
checking every operation's requested-allocation peak against pre-operation
funding and its live allocation against the actual guard after shrink.
It also asserts the guard equals `1,024 + 128 * actual tree entries` at every
cut. Final index destruction requires both net tracked allocations and actual
pool reservation to be exactly zero. The empty-root allocation remaining
after all deletions is covered by the existing fixed allowance; no fee was
raised. Session 75972 passed **1/1**, exit zero (55.27s build, 0.07s runtime).
This is requested native-index allocation evidence, not payload backing or RSS.

After eager indexing, the ordered/random-cut property target passed **4/4**
again (session 43237, exit zero; 28.23s build, 0.43s runtime):

```bash
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow \
  --test stream_join_properties -- --test-threads=1
```

The first scoped Clippy run found unchecked casts on mathematically clamped
i128 time bounds, unnecessary fixture ownership and allocation-test conversion
style. These were corrected with checked conversions and borrowed test inputs,
without suppressions or changing the bounds, fees or drop/refund order.
The scoped Clippy rerun passed with `-D warnings` (session 85698, exit zero;
1m27s). The affected native reference/funding module passed **6/6** after those
corrections (session 6206, exit zero; 58.85s build, 0.18s runtime). The three
first-slice columnar tests passed **3/3**, and the strengthened allocation proof
passed **1/1** again without compilation. Commands:

```bash
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo clippy --locked -p calc-flow --lib --tests -- -D warnings
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib \
  operator::join::tests::native_lookup_tests:: -- --test-threads=1
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib \
  operator::join::tests::columnar_state_tests:: -- --test-threads=1 \
  --skip test_retained_and_dirty_record_containers_scale_with_batches \
  --skip test_known_large_slice_bounds_backing_and_keeps_v1_row_ipc
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib \
  test_native_index_live_allocations_remain_funded_through_insert_and_eviction \
  -- --test-threads=1
```

Lizard reports no over-eight functions in the new native/key/test modules or
the changed compaction fixture file. A function-body comparison against the
parent identifies 29 changed/new functions in `join.rs`, all with CCN at most
eight. Existing over-eight functions elsewhere in that file are unchanged;
no rule suppression was added. `cargo fmt --all --check`, `git diff --check`
and the generated-contract drift command all exited zero. All owned native,
Cargo, test and formatter processes exited, and the shared debug cache was
explicitly released to the parent. No commit or PR is created
while the two later-slice representation tests remain RED.

Container/backing tests remain the previously observed REDs and are not
repeated in this slice. No complete J2a implementation, managed V2 migration,
performance benefit or bounded Arrow/RSS regime is claimed.
Clippy, exact V1 fixtures, affected J1.6 lifecycle checks and ordered properties
form the scoped handoff; full CI/90% coverage and sealed allocation/RSS/latency
measurements belong to the parent workflow.

Any later Arrow owner/capacity proof is anchored to the exact locked
`arrow-buffer`/`arrow-array` **58.3.0**, not a later registry version.
`Buffer::capacity()` returns zero for external owners; that is an unknown-owner
boundary, not evidence of zero retained bytes. Python/FFI-owned unknown buffers
remain legacy retention shapes and are counted separately from any future
proven-native backing/RSS regime. Native-key lookup eligibility is independent
of payload backing eligibility.


## Shared payload locator slice: scoped GREEN, sparse backing still RED

The native-only approved source was copied byte-for-byte into
`target/issue363-j2a-wip/native-source-approved-v1/` before the locator edits.
The independently approved receipt and review remain unchanged.

This slice adds one immutable `PayloadChunk` per eligible input record batch,
shared by retained rows and dirty upserts through row offsets. Persistent
payload containers therefore scale with referenced input batches. V1 row IPC
and output construction use temporary row views; restore still uses legacy
single-row payloads. Native eligibility and legacy Int8/Date error behavior are
unchanged. The persistent container test counts actual owned column-vector
addresses, rather than synthesizing a temporary view for each row.

A chunk owns its immutable record before its real MemoryReservation field.
Admission reserves `1,024 + 256 * columns` controls before inspecting backing,
then reserves the sum of all native buffer capacities before cloning the
record or allocating the shared chunk. This intentionally overcounts aliases.
Every included buffer must have nonzero capacity under the exact locked Arrow
58.3.0 implementation. External capacity-zero owners, unproved payload shapes,
and denied reservations remain legacy retention. Native lookup may still apply
to their eligible keys. The shared-container scope is not a complete bounded
backing/RSS regime: its sparse source backing is currently retained, as the
recorded REDs below require the next slice to fix.

Test-only funding accessors read each live chunk's actual reservation and
independently assert its expected control-plus-known-backing amount. Updated
resource fixtures deduplicate those actual chunk/key guard identities across
live rows and pending upserts and add them to the independently verified native
index funding. No fixture infers state funding by subtracting worker totals.
The existing worker refund equalities, Pending boundaries, home/generation
funding, runtime budgets, pressure/spare values, and final operator-drop exact
zero checks remain unchanged.

Commands below all ran from the isolated columnar-state worktree, with
`CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target`
and `CARGO_BUILD_JOBS=2`:

```bash
cargo test --locked -p calc-flow --lib \
  test_retained_and_dirty_record_containers_scale_with_batches \
  -- --test-threads=1
cargo test --locked -p calc-flow --lib \
  operator::join::tests::columnar_state_tests:: -- --test-threads=1 \
  --skip test_known_large_slice_bounds_backing_and_keeps_v1_row_ipc
cargo test --locked -p calc-flow --lib \
  operator::join::tests::native_lookup_tests:: -- --test-threads=1
cargo test --locked -p calc-flow --lib \
  operator::join::tests::checkpoint_compaction_tests:: -- --test-threads=1
```

Observed results: container **1/1 GREEN** (session 45832, exit zero;
1m57s build); the then-existing columnar subset **4/4 GREEN** (0.02s),
native reference/funding **6/6 GREEN** (0.18s), and checkpoint compaction
lifecycle **14/14 GREEN** (0.08s). The subset command ran before the new
post-eviction test was added. Its exact all-five frozen V1 inventories and
restore/continuation assertions passed. These are locator-slice results;
the earlier native-only Clippy/properties results do not assert that the new
locator production source passed those checks. No whole J2a GREEN is claimed.

The original large-slice backing test remains its previously observed RED.
It was not repeated during this locator boundary. Before any sparse-copy
production code, a new test independently demanded reclamation after ordinary
progress evicts 8,191 of 8,192 physical rows from an initially dense chunk:

```bash
cargo test --locked -p calc-flow --lib \
  test_progress_sparse_chunk_bounds_backing_and_keeps_v1_row_ipc \
  -- --test-threads=1
```

Actual session **19438**, exit **101** (45.33s build, 0.04s execution):
`one post-eviction row retains 196896 bytes` against the test's `<= 4096` bound.
The preceding retained-row count 1, evicted count 8,191, next physical ID 8,192,
and exact original final-row V1 IPC assertions all passed. This is a genuine
backing-density RED, distinct from the already-green container behavior. The
fixture's declared 8,192-row / 4,000,000-byte logical state limit permits its
input and does not alter the engine's runtime budget or FlatV1 charging.

At this boundary no sparse worker production implementation is present.
All owned Cargo/test/formatter processes have exited. The parent requested a
read-only independent review of this exact locator slice before proceeding;
the shared debug cache is released. No commit, publication, performance gain,
new wire codec, or complete sparse-backing safety proof is claimed.


## Locator review correction: interim fixes and two new ownership REDs

Independent locator review requested changes in
`reviews/stream-join-columnar-state-locator.md` (review SHA-256
`3941c817a95680c32002aa89fb8d8906c6b18a6bfcb9378d8fc483be27d2d6a0`).
Its exact-value-equal caller-schema counterexample was reproduced before
production changes. All commands in this section use the same isolated
worktree, shared root `CARGO_TARGET_DIR`, jobs 2, locked dependencies and
Rust 1.88.0 as the preceding slice. Original command outputs below are tool
observations; target excerpts are explicitly transcriptions, not complete
redirected process logs.

```bash
cargo test --locked -p calc-flow --lib test_shared_ -- --test-threads=1
```

Session **80338**, exit **101**, build 1m06s, runtime 0.03s: **two genuine REDs**.
An independently allocated equal caller schema with 1 MiB capacity Strings in
field names, schema metadata and field metadata, plus larger HashMaps, stayed
live after caller handles dropped. Exact schema equality, actual Shared path,
sub-MiB real credit, original V1 IPC suffix, unchanged FlatV1 charge and caller
read-only capacities passed before the Weak-owner assertion failed. Genuine
Shared flat preflight allocated **[8,003, 80,005]** times for 1x/10x fanout;
its distinct-row caches did not prevent temporary views per pair. Neither
finding is a measured latency regression.

The minimal interim production fix rebuilds narrow native array wrappers from
immutable buffer references using the already-owned canonical input SchemaRef
and data types; it therefore drops independent caller schema/field/timezone
controls instead of retaining them. The direct flat cost path reads the
selected row offset, null validity and per-row variable offset differences;
it creates no temporary row views before checking the cost cache. Caller
inputs, FlatV1 charges, caps, framing, semantic capability and native index
Entry128/Base1024 funding are unchanged. No fixed fee increase was made.
Session **12552**, exit zero, build 56.53s, runtime 0.01s: the two tests passed.
This is **interim GREEN only**, not approval of the shared-owner mechanism.

Additional proof tests were then added for actual native owner/control
allocation and slice/null/variable-value preflight equality with Legacy.
The allocation proof prepares canonical schema/runtime outside measurement,
measures native source buffer/wrapper construction, Shared reconstruction,
caller-container drop and final chunk drop separately. Caller-container
release isolates the original buffer owners actually retained by the chunk.
It combines those requested live allocations with measured reconstruction
peak/live allocations and compares them with the real reservation, finally
requiring both net tracked allocation and pool reservation exactly zero.
It is neither whole-process RSS nor an inference from worker totals.

```bash
cargo test --locked -p calc-flow --lib \
  operator::join::tests::columnar_state_tests:: -- --test-threads=1 \
  --skip test_known_large_slice_bounds_backing_and_keeps_v1_row_ipc \
  --skip test_progress_sparse_chunk_bounds_backing_and_keeps_v1_row_ipc
```

Session **1846**, exit **101**, build 51.70s, runtime 0.03s: **7 PASS / 1 FAIL**.
Three and sixteen columns passed; at 128 mixed native flat columns, requested
allocation **peak = live = 113,380 bytes** exceeded actual credit **111,484**.
Thus the existing `1,024 + 256 * columns` allowance also underfunds concrete
native wrappers/buffer-owner controls at width 128. The five exact V1 capture
inventories/continuations, native lookup, legacy preservation, persistent
container bound, Shared slice/null/string preflight parity and both interim
review fixes passed in this run. The new native-allocation failure remains
unfixed. Increasing a constant without a complete control proof is not an
accepted correction.

### Locked Arrow positive-capacity counterexample

The earlier analysis statement equating nonzero `Buffer::capacity()` with
proven native full ownership is **superseded and false for the locked 58.3.0
implementation**. In `arrow-buffer/src/bytes.rs`, `Bytes::capacity()` returns
`Custom(_, size) => size`; `From<bytes::Bytes>` installs `Custom(owner, len)`.
`buffer/immutable.rs::Buffer::from_custom_allocation` likewise passes `len`.
The public capacity rustdoc says external owners return zero, but that is not
what these implementations guarantee. The private allocation kind is not
exposed by a safe read-only API. Cloning then attempting `into_vec` cannot
separate shared Standard and Custom owners. Positive visible capacity cannot
bound an opaque owner's other allocations. No dependency or unsafe change was
made.

A focused test uses the already-available safe
`tokio_util::bytes::Bytes::from_owner`: its OpaqueTimes owner exposes one native
8-byte timestamp Buffer while retaining a hidden 1 MiB Vec and a Weak marker.
Arrow reports capacity **8**, and rebuilding native wrappers does not remove
the opaque buffer owner.

```bash
cargo test --locked -p calc-flow --lib \
  test_shared_payload_refuses_positive_capacity_opaque_bytes_owner \
  -- --test-threads=1
```

Session **10253**, exit **101**, build 50.88s, runtime 0.00s: **actual RED**:
`opaque owner remains live=true despite positive capacity8; shared credit=Some(1816) omits hidden1MiB`.
This confirms the source counterexample without unsafe, new dependencies,
manifest edits or native performance probes. The new test requires conservative
refusal by the raw-input Shared constructor; an approved future owned-copy
constructor may create a separately proven chunk. Existing native-key support
and logical acceptance/errors need not change when payloads fall back.

The three review-boundary excerpts are preserved under
`target/issue363-j2a-red/locator-review.txt`,
`locator-owner-controls.txt`, and `locator-opaque-owner.txt`.
Four currently known behavior failures remain: the two earlier sparse backing
REDs, native wide-control funding, and positive-capacity opaque owner refusal.
No whole current module, funding proof, source approval, Clippy/properties or
bounded backing/RSS claim is made. Lizard on the three changed locator modules
reports no CCN above eight; `git diff --check` passes. Scoped Clippy/properties
are deferred at this correctness/design boundary. The parent requested no
further eligibility or sparse production edits until a new critic gate.

### Proposed minimum owned-copy mechanism for critic decision

This is a concrete proposal, **not approved or implemented**. No public API,
logical fee, capability, fingerprint, shared gather implementation or V2 wire
change is proposed.

1. Raw caller buffers have no proven allocation-kind certificate. They must
   stay on the current Legacy payload fallback unless a new core-owned copy
   is constructed. Native key probing remains independent. A private proven
   chunk constructor accepts only core-built arrays and a real reservation;
   its proof cannot be synthesized from input capacity, ArrayRef type, schema
   equality or a Boolean flag supplied by the caller.
2. For a first narrow owned-copy slice, copy while borrowing the caller record
   within the operator's admission future, in synchronous bounded quanta, with
   cancellation checks/cooperative yields between quanta. An initial proposed
   quantum is at most 4 KiB of payload copying and at most 64 columns/rows;
   the critic must settle the exact bounds. Never send an opaque original
   owner to a detached worker funded only for its visible slice. Rows/columns
   exceeding the small-copy proof remain Legacy rather than creating a new
   validation error. The original input remains subject to the existing input
   lifecycle and is excluded from a new proven-resident/RSS claim.
3. Accumulate those bounded fills into one prepaid core-owned flat builder
   set per eligible source record, rather than one payload container per row
   or per quantum. Compute selected/admitted positions and exact existing
   physical IDs before copying; packed offsets map back to those IDs, event
   times and retain flags. Preserve incoming order and existing framed-key/
   opposite-time ordering. Preflight old state/match/refusal errors before
   installing state. Allocate native buffers, selection descriptors, builders,
   array/vector/control owners and replacement capacities only after their
   checked actual reservation. Reuse canonical schema/type controls already
   owned by the operator; explicitly certify their lifetime/funding boundary
   if a future worker can outlive that owner.
4. Initial copy eligibility should be the proven flat subset with no source
   null buffer, subject to actual full V1 byte tests for every shape, bool bit
   offset and timestamp unit/timezone. Nullable/unsupported/dictionary/nested
   payloads remain Legacy until their byte proof passes; key support remains
   unchanged. Locked `take_primitive` copies values, `take_bits` creates fresh
   bits and `take_bytes` allocates fresh offsets/values. However `take_bytes`
   omits raw bytes in null slots, which can differ from a legacy row's physical
   IPC contents; builders that append a null can also replace underlying null
   values. Neither logical Arrow equality nor new allocation proves V1 byte
   equality. Identity indices, all-valid/null masks, sliced booleans, empty
   strings and source Weak release must be directly proved before expansion.
5. Once a chunk is core-owned by construction, its full buffer capacities and
   concrete native buffer/array/control allocations can be funded and tested.
   Sparse replacement may then use existing `OwnedCpuWork` and opt-in
   `submit_observed_work`, with caller cleanup observer installed before its
   first cancellable await and a real job retirement guard. Pay old inputs,
   indices, output resident capacity, workspace and controls before dispatch;
   successful installation occurs while output credit is still held. Input
   owners release before their credits, metadata/signal and final retirement
   guard. Dropped futures, reset/restore and managed close must retain tracking
   until actual refund; no raw spawn and no stale automatic installation.
6. Admission builder cancellation uses explicit owner-before-credit-before-
   guard field ordering. Borrow-only quanta have no detached source worker;
   any cleanup moved to native owned work may contain only already-proven,
   fully funded core objects. Failed admission/copy reservation retains the
   reviewed valid Legacy/original state and does not count toward a bounded
   columnar regime. A partial copy is never installed.
7. After that certificate exists, retain incremental per-chunk live locators/
   counts and mark only affected sparse chunks at eviction, preserving the
   no-expiry/no-full-state-scan property. Density thresholds and full backing
   bounds, metadata/node allocation proof, old aliases in dirty/snapshot/output
   owners and actual cancellation/reset/refund tests require their own gates.

The critic must decide the borrow-only quantum/whole-record builder mechanism,
its schema/type-control lifetime, exact complete allocation bound, initial byte-
compatible subset, budget fallback and cleanup ordering before production work.
Alternative: temporarily keep every uncertified input on Legacy and ship only
native lookup; this does not complete the requested columnar payload stage.
No owned-copy or sparse implementation was written after these findings.
All owned processes exited and the debug cache is released at this boundary.

## Owned-ingress correction (uncommitted, source gate pending)

The frozen [ownership critique](../critiques/stream-join-columnar-state-ownership.md)
approved a bounded borrow-only core constructor, superseding the historical
proposal and raw positive-capacity eligibility above. The new constructor
never installs caller buffers. It packs admitted source positions into core
PrimitiveBuilder/StringBuilder allocations while borrowing the original input
inside the admission future. There is no ingress worker, raw spawn or automatic
installer. Original inputs and Legacy payloads are excluded from a new proven
core-resident claim.

`PayloadChunk` now owns one column Vec, the canonical SchemaRef and its real
credit, in that drop order. Persistent RecordBatch construction was removed:
locked Arrow's safe RecordBatch constructor performs three full column loops,
which cannot obey a 64-visit quantum at width 128. IPC uses temporary RowView
objects holding their chunk owner. Native keys, FlatV1 charges and flat fanout
preflight read columns and packed offset directly. No V1 framing, state cap,
semantic capability, fingerprint, Arrow dependency, ASOF or shared gather code
changed. Bool payloads and any actual NullBuffer (including all-valid masks)
remain Legacy; native Bool equality remains available. Int8/Date/nested/dictionary
fallback and their existing errors are preserved.

A cold numeric canonical-schema inventory models Schema/Fields/Field Arcs,
name capacities, metadata HashMap bucket capacity and key/value String capacities,
and timestamp timezone Arcs. It does not claim funding at construction. Each
chunk pays that candidate inventory through the actual runtime guard before
capturing metadata/type owners. HashMap capacity uses a conservative two-times
capacity-plus-one bucket bound with control bytes/group padding, anchored to
the locked Rust 1.88 hashbrown layout. Actual fresh-map dynamic-schema allocation
is independently measured by the lifetime fixture, not inferred from worker
totals. The deletion-history caveat below remains unresolved; this is not a
complete proof for every caller-provided canonical schema.

The checked constructor inventory also includes the actual array wrapper and
Arc header, the locked Arrow Bytes owner (ptr, len, three-word Deallocation plus
Arc header), column/plan Vecs, Arrow's four-entry temporary buffer Vec and reset
string-offset allocation, reservation registration/name/reallocation overlap,
and the credit Arc. Exact selected primitive bytes and string value/offset
capacities are added before allocation. Selection descriptors have a separate
real guard. Construction and builder structs release owners before their guard;
chunk aliases retain the same guard rather than charging the same backing twice.
The 3/16/128 allocator fixture now exercises the permitted nonnull, non-Boolean
copy subset. The former masked/Boolean 128-column allocation counterexample is
preserved as historical RED evidence; those shapes now exercise exact Legacy
fallback rather than a fictitious owned-copy proof.

Copy/header/selection planning uses a conjunctive <=4,096 touched-byte and
<=64 conservative visit counter, including wrapper construction and finishing.
Primitive steps account both read/write bytes. String fragments reserve four
boundary-inspection bytes and copy at most 2,046 bytes per step (read+write),
without reallocating a partially filled value buffer. Each boundary yields and
checks the existing context cancellation/deadline. The metadata/key eligibility
summary is computed at cold operator construction, so runtime eligibility does
not introduce an unbounded schema/key header scan.

### Actual RED and GREEN evidence

Commands below use the root warm target, `CARGO_BUILD_JOBS=2`, Rust 1.88,
`cargo test --locked -p calc-flow --lib`, serial test execution and `--nocapture`.
No native benchmark or performance timing was run.

- **72365, exit101**, 58.32s build: all three new `test_owned_ingress` cases
  failed for the intended behavior. Boolean/mask entered Shared; core ingestion
  kept the opaque hidden 1MiB owner; the 32KiB string completed without yielding.
  The earlier first compile had only a corrected Cancelled-variant pattern and
  is not a behavior RED.
- **77314, exit0**, 2m14s build: initial three cases passed. This was before
  later conservative accounting/refactoring and the stronger real-owner gate.
- **60687, exit0**, 42.57s build: thirteen non-post-eviction columnar cases passed,
  including all five frozen V1 inventories and each recovery/continuation, SQL0
  native lookup, persistent container bound, independent 3/16/128 allocator
  inequalities and exact final refunds, opaque/caller-schema release, slice/
  string preflight, and legacy Int8/Date behavior.
- **63023, exit0**, 52.59s build: four additional existing-contract proofs passed:
  128-header yield/cancel before retention; full-budget Legacy fallback; dynamic
  canonical metadata funded while a payload outlives its operator; selected
  late-drop IDs2/3 with packed offsets0/1, next physical ID4 and dirty-only credit
  ownership followed by exact zero.
- **58486, exit0**, 1m24s build: all six native lookup regressions passed, including
  all key types/units/zones against SQL order, dense eviction, budget refusal,
  exact zero-result key lifetime and eager index admission.
- **18903, exit101**, 52.96s build: sixteen cases passed; the strengthened real
  StringBuilder/Wake case reached >=32KiB actual requested allocation and Pending
  but its manually polled loop had not returned to Tokio's deferred-wake scheduler.
  The fixture must yield back before checking the real Wake counter. This is a
  test-scheduler correction, not evidence of a production lost wake or a new
  behavior RED. The exact allocation/funding and Wake assertions remain.
- **2811, exit0**, 40.22s build: the affected long-string fixture passed after
  yielding back to Tokio before reading its Wake counter. The sixteen unchanged
  passing cases from 18903 were not rerun. Together those executions cover
  seventeen cases, excluding the two explicitly unresolved tests below; this
  is not a fresh all-module GREEN run.
- The existing `checkpoint_compaction_tests::` filter passed **14/14**, exit0,
  runtime0.10s. It preserves the actual input/credit retirement waits, pre-ticket
  tracking, reset/restore, managed drain, deadline/cancellation and checkpoint
  framing proofs with the new carrier.
- **81810, exit0**, 1m27s build: `--test stream_join_properties` passed **4/4**,
  preserving exact per-schedule output order and physical IDs plus random
  checkpoint cuts and restored counters.
- **82695, exit0**, 1m59s build, runtime0.04s: the seventeen directly affected
  columnar tests passed after lint fixes and admission helper extraction. The
  invocation explicitly skipped the unchanged sparse test and the new SQL
  scratch RED; neither test was ignored or removed. This is a selected subset
  run, not whole J2a GREEN.
- **50479, exit0**, 30.74s build, runtime0.01s: the full-five-capture V1
  restore/continuation test passed after the last ordering/encoding helper
  extraction. No unrelated passing group was rerun for this last refactor.

Subsequent edits were limited to scoped lint corrections: explicit imports,
borrow/closure simplification, checked bounded fixture conversions, test-only
helper gating and explicit ownership-only field destructuring. They do not
change reservation sizes, fallback, ordering or SQL scratch behavior. In the
constructor inventory, registration/name capacity and its format/reallocation
overlap use three times the complete registration name length. Final complexity
cleanup extracts admission append and canonical row ordering/encoding helpers;
the original cancellation checks and fallible operation order remain unchanged.
Final scoped Clippy and static results are recorded in the frozen receipt.

The initial 8,192-source-row slice to one row now passes naturally because
owned ingress copies only selected positions. It does not prove the separate
post-eviction density obligation. That unchanged test remains present, with
its earlier actual **196,896 >4,096** RED; no sparse replacement was implemented.

### New SQL scratch ownership boundary: actual RED, design unresolved

**37153, exit101**, 44.87s build, runtime0.02s:
`test_owned_copy_key_scratch_does_not_borrow_whole_chunk_backing` first confirms
key7, then measures **32,864 bytes >4,096** for one key row taken from a
4,096-row certified chunk. Existing key_probe_batch builds full RowViews and
Arrow singleton concat retains the whole original backing. This is an actual
backing/owner boundary failure, not a measured latency regression.

Simply forcing concat to copy is insufficient: newly allocated scratch/cached
arrays and metadata need complete funding and credits surviving DataFusion
cancellation/task cleanup, while funding refusal must preserve the previous
valid SQL fallback rather than introduce a new rejection. A function-local
reservation or implicit singleton-copy assumption is not that proof. The
parent has routed the private paid-key-scratch carrier, exact DF cleanup hooks
and refusal behavior to a further critic design checkpoint. No speculative
fee increase, budget increase, public/gather change or SQL scratch production
fix has been made. Source remains pending, whole J2a remains incomplete, and
current tests/source must not be committed or published as passing.

A further **static proof caveat**, not an observed native failure in this slice:
`metadata_inventory` currently bounds HashMap buckets from public `capacity()`.
Rust 1.88 std delegates to hashbrown 0.15; RawTable capacity includes remaining
growth plus items, and erase can leave DELETED control bytes without refunding
growth. Arbitrary caller-provided canonical metadata with deletion/tombstone
history needs an independent bucket-allocation bound or a core-normalized,
constructor-proven metadata owner. The fresh-map allocator sample does not
prove every such history. The current two-times capacity formula must not be
represented as a completed general funding proof before that gate. No metadata
normalization, blind fee increase or additional production implementation was
made at the parent's requested freeze boundary.

### Partial-source freeze and cache handoff

**39284, exit0**, 1m03s: final `cargo clippy --locked -p calc-flow --lib --tests
-- -D warnings` passed after the last helper extraction. The earlier lint-only
diagnostics in 23210 and 61118 were corrected without suppressions and are not
behavior RED evidence. `cargo fmt --all --check`, `git diff --check` and the
three generated-contract comparisons passed. Lizard's six new Rust modules
reported no CCN above eight; a source-body comparison against HEAD identified
sixty changed functions in the three tracked files, with maximum CCN eight.

The exact nine Rust files, this analysis, upstream design inputs, command
ledger and labelled transcribed failure excerpts are frozen under
`target/issue363-j2a-wip/owned-ingress-partial-v4/`. Historical Approved native
and locator snapshots remain unchanged. This freeze is uncommitted WIP for
the next design/source review; it is not SOURCE APPROVED, whole J2a GREEN,
merge-ready, or a performance result. No CI, remote publication, sparse copy
or SQL scratch implementation was performed. All owned sessions have exited;
the root debug cache is released.

### Paid SQL key scratch: working primitive slice after partial-v4

The [approved correction direction](../critiques/stream-join-columnar-state-key-scratch.md)
is now being implemented. The immutable partial-v4 snapshot above remains
unchanged. Current source is uncommitted work and is not a complete SQL owner,
metadata-normalization or J2a approval.

The first private scratch constructor copies non-null Int64/UInt64 keys and
physical IDs directly from payload columns and offsets. It prepays its actual
core buffers, validity IDs, fresh schema/fields, wrapper controls and builder
overlap. A safe locked Bytes1.12.1 `Bytes::from_owner` wrapper attaches an
acyclic funding lease to each native Arrow buffer: the lease owns metadata,
credit and a job retirement guard, and does not own its own arrays. Actual
buffer owners drop before credit, then the managed retirement guard. Cache
entries retain that lease, while the original 32 MiB cache policy and Legacy
SQL remain unchanged. This first constructor supports only the stated
primitive subset; other approved key types still need implementation.

Actual focused commands use the root warm debug target, jobs2, `cargo test
--locked -p calc-flow --lib <filter> -- --test-threads=1 --nocapture`:

- **86085, exit101**, 1m29s build / 0.03s runtime: the genuine DF
  `RepartitionExec` producer cancellation fixture keeps readable key7 behind
  its real input-stream Drop gate after query and operator drop, but original
  funding has refunded. The `pool.reserved()>0` assertion fails. This is a
  real escaped-producer RED, not a local Arc-only ownership test.
- **25236, exit0**, 1m39s build / 0.02s runtime: that same producer fixture
  passes after the buffer lease. Managed job drain stays Pending at the gate,
  then releases after the producer's actual owners and credit; final Weak
  array and exact pool zero assertions pass.
- The original singleton backing fixture passes: key7 remains exact, backing
  is at most4,096 bytes, and scratch credit survives operator/cache drop until
  the returned keys are dropped. This strengthens its former unsafe
  pool-zero-while-keys-live assertion; the original32,864>4,096 RED remains
  preserved above.
- **78144, exit0**, 1m10s build / 0.01s runtime: two additional primitive
  proof cases pass. Per-poll allocation-counter live and peak bytes fit the
  independently read actual scratch reservation. A64-row old-SQL acceptance
  comparison with4,096 bytes free passes with exact pairs and unchanged
  operator status. These already-GREEN cases are additional proof, not REDs.
- **86752, exit0**, 53.66s build / 0.01s runtime: old SQL first succeeds with
  remaining capacity equal to actual scratch credit plus1,024 bytes; paid
  cache admission is independently confirmed and the optimized query also
  succeeds. This does not demonstrate typed optimized-execution OOM cleanup
  or retry. A128-byte headroom variant is prepared but not yet executed.

Sessions35779 and22739 were fixture signature/formatting compilation errors
and are not behavior RED evidence. The supported-shape scan, wider key types,
probe constructor, explicit constructor cancellation carriers, typed SQL OOM
cleanup/retry and private owned DataFusion entry are still unfinished.

The current producer proof covers buffers carrying the lease. DF-created
replacement buffers and consumers retaining only Schema/Field metadata require
additional lifetime protection. A prepared metadata-only real-producer gate
will test that boundary; it has not run yet. A query-local guard, TaskContext,
alias deregistration, or scoped pool alone is not represented as completion
proof. Fresh canonical metadata ownership and the unchanged post-eviction
196,896>4,096 RED also remain open. No source commit, publication, performance
measurement or whole-J2a GREEN claim has been made.

## Main-based continuation: private metadata constructors

The next isolated branch is `feature/stream-join-columnar-state-j2a`, under
`target/worktrees/stream-join-columnar-state-j2a`, based exactly on merged main
`0d94b2f160a9c96753c7271e4f152d5647519da4`. Four tracked diffs were transplanted
with `git apply --3way` without conflict; seventeen scoped untracked files
were copied byte-exact. The original prototype worktree remains unchanged.
The transplant receipt and retained patch are under the new worktree's
`target/issue363-j2a-main-transplant-v1/`. Main's empty-compaction progress fast
path and merged ASOF retirement/coverage corrections were preserved.

The first shared-cache invocations discovered zero tests because Cargo's
dep-info still named another `target/worktrees` source root. They are not
behavioral passes. Only stale core fingerprint records were retained and
invalidated; warm dependencies and immutable release/build seals were not
changed. The genuine current-source baseline compiled in 3m07s and found the
requested container case. Container count and initial slice backing passed.
Post-eviction backing still failed at **196,896 >4,096**; the real metadata-only
DF producer still failed because readable scratch metadata survived its
credit. These remain explicit open gates, not fixed by metadata normalization.

### Actual metadata RED and correction

`test_owned_copy_normalizes_metadata_history_without_retaining_caller_owners`
failed before production changes: a one-entry schema/field metadata history
with large spare strings retained **3,533,224 bytes** of chunk credit. It now
requires private logical-value construction, caller Schema/Field Weak release,
unchanged standalone V1 row IPC, live exact pool/guard equality, and final zero.
The corrected case passed in session39669 (1m08s compile, 0.01s test); initial
visibility/qualification compilation fixes are not RED evidence.

`columnar/metadata.rs` replaces arbitrary caller-capacity inference with fresh
Schema/Field/name/map/timezone construction under the chunk's real prepaid
guard. Fresh HashMap bucket allocation follows locked Rust/hashbrown
entry-count construction: 4/8/16 small buckets, then checked 8/7 load and power
of two, with 16 control/group bytes. Field and schema Arc controls, String
length capacities, pointer-vector/Arc overlap, and timezone String-to-Arc
overlap are paid. No caller metadata map or string-capacity clone is retained.
Maps above 32 entries, keys/timezones above the bounded fragment size, or
schemas above the pointer-transfer bound conservatively retain Legacy.
Values and names are copied in UTF-8-safe fragments under the existing byte
and visit quantum. Original schemas, metadata, flags, timezone spelling,
FlatV1 fees and V1 row serialization remain unchanged.

Per-field inspection/construction is quantum-stepped. Final `Fields` creation
is separately a contiguous `Vec<Arc<Field>>` to Arc-slice transfer, not a new
per-field inspection loop: 128 fields copy 1KiB of initialized pointers;
read/write bytes are explicitly counted. This does not assert that a
64-inspection bound alone proves every constructor's allocation latency.

The 3/16/128 requested-allocation fixture initially reported peak2836 versus
payload guard2140 because it also constructed its test job and separately
funded selection inside the payload measurement. The corrected attribution
creates those fixture inputs before measuring `owned_payload` itself; no
unexplained subtraction or fee increase was used. All three widths then pass
actual live/peak-versus-guard inequalities and final allocation/pool zero.
An additional independent constructor fixture with metadata histories and
0/1/7/32 entries passes the same live/peak/final-zero checks. Equal caller
schema release, private schema surviving operator drop, and conservative
33-entry exact Legacy fallback also pass. Five frozen V1 captures and every
continuation passed after the constructor change. The existing 128-header
yield/cancellation boundary passed. These are scoped checks, not full J2a GREEN.

### SQL acceptance and containment gates still open

The tighter zero-headroom case is a genuine new regression test. Original
SQL accepts the input; an independently confirmed optional paid cache reserves
2,852 bytes, then the optimized query fails when HashJoinInput requests another
16 bytes with zero available. Session79723 exited101 (1m02s compile, 0.01s
test). The error was observed after public error projection; a correction
must classify the internal typed error before projection, release actual
failed objects/credits, and preserve original SQL acceptance before claiming
OOM retry coverage. The earlier 128-byte-headroom case passed and is not RED.

The production `Unproved` branch currently passes the same new paid tables to
`sql_validated` and then releases their funding. It remains unsafe for unknown
producer graphs and is not called Legacy provenance. The original real
metadata-only producer RED remains a counterexample to arbitrary naked
SchemaRef lease protection, outside the proposed exact serial-plan subset.
No assertion replacement is claimed to fix that producer. True Legacy
rebuilding or proven actual owner propagation still needs its production
fallback RED and review. An IPC-detachment proposal has no approved temporary
funding contract and was not implemented. Sparse replacement remains unimplemented.

Final scoped lint for the metadata slice includes minimal imported-prototype
cleanup: explicit imports, explicit destructuring aliases preserving owner
drop order, and a static literal return lifetime. New metadata functions and
directly changed constructor functions have maximum CCN8. Whole-source
approval, full-module GREEN, publication, CI and performance remain pending.

## Metadata scan correction: empty-map profile

Independent review requested changes on the first constructor snapshot.
`HashMap::iter().next()` can scan arbitrary caller empty buckets before the
next quantum step. A logical entry count, or public capacity after deletion,
does not bound that work. Its historical 0/1/7/32-entry allocation checks do
not prove the missing scan bound. Their exact source and results remain in
`metadata-first-chunk-v1`; they are not general passes for the new profile.

The optional columnar profile now requires the original schema metadata map
and every field metadata map to be empty. The cold inventory checks each map
with `is_empty()`, without constructing its iterator. Runtime construction
creates fresh empty maps and never enumerates caller metadata buckets.
Nonempty metadata selects the existing Legacy payload path, preserving public
input support, caller objects, exact row IPC and existing logical charges.
No caller-capacity predicate, fee increase or new public rejection is used.

The actual new RED was
`test_sparse_schema_or_field_metadata_uses_exact_legacy`: normal admission
installed Shared payload instead of the required Legacy payload. Session92756
exited101 after 57.74s compilation, with one discovered failing test. After the
minimal correction, session95611 exited0 after 1m23s compilation; both isolated
schema-only and field-only sparse-map cases passed in the one test. They retain
exact original IPC, physical ID, original FlatV1 retained charge, original
SchemaRef, and final exact pool-zero assertions.

Eight directly affected checks also passed separately: nonempty metadata
history remains Legacy and read-only; empty-after-history constructor
allocations are covered by actual funding; 33-entry Legacy remains exact;
equal caller schema/name/timezone owners release; the private empty schema
outlives the operator with real credit; 3/16/128-column constructor allocations
are funded; all five frozen V1 captures restore and continue; and the 128-header
copy yields before retention. The source-name spare-capacity, timezone and
Weak-owner checks remain meaningful on the supported empty-map profile. The
final initialized field-pointer transfer remains separately accounted.

Scoped `cargo clippy --locked -p calc-flow --lib --tests -- -D warnings`
passed on this worktree's source in 28.33s. Formatter, whitespace and generated
contract checks passed. Lizard1.23.0 found no CCN above8 in the two changed Rust
files. Evidence is under
`target/issue363-j2a-main-transplant-v1/metadata-empty-profile-v2/`.

Independent Source Review approved this exact metadata-only correction; see
`target/issue363-j2a-metadata-empty-profile-review.md` in the root worktree.
SQL cleanup/fallback and post-eviction sparse-copy source are unchanged. The
previous real producer and zero-headroom SQL failures and post-eviction
196,896>4,096 failure remain explicit open gates. No full test module,
performance, full workspace, coverage, CI, commit or publication was run.


## SQL owned slice: actual consumers, fallback and teardown

This local slice closes the identified SQL ownership paths for independent
source review. It does not complete J2a or approve Sparse, wire V2, arbitrary
SchemaRef escape, publication or performance. The metadata empty-map profile,
Boolean/null-mask and unsupported-type Legacy fallbacks remain unchanged.

Constructor-owned buffers now carry a private funding lease through the
locked Arrow/Bytes owner graph. The lease owns schema and actual credit, then
the managed retirement guard; it never owns its own arrays. Arrays and buffer
owners precede their leases. Strings are copied from borrowed caller bytes in
the existing bounded quantum, then their builder finish, complete offset and
UTF-8 validation execute in the existing OwnedCpuWork path. No caller buffers
or schemas are detached into workers. Budget denial of this optional worker
selects Legacy; cancellation and other runtime errors keep their original
meaning. A healthy context with closed private gather admission also selects
Legacy, while its actual cancellation/deadline remains an error.

A gather attempt holding a resident lease can temporarily form
Home→Attempt→lease→Home. This is not described as an acyclic whole graph.
Existing finish/abandon detach the attempt before destruction outside the home
state lock; managed tasks drop retained operators before whole-job drain.
Controlled consumer, dropped checkpoint and actual managed runner checks below
exercise those boundaries. The private lease itself has no array back-edge.

The optimized SQL entry continues to admit only its exact serial physical
plan subset. An unproved plan or nonserial runtime releases new paid tables
and optional paid cache before rebuilding genuine Legacy key tables. Legacy
key arrays use the declared canonical input types, preventing private copied
timezone/type controls from escaping through Arrow concat. Existing resident
buffer leases remain attached to any actually shared copied buffers.

Typed DataFusion ResourcesExhausted is detected through the locked DF54
find_root before error flattening. After actual serial cleanup and optional
cache release, the equality path retries old SQL once. A non-budget typed
error never retries regardless of display text. The observed Legacy input
registration count is two tables for exactly one zero-headroom retry, zero
for the accepted 128-byte-headroom path, and zero for a quoted missing column
whose name deliberately contains ResourcesExhausted. Match pairs, statuses,
caller input and original same-budget SQL acceptance remain exact.

Collected output columns use the declared canonical output DataType after
independent concat. This does not copy their buffers again or change IPC,
values or public schema. It severs the otherwise escaped private timestamp
control. Actual end/cancel tests retain output while all task resources and
the runner cleanup observer settle; the retained timestamp and UTF-8 values
remain readable.

| Boundary                          | Actual pre-fix evidence                            | Observed local result                          |
| --------------------------------- | -------------------------------------------------- | ---------------------------------------------- |
| Primitive real DF consumer        | pool 0 while actual payload credit was 2,072       | paid/Pending/Weak and final exact-zero pass    |
| UTF-8 / LargeUTF-8 real consumer  | pool 0 versus actual 3,016 / 3,024                 | both pass, including actual consumer cleanup   |
| Unproved / nonserial SQL fallback | actual registered Legacy consumer kept paid schema | both pass with private Weak gone and pool 0    |
| Collected timestamp controls      | private timezone Weak survived end and cancel      | both pass after canonical type replacement     |
| Zero-headroom optional scratch    | formerly accepted SQL returned typed budget error  | one observed old SQL retry succeeds            |
| 128-byte-headroom scratch         | prior accepted control, no new behavior RED        | same pairs/status, observed old retry count 0  |
| String attempt/generation denial  | direct current admission guard checks              | both actual stage fees/refunds and Legacy pass |
| Healthy closed gather admission   | pre-ticket checkpoint fixture returned Cancelled   | optional Legacy and all old owner proofs pass  |
| Managed end/cancel with output    | new regression fixtures, no claimed behavior RED   | both actual public runner + cleanup pass       |

Initial fixture compiler errors and the unquoted SQL identifier display
assertion are diagnostics, not behavioral RED. Quoting preserves the intended
text; the non-budget test then passes with observed registration count zero.
The managed source fixture initially declared NeverEmits with the default
SourceProvided policy. Its explicit Disabled policy fixes this validation
mistake, not an engine failure.

The historical naked-schema counterexample remains in frozen metadata source
and `sql-owned-slice-v1/naked-schema-original-counterexample.rs.txt`. Its active
replacement still witnesses a real schema-only Repartition producer after
array/operator release and demonstrates that the actual physical plan is
refused by the owned whitelist. That deliberately naked schema has no lease;
only actual Home/generation funding remains. It is not claimed to be a newly
supported escaping schema. The producer is actually released and its Weak
clears; no test is ignored and the old failure evidence is preserved.

### Actual allocation attribution

Allocation-counter 0.8.1 observes only the current thread. The previous mixed
3/16/128-column constructor check, run unchanged after worker introduction,
reported a cold 3-column peak127,242/live124,930 against payload3,086. A warm
check reported peak4,009/live1,697. These are retained diagnostics, not proof
that residents are underfunded: the first includes process service startup,
and both omit worker construction while attributing gather dispatch to the
payload. The existing process InfrastructureCredit owns its bounded joiner
and worker startup separately; this slice does not create or widen that budget.

The final check retains the same three mixed widths. It independently reads
warm Home16,384 + generation16,384 + attempt0 from their actual owners. Fixed
cfg(test) stack statistics return actual string-worker allocation counters;
existing generic work/output fees pay their control size. A nested per-poll
meter separates dispatch from pure construction. Resident bytes combine
caller and worker counters; the conservative peak includes the worker body
and the derived boxed-output size64. The actual work reservation is8,568 in
this instrumented build (the production return contains no statistics).

Observed resident bytes are2,121/10,159/78,339, below actual payload guards
3,086/14,522/112,942. Conservative peaks6,249/15,122/89,126 are bounded by
payload plus the one actual live work credit. Dispatch return controls are
excluded from resident payload attribution and included in work funding.
Final resident allocation/refund equalities and final pool0 remain exact.
This is an allocation/ownership proof, not RSS or a performance measurement.

All five frozen V1 captures still compare byte-for-byte, restore and continue.
The original retained/dirty container scaling and long-string quantum checks
passed. Direct dropped checkpoint, reset/restore, cancellation/deadline,
failed preparation, admission refusal and actual-attempt-refund checks passed;
only their fixture teardown changed to examine actual worker release and live
state funding before operator drop, then whole-job drain and exact pool0.
The SQL source has no shared gather, ASOF, dependency, public API, schema,
semantic capability, logical FlatV1 charge, checkpoint framing or wire-version
change. Remaining post-eviction sparse backing is still the real
196,896>4,096 RED with retained1/evicted8,191/nextID8,192 already correct.

Exact logs, command receipts, source manifest and review copy are under
`target/issue363-j2a-main-transplant-v1/sql-owned-slice-v1/`. Full module,
workspace, coverage, remote CI and performance are unrun; Sparse is not
implemented and the complete J2a source is not PR-ready.

Scoped Clippy `cargo clippy --locked -p calc-flow --lib --tests -- -D warnings`
passed in1m17s. Its first failure was limited to underscore-read names and a
test-only consumed-input style warning. Their field renames preserve the
exact owner/credit/guard order; no rule was suppressed. Final new and changed
SQL functions have CCN≤8. Six older Join functions exceeding8 retain their
exact metadata-v2 function bodies and are not SQL changes. Formatter,
whitespace and generated-contract results are recorded in the final receipt.
# SQL owned slice v2: closure after allocation

The v1 review found a legal pre-ticket interval after the resident retirement
guard is registered: bounded string copying can yield before requesting a gather
scope. Closing the gather home during that interval originally returned a new
`Cancelled` error even with a healthy job context. The focused test reaches an
actual 32 KiB StringBuilder allocation before closing admission. It failed with
`Cancelled { run_id: "1" }`; the real-cancellation control already passed.

Scope and submission cancellation now check the real context first. A healthy
context treats this optional work refusal as Legacy fallback; actual cancellation
or deadline remains an error, and other errors retain their original variant.
Already accepted worker completion retains its previous error handling. Final
source checks pass both mid-copy cases and the direct deadline/non-cancellation
classification cases. The healthy case preserves identity and exact V1 row IPC;
both cases retain exact final pool-zero and owning drain assertions.

Evidence is under `target/issue363-j2a-main-transplant-v1/sql-owned-slice-v2/`.
The earlier poll-macro fixture compile error is separately recorded and is not a
behavior RED. V1 source/proof archives remain unchanged. Sparse eviction density
is still unresolved; this correction does not claim whole-J2a completion.
# Sparse backing slice v1

The preserved fixed-width post-eviction RED retained 196,896 buffer bytes for one
surviving row after 8,191 evictions. Two additional focused REDs showed that a
candidate did not retry after actual credit release, and a short surviving string
kept 254,912 bytes. Identity, logical counters and exact row IPC were checked
before these failures. The original broad test filter also matched one unchanged
ASOF test; its incidental pass is not Join evidence.

Constructor-owned chunks now prepay an immutable per-row identity/time/byte
inventory and their control objects. Owned commit, eviction and installation
maintain live flags/count/bytes incrementally. A touched chunk is copied only when
its known buffer backing exceeds 4,096 bytes and four times its live payload bytes
plus terminal string offsets. Only that chunk's inventory is scanned, through the
existing copy quantum. No full retained-state or dirty-log scan is added.

Each retained side owns a private FIFO whose links live inside prepaid chunk
controls. Admission denial rotates the candidate; cancellation or dropping the
copy future leaves it registered. Progress attempts its entry-time queue length,
once per candidate. Progress without new expiry can retry queued candidates after
funds are released. Progress with no expiry and no candidates retains the existing
ready-free path. Clear/drop iteratively remove links; reset/restore replace the
owned retained side and cannot leave a recursive chain or cycle.

Copying borrows the known chunk columns and uses the existing prepaid constructor,
CopySelection and string CPU worker. Installation resolves live identities through
the expiration index and rebinds the same retained row and existing dirty upsert.
Physical IDs, keys, event times, logical charges, dirty order and V1 framing do not
change. Existing aliases keep the complete original credit until actual release;
allocation denial keeps the original chunk instead of producing a new error.
These are buffer-density and actual-funding claims, not RSS or speedup claims.

Observed final checks: four sparse refusal/string/drop/cancel cases, seven direct
V1/container/allocator/eviction cases (including the original backing case), two
actual managed end/cancel cases with collected output, and four existing ordered
schedule/recovery/reference properties passed. The allocation case retains the
existing 3/16/128-column shapes; the V1 case compares all five frozen captures and
restores/continues each. Clippy lib/tests, format, contracts, whitespace and changed
function complexity checks pass. Raw commands and frozen bytes are in
`target/issue363-j2a-main-transplant-v1/sparse-slice-v1/`. Unchanged SQL-v2 checks were
not repeated. No full workspace, coverage, performance run, commit or push occurred.

## Sparse backing slice v2: mixed-retention admission

Final source review found an admission path that the eviction-only queue did not
cover. A normal batch can admit rows for matching while retaining only its young
row. Copy selection includes those admitted, retainless rows, so a newly committed
chunk can already exceed the density threshold without any later eviction.

The new normal-process test uses four batches of 512 rows, retaining one row from
each batch under existing opposite-side progress. Before any later progress event,
physical IDs `[511, 1023, 1535, 2047]`, next ID 2,048, retained count four, eviction
count zero and exact V1 row IPC all pass. The old implementation then fails with
12,576 retained buffer bytes, above 4,096. This is the recorded behavior RED, not
an inferred timing regression.

After all new retained rows have been marked live, the owned append boundary
checks only those new rows and enqueues already-due chunks in the existing funded
FIFO. Its existing flag deduplicates chunks. Data processing accepts output,
updates emitted metrics and commits the original logical state before attempting
the same sparse repair used by progress. Successful data calls therefore repair
new admission candidates without needing another watermark. No pending candidate
adds no repair await; no full-state or dirty-log scan is introduced. Allocation
denial keeps the original funded chunk queued for a later data/progress retry.

Repair can yield after logical commit. A cancelled or dropped repair future keeps
the accepted logical prefix, dirty entries and registered head candidate; old
resident credit stays with actual owners. It does not roll back accepted output
or committed row IDs. New replacement owners still drop before their credits.
The existing refusal, dropped-future and actual-cancellation controls remain
green. Synchronous installation is an atomic representation commit over prepaid
locators; it is O(survivors) and has no asserted 64-row or whole-handler latency
bound. The copy/planning quantum remains separate from that installation boundary.

The new test passes, including exact live-state funding and final pool zero.
Six directly affected existing tests also pass: refusal retry, dropped future,
actual cancellation, all five frozen V1 captures with restore/continuation,
no-expiry/no-candidate visits and cancel/deadline checks. The remaining final
format, lint and complexity receipts are recorded with the immutable corrective
snapshot in `target/issue363-j2a-main-transplant-v1/sparse-slice-v2/`. Earlier
Sparse-v1 and SQL/metadata snapshots and their raw evidence remain unchanged.
No new performance, full suite, coverage, commit or push is part of this correction.

## PR377 CI correction: actual allocation and credit attribution

The first Windows, Linux parity and combined coverage jobs fail the same five
tests with identical numbers. Windows reports 1,911 passes and five failures;
Linux parity and coverage report 1,921 passes and five failures. Coverage stops
at those test failures, so that run does not establish the line floor. Each of
the five failures was also reproduced once, in its own focused Linux process;
the raw CI and local failure evidence is preserved. This is a cross-platform
fixture-path issue, not a claimed Windows allocator difference.

The native refund assertions omit actual gather home/generation credit: both
native cases observe an extra 32,768 bytes. The fanout fixture creates separate
short-lived jobs for its two chunks and observes another 16,384 bytes from the
string-copy generation. The corrected fixtures use the existing isolated test
service and one explicit owned job. They read home/generation credit directly,
require attempt credit zero, and compare exact pool totals with independently
observed payload/key/index guards. The original key-vector clear assertion,
append grow/drop equality, full-budget failure, SQL fallback count and unchanged
operator status assertions remain intact. Dropping each actual resident owner
is followed by the corresponding exact resident-credit decrease; whole-job drain
then requires generation/attempt zero, and job drop requires final pool zero.
Service shutdown follows runtime drop, outside Tokio, using the existing fixture.

The cleanup-only fixture omits 14,056 bytes of still-live operator payload/key
credit while checking a deliberately blocked compactor refund. It now includes
that independently observed resident state without weakening the gated attempt
credit, Pending, cancellation, deadline or unchanged-status assertions. After
the gate opens, the existing compaction cleanup observer completes before the
live-state pool equality is checked. The operator is dropped before whole-job
drain, matching the managed lifecycle and preserving final pool zero.

The metadata measurement previously wraps runtime driving and payload construction
in one allocation interval, yielding a peak of 6,053 bytes against the 3,130-byte
payload guard. Its corrected interval polls the complete primitive-copy future
directly with an already constructed fixed waker/context and a stack-pinned
future. Copy, fresh schema/fields/strings, buffer owners, inventory and reservation
allocations remain inside the measurement; executor driving is excluded. Actual
wakes are observed, without claiming an additional quantum counter proof. The
owned-copy measurement reports 41 allocations, 1,941 live bytes and a 1,965-byte
peak against the unchanged 3,130-byte guard. Real chunk destruction offsets the
entire measured live allocation and refunds the pool to exactly zero; metadata
and V1 IPC comparisons remain unchanged. No arbitrary allocation allowance or
production funding increase is added.

All five corrected focused tests pass locally, plus one directly affected flat
offset/variable-value control. The first isolated-service adaptation passes all
funding/cleanup assertions but then fails the service's existing prohibition on
shutdown inside Tokio; that diagnostic is preserved separately from the CI RED.
The final wrappers perform shutdown outside the runtime and pass. Metadata and
cleanup-only passing checks were not repeated after unrelated wrapper extraction.
Final commands and frozen evidence are under
`target/issue363-j2a-windows-fix-v1/`. Only three cfg(test) modules and this analysis
change; all twelve other Rust files in the approved nineteen-file scope are
byte-identical, including all production funding and ownership code. No benchmark
rerun, full local suite or coverage run is part of this correction. Windows and
full CI acceptance remain pending the new PR head.

## PR377 review correction: optional-admission fallbacks

The test-only CI correction is preserved separately in its immutable v1 snapshot.
Subsequent review identifies two production fallback issues, both confirmed by
new focused behavior REDs before their fixes. A healthy context with a closed
scratch home receives `Cancelled { run_id: "1" }` from optional admission. The
actual-stop control also observes the home ID instead of the current context's
ID. Scratch construction now classifies that admission refusal through a fresh
context check: a healthy context gets optional `None` and the existing true
Legacy path; actual cancellation/deadline retain their current-context failure.
Schema/credit remain local until a real retirement guard is registered, and
denied construction refunds its complete optional credit.

The native refusal RED first demonstrates that original SQL accepts the same
budget with 1,408 bytes available, exactly the three-row native-index charge.
`evaluate_matches` then successfully builds that index, refuses native scratch
and leaves zero bytes for SQL's HashJoinInput, which fails even a 16-byte growth.
On optional native `None`, the opposite-side unused index is now dropped before
SQL fallback. Its actual entry/key owners drop before its reservation; retained
row IDs, encoded keys, payloads and logical state remain unchanged. Native
success and non-budget errors keep their original paths. The counter check uses
the original SQL's actual same-budget table-build count, including its existing
owned-attempt/Legacy retry; an initial fixture assumption of one build instead
of two is preserved as a diagnostic, not a new production behavior failure.

All three new focused behavior tests pass. Direct controls also pass for all
five frozen V1 captures with each restore/continuation, typed zero-headroom
cleanup followed by one Legacy retry, and non-budget failure without retry.
The earlier five CI-fixture corrections remain preserved as passing evidence
and are not repeated here. Production changes are limited to native optional
refusal cleanup and scratch optional-admission classification; their changed
functions have complexity seven, seven and three. No fee, cap, schema, public
API, checkpoint format or shared gather implementation changes are introduced.
Final v2 source/proof records live under `target/issue363-j2a-windows-fix-v2/`.
The previous performance acceptance belongs to head `88e817`; this additional
production delta has no new timing claim. New independent SourceReview and
cross-platform/full CI acceptance remain required.
