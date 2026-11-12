# Bounded global DISTINCT — accounting blocker evidence

DAL-519, baseline and final product HEAD
`be12fd45a18daa5d2f6415993141a8842e114b1f`, inherited branch
`agent/dal-orchestrator/d83ca711d94c`. The frozen API note is preserved unchanged.
No feature implementation, commit, push or PR is delivered in this run.
The issue remains `in_progress` for coordinator resolution.

## Finding and contractual consequence

The specified backing rule cannot establish the frozen requirement for Python
Arrow C Stream input. A one-row, nonzero-offset slice imports as a native array
with a logical array charge of **112 bytes**, while retaining its **8,388,608
byte** parent allocation. All Python references to the input are dropped before
measuring; the Rust imported table keeps the allocation alive. Dropping the
Rust table releases it. This is a data buffer, not an opaque allocator or owner
control-block overhead and not an RSS measurement.

Actual probe stdout, Linux x86_64:

```text
rows=1 owner_bytes=8388608 rust_array_charge=112 buffer_len=8 buffer_capacity=16 live_bytes_after_python_drop=8389296
live_bytes_after_rust_drop=0
```

The 8,389,296 live bytes include the 8 MiB data buffer plus Arrow export/schema
allocations. The counterexample depends only on the independently asserted
8,388,608-byte data buffer. The imported array exposes an 8-byte visible buffer
and a 16-byte FFI capacity; neither is the full allocation. A slice near the
beginning of an arbitrarily larger parent has the same small reported charge.
Fixed Batch/schema/container surcharges or a fixed multiplier cannot bound it.

The frozen API note requires both `sum(array.get_array_memory_size())` and
coverage of retained backing for nonzero-offset slices. Its acceptance seed 4
explicitly rejects using visible bytes as a substitute. Applying the specified
existing helper would undercharge this permitted input. Reserving DataFusion
workspace, metadata, candidate/result or checkpoint bytes cannot reveal the
missing ingress owner size. Copying keys later still leaves the original
unmeasured ingress alive during projection, which the note also requires
charging. Silently rejecting Python/C Stream ingress or weakening this budget
would materially alter the frozen four-surface contract.

The trigger explicitly directs the implementer to report actual held-memory
coverage failures with source evidence to the coordinator and not expand scope.
That exception is exercised here. The coordinator needs to settle an ownership
accounting seam for foreign ingress (or explicitly revise its admission/charge
contract) before end-to-end implementation resumes. No general fix to legacy
snapshot SQL, bridge ABI, engine policies or team/gates is included.

## Source chain at the controlling SHA

- `crates/calc-flow-python/src/batch.rs:337` (`from_pyarrow`): obtains the
  caller's `__arrow_c_stream__`, imports through
  `PyTable::from_arrow_pycapsule`, consumes its RecordBatches and installs them
  in `calc_flow::Batch::table`; it neither copies buffers nor records the
  parent allocation size.
- `crates/calc-flow/src/operator/snapshot_sql/mod.rs:484` (`backing_bytes`):
  charges `512 + schema_bytes + sum(256 + array.get_array_memory_size())`.
  This is the existing helper cited by the frozen note, not a new DISTINCT
  implementation being represented as already complete.
- Exact registry dependency `pyo3-arrow-0.18.0/src/table.rs:58`: imports an
  ArrowArrayStreamReader and retains its RecordBatches.
- Exact registry dependency `arrow-array-59.2.0/src/ffi.rs:256`:
  `create_buffer` wraps an FFI pointer using
  `Buffer::from_custom_allocation(ptr, len, owner)`. The owner is opaque and
  `len` is inferred from the FFI logical length/offset; the parent's tail
  allocation size is absent from the C Data description.
- `arrow-array-59.2.0/src/array/primitive_array.rs:1256`: memory size uses
  values/null buffer capacities plus the native array descriptor.
- `arrow-buffer-59.3.2/src/bytes.rs:101`: a custom allocation returns the
  supplied size and explicitly notes that underlying capacity can be larger.
  The public comment in `buffer/immutable.rs:187` still says external capacity
  is zero; that comment is stale. The actual implementation and observed
  capacity are **16**, not zero.

The probe locks the same pyo3/pyo3-arrow and Arrow package versions used by
this checkout, including arrow-array/cast/select 59.2.0 and
arrow-buffer/data/schema 59.3.2. `Cargo.lock` is included in the attachment.
No alternate checkout or installed calc-flow binary supplies the observation.
The probe calls exactly the public import seam used by this checkout and needs
no unsafe application code or product changes.

## Reproduction and validation

Environment: Rust/Cargo 1.94.0, uv 0.11.19, CPython 3.13.9, PyArrow 24.0.0,
Linux x86_64. All build/environment/probe outputs are under this checkout's
`target/` tree. Full regression, coverage, Python feature checks, examples,
stubs, formatting of a product implementation and CI are not claimed.

Unpack the attached `foreign-backing-probe.tar.gz` under `target/` at the
repository root, then run:

```sh
UV_PROJECT_ENVIRONMENT=target/venv uv sync --extra dev --no-install-project
target/venv/bin/python target/foreign-backing-probe/run_probe.py
```

The runner derives PyO3's absolute interpreter, site-packages and shared-library
path from that interpreter. It invokes `cargo run --locked` against the probe
manifest with a separate `target/foreign-backing-build` directory.

Observed result: **exit 101**, intentional assertion failure:
`frozen get_array_memory_size rule omits retained foreign backing`.
Before that assertion, the independent checks for an 8,388,608-byte parent,
at least that much live Arrow pool storage after deleting Python references,
and release back to the initial pool level after dropping the Rust owner all
pass. This is a focused RED contract counterexample, not a product test pass.
An initial harness attempt used a relative PYO3_PYTHON path, failed before
running the probe, and was corrected to the interpreter-derived runner above.

Before finding this blocker, the finite prototype followed RED → GREEN:

- `cargo test -p calc-flow --test bounded_global_distinct test_bounded_distinct_finite_plan_isolates_complete_histories --no-default-features`:
  RED, exit 101, missing `DistinctSpec` and `TableExpr::distinct` as expected.
- `cargo check --locked -p calc-flow --test bounded_global_distinct`: caught
  one trait-method import error in the prototype, which was corrected.
- `cargo test --locked -p calc-flow --test bounded_global_distinct test_bounded_distinct_finite_plan_isolates_complete_histories`:
  GREEN, 1 passed. Its literal oracle covered [1,NULL,1,NULL,2] → {1,NULL,2},
  then [7] → {7}, then typed empty, using one finite plan without reset.

That prototype and its smoke test were withdrawn when the held-memory
counterexample invalidated the proposed budget implementation. Their patch and
files remain only in ignored `target/` for local recovery. **The earlier
single GREEN does not validate the final source tree or satisfy DAL-519.**
No stream/source-proof/checkpoint or four-surface feature result is delivered.
The attached Rust probe passes rustfmt; its Python runner passes Ruff lint and
format checks after correcting import order. The final source formatting did
not change the observed ownership counterexample.

## Final workspace

Tracked product files match the original HEAD exactly. Only this evidence file
and the original frozen API note remain untracked. Product changes were reversed
using their own recorded patch; the pre-existing note was not overwritten.
The source-only reproducibility attachment includes the Rust probe, manifest,
lockfile, interpreter-derived runner and actual stdout/stderr. No detached work
or background polling remains. The coordinator receives the blocker rather
than an incomplete feature PR; review/merge and all later DAL work remain
outside this run.

## G3 decision addendum — 2026-11-12, API contract v2

This appendix is the API designer's decision requested by trigger
`01a1d06a-103e-75fe-8e90-8b614fb08ae8`. The preceding implementation evidence is
preserved verbatim as history, including the statement that the original note
was unchanged in that earlier run. The controlling product HEAD is still
`be12fd45a18daa5d2f6415993141a8842e114b1f`; there is no product diff or feature PR.
The amended authority is [API note v2](../api-notes/bounded-global-distinct.md).

### Decision and changed guarantee

The original whole-ingress-backing claim is withdrawn. `get_array_memory_size`
cannot identify the allocation behind a custom FFI owner: the recorded 112 B /
16 B / 8 MiB counterexample remains valid. No probe, feature test or product
matrix was rerun in this decision pass. That RED is evidence for the old
contract failure, not a test of the new normalizer.

The selected policy is **copy_visible_v2** at the common native DISTINCT node's
admission boundary, for Rust/Python, DataFrame/SQL, finite/stream and reloaded
projects. Every permitted input, including ordinary native arrays, is read
through a temporary loan and reconstructed into fresh node-owned buffers. No
caller size assertion, generic Batch flag, foreign owner token or capacity
heuristic can bypass copying. Unknown foreign backing is accepted through this
copy path; unsupported reconstruction of a particular column is diagnosed at
that column rather than disabling Python/Arrow.

`max_bytes` now bounds node-owned participating logical storage: normalization
destinations, committed/candidate history, schema/metadata copies, tracked
DataFusion workspace/results, independently compacted final output, control,
IPC scratch and live checkpoint captures. These are added at their overlapping
lifetimes, with reservation before each allocation or growth. The raw external
ingress owner is **explicitly outside this revised bound even while a native
envelope keeps it alive during synchronous borrowing**. Its size is unknown,
not zero and not the small FFI capacity. The policy does not bound total graph,
Source, queue, caller or process storage.

The feature releases every raw strong reference it obtained before its first
internal await/query/emit and on admission failure/cancellation; it keeps no
raw input, ArrayRef, SchemaRef, metadata Arc, capsule or closure in state,
cache, errors or checkpoint. Upstream owners can remain alive independently.
In particular, `pipeline/batch.rs:556,650` holds input maps and produced values,
and `crates/calc-flow-python/src/arrow_source.rs:19,56` retains reusable source
material. A node-local copy does not release those owners. Release tests must
separate these lifetimes explicitly.

Final results are copied into independent compact owned output before emit;
a one-row result must not pin the full normalized H. Pending output and the
copy's overlap with the query result/H remain charged. After successful emit
and release of the node's last reference, ownership belongs to downstream and
falls outside this per-node budget. Captures remain charged until their lease
is actually released. This is a logical ownership boundary, not RSS or a new
external delivery guarantee.

### Static feasibility evidence and alternatives

- Locked `arrow-buffer-59.3.2/src/buffer/mutable.rs:139` rounds an explicit
  fresh allocation to a known 64-byte layout; `buffer/immutable.rs:560` moves
  that owned MutableBuffer into Buffer. Primitive/Boolean/string values can
  therefore be rebuilt from a counted layout without retaining foreign owner
  references. Ordinary builder growth can double capacity and must not occur
  without prepayment. This is a source-based implementation route, not a new
  runtime pass claim; no unsafe application code is proposed.
- A borrow-only first pass counts accepted rows, null bits and string bytes;
  reserve the checked rounded destinations and descriptors before copying in a
  second pass. Offset=0, capacity or `clone/slice/project/to_data` alone do not
  prove detachment. New data, bitmap and string offsets must be independent.
- `operator/sql/ipc.rs:127` measures in-memory Schema/Field/metadata capacity;
  it is not serialized `IPC(schema).len` as the initial note described.
  Deep-copy schema/metadata under a precomputed content/container bound before
  retaining them. Reuse `operator/sql/metadata.rs:30,51,91` reservation patterns
  for content/encoding/decoding, and the `ipc.rs:153` old+new buffer-growth
  prepayment pattern. Metadata content and all simultaneous copies count;
  fixed descriptors do not stand in for an unknown parent data allocation.
- A certified allocator/owner ledger could support a future zero-copy path,
  but the present C Data ABI and public Batch do not carry that evidence.
  Adding it across bridges, Sources and graph paths is outside this feature's
  minimum change. Self-reported byte counts or Python buffer introspection are
  not an authenticated ownership chain. Blanket rejection of Python/Arrow or
  changing generic bridge/SQL defaults is rejected. Copying later while
  retaining the old whole-ingress claim is also rejected by this probe.

### Identity, acceptance mapping and implementer handoff

The public host signatures and SQL macros remain unchanged. Constructors and
macros emit required spec `ingress_policy="copy_visible_v2"`; omitted, altered
or unknown policy fails closed on import/compile. Native configuration becomes
`distinct_v2`, relation state_version remains 1, state_layout/accounting become
2/2, capability lists layout 2 only, and QueryAnalysis/gates advance to r9.
Policy participates in declaration/config/cache/recovery identity. No shipped
DISTINCT state needs migration: the first prototype was withdrawn. The old
proposed layout/accounting 1 is rejected, not reinterpreted. Restores prepay
fresh owned decode storage before installation and Source open; foreign owner
tokens or payload assertions are not proofs.

The implementation acceptance delta is:

1. Reuse the recorded sliced-input fixture. While copied H stays live, drop
   all raw Python/native loan references and verify the original 8 MiB parent
   is released and value 1 remains. A retained external SourceBinding/input
   mapping must be dropped separately; it is not covered by the node policy.
2. Use actual v2 lease stages for exact B/B-1 cases. Verify allocation hooks
   reject before destination allocation, including schema/metadata, old H,
   candidate, live captures, query result and compact output. Do not derive B
   from the raw parent's unknown allocation or the old 112 B array charge.
3. Cover nullable fixed-width/Boolean/strings, nonzero bit/row offsets, empty
   data, long schema/metadata values and spare capacity. Test normalization,
   budget, schema and cancellation failures for raw-owner non-retention and
   unchanged committed generation/closed state. Content survives deep copies;
   caller-owned values remain unchanged.
4. Confirm all four public units and both lifecycles reach the common seam,
   plus project reload and raw-owner release through a real public finite and
   two-Data stream path. Add restore/active-capture overlap and policy/layout
   negative cases. The original finite GREEN remains withdrawn evidence.
5. Implement the private normalizer/provenance and ledger, ingress/final-output
   wiring, schema/metadata prepayment, strict policy/identity changes and r9
   gates. Update Rust/Python docs/stubs/explain/support tables with the explicit
   borrowed-ingress exception; do not alter generic Arrow/SQL defaults. The
   note contains the six concrete implementation increments and full oracles.

This resolves the design decision, with an explicit reduction from the original
budget promise. Positive implementation/ownership validation remains to be
performed by the implementer, followed by independent review. There is no
remaining user choice or new authorization requirement. The issue stays
`in_progress`. This pass changes only the API note and this appended decision;
it does not implement, commit, push, create a PR or delegate work.


## Implementation resumed under v2 (2026-11-13)

The coordinator resumed implementation on the same branch/checkout and unchanged
base. The original counterexample and the designer's v2 decision above remain
verbatim. The withdrawn prototype was used only as a wiring reference; its
operator implementation was replaced by a private owned-normalization seam.

Each admitted column is counted and reserved before a fresh Arrow MutableBuffer
is allocated. Typed primitive readers, rebuilt bitmaps and string offsets/bytes
produce independent offset-zero arrays. Stream admission explicitly drops its
raw Batch before committing the owned candidate and performs no internal await.
The final DataFusion result is normalized again; its leases remain alive through
stream emit. Both graph modes use the same native node and provenance boundary.

Observed new test-first cycles so far (not final acceptance):

- `cargo test --locked -p calc-flow --test bounded_global_distinct test_distinct_v2_finite_owned_history_and_policy`: RED missing symbols, then
  GREEN 1 passed after implementing owned copying and mandatory exported policy.
  Initial test harness serde conversion was corrected before the recorded RED.
- `cargo test --locked -p calc-flow --test bounded_global_distinct test_distinct_stream_copies_raw_owner_and_waits_for_real_end`: RED missing stream
  lifecycle methods; GREEN 1 passed. The original buffer retains only the test
  observer while committed H remains live. Two Data events produce no output
  before normal EOF, and repeated normal end does not emit twice.
- `cargo test --locked -p calc-flow --test bounded_global_distinct test_distinct_sql_macro_exports_native_policy`: RED table function not found;
  GREEN 1 passed with the macro lowered into a native distinct project node.
- Source boundedness preflight is in progress. A first run exposed that the
  existing public streaming error projection stripped the diagnostic; a narrow
  engine-authored projection was added. This run is not yet counted GREEN.

Native/Python integration, closed windows, restore and remaining boundary
acceptance are still in progress. No commit/PR/CI result is claimed here.

## Delivery validation addendum (v2 implementation, 2026-11-13)

The historical counterexample and designer decision above are preserved. The
results below are from new product tests in the inherited checkout; none of the
withdrawn prototype's GREEN results are used as v2 acceptance evidence.

The implementation uses private owned batches and typed scalar copying into
fresh Arrow buffers, including validity, offsets and string bytes. Raw stream
ingress is dropped before any internal await. Schema/field/metadata are copied
independently under a declaration reservation, including compacting spare
capacity. History counts duplicates, and window membership precedes projection
and row admission. Finite calls keep temporary history and NULL snapshots.
The same DataFusion finalizer serves Rust/Python DF and SQL. Final output is
normalized again and held under its lease through emit. Restored history and
cached capture wire bytes are rebuilt independently before installation.

Native identity is distinct_v2, state_version 1, layout/accounting 2 and required
copy_visible_v2; analysis schema/revision is 2/9. Public source boundedness has a
default Unknown Rust method without changing SourceCapabilities struct literals;
the Python descriptor appends a defaulted enum field. Adapters freeze and
propagate that same fact. Mandatory structural/binding admission runs without
strict/latest opt-in. Bounded sources require real EOF; selected windows require
bounded EOF or a validated watermark route on the original time coordinate.
The Python convenience root is lexical until native admission succeeds, and
binding diagnostics are mapped back to logical input names.

This is an owned participating peak guarantee. Unknown external ingress backing
during synchronous borrowing, other graph/source/caller owners, RSS and
untracked DataFusion private overhead remain excluded exactly as v2 specifies.
The exception is never used for retained history, output, or captures.

### Environment and commands

Base: be12fd45a18daa5d2f6415993141a8842e114b1f on
agent/dal-orchestrator/d83ca711d94c; no new checkout or environment. The fetched
main was de09369a (two unrelated join/rolling performance commits, no overlapping
files). Rust/Cargo 1.94.0, Python 3.13.9, uv 0.11.19, PyArrow 24.0.0, PyO3 0.28.3,
pyo3-arrow 0.18.0, DataFusion 55.0.0, Arrow 59.2.0 (locked buffer/schema patches
59.3.2). The existing target/venv supplies native/Python validation. Embedded
PyO3 tests use that interpreter via absolute PYO3_PYTHON, its site-packages via
PYTHONPATH, and the Conda lib directory via LD_LIBRARY_PATH.

The inherited clone has no configured Git name/email. Delivery uses per-command
Git -c identity Cheng Li <scrappedprince.li@gmail.com>, matching the base author
and runtime's on-behalf-of identity; shared/global Git configuration is untouched.

New RED/GREEN evidence:

- Native finite/stream/macro missing-symbol tests were RED before wiring, then
  GREEN against the new node. Finite literals [1,NULL,1,NULL,2] -> {NULL,1,2},
  then [7], then empty on the same plan prove independent complete histories.
- `cargo test --locked -p calc-flow --test distinct_closure`: RED accepted an
  Unknown source without closure; GREEN rejects before source open. Initial
  test setup was corrected to supply a sink and Disabled watermarks before the
  recorded behavior RED.
- `cargo test --locked -p calc-flow --test distinct_admission`: RED analysis
  reported Supported for snapshot upstream; GREEN reports Unsupported, and
  compile also rejects it without an opt-in.
- `cargo test --locked -p calc-flow --test distinct_analysis`: RED revision 7,
  GREEN revision 9 with conditional finality and the borrowed-ingress exception.
- `cargo test --locked -p calc-flow --test distinct_owned_schema`: RED retained
  the original SchemaRef; GREEN releases original schema/field Arcs while H is
  alive, even with an 8 MiB spare-capacity metadata string under a 1 MiB budget.
- `cargo test --locked -p calc-flow --test bounded_global_distinct
  test_restore_rebuilds_capture_without_retaining_input_wire_owner`: RED input
  wire Arc count 2, GREEN count 1 (only the observer); restored H still finalizes.
- `cargo test --locked -p calc-flow --test bounded_global_distinct
  test_distinct_budget_failure_names_node_policy_role_required_and_limit`: RED
  lacked node/required/limit labels; GREEN preserves them with copy_visible_v2.
- `cargo test --locked -p calc-flow --lib operator::distinct::owned::tests::`:
  three checks GREEN. Normalization literal B=9282 succeeds and B-1=9281 refuses
  before schema/array allocation (hook count 0). A real query/compact-output
  schedule with a live capture succeeds at its measured owned peak and refuses
  at peak-1 before compact targets (only the two input-copy hooks ran). The
  empty-history descriptor test was RED (one unfunded allocation hook), then
  GREEN (zero) after reserving all empty array descriptors before construction.
- `cargo test --locked -p calc-flow --lib distinct`: 33 passed (four directly
  relevant ownership/quota/projection checks plus existing tests whose names
  match the filter). No timing or performance claim follows from these tests.
- PyO3 `_distinct_query` bridge: RED missing function, then GREEN mandatory
  copy_visible_v2 policy and native columns. A genuine Arrow C-stream import of
  a one-row slice of an 8,388,608-byte parent returns the PyArrow pool to its
  baseline after process_data while H remains alive and before EOF; final is
  literal {1}. `cargo test --locked -p calc-flow-python --lib test_distinct_`:
  two passed, with matching managed Python/shared-library paths.
- Python unknown iterator/root test was RED (one temporary root made), then
  GREEN with no iterator acquisition or state root. Its diagnostic was RED at
  physical input, then GREEN at stream.inputs.events.boundedness with logical
  component_id events.
- Python expression explain was RED missing the owned-peak/borrowed-ingress
  statement, then GREEN. Native readonly SQL regression exposed incorrect
  query path/code wrapping; the shared native probe now uses the existing
  sql.query/invalid_literal convention.

The focused Rust integration command covers bounded_global_distinct,
bounded_distinct_window, distinct_admission, distinct_analysis, distinct_closure
and distinct_owned_schema: 14 passed in total, including independent Rust DF/SQL
stream literal finals and two true legacy Data-boundary outputs. Additional
finite_union_frontend/latest_relation targeted regression: 15 passed.

The Python focused set covers this feature, native stubs, query history,
finite event windows/union, latest updates and finite inner/ASOF joins. Before
the final additional diagnostic cases, 223 passed with a freshly rebuilt local
wheel. Final diagnostic/guard results and exact native/source hashes are recorded
below after verification completes.

The focused feature checks include real managed checkpoint continuation: save
one duplicate-history row, cancel, reopen at cursor 1, accept the second duplicate
and emit exactly {1}. They also cover nullable scalar slices across all admitted
key types, empty/nonnullable final schema, mixed/shared DAG/schema/export,
W=9 followed by an in-window late timestamp and W=10 final, real error/cancel
without false EOF, failure to emit without committing, and legacy per-batch SQL.

`cargo run --locked -p calc-flow --example gen_v3_schema` generated only the new
variant/definitions (125 lines). `cargo run --locked -p calc-flow --example
bounded_global_distinct` independently built the core crate and printed 3
distinct tuples; it does not use the PyO3 crate or a Python helper. The Python
example ran and asserted {None,1,2}. Scoped Clippy with -D warnings, Rust format,
Ruff on changed Python/stubs, native-stub checks and the complexity ratchet were
run; actual failures were fixed without changing any gate/baseline. Final source
verification is recorded below. Full regression/coverage is left to CI. No
benchmark or performance measurement was run.

### Material contract gap: late-policy representation

API note v2:152 and :279 require post-close window members to follow the existing
error/drop selection. But the frozen public constructors have no selection, and
DistinctSpec contains only ingress_policy/history/state_budget
(`operator/distinct/spec.rs:34`); project OperatorSpec::Distinct contains only
columns/spec (`config.rs:158`), with unknown fields denied. The two macros likewise
carry no selection. Existing native late policy is stored in rolling/cross-section
specs, not a job-wide runtime field.

A declaration-only probe actually observed that the selected-window exports for
to_project(late_policy="error") and to_project(late_policy="drop") were identical.
This was a design/API deviation, not successful drop support. The native node's
post-close in-window path currently returns an error (`operator/distinct/mod.rs`,
late-data branch). The Python guard now rejects a requested drop before export;
the focused test was RED (did not raise), then GREEN. This prevents silently
losing the requested behavior but does not satisfy the note's drop acceptance.

The implementation therefore remains draft. The coordinator must settle an
explicit persistent selector (including Rust/SQL expression, project identity,
cache and restore semantics), or narrow that requirement. No field/constructor
or alternative policy was invented here, and no new expert stage was dispatched.
DAL-519 stays in_progress; no merge or complete product acceptance is claimed.

### Final ownership and diagnostic iteration

The first final Python run was **226 passed, 1 failed**. Its budget exception
already contained the safe policy/role/required/limit values, but the PyO3
reason-name projection returned None for the newly typed StateBudgetExceeded
variant. The existing focused test was RED at that exact assertion; the native
enum-to-name mapping was completed rather than relaxing the assertion.

The final declaration-ownership test deserializes a two-character time name
with an 8 MiB String capacity and constructs a node under a 1 MiB budget. The
freshly compiled core test executable (`target/debug/deps/calc_flow-a032a9f4fe8a06a3
test_distinct_declaration_compacts_owned_spec_strings --nocapture`) was RED:
the node retained the excess capacity. The minimum fix copies the validated
spec after declaration prepayment, then explicitly drops the incoming spec.
An invalid or oversized unknown time name is still rejected before this copy.
This extends the schema/field/key ownership proof to deserialized config.

The safe budget projection now occurs only at the native DISTINCT boundary.
It emits existing typed OperatorReason failures with policy, whitelisted role
and parsed numeric counters. The general runtime projection still accepts only
typed reasons; it does not infer trusted budget details from arbitrary callback
messages. Python/stubs expose the same state_budget_exceeded reason string.

The final `cargo test --locked -p calc-flow --lib operator::distinct::` is GREEN:
**5 passed**, including the new config capacity test, exact/B-1 allocation
prepayment, live-capture query peak, empty descriptors and safe budget reasons.
The two focused RED assertions above remain unchanged.

Final core integration verification on the completed product sources:

```sh
cargo test --locked -p calc-flow --test bounded_global_distinct \
  --test bounded_distinct_window --test distinct_admission \
  --test distinct_analysis --test distinct_closure --test distinct_owned_schema \
  --test finite_union_frontend --test latest_relation
```

**29 passed**: 14 feature cases and 15 adjacent union/latest regression cases.
This includes the final config-copy and typed budget-reason implementation;
no withdrawn prototype binary was used for these results.

### Completed local verification and build provenance

Implementation commit: `275db61941d8455b9fadec6bedfb692bd0093f3e`, on the
instructed `agent/dal-orchestrator/d83ca711d94c` branch and original checkout,
based on `be12fd45a18daa5d2f6415993141a8842e114b1f`. The following evidence-only
commit changes this analysis document; product sources remain identical.
User Git identity was used for the commit. No new checkout or environment was
created; the matching wheel was installed into the existing environment. No
team configuration, dependency lock, gate or coverage baseline was modified.

Actual environment: Linux x86_64, rustc 1.94.0 (4a4ef493e), cargo 1.94.0
(85eff7c80), uv 0.11.19, existing `target/venv` using Anaconda CPython 3.13.9,
PyArrow 24.0.0, pytest 9.2.0, Ruff 0.16.0 and maturin 1.16.0. The existing
Conda shared library path is `/home/wegamekinglc/anaconda3/lib`.

Final matching native build and install, exit 0:

```sh
PYO3_PYTHON=/home/wegamekinglc/multica_workspaces/dal-4ee34dfed184/dal-519-d83ca711d94c/workdir/calc-flow/target/venv/bin/python target/venv/bin/maturin build --locked --profile dev --out target/wheels
uv pip install --python target/venv/bin/python --reinstall --no-deps target/wheels/calc_flow_python-2026.11.2-cp313-cp313-linux_x86_64.whl
target/venv/bin/python -m pytest python/tests/test_bounded_global_distinct.py python/tests/test_native_stub.py python/tests/test_query_history_portability.py python/tests/test_finite_event_window.py python/tests/test_finite_union.py python/tests/test_update_semantics.py python/tests/test_finite_inner_join.py python/tests/test_finite_asof_join.py -q --maxfail=3
target/venv/bin/python examples/bounded_global_distinct.py
```

**227 passed**. The previously failing reason_code assertion is GREEN and
unchanged. The Python example emitted the literal tuple set {NULL,1,2}.
This is a local CPython 3.13 Linux wheel, not a package publication check.

`calc_flow._native.__file__` resolved to the installed wheel's
`target/venv/lib/python3.13/site-packages/calc_flow/_native.cpython-313-x86_64-linux-gnu.so`.
A Python check read both that file and the wheel's native zip entry and asserted
byte-for-byte equality. SHA-256 values:

- Installed/wheel native: `737739c65f7ef841a343af6440bb8f10b51d4f8cf9fa223463e97f288b99ea00`.
- Wheel: `64b6ad50a9020bde2ae7e0f994c6dc68e03bd5ff8c1ec9e6cc2babe7e2d98589`.
- Product source tree fingerprint: `18c9de458c775aac691e568c85aed75e44136c5d6a8da681a0a13987a0eba7f8`.

The product fingerprint is SHA-256 over `git ls-tree -r HEAD --` with paths
`crates/calc-flow/src crates/calc-flow-python/src python/calc_flow Cargo.toml
Cargo.lock pyproject.toml crates/calc-flow/Cargo.toml
crates/calc-flow-python/Cargo.toml`. It is checked unchanged after the
evidence-only commit. This ties the final PR code to the tested native module.

The final embedded-Python native verification also exited 0, **2 passed**:

```sh
PYO3_PYTHON=/home/wegamekinglc/multica_workspaces/dal-4ee34dfed184/dal-519-d83ca711d94c/workdir/calc-flow/target/venv/bin/python LD_LIBRARY_PATH=/home/wegamekinglc/anaconda3/lib PYTHONPATH=/home/wegamekinglc/multica_workspaces/dal-4ee34dfed184/dal-519-d83ca711d94c/workdir/calc-flow/target/venv/lib/python3.13/site-packages cargo test --locked -p calc-flow-python --lib test_distinct_
```

This rechecks the actual foreign-owner release before EOF, not just the Rust
normalization test or the withdrawn prototype's result.

Final scoped Clippy commands exited 0 with warnings denied:

```sh
cargo clippy --locked -p calc-flow --lib --test bounded_global_distinct --test bounded_distinct_window --test distinct_admission --test distinct_analysis --test distinct_closure --test distinct_owned_schema --example bounded_global_distinct -- -D warnings
PYO3_PYTHON=/home/wegamekinglc/multica_workspaces/dal-4ee34dfed184/dal-519-d83ca711d94c/workdir/calc-flow/target/venv/bin/python cargo clippy --locked -p calc-flow-python --lib -- -D warnings
cargo run --locked -p calc-flow --example bounded_global_distinct
cargo fmt --all --check
git diff --check
git diff --cached --check
python3 scripts/check_spec_references.py
target/venv/bin/python scripts/verify_complexity_gates.py
target/venv/bin/python -m compileall -q python/calc_flow examples/bounded_global_distinct.py
```

The core example independently built only the core example target and printed
`3 distinct tuples (order unspecified)`. Ruff check and format --check over all
21 changed/new Python and stub files passed. Native stubs are included in the
227-case pytest command. The previously valid project-v3 generator result is
reused: no schema-affecting changes followed it. Full regression, platform
matrices and coverage remain CI work; no performance result is inferred.
The one nonblocking CI snapshot is reported with the PR in the issue handoff.

The late-policy contract gap described above remains open. These GREEN local
checks do not turn the draft into complete v2 product acceptance.

## Local late-policy addendum and reviewer F1/F2 correction

This section implements the frozen 118-line addendum to the original API note.
The previous gap is resolved by that approved persistent contract; its historical
counterexample and all earlier evidence remain above. The reviewer artifact is
preserved separately and is neither edited nor staged by the implementer.

The local spec reuses LatePolicySpec. Default Error(envelope) is read when absent
and omitted when serialized, preserving the existing canonical spec/config and
layout2 control. Drop(metrics_version=1) is explicit and window-only. Rust exposes
the consuming builder/getter; Python adds the one trailing keyword; WINDOW SQL
adds one optional seventh string literal. The schema has a strict two-policy
subset and excludes bounded_source+Drop. No Runner option or public policy alias
was added. Each DISTINCT keeps its local policy when global options configure
independent unmarked rolling/cross-section consumers. Unused global options are
rejected on this new path; local declarations and shared fan-out remain distinct.

Native processing recognizes closure before admitting history rows, fixing F1's
full-max_rows counterexample. Closed Drop scans all RecordBatches at one committed
watermark, stages checked delta/totals and the existing metric sink, releases raw,
then installs state, commits metrics and invalidates capture without an await or
fallible operation. NULL/outside rows do not count; affected_batches increments
once per Data; lateness is W-ts. Error rejects the complete envelope. Schema,
conversion, cancellation, reservation, staged-sink and overflow errors remain
failures. No discarded keys or raw owner enters retained state or a metric closure.

Drop adds prepaid declaration/control/transaction storage (64/256/512 bytes) and
1024 bytes to the capture descriptor quote. Its layout3 control persists strict
LateMetricDelta totals; Error remains layout2 without that field. Restore rejects
inconsistent policy/layout/version/cut/metrics before installation. The DISTINCT
entry uses the async restore path, installs metric totals by replacement and
mirrors them into runtime status before source entry acknowledgement. It does
not replay saved totals as new deltas. Version1 layouts (2,3), accounting2 and r9
remain consistent with capability/export/cache identities; an old layout2-only
inventory can still admit Error, while Drop and unknown state versions fail.

F2 is fixed with the shared native window_time_column proof used by compile,
query analysis and source-coordinate tracing. Arithmetic, CAST and truncation
are Unsupported/capability_mismatch at DISTINCT history; unchanged/pure-renamed
time stays Supported for Rust DF and SQL macro declarations. No data is read.

### New RED/GREEN evidence

- `cargo test --locked -p calc-flow --test distinct_analysis
  test_window_analysis_proves_only_original_or_pure_renamed_time`: RED observed
  Some(Supported) for `ts + INTERVAL '1 microsecond'`; GREEN covers arithmetic,
  CAST, truncation and pure rename through both Rust declaration forms.
- The first two `distinct_late_policy` tests were RED: unknown late_policy wire
  field and rejection of the seventh SQL argument. Both are GREEN with canonical
  default/error equality and strict policy/version rejection unchanged.
- The full-history Drop trace was RED at required=4/limit=2 after closure. GREEN
  emits one final, retains accepted_rows=2 and persists literal totals 2/1/10;
  repeated restore/replay at W=12 then ts=1 reaches 3/2/11 without another final.
- Restore mismatch test was RED for positive late totals without a closing
  watermark. GREEN rejects that cut, missing metrics, unknown version, layouts
  and Error/Drop swaps while preserving the prior complete capture.
- Python keyword test was RED at unexpected keyword. The inventory state-version
  test was RED (did not raise); expression explanation was RED missing the local
  policy. These pure declaration cases are now GREEN (4 passed) without claiming
  that the old native module implements Drop.
- The independent JSON Schema probe was RED accepting bounded_source+Drop.
  After generation, Draft202012Validator 4.26.0 accepts three literal default/
  Error/Drop window cases and rejects four bounded/side-output/version/scope
  negative cases. JSON Schema is a validation utility in the reused environment;
  it is not a product dependency.

Harness-only compilation corrections (snapshot field equality, public root
OperatorSpec import and Result-return handling) are not behavior RED evidence.
The public Rust status check identifies DISTINCT by exported node IDs; source
and output passthrough nodes legitimately have zero counters, and Rust exposes
Duration while Python exposes microseconds. No assertion was removed to pass.

### Necessary core checks and Codacy diagnosis

`cargo test --locked -p calc-flow --lib operator::distinct::`: **10 passed**.
This includes the five existing ownership/budget checks plus new Drop storage,
transaction/capture prepayment, staged failure/cancellation, replaced restore
totals and counter-overflow atomicity. Independent new literal quotas are
31368/31367 (stream control), 31880/31879 (512-byte transaction), and
434312/434311 (402944-byte prepaid control). Allocation hooks stay zero at B-1.
The existing literal 9282/9281 normalization and live-capture query checks remain
GREEN after the ownership-helper refactor.

The targeted integration command with bounded_global_distinct,
bounded_distinct_window, distinct_analysis, distinct_late_policy,
distinct_closure, distinct_owned_schema and distinct_admission: **21 passed**.
It includes actual public Rust DF/SQL stream lowering/export and 3/2/11 status
for each DISTINCT, plus the old per-batch compatibility and owner release cases.
The core example independently rebuilt and printed 3 distinct tuples; the
project-v3 generator succeeded. The unaffected 15 adjacent regressions and the
prior broader Python matrix are reused rather than rerun by default.

GitHub check-run 131435216694 was completed/action_required: “37 new issues
(0 max.) of at least minor severity.” Its annotations identify 36 complexity
findings and one executable-example assert. These were actual code findings,
not an account-activation problem. Helpers now separate admission, reservation,
copying, capture, restore and IPC invariants. Tests keep all literal expectations,
using unwrap for fail-fast setup. The example raises RuntimeError on a bad result
even under optimized Python. Local Lizard 1.26.0 reports all affected/new DISTINCT
functions at complexity <=8; no suppression, gate or baseline was changed.

`cargo clippy --locked -p calc-flow --lib --tests -- -D warnings`: GREEN. The
crate test targets were type/lint checked to include the private quota/metric
tests after actual pedantic failures; this was not a full regression run.
Ruff, complexity ratchet, spec references and diff whitespace checks are GREEN.
Final matching wheel, Python/runtime recovery and CI results follow below.

### Matching native and completed addendum checks

Product implementation commit: `2ac85c3d39667dd28bd1294d6933d61cc9772e59`.
The follow-up commit adds only tests, the support-table correction and this
evidence. Its product fingerprint is checked against this commit before push.
The original checkout/branch/environment and user's Git identity are retained.

At that implementation commit, the actual matching build/install commands were:

```sh
PYO3_PYTHON=/home/wegamekinglc/multica_workspaces/dal-4ee34dfed184/dal-519-d83ca711d94c/workdir/calc-flow/target/venv/bin/python target/venv/bin/maturin build --locked --profile dev --out target/wheels
uv pip install --python target/venv/bin/python --reinstall --no-deps target/wheels/calc_flow_python-2026.11.2-cp313-cp313-linux_x86_64.whl
target/venv/bin/python -m pytest python/tests/test_distinct_late_policy.py python/tests/test_bounded_global_distinct.py python/tests/test_native_stub.py -q --maxfail=3
```

The final run is **41 passed**: 15 direct addendum cases, 23 prior feature cases
and 3 stub cases. The new cases exercise real Rust-backed DF/SQL declarations,
default/error/drop canonical export, local policy precedence with independent
rolling and cross-section consumers, marker exclusion, shared ownership, cache
isolation and finite reuse after a row-budget failure, empty history and success.
The initial expanded mixed-graph fixture incorrectly marked its temporal input
nullable; the fixture was corrected to the existing non-null timestamp/entity/
sequence contract. That ordering_required result was not a product RED.

For both Python DF and SQL, actual managed execution produces literal {1},
captures closed Drop with 2/1/10, and restores totals before allowing new source
events. W=12 then ts=1 yields 3/2/11 with no second final. Replaying from a copied
prior cut produces the same totals. Swapping to Error fails before Source.open
(observed open count 0). Both observed reopened cursors are exactly 4. Warm-cache
Error/Drop plans have distinct native fingerprints; old layout2-only capability
still admits Error and refuses Drop, and an unknown state version is refused.

The fresh wheel's native entry and installed native were read and asserted
byte-for-byte equal. Actual hashes:

- Native: `c7810be6d86b5c5798c398589f7d812fae571842bef2ba59eb0677768a1758f2`.
- Wheel: `8c61ae1d07074863e515fda854bff772dfa5b41db3d307500375984eed71d4ae`.
- Product tree: `f05f960ba05248e826de56bd6a648b501a93c7c9cccd15437f8c7fd99fa30531`.

The tree fingerprint uses the same git ls-tree paths documented above (core
and binding src, Python package and dependency manifests/lock). Python loaded
the existing target/venv site-packages native module. Python 3.13.9, PyArrow24,
pytest9.2, Rust/Cargo1.94, Ruff0.16 and maturin1.16 were unchanged; Lizard1.26 and
jsonschema4.26 were added only as local validation utilities, with no dependency
manifest or lock changes. No new environment or checkout was created.

The final PyO3 source check and real C-stream ownership verification were:

```sh
PYO3_PYTHON=/home/wegamekinglc/multica_workspaces/dal-4ee34dfed184/dal-519-d83ca711d94c/workdir/calc-flow/target/venv/bin/python cargo clippy --locked -p calc-flow-python --lib -- -D warnings
PYO3_PYTHON=/home/wegamekinglc/multica_workspaces/dal-4ee34dfed184/dal-519-d83ca711d94c/workdir/calc-flow/target/venv/bin/python LD_LIBRARY_PATH=/home/wegamekinglc/anaconda3/lib PYTHONPATH=/home/wegamekinglc/multica_workspaces/dal-4ee34dfed184/dal-519-d83ca711d94c/workdir/calc-flow/target/venv/lib/python3.13/site-packages cargo test --locked -p calc-flow-python --lib test_distinct_
```

Both exited 0; **2 native tests passed**, including the real 8 MiB foreign owner
release while copied H is live and before EOF. A final native schema-failure
regression also passed: wrong raw key type after Drop closure leaves the complete
capture and zero metrics intact. Its command was `cargo test --locked -p
calc-flow --test distinct_late_policy test_drop_schema_failure_does_not_change_closed_capture`,
followed by scoped Clippy on that test. This brings the unique core integration
cases in this correction to **22 passed** (the earlier 21 plus this case).

Final Ruff check/format covers all 9 changed Python files; complexity ratchet,
spec-reference check, compileall, Rust format, diff whitespace and the Python
example all passed. Local Lizard <=8 is rechecked over the affected DISTINCT
Rust/Python files and new tests. No full regression or benchmark was run; prior
unaffected evidence is reused and full/platform/coverage work remains in CI.

Reviewer file SHA-256 remains
`bd6b67fcc9cc7c2db61db7ec976cef68de813d8794070dfd64f30618a7d42877`.
Its contents remain untracked and unstaged for the reviewer. The frozen note's
diff is exactly 118 added lines with zero removed lines. F1/F2 and their direct
effects are ready for independent recheck; this is not a merge approval. The
final issue handoff records the single nonblocking CI snapshot, including Codacy;
local complexity correction is not a claim that the remote check is green.

## PyO3 lib-test Clippy gate correction

The independent F1/F2 recheck approved `d2a68c85`; its appended reviewer report
is preserved untracked and unstaged. Its current SHA-256 is
`e7681127a4a42ce562cde3ac0201d13ae51a0e861fbde18f72a72d9676c9dce7`.
This correction addresses only the subsequent CI test-compilation failure.

The actual failed Rust core parity job was
https://github.com/wegamekinglc/calc-flow/actions/runs/43306366639/job/131490764624.
Its downloaded log reports at 2026-11-12 21:09:08–21:10:03 UTC:

```text
error: used underscore-prefixed item
--> crates/calc-flow-python/src/config.rs:1195:26
1195 | let result = _distinct_query(
= note: `-D clippy::used-underscore-items` implied by `-D clippy::pedantic`
error: could not compile `calc-flow-python` (lib test) due to 1 previous error
Process completed with exit code 101.
```

The failed step was Clippy, not a product assertion; subsequent Rust test and
rustdoc steps did not run. Earlier binding lint evidence selected `--lib`
without the lib-test target and therefore did not cover this reference.

Fix commit: `1ba4aec984c031424f582ea4655fd9e867fff7a4`.
The existing bridge test now registers the query planners in a PyModule and
calls `getattr("_distinct_query").call1(...)`, then extracts the same JSON string.
Both original policy/column assertions remain unchanged. The production Rust
function, Python export name, registration and signature are unchanged. No test
case, lint suppression, dependency, gate or production behavior was added.

Actual necessary commands in the reused environment:

```sh
PYO3_PYTHON=/home/wegamekinglc/multica_workspaces/dal-4ee34dfed184/dal-519-d83ca711d94c/workdir/calc-flow/target/venv/bin/python cargo clippy --locked -p calc-flow-python --lib --tests -- -D warnings
PYO3_PYTHON=/home/wegamekinglc/multica_workspaces/dal-4ee34dfed184/dal-519-d83ca711d94c/workdir/calc-flow/target/venv/bin/python LD_LIBRARY_PATH=/home/wegamekinglc/anaconda3/lib PYTHONPATH=/home/wegamekinglc/multica_workspaces/dal-4ee34dfed184/dal-519-d83ca711d94c/workdir/calc-flow/target/venv/lib/python3.13/site-packages cargo test --locked -p calc-flow-python --lib test_distinct_
target/venv/bin/python -m pytest python/tests/test_native_stub.py -q
cargo fmt --all --check
git diff --check
```

All exited 0: Clippy now includes lib tests, the two existing native tests passed
(including registered Python-name lookup and real foreign-owner release), and
the three existing stub checks passed. Format and whitespace checks passed.

An initial package-scoped `cargo clippy --locked -p calc-flow-python --lib
--tests --all-features -- -D warnings` attempt exited 101 in the local optional
rdkafka-sys build, before a completed lint result:

```text
rdkafka_conf.c:60:10: fatal error: curl/curl.h: No such file or directory
```

This is a local all-feature dependency limitation, not the observed CI cause.
No product or environment workaround was introduced; the feature-complete
workspace gate remains CI work and is not reported as locally passed.

All changes in config.rs are inside its existing `#[cfg(test)] mod tests`.
A byte comparison against `d2a68c85` confirms the complete non-test prefix is
unchanged (SHA-256 `abd6f90ebfb3b72cf2b2e84e307528166d4c65f7599780fa4170b16cec38fd10`).
The build script only calls use_pyo3_cfgs and embeds no Git revision. Therefore
the validated production native module and prior product/reviewer evidence are
reused; no wheel rebuild, full matrix, performance run or full review restart
is needed for this test-only correction. The final handoff records the pushed
HEAD and one fresh CI snapshot; pending checks are not a successful gate.
