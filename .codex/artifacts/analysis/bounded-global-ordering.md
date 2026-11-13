# DAL-520 implementation evidence

Baseline: `b29b7a9a164a79c85cbf92f8bf1683a1704c0f3f`, branch
`agent/dal-orchestrator/c9c0751ee030`. The API note is a frozen input and is
preserved byte for byte. The assigned checkout and branch are reused.

The controlling issue, trigger thread and parent discussions were read through
the Multica CLI. Repository guidance is `AGENTS.md`,
`.codex/agents/cf-implementer.toml`, `.claude/rules/code-style.md` and
`docs/introduction.md`. The runtime-named `.codex/references` files and a
work-item methodology/spec/critique do not exist here; the approved API note
and issue supply the design checkpoint. No DAL public facade exists in this
repository. Existing examples keep their owning component and purpose.

The implementation design is one native final ordering node behind both
DataFrame and SQL declarations in both languages. It retains complete H,
copies visible input synchronously, validates an explicit sequence over the
whole candidate, and runs one final sort/skip/fetch kernel. It shares existing
owned normalization and transactional capture/late-policy primitives where
their contracts agree. Sort state and protocol identities remain independent
of DISTINCT. Plain SQL keeps its existing per-batch behavior.

## Verification log

The first focused test is
`test_order_finite_top_k_and_same_plan_isolation` in
`crates/calc-flow/tests/bounded_global_ordering.rs`.

RED command:

```sh
cargo test --locked -p calc-flow --test bounded_global_ordering test_order_finite_top_k_and_same_plan_isolation
```

Compilation started against this checkout's lockfile in its local `target/`.
Results are recorded below when the foreground build completes.


First RED completed with exit 101: missing root SortKey/SortSpec and missing
TableExpr.top_k, as expected. An attempted direct rustc metadata shortcut
could not find the still-building core metadata and is not RED evidence.
The Cargo command above is the valid RED result.

First GREEN: the same command exited 0; 1 passed, 0 failed. It executed the
same finite plan successively on [3,1,2], [9] and typed empty and returned
[1], [9], empty. Build source was this checkout's current working tree with
its pinned Cargo.lock, in target/debug.

Next RED command:

```sh
cargo test --locked -p calc-flow --test bounded_global_ordering test_order_sequence_recovery_emits_one_final
```

Exit 101: BoundedOrderOperator did not implement the stream lifecycle
(process_data/checkpoint/restore/on_end), as expected. The next implementation
adds the independent bounded_order_v1 capture namespace and explicit sequence
comparator, then exercises open and closed recovery.


Stream recovery GREEN: `cargo test --locked -p calc-flow --test
bounded_global_ordering` exited 0, 2 passed. Namespace is bounded_order_v1,
accounting1, Error layout1; open restore adds the remaining rows and uses
sequence ascending as the last comparator; closed restore emits nothing.

Sequence atomicity RED: the focused duplicate-envelope test exited 101 because
it accepted a repeated sequence across Data. GREEN after native tuple checking:
the same integration target exited 0, 3 passed. It compares both control and
data snapshot segments before/after a rejected two-RecordBatch envelope.

DF/SQL compile/export RED: after correcting the test's for_mode Result handling
and compile_stream arguments, the focused test failed because ordering offered
only a batch operator. GREEN after adding the native variant and AST macro seam:
4 passed. Both public Rust frontends compile in both modes and export equal
bounded_order operator descriptors.

Native structural admission RED: the snapshot/rewritten-sequence test exited
101 because it accepted an invalid upstream. GREEN after mandatory graph checks:
5 passed. No opt-in portability/update request is needed for this rejection.

Actual-source closure RED: `cargo test --locked -p calc-flow --test
order_closure` exited 101, missing closure accepted. GREEN after source-descriptor
and original-time watermark preflight: both integration targets exited 0,
5 + 1 passed; Source.open remains zero on rejection.

Python RED: `target/venv/bin/python -m pytest -q
python/tests/test_bounded_global_ordering.py` exited 1 with missing SortSpec,
1 failed. This RED intentionally used the pre-existing installed package before
the new bindings existed. Its native is not evidence for the changed product.
The existing CPython environment from DAL-519 is reused via target/venv symlink;
no new virtual environment was created. GREEN will rebuild and install a wheel
from this checkout, then verify imported native/source provenance.

Scoped Ruff check now passes for the changed Python modules/stub/test. Initial
line-length diagnostics were fixed by Ruff formatting; they were not test passes.
Frozen API-note SHA-256: 424f9be512502a9f6e0160c5517bdd7ac634cadcdc08d998fd68ea6858a67c3d.


Strong delivery RED: `cargo test --locked -p calc-flow --lib
test_bounded_order_requires_explicit_sequence_for_strong_delivery` exited 101:
the built-in codec bypassed tie/replay proof. GREEN: same command exited 0,
1 passed. The gate is specific to bounded_order and does not weaken existing
external-provider validation or make frontend ExactlyOnce available.

Python first matching-native GREEN: Maturin built and uv installed the wheel
from this checkout; the original public DF/SQL test passed. The unknown-source
root test then failed because mkdtemp was called once; after deferring that
root for ordering (and adding the new cache tag), a rebuilt/reinstalled wheel
passed both cases. Five scoped Python cases subsequently passed, including
real [3,1]/[2] native Arrow source slices: legacy top1 [1]/[2], final top1 [1].
Two test setup corrections used the actual APIs: ProjectDocument.model_dump
and explicit Program mode, rather than adding compatibility shims.

Analysis RED: the new Rust test and Python test both observed r9 instead of
r10. Rust GREEN after adding native ordering facts and member-selection
obligations: 6 + 1 + 1 tests passed. Unspecified fetch=Some(0) still has
unproved membership for strict unordered comparison; stable sequence supports
conditional declared final order. Python projection remains pending a wheel
containing r10 at this checkpoint in the log.

Millisecond window RED: projected normalization rejected timestamp[ms]. GREEN:
shared owned copying now handles milliseconds while DISTINCT's public key
validation retains its old domain. The test closes [0,10000) at W10000, counts
2 late rows / 1 affected envelope / max lateness10000 after closure, uses Drop
layout2, keeps accepted_rows=2 and does not re-emit after restore.

History quota RED: max_rows1 accepted-H2/top1 failed with an untyped argument
error. GREEN: it now returns StateBudgetExceeded with required/limit and leaves
accepted_rows0, closed=false and no output. The same finite plan isolation,
sequence and closure targets remain green.

The green finalizer was moved to prepaid owning CPU work: finite uses an
awaited blocking worker, stream uses the existing managed gather service. Raw
ingress is not captured by the worker. This refactor passed the 7 + 1 + 1
integration cases before postprocessing work.

Retained-container RED: private test found declaration25616 + H9282 +
stream-control4096 charged, but the committed history container's64-byte
reservation had been dropped. GREEN retains its lease through replacement,
close or restore; same private test passed. Restore reserves its new container
before allocation and installs it with the candidate.

Postprocessing RED: SQL preserving x/seq was still unknown for declared order.
GREEN: descendants of a bounded-order node now use a single physical partition;
native AST passthrough mapping propagates comparator fields only when they
survive unchanged/renamed. Removed or rewritten comparator fields remain
unknown. Actual export/JSON reload, batch_size2, configured partitions8, two
SQL stages and 50 independently expected ordered rows pass. The latest
integration run at this point is 9 + 2 + 1 passed.

A private measurement (not a performance benchmark) recorded deterministic
quota roles for the two-i64-row/one-column fixture: declaration25616, normalized
H9282, committed39058, combined17474, certified backing1105, sort workspace18658.
Observed stream close peak149149 included managed worker/control retirement;
after close84752 still charged real live worker owners. Capture peak432676,
live capture405138; admission with that capture held peaked423830. Restore
peak466322/restored live47314 used the seven-digit budget identity. These are
logical node-owned charges, not RSS or external-parent measurements. Literal
B/B-1 tests for finite sort/output, sequence workspace, capture and restore
are being checked independently of this measurement.

Schema RED: the Python schema test found no UInt32 maximum on offset. The new
schema narrows sorting offset/rows/fetch and NULL placement, requires an
explicit fetch (including null for full sort), restricts local late policy and
sequence names. This does not change the global StateBudget/NullPlacement
schemas or old nodes. Required-fetch uses a dedicated schema type because
schemars's required attribute emitted code rejected by the repository's
unused_qualifications lint; the lint is preserved.


Owned-budget boundary GREEN: `cargo test --locked -p calc-flow --lib
operator::order::tests -- --test-threads=1` exited 0, 5 passed. Exact finite
sort/output peak 62838 vs 62837; capture 432676 vs 432675; six-digit-budget
restore 466290 vs 466289; explicit-sequence workspace 56016 vs 56015. These
literal bounds were checked against Arrow layouts, without deriving the
expectations by calling the implementation's quote. The 16-byte correction
from an initial sequence estimate was verified by an independent Rust layout
probe; production was not relaxed. Utf8 concat offset preflight RED was a
missing function; GREEN checks 0/MAX/MAX+1 without allocating a giant string.
SortHistory is an independent public enum, not a DISTINCT alias.

Erased-native admission RED: `cargo test --locked -p calc-flow --lib
test_erased_builtin_cannot_bypass_order_admission` exited 101 because a
Sequence-declared builtin wrapped in NodeOperator::Stream skipped the mandatory
native graph checks. The fix rejects this erased representation so it cannot
skip source or structural admission.

Unresolved frozen-API/gate conflict (no waiver applied):
`target/venv/bin/python scripts/verify_complexity_gates.py` exits 1 with
`ruff_complexity:python/calc_flow/symbolic/order.py:PLR0913 grew from 0 to 1`.
The frozen `SortSpec.closed_window` has seven explicit non-cls parameters,
while the repository gate permits five. Ordinary Ruff passes. No noqa,
complexity baseline/threshold change, dynamic-signature shim or API change is
used. Coordinator must reconcile the frozen signature with the gate before
acceptance; functional evidence and a concrete reviewable patch continue.


Native allocation source audit (Cargo.lock: Arrow ord/select59.2.0,
array59.2.0, buffer/schema59.3.2, DataFusion55):
`arrow-ord-59.2.0/src/sort.rs:277,341,355,543,646,940,1023` dispatches
only the admitted primitive/Boolean/byte-array paths and uses in-place
unstable sorting. The largest single-column `(u32,&[u8])` scratch is24N;
indices/null routing fit the32N row allowance. Multi-column indices and
boxed comparator descriptors fit that allowance plus4096 per comparator.
`arrow-select-59.2.0/src/take.rs:501,555` uses selected offsets/values and
nullable source ranges (16N); sorted indices are unique, so selected bytes
cannot exceed the certified full input bytes. `concat.rs:334..366` constructs
the concatenated owned arrays. The bound is two certified full table layouts
(concat + take),32N scratch/indices,8192 control plus4096 per key/column;
compact normalization reserves separately while that work remains live.
This multiplier is applied only to private deep-copied OwnedBatch provenance,
not to arbitrary ingress or an unmeasured external backing. Checked string
capacity validation precedes concat. Worker captures/control and retained
history container leases live until their actual owners drop, including
retirement; captured raw ingress never crosses the first await.

The broader `cargo test --locked -p calc-flow --lib order --
--test-threads=1` filter selected211 tests (including legacy SQL ordered
recovery and adjacent rolling/join order tests): all211 passed. This is not a
full harness, coverage run or performance acceptance. A narrower filter will
be used if further changes require rerunning new operator tests.

Window-domain RED: `cargo test --locked -p calc-flow --test order_analysis
test_order_closed_window_comparison_requires_one_interval` exited101:
nonidentical [0,10) and [0,20) outputs were incorrectly Supported. The common
cutoff check now includes ordering histories alongside DISTINCT histories.


## Final local verification

The final source tree keeps the frozen API-note hash above. There is no public
API deviation, gate waiver, old-layout migration, new environment/branch or
new ExactlyOnce promise. The unresolved signature/complexity conflict remains
an acceptance blocker for the coordinator; DAL-520 stays in_progress.

Latest explicit commands and observed results:

```sh
cargo test --locked -p calc-flow --test bounded_global_ordering --test order_analysis --test order_closure --test distinct_late_policy --test distinct_owned_schema --test distinct_analysis --test latest_relation --test finite_union_frontend
# exit0: 40 passed (10+3+2 new ordering; 25 adjacent regression cases)
cargo test --locked -p calc-flow --test project_v3
# exit0: 3 passed, including exact committed/generated schema equality
cargo test --locked -p calc-flow --lib pipeline::stream::proof_tests -- --test-threads=1
# exit0: 2 passed, including existing external proof gate
cargo test --locked -p calc-flow --lib operator::order::tests -- --test-threads=1
# exit0: 5 passed, including four B/B-1 roles
cargo clippy --locked -p calc-flow --lib --test bounded_global_ordering --test order_analysis --test order_closure --example bounded_global_ordering -- -D warnings
# exit0
PYO3_PYTHON="$PWD/target/venv/bin/python" cargo clippy --locked -p calc-flow-python --lib -- -D warnings
# exit0
cargo run --locked -p calc-flow --example bounded_global_ordering
# exit0: complete H=[3,1,2], final top-1=[1]; independent core-only build
cargo fmt --all --check
# exit0
```

First Clippy failures were fixed by extracting local helpers, removing duplicate
branches and formatting rustdoc; no allows were added. Schema RED also found
that serde try_from selects Wire's schema attributes: UInt32/NULL/late-policy
constraints now live on that wire authority too. A further RED caught missing
by.minItems; the new private schema name also avoids renaming DISTINCT's
existing IngressPolicy definition. Project-v3 artifact equality passes.

```sh
JAX_PLATFORMS=cpu target/venv/bin/python -m pytest -q python/tests/test_bounded_global_ordering.py python/tests/test_native_stub.py python/tests/test_bounded_global_distinct.py python/tests/test_query_history_portability.py python/tests/test_update_semantics.py python/tests/test_capabilities.py python/tests/test_symbolic_expr.py::test_public_surface_is_the_frozen_export_list python/tests/test_config.py::test_project_document_delegates_schema_and_rust_defaults
# exit0: 141 passed
JAX_PLATFORMS=cpu target/venv/bin/python -m pytest -q python/tests/test_finite_event_window.py::test_window_analysis_proves_conditional_eof_and_leaves_prefix_unknown python/tests/test_finite_union.py::test_union_reports_append_history_and_no_global_order python/tests/test_finite_inner_join.py::test_inner_join_conditional_eof_analysis python/tests/test_finite_asof_join.py -k analysis
# exit0: 2 passed,40 deselected; -k omitted union/asof, so explicitly checked below
JAX_PLATFORMS=cpu target/venv/bin/python -m pytest -q python/tests/test_finite_union.py::test_union_reports_append_history_and_no_global_order python/tests/test_finite_asof_join.py::test_asof_conditional_final_relation_proof_revision python/tests/test_finite_inner_join.py::test_four_surface_join_fixture
# exit0: 5 passed
# The first adjacent-check invocation misspelled a config test selector:
# exit4/no tests ran; the corrected 141-case command above is the evidence.
target/venv/bin/python examples/bounded_global_ordering.py
# exit0: both public frontends returned x=[1],seq=[11]
```

The 141-case run includes both old DISTINCT and new top-k through the real
ManagedCheckpointRuntime: first Data -> persisted epoch1 -> cancellation ->
source cursor1 restore -> remaining Data -> exactly one final. Original
per-batch SQL still emits [1]/[2] from two actual stream Data; new ordering
emits nothing before EOF and then [1]. Manual native recovery additionally
covers explicit-sequence offset selection and closed restore with no reemit.

The Python projection RED returned Unknown for `.sort(...).select("x","seq")`.
The final adapter sends literal projection columns through the same native AST
passthrough proof on ordering paths only; dropping seq remains Unknown. This
is in the final 141-case run. Native multi-stage SQL postprocessing is separately
checked with configured partitions8/batch_size2 and50 literal expected rows.
No missing comparator/sequence proof is silently upgraded.

Ordinary Ruff check and format checks on changed Python code/tests/examples
and `git diff --check` pass. The complexity ratchet still exits1 solely for
PLR0913 in the frozen closed_window signature. Full suites, coverage, connector
service matrices, release/platform packaging and performance/RSS acceptance
were not run or claimed.

## Build provenance

```sh
PYO3_PYTHON="$PWD/target/venv/bin/python" target/venv/bin/maturin build --locked --profile dev --out target/wheels
uv pip install --python target/venv/bin/python --reinstall --no-deps target/wheels/calc_flow_python-2026.11.2-cp313-cp313-linux_x86_64.whl
```

Both exited0. The last packaging check reused current native build artifacts
(Cargo finished in0.24s) and included the final Python projection fix; no
source edits followed. Earlier Python-only overlays were diagnostic probes,
replaced by the complete wheel reinstall before the final 141-case run.
This is a local CPython3.13.9 dev Linux wheel, not a release/manylinux artifact.
All62 installed Python/stub/native archive entries match the final wheel byte
for byte; all repository package .py/.pyi files match the installation.

Source fingerprint input: sorted relative paths and bytes of all1208 .rs/.py/
.pyi files under core/src, bindings/src and python/calc_flow, plus root/crate
manifests, Cargo.lock and pyproject.toml, each path/bytes NUL-delimited. Vendor
sources are unchanged from the recorded baseline. Fingerprint and artifacts:

```json
{
  "source_sha256": "d9589afc678da78c7d60f2c825970898dd58718f33340a504ac23aa797046e56",
  "source_file_count": 1208,
  "wheel_sha256": "1296e636bddf3bb6172730ee8e7f3bfd8e009d456dd59432c7e55eaa0c0536c0",
  "native_sha256": "6bfbe0798bb231d8acc9c42f91a1f51d0f850e4cefb89e1e6709d634ed54faf1",
  "matched_archive_files": 62,
  "python": "3.13.9"
}
```

## Review map

- `operator/order/{spec,mod,work,sequence,late,checkpoint}.rs`: one final kernel,
  full H, prepaid owners, local policy and independent state namespace.
- Native dataframe/SQL/lowering/config, pipeline admission and runner preflight:
  one shared variant, pre-open source/cut proof and original-field lineage.
- DataFusion/Expression/SQL postprocessing: actual serial physical execution;
  analysis r10 propagates only retained comparator proofs.
- PyO3 seam/stub, Python symbolic adapters/cache/capability exports and schema:
  data-only projections into the native authority; no Python sorting engine.
- Focused tests, docs and independent examples: observable contracts and
  preserved per-batch, DISTINCT, update/recovery behavior.

The commit-to-evidence mapping is appended after the implementation commit;
only this analysis record changes in that follow-up commit.

Implementation commit: `9535b1c47afbf1ee5fe7ec11cd7c67c3cb247c8d`.
The final local verification and source/native/wheel fingerprints above apply
to its product tree. Its parent is the authorized baseline. The follow-up
commit records this mapping only; it does not change or rebuild the product.

The checkout had no configured Git author, so the first commit attempt failed
without creating a commit. Successful commits use per-command identity from
this run's explicit On Behalf Of metadata; no persistent Git identity was set.


## SortWindow v2 and independent review follow-up

This follow-up starts at `ccda5564809b36e222f4b084ec39cdc20d1b3116`
in the same checkout/branch/environment/PR606. The issue/comment delta was read.
The frozen v2 API note is accepted byte for byte with SHA256
`b4a087ef52cb9b3a43803666e13365dfce544205a6af8c97b88cb34dd8758624`.
The original reviewer report is read-only/untracked, SHA256
`43607ee122c093691d89210144a9eba40d4ebdf5a609f032ba2be5c38c176022`;
it is not edited or staged. The coordinator's supplementary message assigns
F1-F3 in the same run; the original v2 work continues alongside these fixes.

The v2 design is a validated immutable host SortWindow containing only the
column/start/end, flattened into the existing SortHistory. There is no wire
wrapper, native Python method, old-signature parser, new schema field,
capability, layout/accounting version or analysis revision. Rust native wire
imports map construction errors back to sort.spec.history. Public host value
errors use the frozen SortWindow/sort_window paths. Existing SQL syntax and
seams/stubs need no signature change. Root/symbolic exports and direct call
sites are synchronized; historical initial-note/analysis references stay history.

### v2 RED and GREEN

- `cargo test --locked -p calc-flow --test sort_window`: initial exit101 for
  missing SortWindow and the four-argument closed_window. Two unrelated serde
  test error conversions were corrected before repeating this valid RED.
  After implementation: exit0,2 passed, including canonical literal equality,
  serde round-trip, i64 endpoints and native error-path preservation.
- `target/venv/bin/python -m pytest -q python/tests/test_sort_window.py::test_sort_window_is_public_frozen_and_explicitly_typed`:
  exit1, missing public SortWindow. Rebuilt/reinstalled v2 wheel then the
  19-case focused new API/export/stub/local-policy command passed. The complete
  directly affected Python subset later passed28 cases on the fixes' native.
- `target/venv/bin/python scripts/verify_complexity_gates.py`: before changes
  exit1, the original PLR0913 7>5 finding; after changes exit0,
  `complexity ratchet holds`. No threshold, baseline, configuration or
  suppression change. Real explicit signatures are SortWindow3 and
  closed_window5 non-receiver parameters.
- `cargo test --locked -p calc-flow --test project_v3 && cargo test --locked -p calc-flow --doc SortWindow`:
  exit0,3 contract tests and1 new public example passed. The unchanged contract
  checks are reused after the review repairs, which touch no schema types.

### Canonical bytes, cache and old checkpoint evidence

Before editing product code, the installed ccda5564 product (native hash
recorded above) produced `target/dal520-v2-baseline.json`, SHA256
`c2993c8f6c0cf217b3905681830f50ac5e46005a366b417f02636c04b50ea29b`.
Two independent Error/Drop configurations use ts:[-10,10), rows1000/bytes8388608,
offset1, sequence[seq], x ASC/NULL last/fetch1. The committed tests contain
independent flat literal expectations and the captured project/program/plan/
preparation-key digests; no expectation is recomputed from the new constructor.
Both host/SQL native operator descriptors are compared byte for byte. A warm
batch preparation returns the same cached plan for separately constructed equal
windows. The tag stays bounded_order_v1/copy_visible_v2/order_accounting1/analysis10.

Old stream plan fingerprints preserved:

- Error: `e6e74846373d400add03aea21a7ff4471c1c614abad6f2ec710d3a9d87e3d75a`.
- Drop: `0cd303ee9e2092295928bf912551a1cbeaa5cad5a1bab35f0a18977df1f26db0`.

`target/venv/bin/python target/dal520_v2_recovery.py capture` ran against the
original installed ccda5564 product before implementation. Each replayable
bounded source delivered the first row, paused, persisted epoch1, and was
cancelled/awaited: opened[0], no final. An initial instrumentation print tried
to read a consumed plan; it was corrected before the complete two-policy
capture, using a fresh root. That harness error is not a product RED.

After the v2 wheel, `... dal520_v2_recovery.py resume` passed Error/Drop:
opened[1], exactly one final x=[3]/seq=[10]. For the final repair wheel,
untouched original epoch1 manifests/segments were copied byte for byte into a
fresh root, omitting later epoch2 from the earlier successful trial. The copy
was hashed/compared, with no JSON/IPC/checksum or identity rewriting. The same
resume command on the repaired native again passed both cases: identical plan
fingerprints, opened[1], exactly one final. The attached recovery harness makes
capture/resume phases explicit; it is not a public compatibility parser.

### Independent review regressions and changes

F1/F2 RED: `cargo test --locked -p calc-flow --test order_review` exited101,
2 failed. After fixing a test-only TableBatch row-count accessor, the public
plan cancellation case reproduced workspace required11281208 > limit11262843,
and the one-MiB name was incorrectly accepted under40000. The F1 fixture uses
200000 rows solely to keep a real blocking worker alive during cancellation;
it is one correctness counterexample, not a performance workload.

F1 fix: each finite process creates a fresh bounded participating quota and
reserves its declaration role there before input/sequence/final work. That
run's history/workspace/declaration owners travel into the blocking work; no
old pool is reset, refunded or transferred to a later execute. Stream keeps
its own state pool. A deterministic owner test retains old work at53817 bytes,
runs a separate finite execution, asserts those53817 bytes remain charged,
then drops the actual old work and checks release to0. The public test drops
the pending execute and immediately reuses the same compiled plan for[9].

F2 fix: before name allocation, checked declaration credit includes len(name)
plus256 bytes for its owned descriptor/reservation representation. The name
is an owned Arc<str>, shared with gather identity instead of cloning its bytes.
GatherOperatorId carries the shared reservation owner; it drops that owner
last, after its name references. A unit test drops the operator with gather
identity still alive and observes unchanged charge until the identity drops.
The one-column long-name constructor succeeds at1074448 and fails at1074447;
40000 is rejected before target copy. This is a known name allocation, not an
external-ingress/RSS inference.

These fixes add exactly261 bytes for the five-character fixture name. Existing
strict B/B-1 boundaries become finite/output63099/63098, capture432937/432936,
restore466551/466550, sequence56277/56276. The independent original payload
bounds stay the same; assertions and gates are preserved. The F1 fixture budget
is the reviewer's11262582 payload peak plus the same261-byte name role; the
old implementation still failed at that slightly adjusted total in RED.
A private finite test now observes the actual per-run pool, not the persistent
stream/declaration pool. Its result is explicitly dropped before asserting a
zero balance; the first stale test executable correctly kept9282 bytes charged
while its OwnedBatch result was still alive.

F3 RED: `cargo test --locked -p calc-flow --test order_analysis typed` exited101,
2 failed: arithmetic seq+1 AS qseq was Supported, retained comparator projection
was Unknown. The fix records ExpressionOperator's real native query text in
Rust typed projection/filter declarations. Sequence origin and post-comparator
mapping therefore reuse the same native AST authority as compilation and SQL.
The derived/new-name sequence is Unsupported; pure rename is Supported.
Post-projection retaining/renaming all keys is Supported; removed or computed
keys are Unknown, and strict compilation agrees. One initial computed-column
fixture attempted replacing x, prohibited by the existing typed API; it was
corrected to the legal new-name computed_x case, without weakening its order
expectation. Final order_analysis5 plus adjacent distinct_analysis2 pass.

The public F1/F2 regression then passed2 cases. The directly affected sort,
window closure, recovery/ownership/atomicity regression set passed16 cases.
Final command/build ledger follows after collecting the remaining checks.


### Final follow-up verification ledger

```sh
cargo test --locked -p calc-flow --test order_review --test sort_window --test bounded_global_ordering --test order_closure
# exit0: 16 passed (2+2+10+2)
cargo test --locked -p calc-flow --test order_analysis --test distinct_analysis
# exit0: 7 passed (5+2)
cargo test --locked -p calc-flow --lib operator::order::tests -- --test-threads=1
# exit0: 7 passed; all exact B/B-1 and both owner-lifetime tests
cargo test --locked -p calc-flow --lib runtime::streaming::gather_work::tests::cleanup -- --test-threads=1
# exit0: 6 passed; existing funding/retirement/drain guards
cargo clippy --locked -p calc-flow --lib --test sort_window --test order_review --test bounded_global_ordering --test order_closure --test order_analysis -- -D warnings
# exit0 after correcting used-underscore bindings and two unnecessary test struct updates
JAX_PLATFORMS=cpu target/venv/bin/python -m pytest -q python/tests/test_sort_window.py python/tests/test_bounded_global_ordering.py python/tests/test_native_stub.py python/tests/test_symbolic_expr.py::test_public_surface_is_the_frozen_export_list
# exit0: 28 passed on the final matching native
 target/venv/bin/python target/dal520_v2_recovery.py resume
# exit0: both old epoch1 Error/Drop states restored, cursor1, one final
 target/venv/bin/python scripts/verify_complexity_gates.py
# exit0: complexity ratchet holds (actual original command)
```

`cargo fmt --all --check`, ordinary Ruff check/format on package and changed
tests, and `git diff --check` pass. The schema artifact, native seam/stub,
complexity script/baseline and pyproject configuration have zero diff from
ccda5564. The final native schema equals the committed artifact as parsed JSON.
The earlier three project-contract tests and one SortWindow doctest remain
valid; the post-review changes do not change their schema/signature inputs.
Other original 148 Python / 211 Rust results are history, not claimed reruns.
No full matrix, coverage, release or performance verification was started.

The frozen reviewer report still says Request Changes for its fixed old HEAD;
these local fixes do not replace the coordinator's required incremental
independent review. No new unresolved local blocker, API deviation, new branch,
new environment, delegation, merge or issue-status change is introduced.

Final build/reinstall commands (same reused checkout/venv):

```sh
PYO3_PYTHON="$PWD/target/venv/bin/python" target/venv/bin/maturin build --locked --profile dev --out target/wheels
uv pip install --python target/venv/bin/python --reinstall --no-deps target/wheels/calc_flow_python-2026.11.2-cp313-cp313-linux_x86_64.whl
```

Both exit0. The last build completed after source edits; all62 installed
package/stub/native entries equal the final wheel, and repository .py/.pyi
files equal installation bytes. The source fingerprint uses the same1208-file
NUL-delimited scheme as the original delivery. The wheel is local CPython3.13.9
dev Linux, not a release/manylinux artifact.

```json
{
  "base_head": "ccda5564809b36e222f4b084ec39cdc20d1b3116",
  "source_sha256": "a21330654ecfa5dad27f52f5a27fb81c3ee2e29884217aa61e3da9377975e144",
  "source_file_count": 1208,
  "wheel_sha256": "8eca9c319f61b51d48e463bcf2cab088ed92021a885fec58811b1a1554953aa5",
  "native_sha256": "bd2fed8ff0b53fd1a594333749fd6e734af5c477e42b32ec8a9e16e45d0cd5e4",
  "matched_archive_files": 62,
  "python": "3.13.9",
  "schema_matches_committed": true
}
```

Commit mapping for this follow-up is appended after the fix commit. Only this
analysis record changes in that mapping commit; review report remains unstaged.

Follow-up fix commit: `31dbcb9118951bea35c9a915d2622cd52f057ec2`. All final follow-up results and the
source/native/wheel fingerprints above apply to this product tree. The next
commit only adds this mapping. The fixed reviewer report remains unchanged
and unstaged; coordinator/reviewer incremental acceptance is still pending.

## F4 incremental identity regression fix

Base: `d1fb0cdc43276445c1ed1240ad95335701e91c14`, same checkout,
branch and PR606. The incremental reviewer accepted G1/F1/F2/F3 and found
one regression: the real typed DF query introduced for F3 caused the SQL
classifier to overwrite the temporal identity already proven during typed
lowering. This section only covers F4 and its directly affected checks;
the earlier verification remains scoped to its recorded product trees.

The existing reviewer reproducer was rerun unchanged before the fix:

```sh
cargo test --locked -p calc-flow --test relational_validation native_input_identity_survives_dataframe_but_is_cleared_by_sql -- --exact --nocapture
# RED exit101: 0 passed, 1 failed, 14 filtered; line242 expected DF identity Some.
```

A focused adjacent regression was then added and run before production edits:

```sh
cargo test --locked -p calc-flow --test relational_validation typed_projection_and_filter_preserve_only_retained_temporal_identity -- --exact --nocapture
# RED exit101: 0 passed, 1 failed, 15 filtered; typed project lost identity.
```

`query_analysis::classify` now restores only `temporal_identity` from the
typed declaration for project/with_columns/filter when the actual query has
`RowLocal` proof. All other SQL facts and ordering classification keep their
existing logic. Typed lowering's real AST source and comparator mappings from
F3 remain unchanged. Ordinary SQL still clears identity, and a later typed
filter cannot resurrect it. The new test checks both batch and stream facts
for retained-key projection, filter, with_columns, dropped sequence, and
SQL followed by filter. Retained identities must have the expected entity,
event-time and sequence fields. No API/schema/layout/analysis revision or
complexity gate changes are part of this fix.

Final-source verification:

```sh
cargo test --locked -p calc-flow --test relational_validation native_input_identity_survives_dataframe_but_is_cleared_by_sql -- --exact --nocapture
# GREEN exit0: 1 passed, 15 filtered.
cargo test --locked -p calc-flow --test relational_validation typed_projection_and_filter_preserve_only_retained_temporal_identity -- --exact --nocapture
# GREEN exit0: 1 passed, 15 filtered; five batch/stream boundary cases.
cargo test --locked -p calc-flow --test order_analysis typed
# GREEN exit0: 2 passed, 3 filtered; F3 arithmetic sequence rejection,
# pure rename admission, retained/renamed versus removed/computed comparator fields,
# and strict compile agreement.
cargo clippy --locked -p calc-flow --lib --test relational_validation --test order_analysis -- -D warnings
# exit0, no warnings.
cargo fmt --all --check
# exit0.
git diff --check
# exit0.
```

Cargo rebuilt the core Rust source in this reused checkout; the test binaries
are `target/debug/deps/relational_validation-1cee3573b1bf334c` and
`target/debug/deps/order_analysis-305bae3649da4286`. Core build fingerprint
`calc-flow-c4242eaab5409a3b` records features `[]`. The same1208-file
source fingerprint scheme described above gives the following final product
tree provenance (the base SHA identifies the parent, not an unchanged tree):

```json
{
  "base_head": "d1fb0cdc43276445c1ed1240ad95335701e91c14",
  "source_sha256": "00407774fb44bbd1c5524a7d4d1a07df97eefa0072f56768214c132bf324f2bb",
  "source_file_count": 1208,
  "rust_artifact": "target/debug/deps/libcalc_flow-c4242eaab5409a3b.rlib",
  "rust_artifact_sha256": "b9b69039d68f37af4513868fee19f32f634fa65b1e320edef05116594d83d655",
  "rust_features": "[]"
}
```

No Python/native product test was run for F4, and no new wheel is claimed.
The prior matching Python wheel/native evidence belongs to the d1fb0cdc
product tree; it is not evidence of testing this identity fix through Python.
G1/F1/F2 and other unaffected results are reused rather than rerun. No full
matrix, coverage, performance, merge or delegation was started.

The frozen v2 API note remains byte-identical, SHA256
`b4a087ef52cb9b3a43803666e13365dfce544205a6af8c97b88cb34dd8758624`.
The reviewer report remains byte-identical and unstaged, SHA256
`9f11bcdadbcaebf04ba95cdf88bb9d385c7c6b39a05bae78c1122e4c018c70b4`.
Only core query analysis, its direct relational regression, and this appended
evidence are included. F4 independent incremental review remains for the
coordinator/reviewer; DAL-520 stays in_progress.

F4 fix commit: `b0d8e63de86d34b1fe42a88b7b2a7fcebfe5eff0`.
All final F4 commands and Rust/source fingerprints above apply to this commit.
The following evidence-only commit appends this mapping; its product files
are unchanged. Reviewer F4 recheck and the final-head CI snapshot are handed
off separately from these completed local results.

## CI gate follow-up after F4 approval

Base/reviewed HEAD: `ab85af2c0e462b49f44d823547ef1ea07a580f67`.
The independent reviewer approved G1/F1–F4. This increment addresses actual
gate failures on that HEAD, in the same checkout/branch/PR606. It does not
reopen the approved feature contract. Product refactoring in this increment
still requires the coordinator's direct incremental review before merge.

### Actual failure evidence and RED

The initial nonblocking check snapshot found21 checks: Codacy's
`131738367954` concluded `action_required`, and Windows Clippy job
`131738212752` concluded `failure`. Other pending checks were not polled.
Codacy's exact output was `33 new issues (0 max.) of at least minor severity.`
The33 annotations were fetched with:

```sh
gh api 'repos/wegamekinglc/calc-flow/check-runs/131738367954/annotations?per_page=100'
```

The GitHub connector supports commit check listings but rejects the direct
annotations endpoint, so the authenticated gh CLI was used for that read.
The raw annotations are saved locally under target and attached with delivery.
The findings comprise29 complexity reports (27 Rust,2 Python), two Bandit
reports, and two intentionally invalid-call test diagnostics. Paths are:

- Rust product/example: operator/order/{mod,checkpoint,sequence,work}.rs,
  pipeline/order.rs, sql/order.rs, examples/bounded_global_ordering.rs.
- Rust tests: bounded_global_ordering.rs (9), order_analysis.rs (5),
  order_review.rs (1), relational_validation.rs (1), sort_window.rs (1).
- Python: symbolic/order.py declaration/analyze, its projection SQL rendering,
  examples/bounded_global_ordering.py assertion, and two invalid constructor
  calls in python/tests/test_sort_window.py.

Windows failure log: https://github.com/wegamekinglc/calc-flow/actions/runs/43380870659/job/131738212752
(Rust1.94.0; PR merge ref90a88f1 for this HEAD). It reports five errors:
`items_after_test_module` for required_order_fetch/OrderFetchSchema below
config.rs's tests, and four `cast_possible_truncation` errors at order/tests.rs
lines75/95/119/171. The actual job exits1.

Before edits, the local reproductions were:

```sh
uvx lizard -C 8 -w crates/calc-flow/examples/bounded_global_ordering.rs crates/calc-flow/src/operator/order crates/calc-flow/src/pipeline/order.rs crates/calc-flow/src/sql/order.rs crates/calc-flow/tests/bounded_global_ordering.rs crates/calc-flow/tests/order_analysis.rs crates/calc-flow/tests/order_review.rs crates/calc-flow/tests/relational_validation.rs crates/calc-flow/tests/sort_window.rs python/calc_flow/symbolic/order.py
# RED exit1: 29 complexity findings; local Lizard1.26.0 reproduces every location.
cargo clippy --locked -p calc-flow --lib --tests -- -D warnings
# RED exit101: the same five Clippy errors.
uvx bandit -q python/calc_flow/symbolic/order.py examples/bounded_global_ordering.py
# RED exit1: B101 example assertion and B608 projection SQL f-string.
PYTHONPATH=python uvx pylint --disable=all --enable=E1120,E1123 python/tests/test_sort_window.py
# RED exit2: local Pylint reports the legacy start/end keyword arguments.
# Codacy additionally diagnoses the missing required budget in the other call.
```

### Minimal gate corrections

Rust declaration validation/prepayment, restore container staging, sequence
keys/uniqueness, sort indices/take, graph admission and SQL key parsing now
have separate helpers for their existing responsibilities. Budget formulas,
limits, reservation owners and prepay-before-copy order are retained; no
history/final/checkpoint commit boundary moves. The workspace reservation
still covers the entire worker through output normalization. The core example
extracts its declaration builder without changing its component/dependencies.

Seventeen reported Rust tests now assert success with unwrap rather than
propagating setup/expected-success errors via `?`. All62 assertion macros,
loops, case values, failure expectations and checked B/B-1 numbers are
preserved; no test is removed. The Clippy fixes move the unchanged config
helpers before the test module and use checked usize conversions in the four
peak assertions.

Python separates key validation, input schema/provenance, native schema
selection and capability admission. Wire/config/cache values and native
authority are unchanged. Projection rendering joins fixed SQL tokens with
the existing escaped identifier list, producing the same query bytes; no raw
column is interpolated. A regression checks exact projection bytes for a
normal name, a quote/SQL-fragment name, and a reserved word. The two portable
names execute; the nonportable name must fail canonical project validation.
The example
uses an explicit RuntimeError so optimized Python cannot omit verification.
The two invalid-signature calls become parameterized runtime input cases,
retaining both TypeError expectations without a static invalid-call expression.

No suppression, excluded path, threshold, baseline, gate script, pyproject,
schema artifact, public signature, API note or revision is changed.
The frozen API note SHA256 is still
`b4a087ef52cb9b3a43803666e13365dfce544205a6af8c97b88cb34dd8758624`.
The reviewer owns its now-appended F4 approval report; this run preserves its
SHA256 `22d70d675a4f23201dde9b53b2cdf6c2fb3dacc4aec7c92b22d9178ea7a37053`
and leaves it unstaged.

### Static GREEN evidence

The exact Lizard command above exits0 after refactoring. Its CSV covers261
functions, maximum CCN8. Both Bandit files pass (exit0); the same Pylint E1120/
E1123 command passes (exit0). These are local reproductions, not a claim that
the external Codacy check has passed on an unobserved HEAD.

```sh
cargo clippy --locked -p calc-flow --lib --tests --examples --all-features -- -D warnings
# GREEN exit0, Rust1.94.0, finished in2m12s; includes all reported lint sites.
target/venv/bin/python -m ruff check python/calc_flow/symbolic/order.py python/tests/test_sort_window.py python/tests/test_bounded_global_ordering.py examples/bounded_global_ordering.py
# exit0.
target/venv/bin/python scripts/verify_complexity_gates.py
# exit0: complexity ratchet holds, original unmodified script/baseline.
cargo fmt --all --check
target/venv/bin/python -m ruff format --check python/calc_flow/symbolic/order.py python/tests/test_sort_window.py python/tests/test_bounded_global_ordering.py examples/bounded_global_ordering.py
git diff --check
# all exit0.
git diff --exit-code -- schemas/project-v3.schema.json .codacy.yml scripts/complexity_baseline.json scripts/verify_complexity_gates.py pyproject.toml .codex/artifacts/api-notes/bounded-global-ordering.md
# exit0, no drift in frozen contract or gate inputs.
```

### Direct behavior/build verification

```sh
cargo test --locked -p calc-flow --test bounded_global_ordering --test order_analysis --test order_review --test sort_window
# exit0: 19 passed (10+5+2+2).
cargo test --locked -p calc-flow --test relational_validation typed_projection_and_filter_preserve_only_retained_temporal_identity -- --exact --nocapture
# exit0: 1 passed, 15 filtered; all five batch/stream cases retained.
cargo test --locked -p calc-flow --lib order::tests -- --test-threads=1
# exit0: 14 passed, 3217 filtered; includes the seven bounded-order ownership/
# exact budget cases, the native-identity gate, and six neighboring rolling
# tests matched by the filter. No broader matrix was run afterward.
cargo run --locked -p calc-flow --example bounded_global_ordering
# exit0: independent core build/run, complete H=[3,1,2], final top-1=[1].
PYO3_PYTHON="$PWD/target/venv/bin/python" target/venv/bin/maturin build --locked --profile dev --out target/wheels
# exit0: current-checkout core/connectors/binding rebuilt,49.94s.
uv pip install --python target/venv/bin/python --reinstall --no-deps target/wheels/calc_flow_python-2026.11.2-cp313-cp313-linux_x86_64.whl
# exit0.
JAX_PLATFORMS=cpu target/venv/bin/python -m pytest -q python/tests/test_sort_window.py python/tests/test_bounded_global_ordering.py
# final exit0: 29 passed, including both invalid-signature cases and all three
# exact projection-byte/portable-name boundary cases.
JAX_PLATFORMS=cpu target/venv/bin/python -O examples/bounded_global_ordering.py
# exit0: both DataFrame and SQL outputs are {'x': [1], 'seq': [11]}.
```

The new quoted-name fixture initially expected successful collection of a
nonportable name. Its actual ConfigError established that the existing project
contract rejects it before execution. The corrected test keeps that exact
adversarial name, adds a golden escaped query assertion, and requires the
specific native ConfigError and portable-identifier diagnostic. An initial
generic ValueError expectation also failed and was corrected to ConfigError.
Neither failure required or justified widening the accepted name domain.
The final two-file run is29/29 green; no production change followed rebuilding
the wheel. Ruff check/format for the changed test passed again after the fix.

The old wheel evidence is not used for these Python results. A fresh local
CPython3.13.9 Linux dev wheel was built/reinstalled from this product tree;
it is not release/manylinux evidence. All62 package/native/stub entries match
the new wheel, and installed .py/.pyi files match repository source bytes.
The matching native's generated schema equals the unchanged committed schema
as parsed JSON. Final-HEAD package/Python/cross-platform repository gates
remain for CI; local targeted results do not replace them.

Current-product build provenance (same1208-file NUL-delimited fingerprint
scheme as earlier deliveries):

```json
{
  "base_head": "ab85af2c0e462b49f44d823547ef1ea07a580f67",
  "source_sha256": "66e10b8a07c57dafa6e2ce64536a6d3dda7ccce31777fd95c53ea9fbc2f1dca9",
  "source_file_count": 1208,
  "wheel_sha256": "66e5b7b27ceb2cb13471fd7b542f06164d5eae988aadb0f4a85c7a368eb3054f",
  "native_sha256": "c9a546db726b223c56349f4197bde6ee97cbc61a24864e3fcbe0f71d7fa08c74",
  "matched_archive_files": 62,
  "python": "3.13.9",
  "schema_matches_committed": true,
  "rust_artifact": "target/debug/deps/libcalc_flow-c4242eaab5409a3b.rlib",
  "rust_artifact_sha256": "839082b320e39eb722a92eeaac8ec8074180b32a02bfab2c1f09e4cbc3033f8a"
}
```

This run performs no full matrix, coverage, performance, branch/PR/environment
creation or delegation. Unaffected prior evidence is reused with its recorded
tree boundaries. The existing checks wakeup is retained, not duplicated.
The final push/readiness/check snapshot is reported in the issue handoff.
Pending checks and the required direct review of this refactoring block merge;
DAL-520 remains in_progress unless a subsequent authorized merge actually
completes. No required check or review is bypassed.

Gate repair product commit: `d354968339feb8a5d3a696f427c2339ae64ae6ab`.
All final targeted results and source/core/native/wheel hashes in this section
apply to that product tree. The following evidence-only commit appends this
mapping without changing the tested product files. The prior independent
approval is for ab85af2c; the new helpers/rendering/test refactoring are the
direct incremental review scope. External Codacy and final-head CI remain
subject to the actual next snapshot and the existing checks wakeup.

## Project-v3 capability inventory test synchronization

Base/reviewed product HEAD: `f1e0e43622fa6a474e699e2de4bdcad0a9fd21b2`.
The coordinator supplied actual Linux job131757725676 (run43384132153) and
Windows job131748742603 (run43384132120) failures: both report only
`TestProjectV3Surface::test_v3_schema_constant`, line180, whose complete
operator-kind tuple omitted the newly declared bounded_order capability.
The reported Linux Ruff/complexity and96.19% coverage results remain evidence
for their original HEAD; no local coverage rerun or final-head extrapolation
is claimed here.

The frozen API note's capability section explicitly requires bounded_order@1
with batch/stream modes and layouts1/2. The existing inventory at
`python/calc_flow/capabilities.py:1020` already places this capability before
cross_section. The test now adds exactly the literal `"bounded_order"` at
that same first position. The complete ordered tuple and every old capability
remain, as do project_format_versions=(3,), the explicit package/revision
fixture and the connector tuple assertion. No inclusion check, dynamically
generated expected list, product edit, version/schema edit or gate change
was introduced.

The same failure was actually reproduced locally before editing:

```sh
JAX_PLATFORMS=cpu target/venv/bin/python -m pytest -q python/tests/test_project_v3.py::TestProjectV3Surface::test_v3_schema_constant
# RED exit1: 1 failed; index0 bounded_order != cross_section, one extra kind.
```

After the one-literal fix:

```sh
JAX_PLATFORMS=cpu target/venv/bin/python -m pytest -q python/tests/test_project_v3.py::TestProjectV3Surface::test_v3_schema_constant
# GREEN exit0: 1 passed in0.23s.
JAX_PLATFORMS=cpu target/venv/bin/python -m pytest -q python/tests/test_project_v3.py
# exit0: all12 file-local cases passed in0.24s.
target/venv/bin/python -m ruff check python/tests/test_project_v3.py
# exit0.
target/venv/bin/python -m ruff format --check python/tests/test_project_v3.py
# exit0: one file already formatted.
git diff --check
# exit0.
```

Before the RED run, the unchanged1208-file product-source fingerprint was
actually recomputed and checked against the preceding build ledger:
`66e10b8a07c57dafa6e2ce64536a6d3dda7ccce31777fd95c53ea9fbc2f1dca9`.
The imported native's SHA256 was also rechecked, unchanged:
`c9a546db726b223c56349f4197bde6ee97cbc61a24864e3fcbe0f71d7fa08c74`.
Thus the currently installed CPython3.13.9 dev wheel from the gate repair is
reused for these checks; no rebuild is needed for a test-only expected tuple.
Its recorded wheel SHA256 remains
`66e5b7b27ceb2cb13471fd7b542f06164d5eae988aadb0f4a85c7a368eb3054f`.
The prior independent reviewer verified the matching62 installed product
files, core/native/source/wheel hashes and approved this unchanged product.

This commit includes only that test literal and this analysis appendix.
The frozen API note stays byte-identical at SHA256
`b4a087ef52cb9b3a43803666e13365dfce544205a6af8c97b88cb34dd8758624`.
The reviewer-owned report, including its product approval of f1e0e436, stays
unchanged/unstaged at SHA256
`a263a8109f433f011def5856e6c6cc5070b501abe62da23ab198344472c3d357`.
Existing product approvals and unaffected verification remain valid; no new
review, full matrix, coverage, performance, delegation or DAL-521 work is
started for this test-list synchronization. Final-head CI/merge state is
reported in the issue comment after the authorized push and one nonblocking
snapshot. Pending checks retain in_progress and the existing checks wakeup;
no required check may be bypassed.
