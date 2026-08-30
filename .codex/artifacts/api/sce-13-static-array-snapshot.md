# SCE-13 Static Array Snapshot - API Note

Artifact slug: `sce-13-static-array-snapshot`.

## Decision Status

- Evidence head: PR #214 at
  `f882d2133e5805e80b729089877baed91bb0b890`; base is
  `origin/main@760eea4`.
- Determining input: the workspace member approved the precise public delta
  in issue comment `01a05247-fb7b-7058-a785-24dd2abf85d8`; the orchestrator
  recorded the binding interpretation in
  `01a05249-14ca-7b30-a605-cc8c1b19b788`.
- Critique input: `.codex/artifacts/critiques/sce-13-static-array-snapshot.md`
  (`Revise`, delivered in issue comment
  `01a0525c-be8b-715c-94fa-2088bf9378c4`).
- Status: **designer-frozen after the one permitted blocking-item correction;
  ready for `cf-implementer` with no further wording/format review loop**.
- Scope: formally expose `StaticArraySnapshot`, `StaticArrayValues`,
  `Batch::static_array_snapshot()`, and only the accessors required to read
  the snapshot across the `calc-flow` -> `calc-flow-python` crate boundary.
- Non-goals: a shared payload crate, a public latched-payload type, a new
  serialized shape, or any Python/Studio signature or OpenAPI delta.

The manual approval supersedes the earlier draft's preference for a single
borrowed view. The correction below closes the critique without reopening or
expanding that choice.

## Audiences

- Rust users: two additive crate-root data types and one low-level `Batch`
  inspection method. These are supported public API even though the immediate
  consumer is the PyO3 crate.
- Python users: no public adapter or stub change. Native placement reads the
  owned snapshot once per job inside a blocking worker.
- Studio clients: n/a. Live static payloads remain unavailable over REST.

## Surface Today

On `origin/main`, the engine owns `LatchedArrayPayload` and
`LatchedArrayValues` as crate-private implementation details.
`StreamOperatorContext::static_input(name)` returns a borrowed public `Batch`,
but an independent crate cannot inspect the latched descriptor or scalar
values. Rust has no friend-crate visibility, so a public seam is necessary.

PR #214 currently adds two `#[doc(hidden)]` crate-root types and an owned
snapshot method. `#[doc(hidden)]` does not remove semver exposure. Its public
enum is exhaustively matchable, and both types derive payload-bearing `Debug`.
The public item set is approved; those evolution and redaction defects are not.

## Activated SCE-11 First-Consumer Validation

SCE-13 is the first consumer that makes both declaration kind and graph-port
kind operational. It therefore activates SCE-11 review record F7, obligation
4: project compilation must cross-check those two kinds in both directions.
This is a validation requirement, not an additional public Rust API.

- `validate_static_inputs` must retain the declared `BatchKind` for every
  unconnected graph input instead of reducing inputs to a set of port names.
- `StaticInputSpec::Table` matches only an unconnected `BatchKind::Table`
  input port; `StaticInputSpec::Array` matches only an unconnected
  `BatchKind::Array` input port.
- A mismatch produces a `ValidationIssue` with code
  `incompatible_batch_kind` and path `static_inputs[{index}].kind`. Its message
  names the declaration kind, input-port name, and required port kind without
  payload values.
- This check runs during the shared project-validation phase, before
  `StreamExecutionPlan` construction and before any source or provider
  lifecycle work. Both `compile_stream_project` and
  `compile_stream_project_graph` must reject the same mismatch; there is no
  runtime fallback or coercion.

`crates/calc-flow/tests/project_connector_compile.rs` must contain project
tests for both mismatch directions: an array declaration targeting a table
port, and a table declaration targeting an array port. Each case exercises
both compile entry points and asserts the exact issue code and indexed path.
A matched array-declaration/array-port case remains a positive witness.

## Frozen Public Surface

The implementation must project exactly this API:

```rust
// crates/calc-flow/src/batch.rs

/// Host-neutral scalar carriers owned by one static-array snapshot.
#[non_exhaustive]
pub enum StaticArrayValues {
    /// Logical boolean values in compact non-null C order.
    Bool(Vec<bool>),
    /// Logical signed-integer values carried at `i64` width.
    Int(Vec<i64>),
    /// Logical unsigned-integer values carried at `u64` width.
    Uint(Vec<u64>),
    /// Logical float32/float64 values carried at `f64` width.
    Float(Vec<f64>),
}

/// An owned, host-neutral snapshot of one engine-latched static array.
#[must_use = "the owned snapshot contains a complete static-array value copy"]
#[non_exhaustive]
pub struct StaticArraySnapshot {
    // backend: String,
    // dtype: String,
    // shape: Vec<u64>,
    // nulls: Option<Vec<bool>>,
    // values: StaticArrayValues,
    // All fields remain private.
}

impl StaticArraySnapshot {
    #[must_use]
    pub fn backend(&self) -> &str;

    #[must_use]
    pub fn dtype(&self) -> &str;

    #[must_use]
    pub fn shape(&self) -> &[u64];

    #[must_use]
    pub fn nulls(&self) -> Option<&[bool]>;

    #[must_use]
    pub fn values(&self) -> &StaticArrayValues;
}

impl Batch {
    /// Copies this batch's engine-latched static array into an owned snapshot.
    #[must_use]
    pub fn static_array_snapshot(&self) -> Option<StaticArraySnapshot>;
}

// crates/calc-flow/src/lib.rs
pub use batch::{
    Batch, BatchKind, BatchMetadata, ExternalPayload, StaticArraySnapshot,
    StaticArrayValues, TableBatch,
};
```

Visibility and trait constraints are normative:

- Both types, all five listed accessors, and the `Batch` method are ordinary
  documented `pub` API and both types are re-exported at the crate root.
- `batch` remains a private module. `LatchedArrayPayload`,
  `LatchedArrayValues`, their fields, and `Batch::latched_array_payload`
  remain no wider than `pub(crate)`.
- Snapshot fields remain private. There is no public constructor or mutation
  method for `StaticArraySnapshot`.
- Do not use `#[doc(hidden)]`, a Cargo-feature visibility workaround, or a
  consumer allowlist. Enabled public Rust API remains semver surface.
- Do not implement or derive `Debug`, `Display`, `Serialize`, `Deserialize`,
  `IntoIterator`, `AsRef<[u8]>`, or any Python conversion on either type.
- Do not expose additional typed accessors on `StaticArraySnapshot`.
  `values()` plus the approved enum is the complete scalar read surface.
- Neither type requires `Clone`, `Copy`, `Eq`, `Hash`, `Default`, or ordering
  traits. In particular, remove the PR head's current `Clone` derivation and
  avoid freezing traits not needed by the cross-crate seam.
- Neither type has a stable layout and neither receives `#[repr(...)]`.

## Snapshot and Accessor Semantics

- `static_array_snapshot()` returns `Some` only for the existing
  engine-latched static-array representation. `BatchKind::Array` alone is not
  enough: a general `ExternalPayload` array returns `None`; a table returns
  `None`.
- Every call returning `Some` creates an owned, detached copy of backend,
  dtype, shape, optional null bitmap, and compact scalar carriers. No snapshot
  field aliases the latched payload or caller-mutable host memory.
- The snapshot remains readable after the source `Batch` is dropped. This is
  the intentional owned boundary selected by the manual approval.
- Accessors are read-only borrows from the snapshot. No accessor clones or
  exposes mutable storage.
- `nulls() == None` means no logical null bitmap was supplied. `Some` preserves
  every logical C-order position, including an all-false bitmap.
- Placement rejects a bitmap only when at least one entry is `true`.
  `Some(all_false)` is semantically equivalent to no logical nulls for
  placement, must succeed, and remains distinguishable from `None` in the
  snapshot. `Some(&[])` for a zero-element array must also succeed.
- The value vector contains only non-null logical scalars in C order. When a
  null bitmap is present, values are consumed at non-null positions; no
  placeholder is stored for a null cell.
- Empty arrays preserve their carrier family: for example a float snapshot
  contains `StaticArrayValues::Float(vec![])`, not an absent value family.
- Carrier type does not replace `dtype()`: signed widths use `i64`, unsigned
  widths use `u64`, and both float widths use `f64`. Consumers reconstruct the
  declared width from `dtype()`. A `float32` carrier is already validated as a
  lossless representation of the latched float32 logical value.
- Descriptor/value consistency is established by the existing latch
  constructors. The snapshot method does not accept or normalize an arbitrary
  public `StaticArrayValues` into a `Batch`.

## Ownership and Placement Boundary

The approved owned snapshot is a deliberate exception to keeping the latched
handle as the sole allocation; it does not weaken ownership of the running
job. Its frequency and transient lifetime are bounded, but its peak byte cost
is not:

1. Before source/provider lifecycle work, the Python runner has already copied
   caller-mutable NumPy/JAX memory into the engine-owned latch.
2. The stream operator moves an owned cloned `Batch` handle and the static
   input name into `spawn_blocking`.
3. The worker calls `static_array_snapshot()` *inside* that closure. Snapshot
   vector cloning, Python-list construction, NumPy reshape, and JAX placement
   all occur off the Tokio operator task.
4. The worker reads the snapshot through its five accessors and an enum match
   that includes a wildcard arm for future variants.
5. The snapshot and temporary Python host list are dropped before the worker
   returns. Only after successful placement may the operator atomically cache
   the provider-owned `Batch` for later micro-batches.

The core and binding must not cache, checkpoint, serialize, globally register,
or place this owned snapshot in `BatchMetadata`. Public Rust callers who
explicitly request and retain a snapshot own that detached copy; library-owned
runtime paths do not retain one beyond first placement.

D10's exactly-once release rule continues to govern the engine-latched job
handle. Explicit caller-owned snapshots are independent return values, while
the binding's transient snapshot is created and destroyed exactly once per
placed static input per job.

`static_placement_bytes` remains the checked logical provider-transfer count
(`dtype` width times element count). It is nonzero on first placement and zero
for cached later micro-batches. The internal Rust carrier clone is not counted
again and must not change the frozen metric meaning. It is not a measurement
of peak memory, RSS, or internal cloning.

Every successful public snapshot call is `O(rank + bitmap length + non-null
value count)` in time and additional memory, plus copied descriptor strings.
During Python placement, the engine latch, snapshot carriers, Python host list,
NumPy host storage, and a provider-owned JAX result may coexist. Therefore the
transient peak is `O(n)` and has no byte bound beyond the validated input size.
The runtime confines that peak to first placement of each static input per
job; retained public snapshots remain the Rust caller's responsibility.

### Deterministic blocking-worker and cache gate

The binding tests must use a private `#[cfg(test)]` placement gate; it is not a
new public hook or API. The gate proves scheduling and cache order without a
large payload, elapsed-time threshold, sleep, timeout, or poll-count
heuristic:

1. Run a `#[tokio::test(flavor = "current_thread")]` and record the Tokio
   runtime thread ID. The placement hook is the first instruction inside the
   `spawn_blocking` closure, before snapshot cloning, Python-list construction,
   NumPy conversion, or JAX placement.
2. The hook sends an `entered(actual_thread_id)` event through an async-safe
   channel. If that ID equals the runtime thread ID, it reports a wrong-thread
   failure and does not block, avoiding a deadlock while making incorrect
   placement deterministically fail.
3. On a real blocking thread, the hook waits on a synchronous release channel.
   While it is held, an independently spawned Tokio task must send and complete
   a sibling progress event. The test then releases the worker.
4. Private test events or counters observe the actual cache-commit boundary.
   The worker returns provider outputs plus newly placed cache entries; the
   async side rechecks cancellation and only then commits them atomically.

The same gate covers three required outcomes. An injected placement failure
after release returns an error with zero cache commits and no map entry.
Cancellation while the worker is held, followed by a successful worker return,
is rejected by the post-worker cancellation check with zero cache commits and
no map entry. On success, commits remain zero while held, then become exactly
one only after the non-cancelled worker returns; a second micro-batch uses the
cache without re-entering placement, so the placement-call count remains one.

## Redaction Boundary

The values enum grants intentional in-process read authority to code holding an
explicit snapshot. It is not permission for calc-flow-generated observability
to disclose those values.

- Library errors, status, metrics, logs, panic text, and provider diagnostics
  must never format `StaticArrayValues`, a scalar vector, null positions, the
  snapshot, or the underlying `Batch`/payload.
- Backend, dtype, shape, input names, versions, counts, and digests remain
  allowed descriptors under D12. Scalar values, null positions, arbitrary
  metadata, secrets, and memory addresses remain forbidden.
- Null redaction includes both the rendered bitmap and any equivalent index or
  position list. Diagnostics may say that null elements exist, but may not say
  where they occur.
- Both public types deliberately have no `Debug`, `Display`, or serde surface.
  This removes the current PR bridge's new accidental value disclosure.
- Copy-byte metrics contain counts only. Checkpoints continue to carry only
  bounded static-input digest evidence and never a snapshot.

## Error Semantics

The Rust method is an infallible type probe. `None` is the complete public
result for a table or non-latched external array. Snapshot materialization has
no fallible allocation API distinct from Rust's ordinary allocation behavior.

Project compilation reports declaration/port incompatibility before runtime:

| Violation                            | Code                      | Path                            | Message shape                                                           |
| ------------------------------------ | ------------------------- | ------------------------------- | ----------------------------------------------------------------------- |
| array declaration targets table port | `incompatible_batch_kind` | `static_inputs[{index}].kind`   | `array static input {name:?} requires an array port, found table`       |
| table declaration targets array port | `incompatible_batch_kind` | `static_inputs[{index}].kind`   | `table static input {name:?} requires a table port, found array`        |

The PyO3 consumer preserves the existing
`CalcFlowError::ExternalProvider`/streaming projection, carries `name`
separately, and uses these path-first inner messages:

| Violation                                      | Inner message shape                                                                 |
| ---------------------------------------------- | ----------------------------------------------------------------------------------- |
| no latched snapshot for an array static input  | `static_inputs.{name}: expected an engine-latched array batch`                      |
| any logical null is present                    | `static_inputs.{name}.nulls: Array API placement does not support null elements`    |
| backend has no approved placer                 | `static_inputs.{name}.backend: unsupported static array backend {backend:?}`        |
| dtype has no approved host width               | `static_inputs.{name}.dtype: unsupported static array dtype {dtype:?}`              |
| dtype and carrier family disagree              | `static_inputs.{name}.dtype: latched value carrier does not match {dtype:?}`        |
| enum contains a future unsupported variant     | `static_inputs.{name}.dtype: snapshot value family is not supported by this host`   |
| checked placement-byte product exceeds `usize` | `static_inputs.{name}.shape: placement byte count exceeds usize`                    |
| blocking worker terminates                     | existing payload-free `stream callback worker terminated` provider error            |

Descriptor/carrier disagreement is an internal invariant failure but returns an
error instead of panicking, asserting, or selecting a fallback. The wildcard
enum arm returns the future-variant error; it is never `unreachable!()`.
Python/NumPy/JAX failures continue through the existing sanitized provider
projection with the static-input path prefixed and no appended snapshot/value
representation. Cancellation is checked before and after the worker. Failure
or cancellation must not populate `placed_static_inputs`.

## Non-Exhaustive and Semver Contract

This is an explicit additive exception to SCE-11's frozen list of four
static-input crate-root exports. The approved SCE-13 delta adds exactly
`StaticArraySnapshot` and `StaticArrayValues`; the inherent `Batch` method and
five accessors are part of the same public contract.

`#[non_exhaustive]` is mandatory on both types:

- private snapshot fields may evolve without enabling downstream construction
  or destructuring; and
- consumers of `StaticArrayValues` must include a wildcard arm, allowing a
  future digest-version family to add a variant compatibly.

Once released, the following require a major version: removing/renaming either
type, variant, method, or accessor; narrowing visibility; changing a variant's
`Vec<T>` carrier; changing the owned return to a borrow; exposing snapshot
fields; changing `None`, empty-array, or compact-null semantics; or adding a
payload-bearing formatting/serialization contract.

The following are compatible additive evolution: adding private snapshot
fields, adding a new enum variant, and adding a read-only accessor required for
a new descriptor. An enum variant may be added only with a new supported
digest/provider contract; it is not a silent reinterpretation of an existing
dtype.

The approved carriers make both public types observably `Send`, `Sync`,
`Unpin`, `std::panic::UnwindSafe`, and `std::panic::RefUnwindSafe`. An external
compile-time assertion must freeze those auto-traits. Adding a private field or
enum variant is compatible only if it preserves all five; weakening one after
release requires a major version even though no explicit impl is written.
`Clone`, `Copy`, `Debug`, and serde traits remain deliberately outside the
contract.

The snapshot is not a wire format, FFI layout, canonical digest encoding, or
serde contract. Digest bytes remain private to `calc-flow` and must not be
reused for provider placement.

## Documentation and Freeze Artifacts

Implementation must update all public witnesses in the same PR:

- full rustdoc on both types, every variant/accessor, and the `Batch` method,
  including one happy-path example and owned-copy/non-latched/compact-null
  semantics, the per-call `O(n)` allocation, and the larger transient Python
  placement peak;
- `docs/api-reference.md`: add both new types to the static-input crate-root
  export row, state that they are the complete SCE-13 export delta, and explain
  that `static_placement_bytes` is logical transfer rather than peak memory;
- `docs/rust-api.md`: extend the exact frozen static-input export list from the
  four SCE-11 items with these two SCE-13 items and document the low-level,
  non-serialized owned snapshot boundary;
- `CHANGELOG.md`: record the additive Rust public API; and
- this SCE-13 note/review record as the additive freeze witness.

Do not edit the byte-pinned
`.codex/artifacts/api-notes/sce-11-static-stream-inputs.md`. The reviewed
SCE-13 note supersedes only its exact-export-count statement. The SCE-11
ownership, digest, recovery, and redaction contracts otherwise remain intact.

No checked-in `cargo public-api`/rustdoc-JSON snapshot exists at the evidence
head. Do not invent a second generated format for this fix. The checked-in
freeze witnesses are `docs/api-reference.md`, `docs/rust-api.md`, this reviewed
note, and the external-crate compatibility test. After the one critic pass,
record this note's git blob hash in the SCE-13 review handoff.

The proportional documentation gate is:

```text
RUSTDOCFLAGS="-D warnings" cargo doc -p calc-flow --no-deps
RUSTDOCFLAGS="-D warnings" cargo test -p calc-flow --doc
```

The second command actually executes the `compile_fail` rustdoc witnesses;
`cargo doc` alone does not. These run with repository Markdown
link/table/trailing-whitespace checks. No Studio OpenAPI, generated TypeScript,
Python public signature, or `_native.pyi` change is permitted for this delta.

## Directed Compatibility Tests

Implementation is incomplete without these focused tests:

1. Add an external integration test under `crates/calc-flow/tests/` importing
   `calc_flow::{Batch, StaticArraySnapshot, StaticArrayValues}`. Type-check a
   helper returning `Option<StaticArraySnapshot>`, inspect all five accessors,
   and match values with a mandatory wildcard arm. In the same external test,
   assert that both public types satisfy `Send + Sync + Unpin + UnwindSafe +
   RefUnwindSafe` without adding new explicit impls.
2. Prove owned lifetime: obtain a snapshot, drop its source `Batch`, then read
   every descriptor and scalar value successfully.
3. In `batch.rs`, cover bool, all signed/unsigned width families, float32,
   float64, an empty array, and nulls. Assert exact descriptor copies, compact
   non-null ordering, the expected enum variant, empty variant retention,
   preservation of `Some(all_false)`, and `Some(&[])` for a zero-element array.
4. In the same module, use private access to prove descriptor/bitmap/value
   storage is detached from the latched vectors rather than aliased. Assert a
   table and a general custom `ExternalPayload` return `None`.
5. Add rustdoc `compile_fail` examples proving neither public type implements
   `Clone` or payload-bearing `Debug` and that a caller cannot construct or
   mutate a `StaticArraySnapshot` through private fields/accessors. Execute
   them with
   `RUSTDOCFLAGS="-D warnings" cargo test -p calc-flow --doc`; a successful
   `cargo doc` invocation is not a negative-API witness.
6. In `calc-flow-python`, place all current enum variants, including float32
   narrowing, `Some(all_false)`, and `Some(&[])` zero-sized arrays. Reject only
   a bitmap containing `true`. Also cover unsupported backend/dtype, byte
   overflow, and synthetic carrier mismatch through the narrowest internal
   helper; the external test in item 1 enforces the wildcard future-variant
   arm.
7. Implement the deterministic private blocking-worker gate specified above.
   It must prove current-thread Tokio schedulability, zero cache commit on
   failure, zero cache commit on cancellation, commit only after a successful
   non-cancelled worker return, and cache reuse without a second placement.
   Retain first-micro-batch bytes/nonzero and later-micro-batch bytes/zero.
8. In `crates/calc-flow/tests/project_connector_compile.rs`, add both
   declaration-kind/port-kind mismatch projects. Run each through
   `compile_stream_project` and `compile_stream_project_graph`, assert
   `incompatible_batch_kind` at `static_inputs[{index}].kind`, and retain a
   matched array/array positive project.
9. Seed failures with raw-value and memory-address redaction sentinels and use
   a distinctive mixed mask such as `[false, true, false, true]`. Assert final
   Rust/Python streaming errors preserve the exact `.nulls` path while omitting
   the rendered bitmap, equivalent position text such as `positions [1, 3]`,
   `index 1`/`index 3`, scalar canaries, and address canaries.
10. Run the affected core snapshot tests, external compatibility test, project
    connector compile tests, PyO3 mapped-static-input tests, both targeted
    rustdoc commands, fmt/clippy for `calc-flow` and `calc-flow-python`, and
    `git diff --check`. Full-workspace tests and a performance gate are not
    local requirements.

## Critique Closure Checklist

This is the one permitted blocker correction. Each `Revise` item is now a
normative implementation or compatibility-test obligation:

| Critique item           | Closed by                                             |
| ----------------------- | ----------------------------------------------------- |
| B1: kind versus port    | SCE-11 validation section and directed test 8         |
| B2a: negative witness   | Documentation gate and directed test 5                |
| B2b: worker/cache proof | Deterministic worker gate and directed test 7         |
| C1: all-false placement | Snapshot semantics and directed tests 3 and 6         |
| C2: auto-trait limits   | Semver contract and directed test 1                   |
| C3: owned-copy peak     | Ownership and placement boundary                      |
| C4: null-position leak  | Redaction boundary and directed test 9                |

## Why This Shape

- The human-approved two-type surface preserves the current PR's direct
  host-neutral reconstruction model and avoids an architecture detour.
- An owned snapshot lets the sibling crate perform blocking placement without
  exposing the latched payload/downcast target and without tying public users
  to internal `Arc<dyn ExternalPayload>` lifetimes.
- One `values()` accessor plus a non-exhaustive enum is the minimum reader for
  the approved values type; four additional typed snapshot accessors would
  duplicate it.
- Private snapshot fields keep construction and descriptor/value invariants in
  the core. Non-exhaustive types preserve controlled evolution.
- Removing `#[doc(hidden)]` and documenting the cost/redaction contract makes
  the real Rust semver surface explicit.

## Rejected Alternatives

- **Replace both types with one borrowed snapshot view.** Rejected by the
  determining manual approval. It would reopen the selected item set and
  change the approved owned lifetime.
- **Keep PR #214 unchanged.** Rejected: `#[doc(hidden)]`, exhaustive matching,
  payload-bearing `Debug`, missing docs, and unrecorded semver constraints are
  not approved.
- **Add four typed slice accessors beside `values()`.** Rejected as duplicate
  surface; the enum match already provides typed vectors.
- **Expose `LatchedArrayPayload`, `LatchedArrayValues`, or the `Any` downcast
  target.** Rejected: freezes engine storage/digest internals and permits
  unlatched provider payloads to masquerade as static values.
- **Return canonical bytes, JSON, Arrow, or serde values.** Rejected: they
  allocate, alter dtype/null semantics, or create a wire format; digest bytes
  are not a reconstruction format.
- **Put raw values in metadata, checkpoint state, a global registry, or another
  side channel.** Rejected by D10-D12 ownership/redaction/lifetime rules.
- **Use a feature-gated public bridge or retain `#[doc(hidden)]`.** Rejected:
  enabled-feature and hidden public items remain public semver API.
- **Move NumPy/JAX placement into `calc-flow`.** Rejected: reverses the crate
  dependency and couples the core to one host binding.
- **Create a shared payload crate.** Rejected as an expressly unapproved,
  larger architecture and publication change.
- **Place snapshots on the Tokio task before entering `spawn_blocking`.**
  Rejected: snapshot cloning and all host/provider conversion belong inside
  the blocking worker.
- **Use payload size or elapsed time as a schedulability gate.** Rejected as
  nondeterministic; the private entry/release protocol provides causal proof.

## Example

```rust
use calc_flow::{Batch, StaticArrayValues};

let weights = Batch::static_array_float(
    "numpy",
    "float32",
    vec![2, 1],
    None,
    vec![0.25, 0.75],
)?;
let snapshot = weights
    .static_array_snapshot()
    .expect("engine-latched array");

assert_eq!(snapshot.backend(), "numpy");
assert_eq!(snapshot.dtype(), "float32");
assert_eq!(snapshot.shape(), &[2, 1]);

match snapshot.values() {
    StaticArrayValues::Float(values) => {
        assert_eq!(values.as_slice(), &[0.25, 0.75]);
    }
    _ => return Err("unexpected or unsupported static-array value family".into()),
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Open Questions and Remaining Blockers

- There is no API-design open question: the member's approval fixes the two
  types, method, owned lifetime, and minimal accessor set above.
- The single critique pass is closed by the checklist above. There is no
  remaining API-note blocker and no further wording or formatting review loop.
- `cf-implementer` can resume formalizing this bridge and complete the other
  PR #214 reviewer blocker classes. This note alone does not make the PR
  merge-ready or clear pending CI.
