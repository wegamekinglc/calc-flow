# Immutable Static Stream Inputs (SCE-11) - API Note

Artifact slug: `sce-11-static-stream-inputs`.

## Control and Checksum

- Baseline: `main@079d0ed813c61006e23add311b9dc164cfc9bfc2` (SCE-07 squash).
- Controlling inputs (SCE-00 frozen contract, in precedence order):
  - `.codex/artifacts/specs/symbolic-computation-contract.md` (D10/D11/D12),
    SHA-256 `af9e1a68d6067c221e793703e20b08f2d3a2dc1b66a5ab3ffa133418777b6c21`
  - `.codex/artifacts/api-notes/symbolic-computation-engine.md` (§2.5, §3.1,
    §3.2, §3.3, §4, §7, §8, §9), SHA-256
    `be93ca04affb6fe5e4fbb87d6bb28c89e2a182c594911547b04ff9d4d695ccf2`
  - `docs/superpowers/specs/2026-08-22-symbolic-computation-engine-design.md`
    (Matrix Operations section), SHA-256
    `d742d311a5fc9e89c3763a6836a01e02092db04f2645606f13e38b099e521175`
- Status: **frozen for SCE-11 implementation**. This note extracts and restates
  the SCE-11-relevant slice of the approved SCE-00 contract; it opens no new
  design question. Where this note and the controlling inputs disagree, the
  controlling inputs win.
- Reference integrity: this note's own SHA-256 is recorded in the DAL-143
  delivery comment (thread `01a04593-2a0c-7ef3-8874-f4116b831993`) and is the
  reference hash cited by the SCE-11 implementer/tester/reviewer stages.

## Audiences

- Rust users: additive builder for immutable static `Batch` side inputs; no
  change to existing `StreamingRunner::new` call sites.
- Python users: one new keyword-only `static_inputs` mapping on
  `calc_flow.StreamingRunner`; `Batch` values only; validated once before
  sources open.
- Studio clients: project-v3 documents gain a data-only `static_inputs`
  declaration array in `web-ui/openapi.json` and generated types; no live
  payload ever crosses REST.

## Surface Today (baseline `079d0ed8`)

- Rust (`crates/calc-flow/src/continuous.rs:894`):
  `StreamingRunner::new(plan, sources, sinks, checkpoints)` plus
  `with_runtime_config(config)`. No static-input surface.
- Python (`python/calc_flow/runtime.py:891`):
  `StreamingRunner(plan, sources=None, sinks=None, checkpoints=None, *,
  config=None)`. Project-backed plans reject all four optional arguments.
- PyO3 (`crates/calc-flow-python/src/continuous.rs:884`): `_StreamingRunner`
  takes five required positional arguments; `_native.pyi` mirrors it.

## Frozen Surface

### Python public constructor

```python
class StreamingRunner:
    def __init__(
        self,
        plan: StreamExecutionPlan,
        sources: Mapping[str, SourceBinding] | None = None,
        sinks: Mapping[str, Sequence[SinkBinding]] | None = None,
        checkpoints: ManagedCheckpointRuntime | None = None,
        *,
        config: StreamRuntimeConfig | None = None,
        static_inputs: Mapping[str, Batch] | None = None,
    ) -> None: ...
```

- Additive, keyword-only, second keyword after `config`; existing call sites
  are unaffected.
- `None` normalizes to an empty mapping. Keys must be `str` and values must be
  `calc_flow.Batch`; anything else raises `TypeError` before any native
  construction.
- The mapping is defensively copied immediately; later caller mutation of the
  passed mapping has no effect.
- Project-backed plans keep owning sources/sinks/checkpoints/config (those
  four remain rejected when a project plan is supplied) but do **not** own
  static payload values: `static_inputs` is exempt from that rejection and is
  required when the plan declares static inputs.
- `StreamExecutionPlan.static_input_ids` is a property returning the sorted
  exact names as `tuple[str, ...]`; the existing `source_binding_ids` property
  excludes those names.

### Rust crate surface

```rust
impl StreamingRunner {
    #[must_use = "the runner owns the supplied static input handles"]
    pub fn with_static_inputs(self, inputs: BTreeMap<String, Batch>) -> Result<Self>;
}

impl StreamExecutionPlan {
    pub fn static_input_ids(&self) -> Vec<&str>;
    pub fn static_inputs(&self) -> &BTreeMap<String, StaticInputSpec>;
}
```

- `StreamingRunner::new` is unchanged and remains source compatible. The
  Python binding constructs it, then calls `with_runtime_config`, then
  `with_static_inputs`.
- `Batch` is clone-by-owned-handle: the runner acquires an immutable
  engine-owned handle and never mutates caller data.
- SCE-11 crate-root exports are exactly the static-input subset of API-note
  §3.1: `StaticInputDigest`, `StaticInputSpec`, `StaticMutability`,
  `STATIC_INPUT_DIGEST_VERSION`. The rolling/cross-section exports belong to
  SCE-08/SCE-09 and land on their own branches; whichever merges later rebases
  the shared export list and serialized contract files.

### PyO3 binding and stub

`_StreamingRunner.__new__` gains a sixth required positional argument
`static_inputs: Mapping[str, object]` (empty mapping when the public argument
is `None`; the adapter normalizes before calling native). `_native.pyi`
mirrors the new parameter. The private binding stays all-positional; making
the new parameter keyword-only in the private binding is an accepted
implementer variation (non-blocking) as long as the public Python surface
above is exact.

### External-input descriptor

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum StaticMutability { Static }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StaticInputSpec {
    Table {
        name: String,
        mutability: StaticMutability,
        schema: Vec<ArrowFieldSpec>,
    },
    Array {
        name: String,
        mutability: StaticMutability,
        backend: String,
        dtype: String,
        shape: Vec<u64>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StaticInputDigest {
    pub digest_version: String,
    pub sha256: String,
}
```

- `StaticInputSpec` deliberately does not derive `Eq` (its table variant holds
  `ArrowFieldSpec`, which is `PartialEq`-only); `StaticInputDigest` stays `Eq`.
- `ProjectSpec` gains
  `#[serde(default, skip_serializing_if = "Vec::is_empty")] pub static_inputs:
  Vec<StaticInputSpec>`. Names are unique and must name graph external input
  bindings; a binding named by `static_inputs` is not a source binding. Empty
  lists are omitted, so previously valid project-v3 documents canonicalize
  byte-for-byte unchanged. `PROJECT_FORMAT_VERSION` stays `3`.
- `schemas/project-v3.schema.json`, the Python `ProjectDocument`, Studio
  OpenAPI, and generated TypeScript move in the same commit as the Rust
  serialized types (atomic, additive).

### Digest and lineage field spelling

- `pub const STATIC_INPUT_DIGEST_VERSION: &str = "calc_flow.static_input.digest.v1";`
- The digest is lowercase hexadecimal SHA-256 over the canonical tagged bytes
  of API-note §7 (MAGIC `ASCII("calc_flow.static_input.digest.v1") || 0x00`,
  then `0x01`, `TEXT(input_name)`, then the `0x10` table / `0x11` array
  payload). §7 is the byte-exact authority; SCE-11 implements it verbatim and
  never hashes through a provider-specific fallback. `BatchMetadata`, chunk
  boundaries, dictionary indices, and device/stride layout are excluded; NaNs
  canonicalize per dtype.
- `CheckpointManifestFields` and `CheckpointManifest` gain
  `pub static_inputs: BTreeMap<String, StaticInputDigest>` serialized under
  root field `static_inputs` with
  `#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]`. Existing
  manifests with no static inputs stay readable and keep their canonical
  bytes. When non-empty, the state-checksum input becomes the exact
  four-field object `{"operators":...,"sinks":...,"sources":...,"static_inputs":...}`;
  when empty it remains the existing three-field object. Keys use canonical
  order. `ManifestExpectation` gains
  `pub static_inputs: &'a BTreeMap<String, StaticInputDigest>`.
- Prepared-job identity and the checkpoint expectation carry sorted
  `(name, digest_version, sha256)` triples.
- Static **declarations** participate in the semantic plan fingerprint.
  Payload digests MUST NOT enter `StateLineageKey`, the lineage-directory
  hash, the semantic plan fingerprint, or manifest discovery filters. This is
  the load-bearing choice: changed values must reach the *existing* lineage
  and fail there, never silently select a fresh lineage.
- Frozen recovery/preflight order:
  1. copy and validate the exact static input mapping;
  2. acquire engine-owned immutable `Batch` handles;
  3. compute all canonical digests;
  4. open the `StateLineageKey` selected by only the existing pipeline name
     and semantic plan fingerprint;
  5. if a manifest exists, compare the exact static name/version/digest map;
  6. validate all remaining manifest/segment/state invariants;
  7. install state and static handles; only then
  8. open sources and operator/provider/sink lifecycles.
- A first launch with no manifest starts with the validated set and records it
  in the first checkpoint. A launch against a terminal manifest performs the
  same comparison before returning the terminal outcome.

### Python ownership semantics

- Construction copies the mapping only. Validation, digest computation, and
  latching happen exactly once per job — during runner start, completing
  before any source, operator, sink, or provider lifecycle method runs
  (acceptance: "operator 在 source open 前可见").
- After latch, the job-visible value and digest are frozen: caller-side
  mutation of the passed mapping, of the original Python `Batch` objects, or
  of externally mutable backing array memory must not change what operators
  observe or what a later recovery compares. An engine-owned immutable handle
  cannot alias caller-mutable memory, so payloads that are externally mutable
  are snapshotted at latch.
- Static handles are released exactly once on every exit path — success,
  cancellation, startup failure, or recovery failure — and are never
  duplicated into a checkpoint.
- A static value is never re-sent as stream data and never treated as an
  infinite source (SCE-11 non-goal).

### Redaction contract

Status, metrics, logs, and errors expose static-input **names, digest
versions, and digests at most**. They never expose raw payloads, array or
table values, secrets, arbitrary metadata, or Python memory addresses. The
one exact frozen recovery message (via `CalcFlowError::CheckpointMismatch` /
Python `CheckpointError`, raised before any source opens):

```text
static_inputs.{name}.digest: checkpoint digest {stored} does not match prepared digest {prepared} for calc_flow.static_input.digest.v1
```

Missing and extra names on recovery use the `static_inputs.{name}` path.

### Studio worker restoration

- REST/OpenAPI carries declarations only. Live `Batch`/Arrow/NumPy/JAX values
  are never serialized into a project document or any REST payload (SCE-11
  non-goal).
- A Studio-submitted run whose project declares static inputs cannot be
  satisfied over REST in this release: run-manager validation fails closed
  **before a worker is spawned**, with an error naming
  `static_inputs.{name}` as unresolvable. Engine-level preflight (missing
  input before source open) remains the backstop for any in-process caller.
  No new REST endpoint or route is added. Exact Studio wording follows the
  existing `RunManagerError` conventions (implementer freedom, non-blocking).

## Why This Shape

- Keyword-only `static_inputs` after `config` beats a positional slot: it is
  the rare argument, and appending it keeps every existing positional call
  site valid.
- Rust builder `with_static_inputs` beats widening `new`: the Rust `new`
  signature stays source compatible per API-note §9, matching the existing
  `with_runtime_config` pattern.
- Declarations in the plan fingerprint but digests only in prepared-job and
  manifest identity beats putting digests in the lineage key: digests in the
  key would let a changed value silently select an empty fresh lineage and
  bypass recovery validation — the exact failure the acceptance criteria
  forbid.
- Mapping-of-`Batch` beats per-input wrapper objects: matches the
  `sources`/`sinks` mapping convention and keeps the common case one line.

## Error Cases

| Input violation                                | Path / message shape                                                                                        | Exception                  |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------- |
| non-`str` key or non-`Batch` value             | `static_inputs must be a mapping of calc_flow.Batch values`                                                 | `TypeError` (pre-native)   |
| plan declares an input, none supplied          | `static_inputs.{name}: required static input is missing` (before source open)                               | `StreamingRuntimeError`    |
| supplied input not declared by the plan        | `static_inputs.{name}: unexpected static input is not declared by the plan` (before source open)             | `StreamingRuntimeError`    |
| wrong batch kind                               | `static_inputs.{name}.kind: ...` (before source open)                                                       | `StreamingRuntimeError`    |
| table schema mismatch                          | precise schema path under `static_inputs.{name}.schema...` (before source open)                             | `StreamingRuntimeError`    |
| array backend/dtype/shape mismatch             | `static_inputs.{name}.backend` / `.dtype` / `.shape` (before source open)                                   | `StreamingRuntimeError`    |
| unsupported dtype for digest v1                | `static_inputs.{name}.dtype` or the precise table schema path; never a provider fallback hash               | `StreamingRuntimeError`    |
| recovery digest mismatch                       | exact frozen message above (byte-exact)                                                                     | `CheckpointError`          |
| recovery with missing/extra static name        | `static_inputs.{name}` (before source open)                                                                 | `CheckpointError`          |
| Studio run declaring static inputs over REST   | run-manager validation names `static_inputs.{name}` before worker spawn                                     | Studio validation error    |

Exact wording of the preflight rows is normative in path and failure moment
only; the sentence text is implementer freedom (non-blocking caveat). The
recovery-mismatch row is byte-exact.

## Example

```python
from datetime import timedelta

import pyarrow as pa

from calc_flow import (
    Batch,
    ManagedCheckpointRuntime,
    SinkBinding,
    SourceBinding,
    StreamRuntimeConfig,
    StreamingRunner,
)

# plan declares static input "weights" (project-v3 "static_inputs" array or
# compiled symbolic program); quotes is a continuous source binding.
plan = compile_weights_plan()          # StreamExecutionPlan
declared = plan.static_input_ids       # ("weights",)

weights = Batch.from_arrow(
    pa.table(
        {"factor": [1.0, 2.0, 3.0]},
        schema=declared_schema_for("weights"),
    )
)

runner = StreamingRunner(
    plan,
    sources={"quotes": quote_binding},
    sinks={"signals": [signal_sink]},
    checkpoints=ManagedCheckpointRuntime("./state"),
    config=StreamRuntimeConfig(checkpoint_interval=timedelta(seconds=5)),
    static_inputs={"weights": weights},
)

job = runner.start()                   # latch + digest happen here, pre-source
```

Restarting with a different `weights` value raises `CheckpointError` with the
frozen digest-mismatch message before any source opens.

## Open Questions and Caveats

- **Studio fail-closed point (choice flagged):** SCE-00 does not say where a
  REST-launched static-declared run fails. This note freezes "run-manager
  validation, before worker spawn" over "worker preflight failure" for error
  locality. cf-reviewer should confirm this reading.
- **Mutable array memory:** "engine-owned immutable handle cannot alias
  caller-mutable memory" is this note's explicit reading of D10 so the
  caller-mutation acceptance criterion is testable for NumPy-backed values;
  flagged for reviewer emphasis.
- Non-blocking: private binding argument mode, preflight sentence wording,
  Studio `RunManagerError` wording.
- Parallel work (SCE-08/SCE-09) shares the project-schema/manifest JSON
  surface; the later merge rebases (orchestrator's standing instruction).
