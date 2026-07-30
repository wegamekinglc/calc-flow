# DAL-5 5.1 Studio Capabilities Contract - API Note

Status: revised after blocking critique, implementation not started.

Current public-surface baseline:
`2413ffdb74e2a004b3f1b541e9c2917c3b8f5ef2`
(`origin/main` on 2026-07-25, merge of PR #27).

The three upstream analyses below remain frozen evidence for their earlier
`f71e49d7` audit window. This note supersedes only their implementation
assumptions that changed when PR #27 merged.

## Audiences

- Rust users: the existing `calc-flow` exports, project format, validation
  structs, execution result, and provider registry must not acquire a
  Studio-specific contract.
- Python users: `Runtime` needs one immutable, data-only snapshot of the
  capabilities registered in that runtime session. Existing registration and
  `catalog()` calls must remain source-compatible.
- Studio clients: `/api/v2/catalog` must keep its current UDF-only array
  response, while `/api/v2/capabilities` becomes the typed discovery endpoint.
  Validation, run state, and table/array previews must be generated from
  OpenAPI rather than duplicated in TypeScript.

## Inputs and Scope

This design resolves the open decisions in:

- `.codex/artifacts/analysis/dal-5-weekly-capability-progress-2026-07-25.md`
- `.codex/artifacts/api-notes/dal-5-public-surface-audit.md`
- `.codex/artifacts/critiques/dal-5-weekly-capability-critique-2026-07-25.md`

It uses the vocabulary and boundaries in `docs/introduction.md`: `Batch`,
`Port`, `Operator`, `Pipeline`, runtime, and Studio preview. Runtime capability
data describes what one live parent `Runtime` session can compile. A separate
preview projection describes whether Studio can reconstruct a registration in
a spawned worker. Neither projection is a project, catalog of installed Python
packages, source bundle, authorization grant, or unconditional promise that an
arbitrary project will execute.

In scope:

- a versioned runtime-session capability snapshot;
- provider metadata that is safe to serialize;
- an explicit parent-runtime versus spawned-worker capability boundary;
- a new additive Studio route;
- typed validation, run-state, and table/array preview responses;
- compatibility, migration, errors, and acceptance gates.

Out of scope:

- changing project `format_version: 2` or checkpoint formats;
- exposing callbacks, source, import paths, environment data, or secrets;
- remote/hosted Studio, authentication, or authorization;
- adding an external/array editor or a new Operator;
- changing execution, checkpoint, or lazy-DataFusion behavior. In particular,
  schema v1 describes the lazy built-ins that `2413ffdb` actually supports; it
  does not silently add lazy `table_matmul@1` reconstruction.

## Surface Today

### Rust crate

`crates/calc-flow/src/lib.rs` exports `ValidationReport`, `RunResult`,
`ProviderRegistry`, `UdfCatalogEntry`, `PROJECT_FORMAT_VERSION`, and
`VERSION`. There is no global Rust `Runtime`: applications own registries and
plans directly.

### Python

```python
class Runtime:
    def register_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: object,
    ) -> None: ...

    def catalog(self) -> list[dict[str, object]]: ...
    def validation_report(self, project_json: str) -> dict[str, object]: ...

    # Private projection used by trusted built-ins, not a new public entry
    # point for application code.
    def _register_mapping_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: object,
        *,
        input_ports: Sequence[tuple[str, str]],
        output_ports: Sequence[tuple[str, str]],
    ) -> None: ...
```

`catalog()` contains only registered scalar UDFs. It does not contain external
providers, built-in Operators, Arrow types, or Studio limits.

At `2413ffdb`, `PipelineBuilder.table_matmul(...)` is public and
`register_numpy(runtime)` / `register_jax(runtime)` each make two successful
provider registrations:

- `expression@1`: required array `input`, required array `output`;
- mapped `table_matmul@1`: required table `table`, required array `weights`,
  required array `output`.

The mapped native/Python registration seam is intentionally private, but its
successful registration record and declared Port order are observable inputs
to `Runtime.capabilities()`. It must not be collapsed to the legacy
single-array contract.

### Studio REST

```http
GET /api/v2/catalog
200 OK
[
  {
    "provider": "python",
    "name": "identity",
    "version": "1",
    "kind": "data_fusion_scalar",
    "signature": {
      "input_types": ["int64"],
      "return_type": "int64"
    },
    "volatility": "immutable"
  }
]
```

The top-level array and all field names above are compatibility commitments.
Provider registrations must never appear in this response.

`POST /api/v2/projects/{project_id}/validate` currently returns an untyped
object. `RunResponse.result` is an unconstrained JSON object, although the
worker already emits two output shapes selected by `kind: "table"` or
`kind: "array"`.

## Approaches Considered

1. **Add one versioned `/capabilities` response with two scopes (selected).**
   This preserves `/catalog`, gives clients an atomic view of one parent
   runtime session, and separately projects Studio worker reconstruction and
   preview limits.
2. **Expand `/catalog` into an object (rejected).** Changing its top-level
   array, field names, or UDF-only meaning breaks current REST and Python
   clients.
3. **Split operators, providers, types, and limits into separate routes
   (rejected for v1).** Multiple reads can observe different registration
   revisions and require more caching and error handling without a current
   user need.

## Proposed Surface

### Rust crate

No new public Rust export is approved for 5.1.

`ValidationReport`, `RunResult`, `ProviderRegistry`, project
`format_version`, and checkpoint formats remain unchanged. The binding may use
private helpers to obtain canonical Operator, Arrow-type, UDF, and provider
data, but Studio preview limits must not enter the Rust engine API.

This boundary is deliberate: `/capabilities` describes a Python runtime
session plus a Studio process, not every `ProviderRegistry` that a Rust
application may construct.

### Python runtime

The official Python adapter gains immutable data values and one snapshot
method. Names are snake_case in Python.

```python
from dataclasses import dataclass
from typing import Literal

type BatchKind = Literal["table", "array"]
type OptionValueType = Literal["string", "integer", "number", "boolean"]


@dataclass(frozen=True, slots=True)
class ProviderOption:
    name: str
    value_type: OptionValueType
    required: bool = False


@dataclass(frozen=True, slots=True)
class ProviderOptionsSchema:
    fields: tuple[ProviderOption, ...] = ()
    additional_properties: Literal[False] = False


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    schema_version: Literal[1]
    scope: RuntimeSessionScope
    package_version: str
    project_format_versions: tuple[int, ...]
    batch_kinds: tuple[BatchKind, ...]
    portable_arrow_types: tuple[str, ...]
    operators: tuple[OperatorCapability, ...]
    udfs: tuple[UdfCapability, ...]
    providers: tuple[ProviderCapability, ...]


class Runtime:
    def register_provider(
        self,
        provider: str,
        name: str,
        version: str,
        callback: object,
        *,
        options_schema: ProviderOptionsSchema | None = None,
    ) -> None: ...

    def capabilities(self) -> RuntimeCapabilities: ...

    # Unchanged compatibility endpoint.
    def catalog(self) -> list[dict[str, object]]: ...
```

The private `_register_mapping_provider(...)` seam also gains a keyword-only
`options_schema: ProviderOptionsSchema | None = None` transport so an explicit
registration record can represent both registration modes. It remains private
and is not re-exported as a second public provider constructor.

The remaining frozen values have these fields:

- `RuntimeSessionScope(kind="runtime_session", session_id: str,
  revision: int)`;
- `OperatorCapability(kind: str, input_kinds: tuple[BatchKind, ...],
  output_kinds: tuple[BatchKind, ...], requires_datafusion: bool)`;
- `UdfCapability(provider, name, version, kind, input_types, return_type,
  volatility)`;
- `ProviderPort(name, kind, required)`;
- `ProviderCapability(provider, name, version, input_ports, output_ports,
  options_schema)`.

The public `register_provider(...)` factory accepts exactly one required array
`input` Port and one required array `output` Port. A successful mapped
registration instead records its exact declared Port contract. In both modes
Port declarations come from the trusted registration path and cannot be
overridden by `options_schema`.

`options_schema=None` means the provider is registered and usable but has no
declarative options editor. It does not mean that all options are accepted.
The provider callback remains authoritative for compile-time option
validation.

`ProviderOptionsSchema` is intentionally not arbitrary JSON Schema. Version 1
allows only named scalar fields, their scalar value type, requiredness, and
`additional_properties=False`. It has no default, example, description,
format, `$ref`, code, path, or free-form extension field. A provider with
nested or otherwise non-declarative options uses `None` until a later
capability schema version adds a safe representation.

`register_numpy()` and `register_jax()` supply the following exact capability
entries for their respective `numpy` or `jax` provider:

```python
ProviderCapability(
    provider=backend,
    name="expression",
    version="1",
    input_ports=(ProviderPort("input", "array", required=True),),
    output_ports=(ProviderPort("output", "array", required=True),),
    options_schema=ProviderOptionsSchema(
        fields=(ProviderOption("expression", "string", required=True),)
    ),
)
ProviderCapability(
    provider=backend,
    name="table_matmul",
    version="1",
    input_ports=(
        ProviderPort("table", "table", required=True),
        ProviderPort("weights", "array", required=True),
    ),
    output_ports=(ProviderPort("output", "array", required=True),),
    options_schema=None,
)
```

`table_matmul@1` uses `options_schema=None` in capability schema v1 because its
required `columns` value is a non-empty unique list of strings, which the
scalar-only schema cannot express. The provider remains compile-capable and
usable; clients must not substitute the `expression@1` schema or infer that
`columns` accepts a scalar string.

The resulting four built-in identities are exactly
`numpy:expression@1`, `numpy:table_matmul@1`, `jax:expression@1`, and
`jax:table_matmul@1`.

`portable_arrow_types` is the exact portable type-name vocabulary accepted in
v2 project Port schemas and scalar-UDF declarations. It is not the set of all
Arrow payload types, all DataFusion types, or all PyArrow types.

The native `_native.pyi` projection must expose `Runtime.capabilities()` and
the keyword-only `options_schema` transport on both registration modes if the
snapshot is assembled inside PyO3. The top-level Python adapter still returns
the frozen values above, never a native callback or internal registry object.
`ProviderOption`, `ProviderOptionsSchema`, `ProviderPort`,
`ProviderCapability`, `UdfCapability`, `OperatorCapability`,
`RuntimeSessionScope`, and `RuntimeCapabilities` are all public imports from
`calc_flow`, are present in `calc_flow.__all__`, and have adapter/stub/API-doc
parity.

### Runtime-session scope

The snapshot has scope `runtime_session`, with these semantics:

- `session_id` is an opaque UUID created with the Python `Runtime`; it is
  stable for that instance and changes when the runtime is recreated.
- `revision` starts at zero and increments once per successful registry entry:
  one UDF, one single-array provider, or one mapped provider. Rejected or
  duplicate registration does not change it.
- `register_numpy()` and `register_jax()` each add two entries and therefore
  increment revision twice. The helper is not an atomic revision unit. If its
  first registration succeeds and its second fails, the surviving first entry
  and its one increment remain observable.
- there is no current unregister operation.
- `capabilities()` takes one registration lock, returns a defensive snapshot,
  and sorts all observable collections deterministically.
- a spawned preview worker receives only registrations referenced by its
  project. `runtime.udfs` and `runtime.providers` describe only what the parent
  runtime can compile; worker reconstruction is reported separately under
  `preview.workerRegistrations`.
- querying the snapshot must not compile a Pipeline, create an
  `ExecutionPlan`, or initialize DataFusion.

Clients cache by `(schemaVersion, sessionId, revision)`. Capabilities are not a
machine-global installation inventory and must not be cached across
`sessionId` changes.

The Studio process freezes preview limits and its lazy-built-in preflight for
the lifetime of the parent `Runtime` / `RunManager` pair. Replacing either pair
creates a new parent runtime session. The backend may cache the immutable
serialized response by `(sessionId, revision)`; it must not re-sort registries
or retry callback serialization on every poll.

Legal revision transitions are:

| Operation                                            | Before | After | Observable entries                                |
| ---------------------------------------------------- | ------ | ----- | ------------------------------------------------- |
| Create `Runtime`                                     | n/a    | `0`   | none                                              |
| Register one scalar UDF                              | `0`    | `1`   | one UDF                                           |
| Reject its duplicate                                 | `1`    | `1`   | unchanged                                         |
| Register one public single-array provider            | `1`    | `2`   | one UDF, one provider                             |
| Register one mapped provider                         | `2`    | `3`   | one UDF, two providers                            |
| `register_numpy()` on an empty runtime               | `0`    | `2`   | `numpy:expression@1`, `numpy:table_matmul@1`      |
| Second compound entry fails after the first succeeds | `0`    | `1`   | the successful first entry only                   |
| Recreate the runtime                                 | any    | `0`   | new `sessionId`; no entries until re-registration |

### Capability schema version

`CAPABILITY_SCHEMA_VERSION` is integer `1` and is independent of:

- HTTP route prefix `/api/v2`;
- project `format_version: 2`;
- checkpoint format version;
- package version `2.0.0`.

Schema v1 is closed: every Pydantic capability object uses `extra="forbid"`,
and TypeScript decoders reject unknown fields. Adding any object field,
including an optional field with a default, or adding any union member /
discriminator value requires a new capability schema version. Removing or
renaming a field, changing its type or meaning, or changing scope semantics
also requires a new version. The server must not emit a version it does not
fully satisfy.

Adding a newly registered provider/UDF, an Operator entry, a
`portableArrowTypes` value, or a `workerRegistrations` entry that already fits
the v1 item schema is a collection data change and does not bump the version.
Changing an item's field set or adding a new reconstruction discriminator does
bump it.

A version bump never weakens the provider or worker-registration data-exposure
whitelist. Callback, source, bytecode, import/path, environment, serialized
callback, options values, credentials, and secrets remain forbidden in every
version.

The root discriminator is read before version-specific decoding:

- a v1 client receiving `schemaVersion: 2` fails with the unsupported-version
  error and must not attempt to parse v2 as v1;
- a future v2 client used during migration must retain an explicit v1 decoder
  until the bundled server/frontend rollout no longer needs it;
- a response with `schemaVersion: 1` plus an unknown optional field is
  malformed v1 and fails both server-model and client-decoder tests.

### Studio REST

New JSON field names use camelCase. Existing validation and run-result wire
fields retain their snake_case spellings because renaming them inside
`/api/v2` would be breaking.

```http
GET /api/v2/capabilities
```

The route declares `response_model=CapabilitiesResponse`. The backend takes
`schemaVersion` from `RuntimeCapabilities.schema_version` and does not repeat
it inside `runtime`.

```json
{
  "schemaVersion": 1,
  "runtime": {
    "scope": {
      "kind": "runtimeSession",
      "sessionId": "5afb17ea-d456-4a22-a53d-321d0886add7",
      "revision": 2
    },
    "packageVersion": "2.0.0",
    "projectFormatVersions": [2],
    "batchKinds": ["array", "table"],
    "portableArrowTypes": ["bool", "date32", "date64", "float32", "float64", "int8",
      "int16", "int32", "int64", "large_string", "string", "time32[s]",
      "time64[us]", "timestamp[ms]", "timestamp[us]", "uint8", "uint16",
      "uint32", "uint64"],
    "operators": [
      {
        "kind": "expression",
        "inputKinds": ["table"],
        "outputKinds": ["table"],
        "requiresDataFusion": true
      },
      {
        "kind": "sql",
        "inputKinds": ["table"],
        "outputKinds": ["table"],
        "requiresDataFusion": true
      }
    ],
    "udfs": [],
    "providers": [
      {
        "provider": "numpy",
        "name": "expression",
        "version": "1",
        "inputPorts": [
          {"name": "input", "kind": "array", "required": true}
        ],
        "outputPorts": [
          {"name": "output", "kind": "array", "required": true}
        ],
        "optionsSchema": {
          "fields": [
            {"name": "expression", "valueType": "string", "required": true}
          ],
          "additionalProperties": false
        }
      },
      {
        "provider": "numpy",
        "name": "table_matmul",
        "version": "1",
        "inputPorts": [
          {"name": "table", "kind": "table", "required": true},
          {"name": "weights", "kind": "array", "required": true}
        ],
        "outputPorts": [
          {"name": "output", "kind": "array", "required": true}
        ],
        "optionsSchema": null
      }
    ]
  },
  "preview": {
    "inputBatchKinds": ["table"],
    "requestInputFormats": ["arrow_ipc", "columns", "records"],
    "projectInputFormats": ["arrow_ipc", "csv", "inline_json", "json"],
    "workerRegistrations": [
      {
        "registrationKind": "provider",
        "provider": "numpy",
        "name": "expression",
        "version": "1",
        "reconstruction": "serialized"
      },
      {
        "registrationKind": "provider",
        "provider": "numpy",
        "name": "table_matmul",
        "version": "1",
        "reconstruction": "serialized"
      }
    ],
    "limits": {
      "maxInputBytes": {"default": 10485760, "minimum": 1, "maximum": 10485760},
      "maxRows": {"default": 100000, "minimum": 1, "maximum": 100000},
      "timeoutSeconds": {"default": 30, "minimum": 1, "maximum": 300},
      "memoryLimitMb": {"default": 512, "minimum": 64, "maximum": 4096},
      "outputRows": {"default": 1000, "minimum": 1, "maximum": 10000}
    }
  }
}
```

The concrete response model contains:

- `CapabilitiesResponse(schemaVersion: Literal[1], runtime,
  preview)`;
- `RuntimeCapabilitiesResponse(scope, packageVersion,
  projectFormatVersions, batchKinds, portableArrowTypes, operators, udfs,
  providers)`;
- `UdfCapabilityResponse(provider, name, version, kind, inputTypes,
  returnType, volatility)`;
- `ProviderCapabilityResponse(provider, name, version, inputPorts,
  outputPorts, optionsSchema)`;
- `ProviderOptionsSchemaResponse(fields, additionalProperties=False)`;
- `PreviewCapabilitiesResponse(inputBatchKinds, requestInputFormats,
  projectInputFormats, workerRegistrations, limits)`;
- `SerializedWorkerRegistration(reconstruction: Literal["serialized"],
  registrationKind, provider, name, version)`;
- `LazyBuiltinWorkerRegistration(reconstruction: Literal["lazyBuiltin"],
  registrationKind, provider, name, version)`;
- `UnavailableWorkerRegistration(reconstruction: Literal["unavailable"],
  registrationKind, provider, name, version,
  reasonCode: Literal["serializationFailed"])`;
- `WorkerRegistrationCapability = Annotated[
  SerializedWorkerRegistration | LazyBuiltinWorkerRegistration |
  UnavailableWorkerRegistration, Field(discriminator="reconstruction")]`.

`registrationKind` is closed to `"provider"` and `"dataFusionScalar"` in
schema v1. A new value is a union-domain change and therefore requires a
capability schema bump.

Every object uses `extra="forbid"` in its Pydantic model. All lists use stable
ascending order:

- Operators by `kind`;
- UDFs and providers by `(provider, name, version)`;
- portable Arrow types, Batch kinds, formats, and option fields
  lexicographically;
- provider Ports in declared Port order.
- worker registrations by `(registrationKind, provider, name, version)`.

`preview.inputBatchKinds=["table"]` is intentional. Array results can be
typed and rendered without claiming that the current Studio request decoder
accepts array graph inputs.

### Parent compile scope and spawned-worker scope

The two projections answer different questions:

- `runtime.udfs` and `runtime.providers`: the parent Python `Runtime` has these
  successful registrations and can use them during project validation and
  compilation.
- `preview.workerRegistrations`: the `RunManager` can reconstruct, lazily
  register, or cannot transport this identity into a spawned worker at this
  `(sessionId, revision)`.

The route captures safe metadata and the corresponding private trusted
registration records under the same parent registration lock and revision.
Worker classification operates on that captured private copy; a concurrent
successful registration belongs to the next revision and cannot leak into
only one projection.

For a parent registration, Studio classifies the exact callback-bearing
registration record with the same `cloudpickle` transport boundary used by
`RunManager.submit()`. A successful serialization produces
`reconstruction="serialized"`. Failure produces
`reconstruction="unavailable"` and the fixed redacted
`reasonCode="serializationFailed"`; raw exception text and callback
representations are never emitted. Classification is recomputed only when
revision changes.

`reconstruction="lazyBuiltin"` is reserved for a Studio-owned built-in that
was successfully preflighted for the current process and is absent from the
parent registration snapshot. At `2413ffdb`, the lazy selector recognizes
only `numpy:expression@1` and `jax:expression@1`. `table_matmul@1` appears as
worker-reconstructible only when its mapped registration is present in the
parent and serializes successfully. Adding lazy `table_matmul@1` later is an
execution change and requires its own tests; once implemented it is a new
collection entry, not a schema-field change.

A parent entry may therefore be compile-capable but
`reconstruction="unavailable"`. A lazy built-in may be worker-reconstructible
while absent from the parent compile snapshot. End-to-end Studio preview
requires all three independently checked conditions: parent project
validation/compilation, worker reconstruction for every selected
registration, and support for the project's unconnected graph input kinds.
`workerRegistrations` alone is not an execution guarantee.

### Provider metadata whitelist

`runtime.providers[]` may contain only:

| Field           | Meaning                                           |
| --------------- | ------------------------------------------------- |
| `provider`      | Portable provider identifier                      |
| `name`          | Portable Operator name                            |
| `version`       | Exact portable provider version                   |
| `inputPorts`    | Port name, `table`/`array` kind, and requiredness |
| `outputPorts`   | Port name, `table`/`array` kind, and requiredness |
| `optionsSchema` | The scalar-only version-1 schema above, or `null` |

The following must never be read from a callback or emitted: callable
identity, `repr`, module, class, source, bytecode, import path, filesystem
path, environment, serialized callback, credentials, token, secret, arbitrary
metadata, options values, defaults, examples, or descriptions.

Capabilities are built from explicit successful registration records. The
runtime must not use reflection over callback attributes to discover metadata.

`preview.workerRegistrations[]` has its own closed whitelist:
`registrationKind`, `provider`, `name`, `version`, `reconstruction`, and only
for the unavailable variant `reasonCode`. It must not repeat Port or option
metadata, include raw serialization errors, or expose any executable value.
The callback-bearing trusted registration store remains private and separate
from both public projections.

### Validation discriminated union

The response keeps the existing `valid`, `issues`, and `fingerprint` fields
and additively introduces `kind`:

```python
class ValidValidationReport(StrictModel):
    kind: Literal["valid"] = "valid"
    valid: Literal[True] = True
    issues: tuple[ValidationIssue, ...] = Field(default=(), max_length=0)
    fingerprint: str


class InvalidValidationReport(StrictModel):
    kind: Literal["invalid"] = "invalid"
    valid: Literal[False] = False
    issues: tuple[ValidationIssue, ...] = Field(min_length=1)
    fingerprint: None = None


ValidationReport = Annotated[
    ValidValidationReport | InvalidValidationReport,
    Field(discriminator="kind"),
]
```

`ValidationIssue` remains `{path, code, message}`. The backend preserves the
Rust issue order. It does not rewrite a project validation failure into an
HTTP error: both valid and invalid reports return `200 OK`.

### Run-state and result discriminated unions

`RunResponse` becomes an OpenAPI union discriminated by its existing `status`
field. Its JSON fields and status values do not change.

| `status`    | `result`           | `error`          | Time invariant                                |
| ----------- | ------------------ | ---------------- | --------------------------------------------- |
| `pending`   | `null`             | `null`           | `started_at` and `finished_at` are `null`     |
| `running`   | `null`             | `null`           | `started_at` set; `finished_at` is `null`     |
| `completed` | `RunResultPreview` | `null`           | `started_at` and `finished_at` set            |
| `failed`    | `null`             | non-empty string | `started_at` and `finished_at` set            |
| `timed_out` | `null`             | non-empty string | `started_at` and `finished_at` set            |
| `cancelled` | `null`             | `null`           | `finished_at` set; `started_at` may be `null` |

`RunResultPreview.outputs` contains the output-level discriminated union:

```python
class TableOutputPreview(StrictModel):
    kind: Literal["table"] = "table"
    total_rows: int = Field(ge=0)
    truncated: bool
    schema: tuple[OutputFieldPreview, ...]
    rows: tuple[dict[str, JSONValue], ...]
    metadata: dict[str, JSONValue]


class ArrayOutputPreview(StrictModel):
    kind: Literal["array"] = "array"
    backend: str
    total_rows: int = Field(ge=0)
    truncated: bool
    data: JSONValue
    metadata: dict[str, JSONValue]


OutputPreview = Annotated[
    TableOutputPreview | ArrayOutputPreview,
    Field(discriminator="kind"),
]


class RunResultPreview(StrictModel):
    outputs: dict[str, OutputPreview]
    node_timings: dict[str, NodeTimingPreview]
    datafusion_metrics: tuple[DataFusionMetricPreview, ...]
    metadata: dict[str, JSONValue]
```

`OutputFieldPreview`, `NodeTimingPreview`, and
`DataFusionMetricPreview` use the fields already emitted by
`_result_payload()`. The typed model does not add an object-identity promise:
it describes serialized values only.

Unknown output `kind` values fail before a completed `RunResponse` is stored or
rendered. They are never coerced into table, array, string, or generic JSON.

`RunManager._finish_from_message()` validates the complete
`RunResultPreview` atomically before calling `_finish(COMPLETED)`. A malformed
worker result transitions the run to `failed` with a deterministic redacted
contract message and stores no result. FastAPI response-model validation is a
second boundary, not the first place malformed state is detected.

### TypeScript runtime decoding boundary

Generated TypeScript types do not validate network values. The API client
therefore adds these production decoders:

```typescript
export class ApiContractError extends Error {}

export function decodeCapabilitiesResponse(
  value: unknown,
): CapabilitiesResponse;
export function decodeValidationReport(value: unknown): ValidationReport;
export function decodeRunResponse(value: unknown): RunResponse;
```

`request<T>()` is replaced by a decoded request boundary that accepts a decoder
and passes it raw `response.json()` output before returning. Capabilities,
validation, and every create/get/cancel run response use the corresponding
decoder. The decoders return OpenAPI-generated types; they do not introduce
parallel hand-written interfaces. They validate closed object fields,
discriminators, required variant fields, numeric bounds, finite/depth-bounded
JSON values, and cross-field status invariants.

Raw mocked HTTP tests must reject unsupported capability `schemaVersion`,
unknown run `status`, unknown output `kind`, missing variant fields, and extra
fields with `ApiContractError` before React receives a value. Component tests
remain useful for rendering but are not accepted as evidence for this network
boundary.

## Error Semantics

The existing `/api/v2` error body keeps `detail` as either the current string
or FastAPI request-validation array. This design does not replace it with a
new envelope inside v2.

| Condition                                        | Surface/status                | Required message semantics                                                                 |
| ------------------------------------------------ | ----------------------------- | ------------------------------------------------------------------------------------------ |
| Unsupported capability schema at client          | Python/TypeScript local error | `capabilities schema version <found> is unsupported; expected 1`                           |
| Capability runtime session is unavailable        | REST `503`                    | `runtime capability snapshot is unavailable for this session`                              |
| Runtime emits a malformed capability snapshot    | REST `500`                    | `runtime capability snapshot violates schema version 1 at <path>: <constraint>`            |
| Runtime emits a malformed validation report      | REST `500`                    | `runtime validation report violates the v1 contract at <path>: <constraint>`               |
| Parent registration cannot cross worker boundary | Capability data               | `reconstruction="unavailable"`, `reasonCode="serializationFailed"`; no raw exception       |
| Stored project is absent                         | REST `404`                    | Existing project-not-found message                                                         |
| Project is semantically invalid                  | REST `200`                    | `kind="invalid"` plus one or more `{path, code, message}` issues                           |
| `options_schema` has the wrong object type       | Python `TypeError`            | `options_schema must be a ProviderOptionsSchema or None; found <type>`                     |
| Provider option name has a non-data value        | Python `TypeError`            | `provider options_schema field name must contain strict data; found <type>`                |
| Provider option value type has a non-data value  | Python `TypeError`            | `provider options_schema field <name!r>.value_type must contain strict data; found <type>` |
| Provider option field uses an unsupported type   | Python `ValueError`           | Name the option field and list `string`, `integer`, `number`, and `boolean`                |
| Worker returns an unknown output kind            | Run becomes `failed`          | `run result output <name> has unsupported kind <kind>; expected 'table' or 'array'`        |
| Worker returns another malformed result          | Run becomes `failed`          | `run result violates the v2 preview contract at <path>: <constraint>`                      |
| Browser receives an unknown output kind          | TypeScript local error        | Throw `ApiContractError`; do not render a fallback                                         |

Internal runtime/PyO3 contract violations are `500`, not `422`: the caller did
not supply an invalid capability or validation document. Request body errors
remain `422`. A failed asynchronous preview remains a run status, not an HTTP
error on later `GET /runs/{id}`.

`runtime_validation_report()` parses the runtime value through one strict
validation-report adapter shared by all callers:

| Caller                                     | Semantic invalidity          | Malformed runtime report | Storage effect |
| ------------------------------------------ | ---------------------------- | ------------------------ | -------------- |
| `POST /projects/{id}/validate`             | `200` with `kind="invalid"`  | `500` contract violation | none           |
| `POST /projects`                           | existing caller-facing `422` | `500` contract violation | no create      |
| `PUT /projects/{id}`                       | existing caller-facing `422` | `500` contract violation | no overwrite   |
| `POST /projects/import` (`replace` either) | existing caller-facing `422` | `500` contract violation | no write       |

Malformed runtime output is never blamed on the request and never reaches a
storage method. The implementation must not catch the internal contract error
and remap it through the existing semantic-project `422` path.

Error messages may include the offending field or public identity, but must
not echo callback representations, options values, source, paths, or secrets.
Contract messages use an allowlisted path and constraint; they never embed a
Pydantic `input_value`, raw worker message, or serialization exception.

## Compatibility and Migration

1. **Ship producers and models first.** Add the immutable Python/native parent
   snapshot, both provider registration modes, explicit provider options
   schema, worker-reconstruction classification, strict Pydantic models, and
   contract tests without changing the frontend.
2. **Add the route.** Add `GET /api/v2/capabilities`; regenerate
   `web-ui/openapi.json` and `web-ui/src/api/schema.d.ts`.
3. **Promote existing wire data to typed models.** Add validation `kind`,
   model `RunResponse` by status, and model output previews by `kind`.
   Existing validation fields, run statuses, run-result field names, and
   emitted success values stay unchanged.
4. **Switch the frontend and backend atomically.** Use generated
   `CapabilitiesResponse`, `ValidationReport`, `RunResponse`, and
   `OutputPreview`; install the runtime decoders; delete the corresponding
   hand-written interfaces and the `as unknown as RunResultPreview`
   assertion. A new decoder must not be deployed against a server that still
   omits validation `kind`.
5. **Correct documentation.** Document `/catalog` as the UDF compatibility
   endpoint and `/capabilities` as runtime-session discovery.

`/api/v2/catalog` is not deprecated in 5.1 and has no removal date. A client
that receives `404` for `/capabilities` may fall back to `/catalog` only for
its UDF picker. It must not infer provider, Operator, Arrow-type, or preview
support from that fallback.

Adding `kind` to validation is an additive JSON change for the current
hand-written Studio client, but a new generated/runtime-validating client is
intentionally incompatible with an old server response that lacks `kind`.
Backend and frontend are bundled and must roll out together. External
generated-client users must regenerate and may need source changes.

Tightening `RunResponse` from one object to a six-variant generated union is
also intentionally source-breaking for exhaustive generated-client code, even
though the current successful JSON payloads and status strings do not change.
Old run-response fixtures must parse with the new decoder; the existing
pre-5.1 validation fixture without `kind` must fail with `ApiContractError`.
New responses are checked against the documented current client tolerance:
validation's extra `kind` is ignored by the repository's old structural
client, and run response JSON is unchanged. No promise is made for external
old clients that reject unknown properties.

A new result `kind` in the future is breaking for exhaustive v2 clients and
requires an explicitly versioned result/API migration; incrementing the
capability schema alone is not permission to add it. Capability
`schemaVersion` versions only the `/capabilities` document, not validation or
run-result responses.

## Why This Shape

- One parent runtime-session snapshot gives the UI a coherent compile view
  while `sessionId` and `revision` make its cache lifetime explicit; a
  separate worker projection avoids turning compile metadata into an execution
  promise.
- A separate route protects the deployed UDF-array contract and lets the new
  response use camelCase without aliasing old fields.
- The scalar-only provider schema is less expressive than arbitrary JSON
  Schema, but it cannot carry callbacks, source, import paths, defaults, or
  extension bags. Providers with complex options remain usable and
  non-declarative.
- Rust stays engine-focused. Python owns dynamic trusted registrations, and
  Studio adds its own preview constraints.
- Validation and run-state unions encode invariants that already exist.
  Output `kind` then selects exactly one table or array shape at compile time
  and runtime.

## Critique Disposition

All findings in
`.codex/artifacts/critiques/dal-5-studio-capabilities-contract-critique.md`
have an explicit disposition:

| Finding                                   | Disposition | Design resolution                                                                                                                            |
| ----------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Stale pre-#27 provider surface            | Accepted    | Baseline is `2413ffdb`; both registration modes and exact NumPy/JAX `expression@1` / `table_matmul@1` entries are normative.                 |
| Parent runtime versus worker availability | Accepted    | `runtime` is compile scope; `preview.workerRegistrations` is a separate reconstruction projection with serialized/lazy/unavailable variants. |
| Closed schema versus optional-field rule  | Accepted    | Schema v1 is closed; every new object field or union member, optional or not, requires a capability schema bump.                             |
| Illegal empty revision-2 example          | Accepted    | The revision-2 example now contains the two NumPy registrations; a full transition table defines legal and partial compound states.          |
| Generated-client source compatibility     | Accepted    | Source break and backend/frontend coupling are explicit; old/new fixture behavior is required.                                               |
| Missing browser runtime parser            | Accepted    | Production decoders run on raw JSON before `request` returns; component-only casting is insufficient.                                        |
| Validation malformed-report callers       | Accepted    | Validate/create/put/import share strict parsing; malformed runtime output is `500` and cannot mutate storage.                                |
| Result validation after state transition  | Accepted    | Full result validation occurs in `RunManager` before `COMPLETED`; contract failure stores no result and becomes `failed`.                    |
| Coarse result edge cases                  | Accepted    | Empty, zero-length, single-row, null-only, empty-output, Unicode, and exact-boundary cases are required end-to-end fixtures.                 |
| Missing Python export/parity gate         | Accepted    | Every new frozen value is listed as a top-level import with `__all__`, stub, and docs parity tests.                                          |
| Ambiguous `arrowTypes`                    | Accepted    | Renamed to `portableArrowTypes` / `portable_arrow_types` and defined as the v2 declaration vocabulary.                                       |
| Ambiguous generated-file drift check      | Accepted    | Implementation checks distinguish intentional generated changes from post-generation drift and separately freeze project schema.             |
| Repeated snapshot allocation              | Accepted    | Backend caches the immutable serialized response by `(sessionId, revision)`; no v1 benchmark is required.                                    |

The critique also correctly marked checkpoint and at-least-once runner cases
as not implicated. They remain out of scope because this contract changes no
runner or checkpoint behavior.

## Error Cases

| Input violation                        | Message text                                                                                                      |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Capability schema version `2`          | `capabilities schema version 2 is unsupported; expected 1`                                                        |
| Extra field in a schema-v1 object      | `runtime capability snapshot violates schema version 1 at <path>: extra fields are forbidden`                     |
| Option field named twice               | `provider options_schema contains duplicate field 'expression'`                                                   |
| Wrong schema object                    | `options_schema must be a ProviderOptionsSchema or None; found dict`                                              |
| Option type `object` in schema v1      | `provider options_schema field 'expression.value_type' must be string, integer, number, or boolean; found object` |
| Callable as an option field name       | `provider options_schema field name must contain strict data; found function`                                     |
| Runtime reports valid with no hash     | `runtime validation report violates the v1 contract at fingerprint: valid reports require a fingerprint`          |
| Runtime reports invalid with no issues | `runtime validation report violates the v1 contract at issues: invalid reports require at least one issue`        |
| Worker output kind `tensor`            | `run result output 'output' has unsupported kind 'tensor'; expected 'table' or 'array'`                           |
| Callback cannot serialize for preview  | Capability entry uses `reconstruction="unavailable"` and `reasonCode="serializationFailed"`                       |

## Example

```python
from calc_flow import (
    ProviderOption,
    ProviderOptionsSchema,
    Runtime,
)

runtime = Runtime()
runtime.register_provider(
    "acme",
    "normalize",
    "1",
    normalize_callback,
    options_schema=ProviderOptionsSchema(
        fields=(
            ProviderOption("method", "string", required=True),
            ProviderOption("center", "boolean"),
        )
    ),
)

snapshot = runtime.capabilities()
assert snapshot.schema_version == 1
assert snapshot.scope.kind == "runtime_session"
assert snapshot.providers[0].provider == "acme"
```

The Studio projection is:

```http
GET /api/v2/capabilities
```

The response advertises `acme:normalize@1` for the current runtime session. It
does not serialize `normalize_callback` or inspect it for metadata.

## Acceptance Matrix

| Surface                          | Acceptance                                                                                                                                                                               | Required gate                                                                                                                                                       |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rust core                        | No change to exports, validation/result, project v2, checkpoints, or lazy DataFusion.                                                                                                    | Workspace tests, rustdoc, and public-export review.                                                                                                                 |
| PyO3 binding                     | Single-array and mapped registration records project exact immutable Port contracts and scalar option metadata.                                                                          | Inline tests, `_native.pyi` parity, rejected-registration tests, and exact mapping Port-order fixture.                                                              |
| Python adapter                   | Only successful entries increment revision; compound helpers increment twice and expose no partial-state fiction.                                                                        | Full revision transition table, defensive-copy tests, and duplicate/partial compound tests.                                                                         |
| Python public exports            | Every proposed frozen value and `Runtime.capabilities` is importable and documented consistently.                                                                                        | `calc_flow` import, `__all__`, adapter, `_native.pyi`, and Python API-doc parity test.                                                                              |
| NumPy/JAX provider entries       | Each helper yields exact `expression@1` plus mapped `table_matmul@1`; table-matmul Ports retain `table, weights` order and `optionsSchema=null`.                                         | Parameterized NumPy/JAX fixtures asserting identities, revision `2`, Ports, option schema, and stable sort.                                                         |
| `/api/v2/catalog`                | Empty and populated fixtures are byte-for-byte compatible in shape and provider registrations never appear.                                                                              | Exact contract tests for top-level array, wire fields, order, and UDF-only semantics.                                                                               |
| `/api/v2/capabilities`           | Closed concrete OpenAPI schema v1 has separate parent compile and worker reconstruction projections.                                                                                     | Empty/UDF/single/mapped/compound/mixed states, malformed snapshot, and OpenAPI discriminator tests.                                                                 |
| Capability version compatibility | Optional extra fields fail under v1; v1 clients reject v2 before parsing; future v2 clients retain explicit v1 decoding during migration.                                                | Old-client/new-server and new-client/old-server fixtures, including `schemaVersion=1` plus an extra field.                                                          |
| Provider and worker safety       | Public snapshots contain only their respective whitelists and never callback/source/path/secret/raw exception data.                                                                      | Recursive key whitelist and hostile callback objects whose attributes raise if inspected.                                                                           |
| Worker reconstruction            | Ordinary callbacks classify serialized; runtime-capturing callbacks classify unavailable; preflighted lazy expression built-ins classify lazy; mapped registrations restore exact Ports. | Spawned-process tests for serializable/unserializable callbacks, lazy NumPy/JAX expression, mapped provider, UDF, and compile-capable/preview-unavailable identity. |
| Validation REST                  | `kind` enforces valid/hash/no issues versus invalid/issues/null hash, and malformed runtime output is `500` for every caller without storage mutation.                                   | Parameterized validate/create/put/import tests plus generated TypeScript narrowing.                                                                                 |
| Run REST                         | `status` enforces the result/error/time matrix; only a fully validated `completed` state stores a result.                                                                                | State matrix, manager-before-finish contract tests, and OpenAPI `oneOf`/discriminator assertions.                                                                   |
| Table result                     | Empty, single-row, null-only, Unicode-name, exact-limit, and one-above-limit tables preserve required schema/rows/count/truncation/metadata.                                             | Parameterized worker-to-model-to-HTTP-to-decoder-to-ResultsPanel fixtures.                                                                                          |
| Array result                     | Zero-length, single-row, Unicode-name, exact-limit, and one-above-limit arrays preserve backend/data/count/truncation/metadata without table fields.                                     | Parameterized worker-to-model-to-HTTP-to-decoder-to-ResultsPanel fixtures.                                                                                          |
| Empty/external-only result       | Empty outputs are valid; external-only results have empty DataFusion metrics and discovery creates no DataFusion session.                                                                | Empty-result and lazy-DataFusion regression tests.                                                                                                                  |
| Malformed worker result          | Unknown kind, missing/extra fields, non-finite values, and over-depth JSON fail before `COMPLETED`, store no result, and expose a redacted error.                                        | Direct manager-message tests plus stored-state and HTTP assertions.                                                                                                 |
| TypeScript runtime boundary      | Raw unsupported version/status/kind and missing/extra-field payloads reject with `ApiContractError` before React.                                                                        | Mocked-fetch API-client tests; component casts do not satisfy this gate.                                                                                            |
| Generated frontend types         | No hand-written discovery/result shapes or double assertion remain; decoders return generated types.                                                                                     | API sync, build, frontend tests, and generator drift procedure below.                                                                                               |
| Documentation                    | `/catalog` is UDF-only; `/capabilities` documents parent/worker scope, closed versioning, fallback, and source compatibility cost.                                                       | Markdown link/style checks and API-reference review.                                                                                                                |

## Design Verification Performed

- Confirmed the current public-surface evidence checkout has `HEAD ==
  origin/main ==
  2413ffdb74e2a004b3f1b541e9c2917c3b8f5ef2`.
- Read the three named upstream artifacts in full and checked their conclusions
  against the current critique, `docs/introduction.md`,
  `crates/calc-flow/src/lib.rs`, the relevant config/provider/runtime code,
  `python/calc_flow/`, the Studio backend, `web-ui/openapi.json`, generated
  TypeScript, and ResultsPanel.
- Checked the `f71e49d7..2413ffdb` diff and current tests. Confirmed public
  `PipelineBuilder.table_matmul`, the private mapped registration projection,
  exact `table, weights -> output` Port order, two registrations per
  NumPy/JAX helper, mapping restoration in workers, and the regression where a
  callback capturing `Runtime` cannot be serialized.
- Used `jq -e` to confirm the current `/catalog` response is
  `array<object>`, `/capabilities` is absent, validation is an unconstrained
  object, and `RunResponse.result` is present but unconstrained.
- Used `rg` to confirm the current wide backend models and
  `as unknown as RunResultPreview` cast, and to confirm there is no existing
  capabilities route or Python method.
- Restored the three upstream originals byte-for-byte and checked their
  SHA-256 digests against the source copies:
  `52fbe05aafaa658c42c4ffe8ece2924690045b1e1d0a67091910fdbf3f494145`,
  `859ea4d05690f490e899a0d4d9f4d472fd6a3d66561a6d99f12e5fc059549e35`,
  and
  `ee204ab9f6a8d4d0370c85a62265a57f3626f523d2d2378aad23efbcb0fe3213`.
- Ran `python -m unittest scripts.test_release_config` at `2413ffdb`: 11 tests
  passed.
- Parsed the canonical response example as JSON and ran the final required
  decision, 13-item disposition, code-fence, placeholder, absolute-path,
  trailing-whitespace, and aligned-table checks: all passed.
- Checked worktree scope: changes are limited to this revised note and the
  three restored upstream artifacts; no implementation or generated file
  changed.

No build, generated-file rewrite, engine test, or benchmark was run: this note
changes no implementation or generated contract. The full commands below are
implementation acceptance gates, not evidence that 5.1 has been implemented.

## Verification Required at Implementation

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps

uv run ruff check .
uv run ruff format --check .
JAX_PLATFORMS=cpu uv run pytest python/tests -q
uv run --project web-ui/backend --extra dev pytest

cd web-ui
npm run sync:api
npm run build
npm test
```

For generated-contract synchronization:

```bash
# From a clean CI checkout after the intended generated files are committed:
npm --prefix web-ui run sync:api
git diff --exit-code -- web-ui/openapi.json web-ui/src/api/schema.d.ts

# During local implementation, stage the intended OpenAPI/generated changes,
# rerun the generator, and require no unstaged generator drift.
git diff --exit-code -- web-ui/openapi.json web-ui/src/api/schema.d.ts

# 5.1 must not change the project format/schema.
git diff --exit-code 2413ffdb74e2a004b3f1b541e9c2917c3b8f5ef2 -- \
  schemas/project-v2.schema.json

git diff --check
```

For 5.1, the project schema must remain unchanged; OpenAPI and generated
TypeScript must change together.

## Remaining Risks

- The version-1 provider options schema deliberately cannot describe nested
  options. Such providers remain discoverable but not declaratively editable.
- `sessionId` identifies an in-process Python `Runtime`, not a daemon, host,
  package installation, or spawned worker. UI copy must not imply a broader
  scope.
- `/api/v2` still uses legacy string/array `detail` error bodies. A
  machine-readable API-wide error code envelope needs a separate compatibility
  design.
- Dynamic registration during active preview submission remains possible.
  The snapshot revision tells clients that discovery changed; an already
  submitted run keeps the registration snapshot captured for that run.
- Worker reconstruction classification proves only that the registration can
  cross the current transport boundary. It cannot prove that arbitrary
  provider options or input values will execute successfully.
- At `2413ffdb`, lazy worker registration covers NumPy/JAX
  `expression@1`, not `table_matmul@1`. The latter is still discoverable and
  worker-reconstructible when the parent helper registration is transported.
- Generated-client source compatibility is intentionally tightened inside the
  existing `/api/v2` prefix. The bundled Studio can deploy atomically;
  external generated-client consumers require explicit migration notice.

## Open Questions

None. The three blocking findings and every significant/minor recommendation
have a disposition above. The public shape, closed schema version, parent and
worker scopes, whitelists, unions, error semantics, migration rules, and
acceptance gates are fixed by this revision.
