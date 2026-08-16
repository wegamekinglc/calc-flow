# Continuous Streaming 3.0 — M6 Connectors and Project v3 API Note

| Field         | Value                                                            |
| ------------- | ---------------------------------------------------------------- |
| Status        | **Proposed — companion to the M6 specification**                 |
| Baseline      | `main@858199f6df0161801bb6028f37f3ebbeb1684e3e` (post Public A6) |
| Artifact slug | `m6-connectors-project-v3`                                       |
| Spec          | `../specs/m6-connectors-project-v3.md`                           |

## 1. Purpose and compatibility boundary

This note freezes the exact public shapes for M6. It adds new connector,
secret, format, project-v3, and Studio surfaces; it MUST NOT alter the A6
runner contract, manifest wire format, or checkpoint semantics. Signatures
below are Rust 2024 with `unsafe_code = "forbid"`; all fallible operations
return `calc_flow::Result<T>`.

## 2. Module layout and ownership

```text
crates/calc-flow/src/connector/
  mod.rs         re-exports and registry error types
  capability.rs  identity, capability vocabulary, descriptors
  format.rs      FormatDecoder / FormatEncoder contracts and bounds
  registry.rs    ConnectorRegistry, plan-scoped snapshot
  secret.rs      SecretReference, SecretResolver, redaction markers

crates/calc-flow-connectors/src/
  lib.rs         feature-gated re-exports (file default-on)
  csv.rs         CSV codec (always compiled)
  json_lines.rs  newline JSON codec (always compiled)
  parquet.rs     Parquet codec (feature: file)
  file.rs        file/directory snapshot source (feature: file)
  file_sink.rs   transactional Parquet sink (feature: file)
  kafka.rs       feature: kafka
  postgresql.rs  feature: postgresql
  database_types.rs  shared Arrow/SQL type matrix (features: postgresql, clickhouse)
  clickhouse.rs  feature: clickhouse
  http.rs        feature: http
  websocket.rs   feature: websocket
```

The core crate owns contracts and validation; the connectors crate owns
implementations and heavy dependencies. Nothing in `calc-flow` may import
`calc-flow-connectors`.

## 3. Identity, capabilities, and descriptor

```rust
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ConnectorIdentity {
    pub provider: Arc<str>,
    pub name: Arc<str>,
    pub version: Arc<str>,
}

pub enum DeliveryCapability { BestEffort, AtLeastOnce, ExactlyOnce }

pub enum ReplayCapability { ReplayableExact, Unreplayable }

pub enum WatermarkSupport { Native, GeneratedOnly }

pub enum TransactionSupport { None, PreCommitCommit, LedgerIdempotent }

#[derive(Clone, Debug)]
pub struct ConnectorCapabilities {
    pub delivery: DeliveryCapability,
    pub replay: ReplayCapability,
    pub watermark: WatermarkSupport,
    pub transaction: TransactionSupport,
    pub snapshot: bool,
    pub polling: bool,
    pub cdc: bool,
    pub lookup: bool,
}

#[derive(Clone, Debug)]
pub struct ConnectorDescriptor {
    pub identity: ConnectorIdentity,
    pub kind: ConnectorKind,           // Source | Sink | Both
    pub capabilities: ConnectorCapabilities,
    pub formats: Vec<FormatIdentity>,  // formats this connector accepts
    pub config_schema: JsonMap,        // data-only, bounded, no secret fields
}
```

`ConnectorCapabilities` converts into the existing A6-native
`SourceCapabilities` and `SinkDelivery` values; the registry MUST NOT define
a second vocabulary consumed by preflight.

## 4. Registry and plan-scoped snapshot

```rust
pub struct ConnectorRegistry { /* crate-private map keyed by identity */ }

impl ConnectorRegistry {
    pub fn new() -> Self;
    pub fn register_connector(
        &mut self,
        descriptor: ConnectorDescriptor,
        factories: ConnectorFactories,
    ) -> Result<()>;
    pub fn register_format(&mut self, descriptor: FormatDescriptor) -> Result<()>;
    pub fn snapshot(&self) -> ConnectorRegistrySnapshot;
}

// Cloned freely; no `Debug` because the snapshot holds factory trait
// objects, and plans never observe their contents beyond resolution.
#[derive(Clone)]
pub struct ConnectorRegistrySnapshot { /* immutable copy, Send + Sync */ }
```

- `register_connector` binds the trusted factories at registration and
  fails atomically when the declared kind does not carry exactly its
  matching factories (`Source` registers one source factory, `Sink` one
  sink factory, `Both` both), when a factory descriptor names a different
  identity, or when the `(provider, name)` slot is already occupied —
  whether by the same `(provider, name, version)` identity or by a
  different version label.
- Plans compiled with a snapshot never observe later registrations.
- `compile_stream()` accepts the snapshot exactly as it accepts
  `UdfRegistrySnapshot` today.

## 5. Secret resolver

```rust
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SecretReference {
    pub resolver: SecretResolverKind,  // Environment | File | Registered
    pub key: Arc<str>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SecretResolverKind { Environment, File, Registered }

pub trait SecretResolver: Send + Sync {
    fn resolve(&self, reference: &SecretReference) -> Result<SecretHandle>;
}

#[derive(Debug)]  // Debug renders "<redacted secret>"
pub struct SecretHandle { /* crate-private bytes; no Clone, no Serialize */ }
```

- Connector configuration types expose `secrets: BTreeMap<String, SecretReference>`
  slots; no config field is a raw credential string.
- Resolution happens once per connector open inside the owning task.
- `SecretHandle` derefs to bytes only within connector code, is not
  `Serialize`, and its `Debug`/`Display` are the fixed marker
  `"<redacted secret>"`.
- Fingerprints hash the reference, never the value.

## 6. Format layer

```rust
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct FormatIdentity { pub name: Arc<str>, pub version: Arc<str> }

pub struct DecodeBounds { pub max_rows: u64, pub max_bytes: u64 }

pub trait FormatDecoder: Send + Sync {
    fn identity(&self) -> FormatIdentity;
    fn decode(&self, bytes: &[u8], bounds: &DecodeBounds, schema: &[ArrowFieldSpec])
        -> Result<Batch>;
}

pub trait FormatEncoder: Send + Sync {
    fn identity(&self) -> FormatIdentity;
    fn encode(&self, batch: &Batch) -> Result<Vec<u8>>;
}
```

Decoders MUST fail with the format identity and the violated bound when
expansion exceeds `DecodeBounds`; they never return an oversized batch to an
edge.

## 7. Factories and A6 binding integration

```rust
#[async_trait]
pub trait ConnectorSourceFactory: Send + Sync {
    fn descriptor(&self) -> &ConnectorDescriptor;
    async fn open(&self, options: &JsonMap, secrets: &dyn SecretResolver)
        -> Result<Box<dyn StreamSource>>;
}

#[async_trait]
pub trait ConnectorSinkFactory: Send + Sync {
    fn descriptor(&self) -> &ConnectorDescriptor;
    async fn open(&self, options: &JsonMap, secrets: &dyn SecretResolver)
        -> Result<Box<dyn StreamSink>>;
    async fn open_transactional(
        &self, options: &JsonMap, secrets: &dyn SecretResolver,
    ) -> Result<Option<Box<dyn TransactionalStreamSink>>>;
}
```

Factories construct the A6-public `StreamSource` / `StreamSink` /
`TransactionalStreamSink`; they introduce no new lifecycle or ownership.
`open_transactional` returns `None` for sinks whose
`TransactionSupport::None` capability already ruled out exactly-once during
compilation.

## 8. Delivery guarantee compilation

`StreamRequirements` gains the registry snapshot. When the requested
delivery is `ExactlyOnce`, compilation walks every reachable
source/operator/edge/sink, derives the effective minimum, and on failure
returns an error listing the precise participant path and the unmet axis.
The check runs before any factory `open()` call. At-least-once and
best-effort requests never upgrade silently.

## 9. Project v3 document

```json
{
  "format_version": 3,
  "name": "pg-cdc-to-parquet",
  "runtime": { "mode": "stream", "options": { "checkpoint_interval_ms": 30000 } },
  "graph": { "nodes": [], "edges": [] },
  "sources": [{
    "binding": "pg-src",
    "connector": { "provider": "calc-flow-connectors", "name": "postgresql", "version": "2.0.0" },
    "format": null,
    "options": { "mode": "logical_cdc", "publication": "calcflow", "slot": "calcflow_slot" },
    "watermark": { "policy": "bounded_out_of_orderness", "column": "commit_time", "delay_ms": 5000 },
    "secrets": { "password": { "resolver": "environment", "key": "PG_PASSWORD" } }
  }],
  "sinks": [{
    "binding": "parquet-out",
    "connector": { "provider": "calc-flow-connectors", "name": "file", "version": "2.0.0" },
    "format": { "name": "parquet", "version": "1" },
    "options": { "path": "out/", "mode": "transactional" },
    "delivery": "exactly_once"
  }],
  "state": { "root": "state/", "retention": 3 }
}
```

Rules frozen beyond the master plan:

- `runtime.mode: "batch"` keeps inline data fixtures and no connector
  bindings; `"stream"` requires connector bindings and rejects fixtures.
- `options` and `config_schema` are bounded JSON with strict unknown-field
  rejection at every layer.
- Secret values are structurally impossible: any string that is not a
  `SecretReference` object in a `secrets` map is rejected before native
  factory invocation.
- The canonical serialization of this model determines the stream
  fingerprint exactly as v2 does today; connector identity, format identity,
  options, watermark policy, and delivery requests are all fingerprinted.
- `PROJECT_FORMAT_VERSION` becomes `3`; documents declaring `2` fail with
  `UnsupportedVersion` carrying the expected version.

## 10. Python surface

```python
calc_flow.compile_stream_project(project: Mapping[str, object], delivery: str, ...) -> StreamExecutionPlan
calc_flow.capabilities() -> CapabilitiesReport
CapabilitiesReport.connectors: tuple[ConnectorCapabilityInfo, ...]
```

- `CapabilitiesReport.connectors` lists only transports compiled into the
  native module (feature-detection constants exported from PyO3).
- `compile_stream_project` validates defensively (copying caller mappings),
  rejects secret values pre-native, and preserves field paths in errors.
- No new runner classes, ownership methods, or blocking behaviors.

## 11. Studio v3 routes and limits

Routes replace `/api/v2` exactly as the master plan lists them. New
configuration model:

```python
class JobLimits(TypedDict):
    max_concurrent_jobs: int              # >= 1
    max_job_resident_memory_bytes: int    # per-job ceiling
    max_global_resident_memory_bytes: int # studio-wide ceiling
    max_checkpoint_disk_bytes: int        # per-job state/checkpoint usage
```

- Exceeding any limit transitions the affected job to a typed failed state
  (`job_limit_exceeded`) with the violated limit named; it never kills the
  process silently.
- The legacy run timeout is deleted only once these limits are enforced and
  tested; jobs otherwise end only by user stop, cancel, or failure.
- SSE event names: `state`, `progress` (epoch, watermark, throughput,
  backpressure, late rows), `checkpoint`, `terminal`. Events are
  payload-free summaries; no batches, no secrets.

## 12. Error projection and redaction

All connector failures surface as `CalcFlowError::Streaming(
StreamingError)` or a new `CalcFlowError::Connector(ConnectorError)` whose
fields carry connector identity, operation, and a payload-free detail
string. The redaction census forbids: secret values, full URLs with
credentials, raw frames, and query bodies. Error display MUST include the
connector identity and stable operation name.

## 13. Test placement and named evidence

- Core registry/capability/secret/format tests:
  `crates/calc-flow/tests/connector_registry.rs`.
- Connector crate unit tests beside source; fake-transport logic tests
  remain container-free.
- Container tests: `#[ignore]` + `CALC_FLOW_CONNECTOR_CONTAINERS=1`, one
  module per transport under `crates/calc-flow-connectors/tests/`.
- Python: `python/tests/test_capabilities.py`,
  `python/tests/test_project_v3.py`.
- Studio: `web-ui/backend/tests/test_jobs_v3.py`, Playwright spec
  `web-ui/tests/e2e/stream-job.spec.ts`.
- Every RED list item in the M6 execution plan maps to a named test
  recorded in the PR body.

## 14. CI and coverage interfaces

- Per-connector coverage decision (D3 review clause, M6.3): the Kafka
  transport's broker-bound module `crates/calc-flow-connectors/src/kafka.rs`
  is omitted from the workspace line-coverage gate because its runtime
  paths require a live broker; those paths are proven by the gated
  connector-containers leg, while the module's offline logic (option
  parsing, cursor replay assignment, identity derivation, offline
  construction) stays inside the measured set. Remove the omission when
  a broker-less harness covers the runtime paths.


- ci-linux gains a `connectors` job: `--all-features` build plus container
  services (`ghcr.io/redpanda-data/redpanda`, `postgres:16`,
  `clickhouse/clickhouse-server`) running the gated `--ignored` tests.
- Local parity via `crates/calc-flow-connectors/compose.yaml` exporting the
  same connection settings and the same env gate.
- Coverage legs never enable the container gate; the workspace 90% line
  floor is unchanged for non-container code.

## 15. Implementation commit sequence

1. M6.1 core `connector/` module + `tests/connector_registry.rs` (RED
   first).
2. M6.2 connectors crate skeleton with csv/json-lines codecs and the file
   connector + transactional Parquet sink; audit/deny/CI updates.
3. M6.3-M6.6 transports in frozen order with container legs.
4. M6.7 project v3 on `feature/m6-integration`; M6.8 Python; M6.9 Studio.
5. M6.10 cutover: v2 removal, contract regeneration, soak, atomic merge.
