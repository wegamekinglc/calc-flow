# Continuous Streaming 3.0 — M6 Connectors and Project v3 Specification

| Field             | Value                                                                     |
| ----------------- | ------------------------------------------------------------------------- |
| Status            | **Proposed — blocks M6 implementation until review is complete**          |
| Priority          | **P0 — connector safety, project v3, and Studio v3 depend on this delta** |
| Baseline          | `main@858199f6df0161801bb6028f37f3ebbeb1684e3e` (post Public A6)          |
| Milestone         | M6 — connectors, project v3, Python capability wiring, Studio v3          |
| Artifact slug     | `m6-connectors-project-v3`                                                |
| Intended audience | connector, compiler, runtime, Python, Studio, CI, test, and review owners |

## 1. Authority and precedence

This document is the controlling M6 delta wherever the following inputs are
silent, stale, or inconsistent with the current `main` implementation:

1. `docs/superpowers/plans/2026-08-16-m6-execution-plan.md` (PR #132);
2. `docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md` section 12;
3. the Public A6 delta package
   (`.codex/artifacts/{specs,api-notes,critiques}/a6-public-continuous-runtime.md`)
   and its implementation evidence;
4. the M5, M4, M3 delta packages and the continuous-streaming runtime
   specification;
5. baseline commit `858199f6df0161801bb6028f37f3ebbeb1684e3e`.

Precedence for an M6 conflict is:

```text
this M6 delta
> the M6 execution plan
> compatible Public A6 delta requirements
> compatible M5/M4/M3 delta requirements
> main-plan section 12 historical text
```

The key words **MUST**, **MUST NOT**, **REQUIRED**, **SHOULD**, **SHOULD
NOT**, and **MAY** are normative. This document freezes contracts only; it is
not implementation evidence and grants no merge approval.

## 2. Current-main reconciliation

Post-A6 facts that M6 builds on and MUST NOT re-implement or regress:

- The crate-root public surface already exports the source-driven
  `StreamingRunner`, the owning `StreamingJob`, `StreamSource`,
  `StreamSink`, `TransactionalStreamSink`, source/sink bindings, watermark
  policy, and the managed checkpoint runtime with v3 manifest recovery.
- Whole-job preflight already validates source capabilities, replay
  positioning, delivery capability, byte/row bounds, and reachable-path
  guarantees before any connector lifecycle side effect.
- PyO3 and Python expose the same lifecycle; Python is not a second runtime.
- Project format is still version 2; stream mode is expressed only through
  `compile_stream_project()` and has no connector, secret, or
  `runtime.mode` fields.
- Studio still serves `/api/v2` with private checkpoint persistence wiring.
- All package versions are `2.0.0`; the `3.0.0` bump belongs to M7.4.
- No `calc-flow-connectors` crate, no `connector/` module, and no
  `SecretResolver` exist yet.

## 3. Scope

M6 delivers, in task order tracked by issues #133 through #143:

- a core connector registry with plan-scoped immutable snapshots, an
  independent capability vocabulary, a secret-reference resolver, and a
  transport-orthogonal format layer (M6.1);
- a `calc-flow-connectors` workspace crate with per-transport cargo features
  and the file/CSV/JSON/Parquet connector as its first delivery, including a
  transactional Parquet sink proven by the M5 fault matrix (M6.2);
- Kafka, PostgreSQL (snapshot/incremental/CDC), ClickHouse
  (snapshot/incremental), HTTP polling, and WebSocket connectors (M6.3-M6.6);
- a strict data-only project format v3 with `runtime.mode`, connector and
  secret-reference bindings, and capability-checked delivery requests
  (M6.7);
- Python compilation of project v3 onto the existing A6 native job handle
  plus honest connector capability enumeration (M6.8);
- Studio `/api/v3` continuous job routes, SSE, a continuous-job UI, and
  resource limits that replace the bounded-run worker timeout (M6.9);
- the atomic integration cutover that removes project/REST v2 and regenerates
  every contract file in one commit (M6.10).

## 4. Non-goals and milestone firewall

- No distributed workers, shuffle, slots, rescaling, or controller HA.
- No session windows, early triggers, allowed lateness, or side outputs.
- No engine-level changelog/retract/upsert streams; PostgreSQL CDC remains an
  append-only change-event envelope.
- No multi-input incremental SQL join or event-time temporal join.
- No automatic schema evolution; DDL stops the source and reports schemas.
- No connector code inside the core `calc-flow` crate and no dependency from
  `calc-flow` onto `calc-flow-connectors`; the dependency edge points the
  other way only.
- No second capability system: the registry MUST produce the A6-native
  `SourceCapabilities` / `SinkDelivery` descriptors that preflight already
  consumes.
- No new runner semantics, checkpoint model, or manifest wire format.
- No v2 compatibility, alias, deprecation period, or automatic migration for
  project, REST, or Python surfaces.
- M6 does not publish `3.0.0` and does not decide crates.io publication of
  the connectors crate (both are M7.4).

## 5. Controlling decisions

### D1 Dependency edge (frozen)

`calc-flow-python` gains a feature-gated optional dependency on
`calc-flow-connectors`. The connectors crate exposes per-transport features
`file`, `kafka`, `postgresql`, `clickhouse`, `http`, and `websocket`, all
default-off except `file`. `calc-flow-python` mirrors them as
`connector-<transport>` features; the default Python build enables
`connector-file` only. Consequences:

- The default published wheel carries file/CSV/JSON/Parquet connectivity
  without heavy native dependencies.
- The master-plan acceptance gate "define and run PostgreSQL CDC ->
  ClickHouse/Parquet from one project v3" is satisfied in feature-enabled
  builds exercised by CI container legs, not by the default wheel.
- `python/calc_flow/capabilities.py` MUST enumerate only connectors compiled
  into the running native module.

### D2 Feature gates (frozen)

One cargo feature per transport, named exactly as in D1. Lightweight
pure-Rust format codecs (CSV, newline JSON) compile unconditionally. Every
feature combination MUST compile with `unsafe_code = "forbid"` preserved.

### D3 CI, `--all-features`, and coverage (frozen)

- `rdkafka` builds vendored from source (cmake) so workspace
  `--all-features` targets compile without system librdkafka; PostgreSQL,
  ClickHouse, HTTP, and WebSocket clients are pure Rust with rustls.
- Container-dependent tests are `#[ignore]` tests gated by
  `CALC_FLOW_CONNECTOR_CONTAINERS=1`; they run in dedicated ci-linux service
  legs and never in ordinary unit-test or coverage runs.
- The workspace `--fail-under-lines 90` gate continues to cover all
  non-container code. Transport clients sit behind thin shim traits so
  ordinary unit tests with fakes cover connector logic. If a connector
  cannot reach 90% on non-container paths, a task-level review MAY grant
  that connector a separate documented floor of at least 85% recorded in
  this spec's revision log; silent relaxation is forbidden.

### D4 Versioning and packaging (frozen)

`calc-flow-connectors` uses `version.workspace = true`. Packaging hygiene
starts at M6.2: license, readme, `exclude` of container assets, audit and
deny coverage for every new dependency. Whether the crate is published to
  crates.io, and with which wheel variants beyond default features, is
  decided by M7.4.

### D5 Delivery order (frozen)

file/Parquet first; Kafka and PostgreSQL next in parallel; ClickHouse after
PostgreSQL shares `database_types` and container infrastructure;
HTTP/WebSocket last. Project v3 (M6.7) may start after M6.1 and extends
capability validation incrementally as connectors land.

## 6. Connector model

### 6.1 Identity and registration

- A connector identity is `(provider, name, version)`; `provider` is
  `"calc-flow-connectors"` for built-ins.
- Duplicate identity registration fails atomically: the registry is either
  unchanged or fully extended.
- Unknown connector or format identities fail during compilation, before any
  source construction or lifecycle side effect.

### 6.2 Capability vocabulary

Capabilities are independent typed axes on an immutable descriptor; a single
coarse flag such as `database = true` MUST NOT exist:

- `delivery`: `best-effort` | `at-least-once` | `exactly-once`;
- `replay`: `replayable-exact` | `unreplayable`;
- `watermark`: `native` | `generated-only`;
- `transaction`: `none` | `pre-commit-commit` | `ledger-idempotent`;
- `snapshot`: optional consistent-snapshot reading mode;
- `polling`: optional strictly-monotonic composite-cursor mode;
- `cdc`: optional append-only change-event mode;
- `lookup`: optional point-lookup mode (deferred; no 3.0 connector ships it).

The registry descriptor converts into the A6-native source/sink capability
values that preflight already validates. Effective delivery is the minimum
of source, operator, edge, and sink capability; a requested
`exactly-once` that any reachable participant cannot satisfy fails
compilation with the precise participant path.

### 6.3 Secret handling

- Connector configuration carries named secret slots as
  `SecretReference { resolver, key }` values; the structured config types
  have no field capable of holding a secret value.
- `SecretResolver` resolves references only at connector open time inside
  the owning task; resolved values are opaque handles, never stored in
  config, status, metrics, logs, errors, fingerprints, or manifests.
- Debug/Display of every secret-bearing type renders a fixed redaction
  marker; canary tests enforce the redaction census.

### 6.4 Format layer

- `FormatDecoder`/`FormatEncoder` are transport-orthogonal and identified by
  `(name, version)`.
- Decoding is bounded: expansion beyond the configured row/byte limits
  fails before enqueue with the offending format identity and bound.
- Formats participate in the plan-scoped registry snapshot exactly like
  connectors.

## 7. Transport scopes (summary)

Per-transport RED lists, implementation detail, and design references remain
in master-plan section 12 tasks M6.2-M6.6; this section freezes only the
contract boundaries:

- **file/Parquet:** finite snapshot source with stable file-identity cursor;
  transactional epoch-renamed Parquet sink; exactly-once under the full M5
  fault matrix on supported local filesystems.
- **Kafka:** deterministic partition-to-task mapping; per-partition offset
  cursors; stable non-secret transactional IDs; explicit fencing, rebalance,
  lost-partition, and recovery handling; exactly-once claims only after the
  Kafka fault matrix passes.
- **PostgreSQL:** snapshot, incremental-query, and logical-CDC source modes
  with commit-LSN cursors and barrier-at-commit-boundary; append, upsert,
  and ledger-transactional sinks; secrets only through the resolver; DDL
  stops the source with old/new schema reporting.
- **ClickHouse:** snapshot and incremental-query sources; at-least-once sink
  with per-epoch stable `insert_deduplication_token` exposed as
  `retry-deduplicated`, never as unconditional exactly-once.
- **HTTP/WebSocket:** bounded responses, optional ETag/Last-Modified replay
  cursors, TLS verification on by default, explicit insecure mode with
  warning, `DropOldest` explicit and incompatible with exactly-once.

## 8. Project v3 model

- Top level: `format_version: 3`, `name`, `runtime.mode` (`batch` |
  `stream`) with mode-specific options, `graph`, stream `sources` and
  `sinks` binding lists, `state`/checkpoint configuration.
- Source binding: binding id, connector identity, format identity, bounded
  data-only options, watermark policy reference, named secret references.
- Sink binding: binding id, connector identity, format identity, options,
  requested delivery.
- Batch projects keep inline data fixtures; stream projects use connector
  bindings; a project mixing both fails validation.
- Every layer rejects unknown fields; documents are size- and depth-bounded;
  canonical serialization determines the fingerprint; secrets enter only as
  references.
- `format_version: 2` documents fail with `UnsupportedVersion`; the v2
  schema file moves to a historical documentation directory that no runtime
  path reads.
- The generated JSON Schema must equal `schemas/project-v3.schema.json`
  byte-for-byte at every merged commit.

## 9. Python surface

- `PipelineBuilder.compile_stream()` accepts a project v3 document and
  compiles onto the existing A6 native job handle; no second owner,
  re-entrancy, or lifecycle bypass is introduced.
- Capability enumeration reflects the compiled native module honestly,
  including the absence of opt-in transports in default wheels.
- Validation errors preserve stable field paths and never leak secrets.
- `_native.pyi` stays exactly consistent with runtime members.

## 10. Studio v3 model

- Routes: `/api/v3/projects`, `/api/v3/jobs`, `/api/v3/jobs/{id}`,
  `/api/v3/jobs/{id}/events` (SSE), `/api/v3/jobs/{id}/checkpoint`,
  `/api/v3/jobs/{id}/shutdown`, `/api/v3/jobs/{id}/cancel`.
- Before `/api/v2` removal, Studio MUST enforce replacement limits: maximum
  concurrent jobs, per-job and global resident-memory ceilings, maximum
  checkpoint/state disk usage, and user-explicit stop as the only natural
  lifecycle end. Setting the worker timeout to nothing is forbidden.
- SSE events carry payload-free job progress (state, epoch, watermark,
  throughput, backpressure, late-row counters); never secrets or raw
  batches. `serve()` stays loopback-only.
- OpenAPI and generated TypeScript types regenerate in the same commit as
  route or model changes.

## 11. Acceptance matrix

### Registry and capability (M6.1)

- **AC-01:** duplicate connector or format identity fails atomically;
  unknown identity fails before source construction.
- **AC-02:** a post-compilation registration cannot affect an existing plan
  snapshot.
- **AC-03:** project documents select connectors and formats by data only;
  no executable object can be injected through a project.
- **AC-04:** connector configuration cannot structurally carry a secret
  value; resolution happens only at open time and never appears in status,
  metrics, logs, errors, fingerprints, or manifests.
- **AC-05:** decoder expansion beyond row/byte limits fails before enqueue
  with format identity and bound in the error.
- **AC-06:** capability axes are independently observable; no coarse
  capability flag exists.
- **AC-07:** a requested `exactly-once` with any incapable reachable
  participant fails compilation with the precise participant path, before
  any connector lifecycle side effect.

### File/Parquet connector (M6.2)

- **AC-08:** deterministic discovery and cursor replay; partial file,
  schema mismatch, corrupt Parquet, and oversized row groups fail closed.
- **AC-09:** staging files are never visible as committed output; retry and
  recovery are idempotent; symlink, traversal, wrong type, and locked files
  fail closed.
- **AC-10:** the real file sink passes the full M5 fault matrix;
  file-to-Parquet is exactly-once on supported local filesystems.

### Kafka connector (M6.3)

- **AC-11:** vendored `rdkafka` builds on Linux, macOS, and Windows wheel
  targets with `unsafe_code = "forbid"` intact.
- **AC-12:** restart replays partitions from checkpointed offsets; the
  transactional sink emits no duplicate records for a committed epoch.
- **AC-13:** fencing, rebalance, lost-partition, and timeout paths are
  explicit, typed, and observable; no exactly-once claim before the Kafka
  fault matrix passes.

### PostgreSQL connector (M6.4)

- **AC-14:** snapshot + CDC hand over without a gap and preserve
  transaction commit order; cursors are commit LSNs.
- **AC-15:** barriers inject only at transaction commit boundaries; slot
  LSN is confirmed only after the manifest is durable.
- **AC-16:** the transactional ledger sink is idempotent across commit-ack
  loss and process restart; append/upsert modes never claim exactly-once.
- **AC-17:** credentials exist only through the secret resolver and are
  absent from every observable surface.

### ClickHouse connector (M6.5)

- **AC-18:** snapshot/polling cursors resume without gaps or infinite
  repeats; unknown types fail at compile time.
- **AC-19:** same-epoch retries reuse the identical deduplication token and
  row order; capability is reported as `retry-deduplicated`, not
  `exactly-once`.

### HTTP/WebSocket connector (M6.6)

- **AC-20:** capabilities distinguish replayable HTTP, unreplayable HTTP,
  and lossy WebSocket; `DropOldest` is explicit and incompatible with
  exactly-once.
- **AC-21:** TLS verification defaults on; insecure mode is explicit and
  warned; authorization material is redacted everywhere.

### Project v3 (M6.7)

- **AC-22:** v2 and malformed v3 documents fail closed with precise paths;
  every layer rejects unknown fields.
- **AC-23:** one project v3 defines PostgreSQL CDC -> window ->
  ClickHouse/Parquet without executable objects or secret values.
- **AC-24:** canonical serialization is deterministic; the generated schema
  equals `schemas/project-v3.schema.json` byte-for-byte.

### Python (M6.8)

- **AC-25:** project v3 compiles onto the A6 job handle with identical
  Rust/Python connector, capability, and delivery views.
- **AC-26:** capability enumeration never advertises connectors absent from
  the compiled native module; the stub matches runtime members exactly.
- **AC-27:** validation errors preserve field paths and leak no secrets;
  repeated cancellation stress stays green.

### Studio v3 (M6.9)

- **AC-28:** replacement resource limits are enforced before `/api/v2`
  removal; exceeding them produces a typed terminal job state.
- **AC-29:** SSE streams carry payload-free progress; browser workflows
  start, observe, checkpoint, stop, and reconnect to terminal status with no
  worker or EventSource leaks.
- **AC-30:** OpenAPI and TypeScript types regenerate in the same commit.

### Integration (M6.10)

- **AC-31:** project and REST v2 are removed atomically with all surfaces
  green at the exact final head; no alias or shim remains.
- **AC-32:** the exact final head passes the full gate set and a 20-minute
  connector-path soak with machine-readable evidence.

## 12. Required implementation order

```text
M6.1 -> M6.2 -> { M6.3 | M6.4 } -> M6.5 -> M6.6
M6.1 -> M6.7 -> M6.8 -> M6.9
all  -> M6.10
```

M6.1 MUST merge before any connector crate exists. M6.7 work starts on the
`feature/m6-integration` branch after M6.1 merges; connector tasks merge to
`main` directly as additive feature-gated changes.

## 13. Pull-request and merge strategy

- One task, one issue, one feature branch, one PR; RED evidence recorded in
  the PR body before implementation.
- Connector PRs (M6.2-M6.6) target `main` and MUST keep ordinary unit and
  coverage runs container-free.
- Product-surface PRs (M6.7-M6.9) stack on `feature/m6-integration`; M6.10
  merges the integration branch atomically after all gates pass at the exact
  final head.
- Any deviation from a frozen decision D1-D5 requires revising this
  specification first in its own reviewed PR.

## 14. Completion definition

M6 is complete when issues #133-#143 are closed with: merged implementations
passing the full repository gate set at exact heads, container legs green in
CI, fault-matrix and soak evidence recorded for transactional paths, all
generated contracts regenerated, and no v2 project/REST surface remaining.
M6 completion does not bump versions or publish artifacts; that is M7.4.
