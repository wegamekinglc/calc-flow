# Continuous Streaming v3 Current-Main Acceptance Audit

| Field      | Value                                                          |
| ---------- | -------------------------------------------------------------- |
| Baseline   | `main@3d7f721b758beede63c4b7fdf232b58e7543b4aa`                |
| Plan       | `docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md` |
| Audit date | 2026-08-19                                                     |
| Candidate  | Review worktree; final PR head recorded by GitHub              |
| Verdict    | **IMPLEMENTED — full gates and exact-head review required**     |

## Executive result

The baseline audit found eleven material gaps behind already-merged M6/M7
labels. This review did not infer completion from those labels. It traced the
public Rust, PyO3/Python, connector-factory, Studio worker, generated-contract,
CI, security, performance, packaging, and release paths and added RED-first
coverage for the missing behavior.

The review candidate closes CFV3-01 through CFV3-11 in implementation. It also
closes four gaps discovered while validating those findings: project-v3 support
for union and window operators, sink-owned checkpoint state segments,
high-cardinality window-state recovery beyond the former 10 MiB JSON boundary,
and an honest three-tier best-effort/at-least-once/exactly-once delivery proof.

This document deliberately separates implementation closure from repository
acceptance. The candidate is not a final pass until the full commands in
`AGENTS.md` pass and GitHub verifies the exact final PR SHA, including Actions,
Codacy, unresolved review threads, conflict state, and mergeability.

## Finding closure ledger

### CFV3-01 — Canonical project v3 cutover

Status: **implemented**

- `crates/calc-flow/src/config.rs` is the single strict project-v3 model and
  compilation path. It resolves graph operators, exact connector and format
  identities, options, source/sink bindings, state, watermarks, and delivery.
- `format_version: 2` fails before compilation or connector work. The Rust
  regression is in `crates/calc-flow/tests/project_v3.rs`; Python and Studio
  cover the same boundary in `python/tests/test_project_v3.py` and
  `web-ui/backend/tests/test_app.py`.
- PyO3, Python `PipelineBuilder`, Studio `/api/v3`, the generated JSON Schema,
  OpenAPI, and TypeScript declarations consume the same v3 contract. The
  parallel public v2 runtime surface and v2 Studio routes are removed.
- Built-in project operators now include stream-only union and window nodes;
  batch compilation rejects them with a precise runtime-compatibility issue.

### CFV3-02 — Binding-scoped secret resolution

Status: **implemented**

- Project bindings carry only `SecretReference` values. `config.rs` validates
  exact required slots and constructs a binding-scoped resolver before a
  factory is invoked.
- The resolver returns redacted, non-serializable `SecretHandle` values. The
  reference map is frozen into the binding, while credential bytes remain
  connector-owned and never enter fingerprints, manifests, status, or public
  diagnostics.
- Core and connector canary tests cover missing/unknown slots, resolver kinds,
  credentialed URLs, client errors, checkpoint recovery errors, and panic
  projection.

### CFV3-03 — Registered connector runtime paths perform real I/O

Status: **implemented**

- Registry factories return the same `StreamSource`/sink objects that own
  credentials, clients, cursors, checkpoint state, and close/join behavior.
  There are no factory-returned perpetual-`Idle` placeholders.
- HTTP and WebSocket factory-path tests use local servers. Kafka, PostgreSQL,
  PostgreSQL CDC, and ClickHouse have opt-in container tests that exercise the
  registered object rather than reproducing its I/O outside the connector.
- The runtime opens, seeks or recovers, polls, checkpoints, and closes these
  objects through the public source/sink lifecycle only.

### CFV3-04 — Studio owns native continuous jobs

Status: **implemented**

- `web-ui/backend/src/calc_flow_studio/run_manager.py` starts a persistent
  spawned worker whose owner is the Python/native `StreamingRunner` and
  `StreamingJob`, not a batch-preview facade.
- `/api/v3/jobs` lifecycle calls await native status, checkpoint, graceful
  shutdown, cancellation, and terminal settlement. SSE projects state, epoch,
  watermark, throughput, queue pressure, backpressure, and late-row values.
- Concurrency, per-job/global RSS, checkpoint/state disk, worker death, and
  explicit stop limits are enforced and covered in
  `web-ui/backend/tests/test_jobs_v3.py`.

### CFV3-05 — Honest connector coverage gate

Status: **implemented; numeric gate must pass on the final tree**

- Linux CI now runs the exact repository command
  `cargo llvm-cov --workspace --all-features --fail-under-lines 90` without
  excluding connector modules.
- Fake/local-server tests cover offline and protocol-independent logic;
  service-backed correctness remains a separate container gate and is not used
  to hide unmeasured source files.

### CFV3-06 — Paired performance and soak release gates

Status: **implemented**

- `scripts/verify_perf_gates.py` requires distinct baseline and candidate
  artifacts, exact ref metadata, complete paired pytest and Criterion labels,
  usable confidence intervals, and the frozen five-percent decision rule.
- `.github/workflows/release.yml` checks out the prior tag and exact candidate,
  executes both benchmark families, verifies their role/ref provenance, and
  invokes the fail-closed comparison.
- The release workflow also runs the high-cardinality state recovery test and
  both 20-minute continuous-runtime soaks; missing evidence fails the release.

### CFV3-07 — Enforcing security and supply-chain gate

Status: **implemented**

- `scripts/verify_security_gates.py` runs `cargo audit`, `cargo deny --locked
  check`, and `npm audit --omit=dev` with explicit argument vectors and failure
  propagation. `--checklist-only` is the only non-enforcing mode.
- Release-helper tests pin command order, checklist evidence, workflow wiring,
  v3 release configuration, and fail-closed behavior.
- Credential redaction, path/symlink containment, bounded decode/message/state
  inputs, schema/project validation, database ledger identity, and ClickHouse
  deduplication boundaries have named tests in their owning modules.

### CFV3-08 — PostgreSQL connection-task ownership

Status: **implemented**

- PostgreSQL source, CDC source, and sink retain their driver task. Failed open,
  connector failure, cancellation, and close cancel when required, join the
  task, and preserve deterministic primary/secondary error order.
- No connection driver is detached with a dropped `tokio::spawn` handle.

### CFV3-09 — PostgreSQL and ClickHouse acceptance semantics

Status: **implemented**

- PostgreSQL snapshot uses a held repeatable-read transaction; incremental
  polling uses an exact lexicographic composite cursor. Logical CDC implements
  publication/slot policy, exported-snapshot-to-LSN continuity, bounded
  transaction splitting, and checkpoint admission only after the final batch.
- PostgreSQL transactional sink checks and writes the epoch ledger and target
  rows in one database transaction; recovery compares durable evidence before
  replay.
- ClickHouse source uses a unique composite cursor. Its ordinary sink performs
  real inserts; retry-deduplicated mode verifies server/table capability,
  preserves exact staged blocks and stable tokens, and never claims generic
  exactly-once.
- Snapshot PostgreSQL, HTTP, and WebSocket capabilities are explicitly
  best-effort/unreplayable. The requested/effective delivery proof never
  overstates a lossy route.

### CFV3-10 — Public documentation and guidance match v3

Status: **implemented**

- README, repository guidance, introduction, API references, getting started,
  runtime envelope, connector guide, migration guide, examples, and Studio
  documentation now describe project v3 and `/api/v3`.
- `docs/connectors.md` records exact connector identity, option/secret examples,
  replay and transaction bounds, three-tier delivery semantics, CDC behavior,
  ClickHouse token limits, and union/window project shapes.
- Connector identity version `2.0.0` is the intentionally stable registered
  protocol identity; package versions are independently synchronized at
  `3.0.0`.

### CFV3-11 — v3 release workflow

Status: **implemented**

- Release tags match `v3.*`; package smoke tests assert `/api/v3` and Studio's
  `calc-flow>=3.0.0,<4` dependency range.
- The workflow builds and inspects core wheel, sdist, Rust crates, and Studio
  wheel, then installs wheels in clean environments and runs public smokes.
- `scripts/test_release_config.py` makes the tag, route, dependency, performance,
  security, soak, and helper-test wiring executable release invariants.

## Additional closure discovered by this audit

- `DeliveryGuarantee` and project `DeliveryRequest` expose `BestEffort`,
  `AtLeastOnce`, and `ExactlyOnce` across Rust, PyO3/Python, schema, Studio, and
  status. At-least-once routes downgrade only when their reachable sources are
  lossy/unreplayable; disjoint outputs remain independent.
- Public job status projects the aggregate event-time watermark through Rust,
  PyO3, Studio RFC 3339 SSE events, and the frontend observatory.
- Checkpoint manifests carry immutable sink state segments as well as operator
  segments. Recovery validates and restores both ownership namespaces.
- The opt-in M7 window-state test creates 400,000 keys, proves the segment is
  larger than the retired 10 MiB inline-JSON ceiling, and restores the exact
  accumulator set.

## Known defect accepted for this candidate

### CFV3-KNOWN-01 — Intermittent managed-recovery start failure under local parallel stress

Status: **open; documented and accepted as non-blocking for this PR**

- One local run of the complete 512-test Rust core binary, with the default
  parallel test scheduler and substantial concurrent checkpoint fault tests,
  observed the checkpoint-restart smoke fail during generation 1 or 2. The
  owning public boundary reported `streaming runtime initialization failed`;
  test-only pre-sanitization diagnostics narrowed the primary failure to
  managed checkpoint recovery, but did not reproduce often enough to identify
  the underlying state, source, operator, or sink recovery operation.
- The focused three-generation restart smoke passes, the repository's serial
  Rust CI configuration passes this boundary, and subsequent complete parallel
  stress runs passed. The observed path fails closed before the restarted job
  accepts new work; there is no evidence of manifest selection corruption,
  cursor advancement beyond durable state, duplicate sink commit, or data
  loss. Temporary raw-error diagnostics were removed from the candidate.
- Follow-up work must retain the sanitized public error contract while adding
  structured, secret-safe internal recovery-stage evidence. Closure requires a
  deterministic reproducer or a production fix backed by repeated parallel
  restart stress, the focused soak, and the serial CI harness. Until then this
  item must not be represented as resolved merely because a later run passes.

## Acceptance evidence policy

The PR body must map each original finding to its RED test and implementation.
Local validation must run every command group in `AGENTS.md`, including the
unmodified 90% workspace coverage gate, generated-contract drift checks,
package inspection, clean-install smokes, connector container tests where the
services are available, and the opt-in soak gates.

After every push, evidence is stale until it is re-collected for the new head.
Final acceptance requires all Actions and status contexts successful, no
actionable Codacy issue, no unresolved actionable review thread, a clean
mergeability state, and verification that the PR head and reviewed tree are
the exact SHA being reported.
