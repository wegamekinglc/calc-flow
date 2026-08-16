# Continuous Streaming 3.0 — M6 Connectors and Project v3 Critique

| Field         | Value                                                            |
| ------------- | ---------------------------------------------------------------- |
| Status        | **Reviewed against spec and API note revision 1**                |
| Baseline      | `main@858199f6df0161801bb6028f37f3ebbeb1684e3e` (post Public A6) |
| Artifact slug | `m6-connectors-project-v3`                                       |
| Spec          | `../specs/m6-connectors-project-v3.md`                           |
| API note      | `../api-notes/m6-connectors-project-v3.md`                       |

This critique adversarially reviews the M6 delta package along eight axes.
Each concern is resolved by a binding requirement on the spec, the API
note, or the task list. Verdicts: **BLOCK** (must change the package) or
**RESOLVED** (requirement already frozen).

## Axis 1 — Build and supply chain risk

**Concern (BLOCK):** D1/D2 put `rdkafka` behind a cargo feature, but the
workspace runs clippy, coverage, and docs with `--all-features`; a
system-librdkafka dependency would break every default developer
environment and the coverage gate.

**Resolution:** Spec D3 freezes vendored/cmake `rdkafka` and pure-Rust
rustls clients for every other transport; container tests are env-gated
`#[ignore]` tests excluded from ordinary runs. RESOLVED — M6.3 must prove
the vendored build on all three wheel targets before any Kafka code lands
(AC-11).

**Concern:** a new third workspace crate multiplies release surfaces.

**Resolution:** D4 keeps `version.workspace = true`, defers publication to
M7.4, and requires packaging hygiene from M6.2. RESOLVED.

## Axis 2 — Capability spoofing and delivery claims

**Concern (BLOCK):** a mis-declared descriptor could let an ordinary sink
claim exactly-once, recreating the false-safety failure mode the research
report warns about.

**Resolution:** the registry MUST NOT define a second vocabulary; it
converts to the A6-native capability descriptors that whole-job preflight
already validates, and AC-07 requires a requested exactly-once to fail at
compilation with the precise incapable participant path, before any
factory `open()`. The ClickHouse token path is explicitly capped at
`retry-deduplicated` (AC-19). RESOLVED.

**Concern:** `open_transactional` returning `None` could silently downgrade
a requested transactional sink.

**Resolution:** the API note requires exactly-once eligibility to be
decided at compilation from capabilities; a runtime `None` can only occur
for a plan that already compiled as at-least-once. M6.1 tests must assert
the downgraded path is unreachable for exactly-once plans. RESOLVED —
carried as a named RED test in M6.1.

## Axis 3 — Secret leakage

**Concern (BLOCK):** connector options are user JSON; a password typed
into `options` would flow into fingerprints, manifests, and logs.

**Resolution:** the spec structurally forbids secret values: config types
expose only `BTreeMap<String, SecretReference>` slots, resolution happens
only at open time into a non-serializable `SecretHandle` with fixed
redaction markers, and canary tests enforce the census (AC-04, AC-17,
AC-27). Project v3 validation rejects any non-reference credential-shaped
value before native code runs. RESOLVED.

**Concern:** error strings from clients often embed URLs with credentials.

**Resolution:** the API note freezes the error projection and its redaction
census (no values, no credentialed URLs, no frames); M6.4 must include a
canary that injects a poisoned error from a fake client and asserts the
public error. RESOLVED — named RED test in M6.4.

## Axis 4 — Coverage gate feasibility

**Concern (BLOCK):** the 90% workspace line gate with container tests
excluded could be unmeetable for protocol-heavy connectors, inviting
silent exclusion of the whole crate.

**Resolution:** D3 requires thin shim traits so unit tests with fakes
cover connector logic; any per-connector floor below 90% requires a
reviewed spec revision recorded in this package, minimum 85%. Container
legs stay outside coverage by design. RESOLVED — treat any proposal to
widen exclusion beyond ignored container tests as a BLOCK at review.

## Axis 5 — Studio resource limits

**Concern (BLOCK):** deleting the worker timeout without an enforced
replacement turns Studio into an unbounded process farm.

**Resolution:** the spec and API note freeze the `JobLimits` model
(concurrency, per-job and global memory, checkpoint disk) with typed
`job_limit_exceeded` terminal states, and AC-28 requires enforcement and
tests before `/api/v2` removal; explicit user stop is the only natural
end. RESOLVED.

## Axis 6 — Project v3 strictness and migration

**Concern:** v2 projects in the wild have no migration path; silent
field reinterpretation would be worse.

**Resolution:** v2 fails `UnsupportedVersion`; the v2 schema moves to a
historical directory no runtime path reads; the M7.3 guide documents the
manual mapping without promising automation. Batch mode keeps fixtures so
non-stream users port minimally. RESOLVED.

**Concern:** fingerprint drift between v2-era and v3 stream plans could
invalidate checkpoints on upgrade.

**Resolution:** M6 is a breaking major-version boundary by design; the
CHANGELOG must state that v3 fingerprints intentionally differ and prior
state roots are not recoverable across the boundary. RESOLVED — carried as
an M6.7 documentation requirement.

## Axis 7 — Container test isolation

**Concern:** container tests leaking into ordinary unit runs would make
local development require Docker and break the WSL2 primary environment.

**Resolution:** `#[ignore]` plus `CALC_FLOW_CONNECTOR_CONTAINERS=1` gate
both CI and local runs; compose files are excluded from packaging; the M6.2
task validates WSL2 Docker feasibility and falls back to CI-only container
evidence with fake-shim local coverage if unavailable. RESOLVED.

## Axis 8 — Process and sequencing

**Concern:** connector tasks merging directly to `main` while product
tasks stack on an integration branch could desynchronize the capability
model.

**Resolution:** the frozen order (spec section 12) requires M6.1 to merge
before any connector crate exists, and M6.7 to rebase onto main after each
connector merge; M6.10 performs the atomic cutover at an exact green head.
Any D1-D5 revision requires a prior spec PR. RESOLVED.

## Verdict

All axes are RESOLVED with binding requirements carried into named RED
tests and acceptance criteria AC-01 through AC-32.

BLOCKS REMAINING: 0
