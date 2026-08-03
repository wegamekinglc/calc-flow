# Continuous Streaming Runtime - Critic Critique

Artifact slug: `continuous-streaming-runtime` (shared with the M0.1
specification and the M0.2 API note, per plan section 16).

## Target

- Spec: `.codex/artifacts/specs/continuous-streaming-runtime.md` (D1-D9,
  S1-S10, I1-I10, NG1-NG13)
- API note: `.codex/artifacts/api-notes/continuous-streaming-runtime.md`
  (A1-A8, B1-B5, C1-C3, D1-D3, E1-E5, F1, G1-G3)
- Plan: `docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md`
  (M0.2 gate: both the `StreamSource` cancellation contract and the
  Drop-cancels-without-join ownership model must be judged free of `Block`
  before M1)
- v2 surfaces checked for fidelity: `crates/calc-flow/src/{lib,operator,io,
  checkpoint,error}.rs`, `python/calc_flow/{runtime,pipeline,store}.py`,
  `web-ui/backend/src/calc_flow_studio/{models,app}.py`,
  `docs/introduction.md`.

## Verdict

**Block.** Four findings are soundness gaps or contradictions in frozen
material: the D4.1 commit order omits the segment-publication step its own
API references; D9.4 contradicts D9.3 on epoch-number reuse; the frozen
`Cursor` type cannot perform the regression detection S2.2/S5.4 assign to
the runtime; and the ownership model as frozen admits three paths to
cancelled-but-never-joined tasks, which D5.1/I5 forbid. All four have
precise, cheap repairs stated below. Everything else is Major or Minor.

## Findings

### Blocking Issues

- **D4.1's commit order never publishes segments to the committed area; a
  manifest can reference a path that does not exist** - spec D4.1 lists
  exactly four steps: stage, durable write ("staging and committed areas on
  the same filesystem so that publication is an atomic rename"), validate,
  then "The checkpoint manifest ... is published **atomically and last**".
  The segment rename promised by the step-2 parenthetical never gets its own
  step. The API note makes the hole load-bearing: A7 declares
  `publish_segment(&self, handle)` with the doc comment "Atomic rename from
  staging to committed (D4.1 step ordering)" - a dangling reference, because
  D4.1 contains no such ordering - and G3's manifest example stores a
  committed-style `relative_path`
  (`".../hourly/0000000000000007-accumulators.arrow"`). An implementation
  that follows the spec text literally publishes the manifest while the
  segment still sits in staging; the crash window between manifest rename
  and segment rename then leaves a published manifest (sole truth, D4.2)
  referencing a non-existent committed path, and recovery fails on a
  completed epoch - exactly what D4.3's last row claims cannot happen. The
  AC-D4 fault matrix also lacks a crash point for this window. The plan got
  this right (M4.1: "校验目标后才 atomic rename"); the spec dropped it.
  - **Suggested fix:** insert a new D4.1 step between validation and
    manifest publication: "Each validated segment is published into the
    committed area by an atomic rename from its staged path, followed by an
    fsync of the committed parent directory, BEFORE the manifest that
    references it is published. The manifest's `relative_path` always
    denotes the committed path." Add a D4.3 crash-table row: "After segment
    rename, before manifest rename | Previous manifest stands; the committed
    but unreferenced segment is garbage and collected (D4.2)". Add the
    matching injection point to AC-D4 and to the M5.5 fault matrix handoff.

- **D9.4 contradicts D9.3: abandoned epoch numbers ARE re-issued, and must
  be** - spec D9.4 states: "Its number is not reused within the lineage
  because the job that owned it is terminal; the recovering job continues
  from the last completed epoch per D9.3." But D9.3 says "the next
  checkpoint after recovery is epoch `R + 1`". Worked counter-example under
  the spec's own definitions: manifest at epoch 6 exists; job A injects
  epoch 7 and hits the D7 timeout (job-fatal, terminal, no manifest); job B
  recovers from the epoch-6 manifest and injects epoch 7 per D9.3. Number 7
  is reused within the lineage (D9.3 keys lineage by fingerprint plus state
  root, spanning jobs). The reuse is not a defect - it is required: G3's
  `pre_commit` example derives `ledger_key` from `(fingerprint, sink,
  epoch)`, and M6.5's stable ClickHouse `insert_deduplication_token` is
  per-epoch; both only dedup correctly if the re-issued epoch produces the
  identical key, which is safe precisely because the replayed prefix is
  deterministic at the exactly-once tier (S9.3) and duplicates are tolerated
  below it (S9.1). The sentence as written tells an implementer the wrong
  invariant.
  - **Suggested fix:** rewrite the second sentence of D9.4 to: "Within the
    abandoning job the number is never reused, because that job is terminal
    (D7, S7.6). The recovering job re-issues the same number as its first
    checkpoint (`R + 1` per D9.3): the abandoned attempt left no manifest,
    and the recovering job replays the same deterministic prefix, so
    re-issued epochs are safe and keep `(fingerprint, sink, epoch)`
    idempotency keys stable (S7.5, S9.3)."

- **`Cursor = serde_json::Value` cannot implement the runtime-side
  regression detection that S2.2 and S5.4 freeze** - spec S2.2: "A repeated
  or regressed cursor fails closed before enqueue (the same pre-side-effect
  treatment as a regressed watermark, S5.4)"; S5.4: "cursor regression
  (S2.2) ... [is] detected and rejected before the offending message is
  enqueued". The enqueuer is the runtime. But the API note A4 freezes
  `pub type Cursor = serde_json::Value;` - an opaque, source-ordered value
  (spec section 1: "a source-defined, totally ordered replay position").
  The runtime holds no comparison rule for a source-defined order, so it
  cannot detect a regressed cursor the way it detects a regressed
  `EventTime` watermark. The two artifacts as frozen assign enforcement to
  a layer the frozen type cannot support; M1.2/M2.2 will implement whichever
  reading is convenient and freeze it as public behavior. Undetected
  regression is not benign: the barrier snapshot cursor (S7.4) "covers"
  enqueued items only if cursors are monotone, so the replayable-prefix
  claim that exactly-once rests on is ill-defined when cursors regress.
  - **Suggested fix:** pick one, now:
    (a) *Ordered envelope (recommended).* Replace the type alias with
    `pub struct Cursor { pub order: Vec<u8>, pub payload: JsonMap }` where
    `order` is a connector-encoded, lexicographically ordered byte key
    (big-endian LSN; composite positions via the G1 order-preserving
    encoding, which already exists for exactly this purpose). The runtime
    compares `order` bytewise and fails closed on repeat/regression per
    S2.2, unchanged in spirit; `payload` stays opaque JSON for the manifest.
    (b) *Source-side obligation.* Amend S2.2/S5.4 so regression detection is
    a hard connector contract obligation ("the source MUST never emit a
    repeated or regressed cursor; the runtime cannot verify source-defined
    ordering and treats violation as a connector defect"), with the runtime
    keeping a hash-based repeat check as a backstop. Weaker - D3.3's
    rationale for model (b) was precisely not to trust per-connector
    diligence - but honest about the layer boundary.

- **Ownership model (D5/A6) is not sound as specified: three paths yield
  cancelled-but-never-joined tasks, and D5.4's guarantee sentence is false
  under the frozen API** - see adjudication (b) below for the full analysis.
  In short: (i) `StreamingRunner::shutdown(self)` joins only reaper entries,
  but a live `StreamingJob`'s tasks sit in the job's supervisor registry,
  not the reaper - so a live job's tasks outlive runner shutdown,
  contradicting D5.4's "no task outlives its owning runner's shutdown";
  (ii) `start(&self) -> StreamingJob` returns an owned handle with no
  lifetime tie, so dropping the runner while a job is alive leaves the job's
  Drop transferring its registry to a runner-scoped reaper whose owner is
  gone - under every implementation consistent with the signatures the
  `JoinHandle`s end up dropped unjoined, which D5.1 forbids ("every
  `JoinHandle` is held by the supervisor or the reaper until it is joined")
  and which makes D5.5's observable-state assertions (mock closed flags)
  timing-dependent on that path, violating I8; (iii)
  `shutdown(self)` consumes the runner, so caller-cancelling that future
  mid-join drops `self` with the reaper unjoined - same hole, reached by
  ordinary async caller cancellation. `Drop for StreamingRunner` is never
  specified.
  - **Suggested fix (full repair in adjudication (b)):** make the runner
    core `Arc`-shared between the runner handle and live job handles; change
    `shutdown(self)` to an idempotent, resumable `shutdown(&mut self)` that
    cancels and joins live jobs first and the reaper second; freeze
    `Drop for StreamingRunner` as token-trigger-only and state that D5.4's
    no-outlive guarantee binds only on the completed-`shutdown` path, with
    the bare-drop path observable (metric/log) and asserted via token state
    and task-completion flags, never sleeps.

### Significant Concerns

- **Sync-Python sources: the teardown-drop mechanics behind "D3.5 across
  the PyO3 boundary" are never stated** - API note B1 claims the A4 rustdoc
  contract applies "including D3.5 across the PyO3 boundary (teardown
  cancellation is delivered once, never mid-stream-and-resume)", and accepts
  "sync or async method returns". For an *async* Python `next()`, dropping
  the Rust-side awaitable cancels the Python task - fine. For a *sync*
  Python `next()` blocking on a thread (the v2 `_resolve` pattern,
  `python/calc_flow/runtime.py` line 18), there is no future to drop: the
  "dropped poll" of D3.5.3 is an abandoned blocking call on a
  `spawn_blocking` thread, which Tokio does not cancel. D3.5.5 requires
  `close()` to "release external resources without requiring the dropped
  call to complete" - for sync Python sources that only works if `close()`
  actively unblocks the in-flight `next()` (close the queue, set the event),
  and the note never says so, nor which thread `close()` runs on while
  `next()` holds another. Adjudication (a) below is conditional on this
  repair.
  - **Suggested fix:** add to B1: for synchronous Python sources, teardown
    abandons the blocked `next()` call; `close()` MUST be callable while a
    `next()` call is blocked on another thread and MUST cause that call to
    return or raise; the runtime never calls `next()` again after initiating
    teardown (D3.5.3) and joins the abandoned thread before `close()`
    returns. Add an M6.8 RED line: a sync mock source blocked in `next()`
    is released by `close()`, observed via flags, not sleeps.

- **Sinks have no teardown-drop tolerance contract; sources got D3.5, sinks
  got nothing** - D5.2 requires every task to "observe the token at every
  await boundary", which means sink `write`/`flush`/`begin_epoch`/
  `pre_commit`/`commit` futures are wrapped in token-aware selects and can
  be dropped at arbitrary await points during cancel/failure convergence.
  The spec carefully released sources from mid-stream cancellation-safety
  (D3.5.1) and stated their residual obligations (D3.5.3-5); the sink
  lifecycle in A5 states no residual obligation at all. A mid-flight dropped
  `commit()` is the commit-ack-loss case and is covered at the *protocol*
  level (S7.5 idempotent retry, M5.4 RED), but nothing covers the *object*
  level: a dropped `write` mid-HTTP-request must not corrupt the sink's
  state for the subsequent `close()`, exactly the D3.5.5 hygiene sources
  get.
  - **Suggested fix:** add an A5 contract paragraph mirroring D3.5: sink
    method futures are dropped only at teardown (graceful drain, cancel, or
    failure convergence), never mid-epoch-and-resume; after such a drop the
    runtime calls no method other than `close()`/`abort()` on that instance;
    `close()` MUST tolerate a mid-flight drop of any prior method.

- **B3's blanket async-cancellation sentence under-specifies per-method
  semantics and conflicts with A6/D5.3** - B3 freezes: "cancelling the
  surrounding task cancels the native operation, native cleanup completes
  before `asyncio.CancelledError` propagates". Imported verbatim from v2's
  one-shot `execute_async`, this is wrong-shaped for a long-lived job
  surface. For `wait_async()` the "native operation" is a wait subscription:
  cancelling it must not cancel the job - A6's rustdoc says exactly that
  ("Dropping this FUTURE does not cancel the job") - but B3's sentence
  invites an implementor to forward cancellation into the job. For
  `shutdown_async()`/`cancel_async()`: does a cancelled awaiter abort the
  native transition (leaving the job half-drained in an unspecified state)
  or detach from it? For `trigger_checkpoint_async()`: does the queued
  request (D9.5: never rejected, never coalesced) survive caller
  cancellation? Three different answers are all textually compatible with
  B3 today.
  - **Suggested fix:** replace the blanket sentence with a per-method table:
    `status_async`/`wait_async` are observe-only - dropping the awaiter has
    no job effect; `trigger_checkpoint_async` - the request stays queued and
    will inject its epoch; the awaiter only loses the resolution;
    `shutdown_async`/`cancel_async` are idempotent transitions - a dropped
    awaiter leaves the native transition running to convergence and a
    re-call re-attaches to the same outcome; `start_async` - a cancelled
    call either returns a live job or converges to none, never a half-started
    job (D5.2 join rules apply).

- **`EpochIdempotent` is an unverifiable self-declared flag that satisfies
  exactly-once compilation, and A5's defense of it is wrong** - A5's
  spoofing paragraph says a false `delivery_capability()` claim "can damage
  only the claimant's own sink output". Not so: S9.2 accepts "a
  transactional or epoch-idempotent sink" for `delivery = "exactly_once"`,
  and S9.4 then reports the pipeline's per-sink guarantee as exactly-once to
  the user. A sink that declares `EpochIdempotent` without dedup behavior
  converts the user's requested exactly-once into at-least-once *with an
  exactly-once label* - the failure mode research section 11 calls out
  ("把普通 `Sink` 宣传为 exactly-once 会产生错误安全感"). The type-level
  `SinkBinding::Transactional` boundary is genuinely spoof-resistant; the
  enum flag is not, and the note's own M6.5 constraint (ClickHouse dedup
  tokens have a bounded retention window and must not be upgraded to
  "无限期 exactly-once") shows the flag carries time-bounded semantics the
  type does not express. Also unspecified: when `delivery_capability()` is
  sampled - a sink returning different values at binding time and mid-job
  silently downgrades the guarantee.
  - **Suggested fix:** (1) sample `delivery_capability()` once at binding
    validation and freeze it in the compiled plan's per-sink delivery
    record; (2) require an `EpochIdempotent` declaration to carry its basis
    (mechanism identifier plus retention bound, e.g.
    `{ "mechanism": "clickhouse_insert_dedup_token", "window": "bounded" }`)
    into the manifest's `sinks.*.delivery` record and into S9.4 reporting,
    so the guarantee is reported with its evidence and expiry class; (3)
    correct A5's sentence: a false claim damages the *user-visible
    guarantee*, which is why the basis must be recorded.

- **`TransactionalSink::recover` lacks the orphan-staging cleanup
  obligation, and G3's derivation rule silently assumes a query capability
  the trait never states** - S7.5: "A failure before manifest publication
  aborts staged sink output". A crash cannot run `abort()`. After a crash
  between sink pre-commit and manifest publication, the new job's
  `recover()` reads the *latest* manifest (epoch R) - which does not
  contain the abandoned epoch E's pre-commit - so no contract path ever
  cleans E's external staged artifacts (staging files, prepared
  transactions, staged inserts). The plan knows this (M5.5 fault matrix
  asserts "staged-artifact cleanup"; M6.2: "staging file 不会被当作
  committed output") but the frozen trait does not. Separately, S7.5/G3's
  chosen derivation rule ("per-sink completion is derived by the sink
  recovery protocol from the manifest's pre-commit metadata") is sound only
  if `recover()` can determine the commit outcome from the manifest's
  `SinkPreCommit` plus the external system alone; A5's rustdoc never states
  that requirement.
  - **Suggested fix:** extend A5's `recover` contract: "recover(manifest)
    MUST be able to determine, from the manifest's `SinkPreCommit` and the
    external system alone, whether each recorded epoch was externally
    committed, completing those that were not; it MUST also abort or remove
    any locally staged epoch artifacts NOT recorded in that manifest
    (orphan cleanup for crash-abandoned epochs)."

- **C2 excludes connector options from the semantic fingerprint wholesale -
  including options that change the data semantics** - C2 states the secret
  reference "does not feed the semantic fingerprint (NFR-5 restricts the
  fingerprint to execution mode, graph, operator configuration, UDF catalog,
  and state-layout-affecting semantics; connector options are none of
  these)". Excluding secret references is correct (rotation must not
  invalidate checkpoints). Excluding *all* connector options is not: C1's
  own source example carries `"publication": "calc_flow_pub"` and
  `"slot": "calc_flow_slot"` in `options`. Two projects differing only in
  `publication` read different tables - different data - yet produce the
  same fingerprint, the same lineage (D9.3), and checkpoint compatibility;
  recovery would silently restore window state computed from one
  publication's data and resume reading another's. Same for sink `table`,
  HTTP `url`, polling `query`.
  - **Suggested fix:** split connector options into data-semantics-affecting
    options (feed the semantic fingerprint: publication, query, topic,
    table, URL, file paths, format choice) and credential/transport options
    (feed nothing, or the runtime-config hash); state that the split is a
    per-connector declared list owned by M6.1's capability surface, with
    secret references never feeding either. Alternatively, cheaper and
    acceptable for M0: keep options out of the fingerprint but freeze a
    source-compatibility rule - the manifest records a connector-declared
    "source identity hash" and recovery fails closed on mismatch.

- **Secret values can leak through client-library error messages at the
  open-time resolution boundary; I4 freezes the surfaces but not the
  sanitization** - I4: "Connectors resolve secrets through references at
  open time only." The resolution boundary is exactly where a resolved value
  first exists in process memory, and where client libraries echo it:
  connection errors from database/HTTP clients commonly embed the connection
  URL or auth material in their `Display` output. If a connector propagates
  such an error into `CalcFlowError`, the value flows into logs, status
  snapshots, `JobOutcome.errors`, SSE `job.terminal`, and Studio
  `JobResponse.error` - every surface I4 names, through a door I4 does not
  close. The plan states the per-connector obligation (M6.4: URL/password/
  TLS key "序列化、日志、error、metrics 中不可见"; M6.6: "authorization
  header、含凭据 URL、payload 全部脱敏"), but an M6 implementor reading
  only the frozen invariant sees no general rule.
  - **Suggested fix:** extend I4 (or C2) with one sentence: "Connector
    errors raised after secret resolution MUST be wrapped so that
    client-library messages - which may echo resolved values, URLs, or auth
    headers - never propagate raw into `CalcFlowError`; the wrapped error
    carries the connector identity and a sanitized message." Add an AC-I
    redaction test that forces a failing open with a canary secret value.

- **Exactly-one-active-job-per-lineage is never stated; two runners on one
  state root break D9.2 and D4.2** - D9.3 keys lineage by fingerprint plus
  state root, but nothing forbids two `StreamingRunner`s (two processes, or
  two runners in one process) sharing one state root and checkpoint
  directory. Both recover from the same manifest, both inject epoch R+1
  (violating D9.2's strictly-increasing rule), and both publish manifests -
  atomic rename gives last-writer-wins, so one job's completed epoch
  silently vanishes from the sole-truth record (D4.2) and its segments
  become orphans. Plan M4.2 assumes a lock ("locked/delete failure 立即
  停止") but the spec never states the invariant the lock enforces. v2 had
  the rule in prose (introduction.md: "One runner has an exclusive lease on
  a stateful plan"); v3's spec lost it.
  - **Suggested fix:** add to S7.1 or D9: "A lineage (pipeline fingerprint
    plus state root) has exactly one active job. The managed state root and
    checkpoint store are exclusively locked by the running job (M4.2); a
    second `start()` against the same lineage fails with
    `CalcFlowError::Conflict` before any source opens."

- **I5's "including connector-internal tasks" is unimplementable through
  the frozen trait surface** - I5 extends the D5 ownership chain to
  "connector-internal tasks and compaction workers". Compaction workers are
  runtime-spawned - registerable. Connector-internal tasks are not: the A4/A5
  traits give connectors `&mut self` methods with no handle to the
  supervisor registry, so a connector that spawns a Tokio task has no
  registration path. Every M6 implementor will resolve this ad hoc, which is
  what spec section 9 forbids for owned decisions.
  - **Suggested fix:** pick one, now: (a) state in I5/A4/A5 that connectors
    MUST NOT spawn Tokio tasks; blocking client work runs on the I6
    mechanisms (blocking pools, dedicated threads) and client-internal OS
    threads (e.g., a bundled client's background thread) are external
    resources released by `close()`, not runtime tasks; or (b) freeze a
    scoped spawn channel on the source/sink context types. (a) is honest
    about what the trait surface already implies.

- **F1/A2.2's synchronous `checkpoint(E)` returning encoded segment bytes
  puts unbounded CPU work on the operator task's executor thread** - A2.2
  freezes `fn checkpoint(&mut self, epoch) -> Result<OperatorStateSnapshot>`
  as synchronous, and A2.3 carries segments as
  `BTreeMap<String, Vec<u8>>` - already-encoded bytes. For the built-in
  window operator, producing those bytes means Arrow IPC encoding of all
  dirty-key state, synchronously, on the Tokio worker, on the barrier path.
  With a large dirty set (first checkpoint after a load spike), that stalls
  every task on the worker - including control forwarding for the very
  checkpoint in flight - and the only bound is the D7 job-fatal timeout.
  I6's enumerated list (file I/O, compression, connector clients,
  compaction) does not literally cover serialization, but the design intent
  of A2.2 ("keep I/O off the barrier path") is undermined when the encoding
  itself is the expensive part.
  - **Suggested fix:** state in A2.2/A2.3 the capture-cost rule:
    `checkpoint()` MUST be O(dirty-key metadata), not O(dirty-key bytes) -
    the built-in window operator maintains segments pre-encoded or encodes
    incrementally on mutation, so the sync capture is a cheap handoff of
    already-formed buffers; any residual encoding runs on the blocking pool
    with the operator task's ingresses paused (F1 already pauses them at
    alignment). Add an M4.4 RED line: with paused time and a large dirty
    set, the operator task processes a following control message within a
    bounded number of executor steps.

- **Orphan collection races an in-flight epoch: `collect_orphans` can
  delete a segment the in-flight manifest is about to reference** - A7's
  `collect_orphans(&self, retained: &[StateHandle])` deletes everything not
  listed. With the repaired D4.1 order (rename before manifest), a segment
  for the in-flight epoch sits in the committed area, referenced by no
  published manifest yet. If retention/compaction (a job-internal worker per
  D5.1) computes orphans against the manifest list in that window, it
  deletes the segment; the manifest then publishes a reference to a deleted
  file and the next recovery fails on a completed epoch. D4.2's "their
  presence MUST NOT fail recovery" covers extra garbage, not missing
  referenced segments.
  - **Suggested fix:** state in D4 or A7: orphan collection and retention
    treat as retained (i) every segment referenced by a retained manifest,
    (ii) every segment belonging to an in-flight epoch, and (iii) every
    segment created after the latest retained manifest; equivalently,
    collection is serialized against checkpoint commits by the coordinator.

### Minor / Style Notes

- **The v2 replacement table is incomplete.** The naming map plus the
  "keeps v2 meaning" list leave the disposition of many `lib.rs` exports
  unstated: the `CheckpointStore` *trait* (A7 silently swaps in a same-name
  trait with a different method set - louder than a rename, undocumented),
  `ExecutionOptions`, `RunResult`, `RunMetadata`, `NodeTiming`,
  `ExternalPayload`, `TableBatch`, `DataFusionRuntime`/`DataFusionConfig`/
  `DataFusionQueryMetric`, the `ProjectSpec`/`PipelineSpec`/`NodeSpec`
  family, `compile_project`/`validate_project`, `canonical_json`, `JsonMap`,
  `ExternalOperatorSpec`, `OperatorDefinition`, `validate_selected_udfs`,
  and the `MAX_*` constants. Also unflagged: v2 `Sink::write` takes a
  `context: &RunContext` parameter that v3 `StreamSink::write` drops - a
  real signature change hidden under the naming-map row. M1.1 will otherwise
  improvise each disposition. Add an explicit "all unlisted v2 public names
  keep their meaning" sentence plus a list of the intentional same-name
  replacements.
- **A2's trait listing omits the default bodies A2.3 describes.** A2.3 says
  a stateless operator gets a default empty `checkpoint`, a rejecting
  `restore`, and an `Ok(())` `reset`; the A2 code block shows no defaults.
  Show them, as A1 does for `BatchOperator`.
- **`POST /api/v3/jobs/{job_id}/checkpoint` can park an HTTP request for
  the full `checkpoint_timeout` (default 10 minutes, A6).** D1 states no
  route-level bound and no 202-and-poll pattern. State the bound or the
  async pattern; the v2 preview route had `timeout_seconds <= 300`.
- **`trigger_checkpoint` caller cancellation is unspecified** (Rust side;
  the Python side is covered once B3 is repaired): does dropping the future
  dequeue the D9.5 request or leave it to inject? State it - the Studio
  route's client-disconnect behavior depends on the answer.
- **Reaper-observed `JoinError`s have no defined surfacing.** A task that
  panics while converging after a job-handle Drop is only observed by the
  reaper at the next `start()` (D5.4 point i). State: such errors are
  logged/metric'd and never fail the new job.
- **`SecretResolver` is referenced but owned by nobody.** C2 speaks of "the
  process-local `SecretResolver` registry (built-in: `\"env\"`)" but no
  section freezes the trait/registry or lists it as a deferral. Add it to
  the API note's open questions with owner M6.1.
- **Python `capabilities()` is a stringly mapping with no validation
  rules** (B1: `Mapping[str, object]`, keys listed in a comment). State the
  adapter's errors for missing keys, wrong types, and negative bounds.
- **Group-key encoding (G1) is silent on key size and float grouping
  consistency.** A 1 GiB `Utf8` group key yields a 1 GiB encoding in the
  sort path; note a bounded-key rule for M4.3. The IEEE total-order map
  distinguishes `-0.0` from `+0.0` and orders NaN deterministically; M4.3
  must state whether grouping equality matches the encoding order.
- **S10.3's wording vs A4.1's split.** The spec says "`compile_stream()`
  MUST validate" the source batch bounds against edge budgets; A4.1 splits
  this between `compile_stream` (project path) and `StreamingRunner::new`
  (builder path). The intent (before any source opens) is met on both
  paths, but the spec's sentence names one function. Align: "compilation or
  runner construction validates ..., always before any source opens".
- **The Python happy-path example asserts something the frozen semantics
  falsify.** B-section example: `epoch = job.trigger_checkpoint()` then
  `assert outcome.state == "completed" and outcome.completed_epoch == epoch`.
  S8.2's `draining -> completed` publishes "a final epoch manifest (numbered
  per D9)", i.e., epoch `epoch + 1` per D9.2's +1-per-injection rule, so
  `completed_epoch` is `epoch + 1`, not `epoch`. Fix the assertion
  (`completed_epoch > epoch`, or drop it). Examples are executable
  documentation; this one teaches a wrong invariant.
- **B3's store parenthetical conflicts with its own frozen message
  pattern.** B3 says store methods are "unchanged from v2" and then freezes
  one message pattern, but v2 store raises a different message
  ("blocking store operation cannot run inside an event loop; await
  {method}()", `python/calc_flow/store.py` line 22-26) and store async
  methods are not `_async`-suffixed (`get` vs `get_blocking`). Say which
  pattern stores use.
- **Python plan/runner exclusivity is unstated.** Rust enforces one runner
  per plan by move (`StreamingRunner::new(plan, ...)` consumes it); B2's
  Python adapter receives a shared plan object. State whether a second
  `StreamingRunner` over the same plan raises (matching v2's exclusive
  lease) and where.
- **Studio has no fresh-lineage escape hatch.** D1 removes the v2
  project-level checkpoint delete routes and declares fresh lineage "a
  filesystem-level decision". Acceptable for a loopback Studio, but M7.3
  must document the manual procedure (stop job, remove/repoint state root
  and checkpoint directory) or users with a poisoned lineage are stuck.
- **SSE in the generated contract.** D2's `text/event-stream` route must
  survive `npm run sync:api` into `web-ui/openapi.json` and `schema.d.ts`
  with a usable type (the v2 route exists in the backend; verify the
  generator emits something the frontend can consume, since v3 adds
  `Last-Event-ID` resume semantics the frontend must implement).

## Adjudication (a): the `StreamSource` cancellation-safety contract

**Verdict: PASS - no Block. The D3 model (b) plus the D3.5 contract is
sufficient and testable, and no full cancellation-safety obligation survives.
One residual obligation does survive; it is already half-stated in D3.5.5 and
must be completed for sync Python sources (the first Major finding above).**

Reasoning. D3.3's rejection of model (a) is sound: "dropping this future at
any await point loses no data, advances no cursor, and the source remains
usable" is uncheckable by the type system and fragile across PyO3, and one
non-compliant connector silently breaks delivery tiers. Model (b)
concentrates the hard property in the pump/source-task/slot structure:

- *The barrier-path claim holds.* D3.4's bound - O(1) control delivery plus
  completion of at most one in-flight edge enqueue per output edge, which
  can block only on downstream backpressure - contains no external poll
  latency, because the pump alone touches external I/O and the source task's
  control path never awaits it. The residual stall (backpressured enqueue)
  is bounded by the D7 checkpoint timeout, which is job-fatal by decision;
  the claim "a quiet-but-healthy source responds to barrier and cancel
  requests immediately" is true under the structure. Graceful shutdown of a
  permanently quiet source also resolves: D3.5.3 lists graceful shutdown as
  a teardown path, so the pending poll is dropped once, the final manifest's
  cursor simply does not cover the unreturned item, and a later job re-reads
  it (replayable tier) - at-least-once preserved.
- *The one-item slot is sufficient under all `SourceEvent` orderings.* The
  pump cannot refill a full slot and the source task drains in FIFO order,
  so for any interleaving of `Data`/`Watermark`/`Idle` there is at most one
  undisposed event at any instant; when a barrier request arrives, the
  snapshot cursor covers exactly the enqueued prefix, the slotted event is
  strictly later in source order, and S7.4's boundary rule ("an item in the
  prefetch slot or in the pump's in-flight poll is post-barrier") is
  consistent with both cursor and watermark monotonicity. `None` (end of
  input) is not slotted - the pump reports it out of band and the task emits
  exactly one `EndOfInput` (S1.6). No ordering hazard exists.
- *The surviving obligation is teardown-drop hygiene, not cancellation
  safety.* Dropping the in-flight poll at teardown *is* a cancellation at an
  arbitrary await point; what the connector is released from is only the
  resume-and-continue half of the property. The residual obligations - the
  drop must not corrupt instance state needed by `close()`, `close()` must
  release resources without the dropped call completing, and the durable
  cursor must never cover unreturned items - are exactly D3.5.3-5, and they
  are mechanically checkable per connector with a standard mock harness
  (AC-D3's paused-time test with flags: blocked poll, barrier observed on
  all edges, cursor covers only pre-barrier items, teardown drops the poll,
  never polls again, durable cursor unchanged).
- *Testability:* AC-D3 covers the Rust async case deterministically. The
  sync-Python case is testable the same way but the *contract* for it is
  missing (the Major finding above): B1 must state that for sync sources the
  "drop" is an abandoned blocking call and that `close()` must unblock it.

So the contract stands. It needs one paragraph in B1, not a redesign.

## Adjudication (b): the Drop-cancels-without-join ownership model

**Verdict: NOT SOUND AS SPECIFIED - Block.** The specified core is right:
Drop cannot await, so Drop must not join; cooperative token cancellation
plus lazy reaper joins at the two defined points (next `start()`, runner
`shutdown()`) is the correct shape; terminal paths joining every registered
task before reporting (D5.2) is the right rule; and the D5.5 test-assertion
strategy (registry lengths, `JoinHandle::is_finished`, mock closed flags,
completed epochs; never sleeps; paused Tokio time for timers) is the right
strategy and is sufficient for every *specified* path. The Block is for
what the frozen API leaves unowned:

1. **Live job at runner shutdown.** A6's `shutdown(self)` is documented as
   "Joins every reaper entry before reporting closed". A live
   `StreamingJob`'s tasks are registered in the job's *supervisor*
   registry, not the reaper - transfer happens only at the job's own Drop
   (D5.3). Nothing in A6 says `shutdown` cancels or even notices live jobs
   (`start`'s `Conflict` rule only blocks a *second* `start`). So a runner
   can report closed while a job's tasks run on - flatly falsifying D5.4's
   guarantee sentence "no task outlives its owning runner's shutdown".
2. **Runner dropped while a job is alive.** `start(&self) ->
   Result<StreamingJob>` returns an owned handle with no lifetime tie to the
   runner. If the runner is dropped first, the job's later Drop transfers
   its registry to "the runner-scoped reaper" - whose owner no longer
   exists. Under every implementation consistent with the frozen signatures
   (reaper inside the runner: transfer target gone, handles dropped
   unjoined; reaper `Arc`-shared: handles dropped when the last reference
   drops, still unjoined), tasks end cancelled-via-token but never joined.
   That violates D5.1's "every `JoinHandle` is held ... until it is joined"
   and I5's "no detached Tokio tasks", and on this path D5.5's
   observable-state assertions (mock closed flags) become timing-dependent,
   violating I8.
3. **Caller-cancelled `shutdown(self)`.** Because `shutdown` consumes the
   runner, dropping the future mid-join drops `self` - runner, reaper, and
   all unjoined handles - at an arbitrary join boundary. Ordinary async
   caller cancellation (a `select!` in user code, a Python `asyncio`
   cancellation crossing B2) reaches this.

The repair that preserves D5's core and costs one signature change:

- Make the runner core `Arc`-shared between the runner handle and every
  live job handle, so the reaper target outlives the runner handle and
  path 2 is well-defined.
- Change `pub async fn shutdown(self)` to `pub async fn shutdown(&mut self)`,
  idempotent and resumable: a caller-cancelled shutdown leaves a live runner
  that can be shut down again, eliminating path 3. `shutdown` cancels and
  joins live jobs (the runner core tracks them) first, then the reaper -
  making D5.4's sentence true - and only then reports closed.
- Freeze `impl Drop for StreamingRunner`: triggers the cancellation token,
  never joins, never blocks. State explicitly that D5.4's no-outlive
  guarantee binds only on the completed-`shutdown` path; the bare-drop path
  is the sanctioned abandon-joins escape hatch, MUST be observable
  (metric plus log), and tests assert it via token state and task-completion
  flags, never sleeps. I5's "no detached tasks" then reads: no task without
  a cancellation token and a registered owner while any handle is alive -
  which the repaired model enforces by construction.

(An alternative - `StreamingJob<'a>` borrowing the runner - enforces the
teardown order at compile time but breaks the plan's own sketch: a
`Drop`-bearing borrowed job holds the borrow until scope end, so
`runner.shutdown()` after `job.wait()` would not compile without an explicit
`drop(job)`. The `Arc` + `&mut self` repair keeps the sketch working and is
recommended.)

With those edits, the model is sound and M1 may enter.

## The S7.3 / F1 friction: resolved

Spec S7.3's narrative - "the task performs its snapshot for epoch `E`,
stages dirty state, acknowledges the coordinator, and forwards the barrier
per the forwarding-timing decision owned by the M0.2 API note (section 9)" -
lists "acknowledges" before "forwards", while F1 forwards immediately after
the synchronous snapshot and hands the acknowledgement to the coordinator
concurrently.

**Ruling: F1 is consistent. S7.3's list is non-ordered.** The spec itself
assigns the ordering decision to the API note twice: in S7.3 ("per the
forwarding-timing decision owned by the M0.2 API note") and in section 9
("Barrier forwarding timing after operator snapshot ... Constraint recorded
here: snapshot success precedes forwarding either way"). F1 satisfies the
one hard constraint (its step 1 completes before its step 2) and correctly
records the latency bound the deferral demanded (O(graph depth x per-edge
traversal), zero coordinator round trips, vs O(graph depth x round trip) for
the rejected alternative, both under the same D7 timeout ceiling). F1's
failure-semantics paragraph is also sound: a forwarded barrier for an epoch
that never commits produces only unreferenced snapshots, collected per D4.2,
and no downstream effect escapes because staging is not publication (D4.1)
and sinks abort pre-manifest output (S7.5, strengthened by the
orphan-cleanup Major above). No Block.

**Required edit (spec owner, one sentence):** in S7.3, replace

> the task performs its snapshot for epoch `E`, stages dirty state,
> acknowledges the coordinator, and forwards the barrier per the
> forwarding-timing decision owned by the M0.2 API note (section 9);

with

> the task performs its snapshot for epoch `E`, stages dirty state, and
> then acknowledges the coordinator and forwards the barrier in the order
> fixed by the M0.2 API note (section 9) - the verb sequence in this
> sentence is not an ordering claim; the only ordering constraint is that
> the snapshot precedes both the acknowledgement and the forwarding;

## Counter-Proposals

- **Cursor as an ordered envelope (for the S2.2 Block).**
  `pub struct Cursor { pub order: Vec<u8>, pub payload: JsonMap }`:
  connectors encode their source-defined position into a lexicographically
  ordered byte key (big-endian LSN for PostgreSQL; `(partition, offset)`
  via G1's existing order-preserving composite encoding for Kafka; file
  identity + row-group ordinal likewise), and keep human/debuggable payload
  in `payload`. The runtime compares `order` bytewise - S2.2/S5.4's
  runtime-side fail-closed works as written, the manifest stores both
  halves, and G1's encoding is reused rather than invented. Cost: one type
  change in A4 and a per-connector encoding obligation stated once in M6.1.
- **Proof-carrying `EpochIdempotent` (for the spoofing Major).** Replace
  the bare enum variant with a structured declaration
  `EpochIdempotent { mechanism: IdempotencyMechanism, retention: RetentionClass }`
  recorded into the plan's delivery record, the manifest's
  `sinks.*.delivery`, and S9.4 reporting. ClickHouse's bounded dedup-token
  window then reports as exactly what M6.5 demands - retry-deduplicated
  within a bounded window - instead of silently upgrading to exactly-once.
- **`Arc`-core runner with resumable `&mut self` shutdown (for the
  ownership Block).** As sketched in adjudication (b): job handles and the
  runner handle share an `Arc` runner core; `shutdown(&mut self)` is
  idempotent and joins live jobs then the reaper; `Drop` for the runner
  handle only fires the token. Zero new runtime machinery beyond a weak
  live-job registry on the core.

## Questions for the Author

1. (Spec) Was the segment rename deliberately left out of D4.1, or is the
   A7 `publish_segment` doc comment's "(D4.1 step ordering)" a reference to
   a step that was meant to be written? The repair above assumes the plan's
   M4.1 order (validate, rename, then manifest).
2. (Spec) D9.4's "not reused within the lineage" - was "lineage" intended
   as "job"? Under the defined meaning of lineage (D9.3, fingerprint plus
   state root) the sentence is false; the proposed rewrite keeps the
   intended safety property.
3. (API note) For `trigger_checkpoint`, should a caller-dropped future
   dequeue the request or let it inject? The Studio route's
   client-disconnect behavior inherits whichever answer you freeze.
4. (API note) Is the operator-scoped `DataFusionRuntime` of plan section
   2.2 (lazy, one per table stream-operator task, none for array-only
   chains) deliberately absent from the note as pure implementation detail,
   or should its UDF-visibility equivalence to v2's run-scoped session be
   asserted somewhere testable (M2.3's RED lines cover it - confirm that is
   the intended home)?
5. (Both) Which artifact owns the per-connector split of options into
   fingerprint-feeding vs credential/transport (the C2 Major)? It needs a
   home before M6.1, or M6.1 will invent it per connector.
6. (API note) `StreamOperator::checkpoint` receives `epoch` but
   `restore`/`reset` receive no lineage context; is a custom operator ever
   expected to distinguish "restored at R" from "fresh"? If not, say so -
   M4 implementors will otherwise thread it through defensively.

---

*Assessment of the artifacts as a whole: the spec's semantic core (D1-D3,
D5-D8, S1-S6, S8-S10) is unusually tight - the D2 late-boundary analysis,
the D3.4 latency budget, and the S9 capability matrix all survived attack.
The Blocks found here are seam bugs between the two artifacts and between
the artifacts and the frozen Rust signatures - exactly the class of problem
M0 exists to catch before it becomes unchangeable public behavior.*

---

# Round 2 - re-review of the revised artifacts

Both revisions were re-read in full from disk (spec revision 2 and the
revised API note, same slug). Verdicts below cite the revised text. Every
round-1 finding was verified against the actual artifacts, not the owners'
reports.

## Round-1 Blocks: resolution verdicts

- **Block 1 (D4.1 commit order) - RESOLVED.** D4.1 is now five steps; the
  new step 4 reads: "Each validated segment is **published** into the
  committed area by an atomic rename from its staged path, followed by an
  fsync of the committed parent directory, BEFORE the manifest that
  references it is published. The manifest's `relative_path` always denotes
  the committed path", and the closing rule is now "A manifest MUST NOT
  reference an unvalidated or unpublished segment." D4.3 carries the new
  row "After segment rename, before manifest rename | Previous manifest
  stands; the committed but unreferenced segment is garbage and collected
  (D4.2)" (and correctly refined the old row to "After validation, before
  segment rename"). AC-D4 now injects crashes "at each row of the D4.3
  table - including the window between the segment rename and the manifest
  rename" and states "This injection-point set is handed to the M5.5 fault
  matrix." FR17 names the five-step order. The note's A7 `publish_segment`
  doc no longer dangles: it states the rename-plus-parent-fsync occurs
  "always BEFORE the manifest that references the segment is published ...
  (D4.1's stage -> durable write -> validate -> rename -> manifest order)",
  and G3's D4-uphold rule repeats the order.
- **Block 2 (D9.4) - RESOLVED.** D9.4 now reads: "Within the abandoning
  job the number is never reused, because that job is terminal (D7, S7.6).
  The recovering job re-issues the same number as its first checkpoint
  (`R + 1` per D9.3): the abandoned attempt left no manifest, and the
  recovering job replays the same deterministic prefix, so re-issued epochs
  are safe and keep `(fingerprint, sink, epoch)` idempotency keys stable
  (S7.5, S9.3)." The contradiction is gone and the dedup-key rationale is
  on record.
- **Block 3 (Cursor ordered envelope) - RESOLVED.** A4 freezes
  `pub struct Cursor { pub order: Vec<u8>, pub payload: JsonMap }` with the
  contract: bytewise comparison of `order` MUST equal source order,
  connectors build it with the G1 order-preserving encoding (LSN as
  big-endian `u64`, Kafka `(partition, offset)` as a G1 composite), the
  runtime compares bytewise and fails closed on repeat/regression before
  enqueue (S2.2, S5.4), and `order` is non-empty and bounded at 16 KiB
  (D4.4). The spec side matches: section 1's Cursor definition now carries
  "a connector-encoded ordered byte key whose bytewise lexicographic order
  IS the source order, plus an opaque payload ... the runtime compares only
  the ordered key"; S2.2 states "The runtime enforces this by comparing
  each item's ordered cursor key bytewise against the ingress's last
  enqueued key"; section 7's table and section 9's deferral list the
  envelope. G3 serializes `order` as a lowercase hex string with
  `payload` as strict JSON; B1 crosses it as
  `{"order": <hex str>, "payload": <mapping>}` with hex/size validation
  (16 KiB) and defensive copies; the Python example builds a big-endian
  `u64` hex key. Cosmetic residue, no severity: S5.4 kept the phrase
  "cursor regression (S2.2)" instead of the note's suggested "cursor
  ordering-key regression (S2.2)" - substance is unaffected because S2.2
  defines the mechanism.
- **Block 4 (ownership model) - RESOLVED, with one new Minor (N2 below).**
  A6 now states the runner core "is `Arc`-shared between this handle and
  every live `StreamingJob`, so the reaper target outlives this handle and
  a job handle's Drop always has a live transfer target", including the
  live-job registry. `shutdown` is now
  `pub async fn shutdown(&mut self) -> Result<()>`, "Idempotent and
  resumable ... cancels and joins every live job first (the runner core
  tracks them), then joins the reaper, and only then reports closed ... A
  caller-cancelled `shutdown` leaves a live runner that can be shut down
  again; a completed `shutdown` leaves no task alive (the D5.4 no-outlive
  guarantee binds on this path, and only on this path)."
  `impl Drop for StreamingRunner` is frozen: token only, never joins, the
  bare-drop path is "the sanctioned abandon-joins escape hatch ... observable
  (metric plus log) and is asserted in tests via token state and
  task-completion flags, never sleeps". The spec side matches: D5.1 (runner
  core tracks live jobs; every task has a token and a registered owner
  while any handle is alive), D5.3 (reaper "outlives the runner handle"),
  D5.4 (shared core, idempotent/resumable shutdown joining jobs first, the
  guarantee binding on the completed path, the observable escape hatch,
  caller-cancelled shutdown converging on retry, and the reaper JoinError
  rule: "logged and recorded in metrics, and never fail the job that
  triggered the reaper pass"). AC-D5 now asserts all four paths
  (shutdown-with-live-job, caller-cancelled shutdown, bare drop, empty
  registries). D9.6 freezes one active job per lineage with
  `CalcFlowError::Conflict` before any source opens, matching A6's
  `start()` contract. The borrowed-job alternative was considered and
  rejected in the note for the reason round 1 raised (an explicit
  `drop(job)` would be forced into the plan's own sketch).

## Round-1 Majors: 11/11 resolved

1. **Sync-Python teardown - RESOLVED.** B1's "Synchronous Python sources"
  paragraph states `close()` MUST be callable while `next()` blocks on
  another thread and MUST cause it to return or raise; the runtime never
  re-polls after teardown starts and joins the abandoned thread before
  `close()` returns; the M6.8 RED line (flags, never sleeps) is present.
  Spec D3.5.6 names the abandoned-blocking-call rule.
2. **Sink teardown-drop contract - RESOLVED.** A5's "Sink teardown-drop
  contract" mirrors D3.5: drops only at teardown, only `close()`/`abort()`
  afterwards, `close()` tolerates mid-flight drops, dropped `commit()` is
  the commit-ack-loss case covered by S7.5 plus `recover`.
3. **B3 per-method cancellation - RESOLVED.** The per-method table is
  present (status/wait observe-only; trigger stays queued; shutdown/cancel
  run to convergence and re-attach; start never half-starts; runner
  shutdown resumable), the v2 precedence rule is preserved, and A6's
  `trigger_checkpoint` rustdoc states the request is NOT dequeued on
  caller cancellation.
4. **EpochIdempotent evidence - RESOLVED.** A5 freezes
  `EpochIdempotent { mechanism: String, retention: RetentionClass }` with
  `RetentionClass::{Unbounded, Bounded}`; `delivery_capability()` is
  sampled once at binding validation and frozen into the plan; the basis
  travels into the plan record, the manifest (G3's three-shape
  `sinks.*.delivery`), and S9.4 reporting; the spoofing paragraph now says
  a false claim damages the *user-visible guarantee*. Spec S9.2/S9.4 carry
  the basis and sampled-once rules. One boundary clause is missing on the
  spec side - see new Block N1.
5. **recover() contract - RESOLVED.** A5's `recover` rustdoc states the
  commit-outcome query (from `SinkPreCommit` plus the external system
  alone) and the orphan-cleanup obligation for staged artifacts no manifest
  records; spec S7.5 names the crash-abandoned-epoch recovery pass and
  section 9 assigns the contract to the note.
6. **C2 options split - RESOLVED (via the identity-hash alternative).**
  NFR-5: connector options stay out of the fingerprint; the manifest
  records a connector-declared source identity hash over
  data-semantics-affecting options; recovery fails closed on mismatch;
  secret references feed neither hash. C2 freezes the two declared classes
  and the `sources.*.identity_hash` placement; G3 defines it as SHA-256
  "computed identically at compile time and at recovery"; sink target
  identity rides in `sinks.*.pre_commit`.
7. **Secret error sanitization - RESOLVED.** I4 gained the wrapping
  sentence (sanitized message plus connector identity); C2 freezes the
  connector-facing obligation; AC-I adds the forced-failing-open canary
  redaction test.
8. **Lineage exclusivity - RESOLVED.** Spec D9.6; A6 `start()` Conflict on
  per-runner and per-lineage (any runner, any process) before any source
  opens; B2 documents the native lease for Python (second runner over one
  plan raises `calc_flow.CompileError`).
9. **I5 connector tasks - RESOLVED (prohibition chosen).** I5: "Connector
  implementations MUST NOT spawn Tokio tasks"; client OS threads are
  external resources released by `close()`. A4 states the rule and extends
  it to sinks and operators; A5 repeats it for sinks.
10. **Capture-cost rule - RESOLVED.** A2.2 states `checkpoint()` MUST be
    O(dirty-key metadata): pre-encoded segment buffers maintained
    incrementally on mutation, capture is a cheap handoff, never a bulk
    encode on the executor thread; the M4.4 paused-time RED line is
    present. I6 now lists "bulk serialization and encoding" and names the
    cheap-metadata-handoff rule.
11. **Orphan-collection race - RESOLVED.** Spec D4.5 freezes the (i)/(ii)/
    (iii) retained set and coordinator serialization; A7's
    `collect_orphans` doc repeats it verbatim; FR17 names it.

## Round-1 Minors: 14/14 resolved

m1 naming map (same-name replacements paragraph plus explicit catch-all,
including the `CheckpointStore` trait swap, `StreamSink::write`'s dropped
context parameter, and `OperatorDefinition`'s disposition); m2 A2 default
bodies now shown; m3 checkpoint route bound (`checkpoint_route_wait_seconds`
default 60, 200 `CheckpointResponse` vs 202 `CheckpointAccepted`, completion
observed via SSE `job.epoch`); m4 trigger-cancellation rule (stays queued);
m5 reaper JoinErrors logged/counted, never fail the new job; m6
`SecretResolver` seam assigned to M6.1 (note open questions item 1); m7
`capabilities()` adapter validation rules (TypeError/ValueError per key
class, no invented defaults); m8 G1 bounded-key note (64 KiB, M4.3 owns)
and float-grouping note with spec section 9 assignment; m9 S10.3/FR23/
AC-S10/A4.1 aligned to "compilation or runner construction, always before
any source opens"; m10 Python example now asserts
`outcome.completed_epoch > epoch` with the S8.2/D9.2 rationale in a
comment; m11 B3 store paragraph (v2 store message and unsuffixed-async/
`*_blocking` pattern kept, frozen pairing applies to the v3 surface only);
m12 Python plan lease; m13 fresh-lineage manual procedure assigned to M7.3
(spec section 9 and note open questions item 4); m14 SSE-through-`sync:api`
verification assigned to M6.9 (open questions item 5). Round-1 questions:
Q1, Q2, Q3, Q5 answered by the revisions; Q6 answered in A3's new "Lineage
context" bullet (restore/reset deliberately receive no lineage context).

## New findings in round 2

### Blocking Issues

- **N1. Spec S9.2 (and the S9.1 matrix cell) do not carry the
  unbounded-retention qualifier that A5 and G3 enforce - the two artifacts
  now disagree on the exactly-once compile boundary.** A5 states: "A
  `Bounded` retention class cannot satisfy `delivery = "exactly_once"`
  (compilation fails naming the configuration path)", and its spoofing
  paragraph requires "a transactional or unbounded epoch-idempotent sink".
  G3 states a bounded-retention record is "never [reported] as exactly-once
  (S9.4)". But spec S9.2 requires only "a transactional sink or an
  epoch-idempotent sink carrying a declared basis (mechanism identifier
  plus retention bound, recorded per S9.4)" - a bounded-retention sink DOES
  carry a declared basis, so under S9.2 it satisfies the sink capability
  and the project compiles. S9.1's matrix cell likewise reads
  "Transactional or epoch-idempotent" with no retention qualifier. A plan
  with a bounded-window dedup mechanism (ClickHouse's token, M6.5's
  explicit non-goal) requesting `exactly_once` therefore compiles per the
  spec and fails per the note. This is a direct cross-artifact
  contradiction on the single most safety-critical boundary in the
  program, and the M0.2 acceptance gate ("no definition in those artifacts
  contradicts this specification"; spec section 8's own gate) forbids it.
  - **Suggested fix (spec owner, one clause plus one cell):** in S9.2,
    change the sink clause to "a transactional sink, or an epoch-idempotent
    sink with an unbounded retention class and a declared basis (mechanism
    identifier plus retention class, recorded per S9.4); a bounded
    retention class fails this requirement with the configuration path
    named"; in S9.1's third-row sink cell, change "Transactional or
    epoch-idempotent" to "Transactional, or epoch-idempotent with
    unbounded retention". No note change is needed - A5/G3 already state
    the stricter rule.

### Minor / Style Notes

- **N2. The Rust happy-path example no longer compiles under A6's own
  signature.** `shutdown` is now `&mut self`, but the example declares
  `let runner = StreamingRunner::new(...)?;` (immutable binding) and ends
  with `runner.shutdown().await`. Fix: `let mut runner = ...`. Examples in
  this repository are compiled artifacts; this one would fail
  `cargo build --examples`.
- **N3. F1's trailing "flagged for the critic" paragraph is now stale.**
  The spec sentence it references has been rewritten per round 1's required
  edit (the new S7.3 text explicitly disclaims ordering and defers to the
  note), so the paragraph that says "this note reads that list as
  non-ordered ... Flagged for the critic as the one sentence in the spec
  whose prose order my choice reorders" now misdescribes the spec. Replace
  it with a one-line reconciliation note ("S7.3 revision 2 defers the
  ack/forward order to this decision; the constraint it freezes -
  snapshot before both - is satisfied by step 1") so M5.3's implementer
  is not sent chasing a friction that no longer exists. No severity beyond
  editorial.
- **Open question carried from round 1 (non-blocking):** the
  operator-scoped lazy `DataFusionRuntime` of plan section 2.2 (one per
  table stream-operator task, none for array-only chains) appears in
  neither artifact; M2.3's RED list covers it, but neither artifact says
  that placement is deliberate. A one-line pointer in the note's open
  questions would close it.

## Round-2 consistency sweep (mandated cross-checks)

All other cross-references verified consistent, with citations: spec
section 1/S2.2 cursor envelope vs A4 `Cursor` (verbatim mechanism match);
spec D5.4 ownership shape vs A6 (`Arc` core, `&mut self` shutdown, Drop
freeze, escape-hatch observability - match); spec S7.5 recovery pass vs A5
`recover` rustdoc (match); spec I6 capture-cost sentence vs A2.2 rule
(match, including "they are already blocked at alignment, S7.3"); spec
D3.5.6 vs B1 sync-Python paragraph (match); spec D9.6 vs A6 `start()`
(both lock the lineage for the job's lifetime and fail with `Conflict`
before any source opens - match); spec D4.5 vs A7 `collect_orphans` (the
(i)/(ii)/(iii) retained set is verbatim - match); spec S10.3 vs A4.1
(match); NFR-5 identity hash vs C2 two-class rule vs G3
`sources.*.identity_hash` (match, including "secret references feed
neither hash" on both sides); S9.4 reporting vs A5 sampled-once rule
(match); the clarified S7.3 sentence vs F1's step order (snapshot precedes
both ack and forwarding - satisfied by F1 step 1; the mandated round-1
edit is present in substance); B3 per-method table vs A6 rustdoc set
(match, including the runner-`shutdown` resumability row).

## Round-2 verdict

Round 1: 4 Blocks, 11 Majors, 14 Minors - all resolved in substance except
as listed above. Round 2 introduces 1 new Block (N1, one clause in the
spec), 1 new Minor (N2, one keyword in the example), and one editorial
item (N3).

**BLOCKS REMAINING: 1**

---

# Round 3 - focused confirmation pass

Both round-3 micro-revisions were verified from disk against the round-2
prescriptions.

## Per-item verdicts

- **N1 (Block) - RESOLVED.** Spec S9.2 now reads: "checkpointing, and a
  transactional sink, or an epoch-idempotent sink with an unbounded
  retention class and a declared basis (mechanism identifier plus
  retention class, recorded per S9.4); a bounded retention class fails
  this requirement with the configuration path named." The S9.1 third-row
  sink cell now reads "Transactional, or epoch-idempotent with unbounded
  retention" (table repadded to the widened column per the markdown style
  rule; other cells unchanged). The spec rule is now identical to A5's
  ("A `Bounded` retention class cannot satisfy `delivery = "exactly_once"`
  (compilation fails naming the configuration path)") and G3's
  ("never as exactly-once (S9.4)"); the spec/note compile-boundary
  contradiction is eliminated and no new inconsistency with A5, G3, or
  S9.4 was introduced. The spec Status line's revision-3 provenance
  sentence accurately records the change and adds no semantics.
- **N2 (Minor) - RESOLVED.** API note line 1971 is now
  `let mut runner = StreamingRunner::new(`; no immutable `let runner`
  binding remains, so `runner.shutdown().await` type-checks under A6's
  `&mut self`.
- **N3 (editorial) - RESOLVED.** F1's tail now reads: "Reconciliation with
  S7.3 (settled in critique round 2): S7.3 revision 2 explicitly disclaims
  an ack/forward ordering and defers it to this decision; the one
  constraint it freezes - the snapshot precedes both the acknowledgement
  and the forwarding - is satisfied by step 1." Accurate; no "flagged for
  the critic" text remains.
- **Open question (DataFusionRuntime placement) - CLOSED.** The note's
  open questions item (7) records that the operator-scoped lazy
  `DataFusionRuntime` is deliberately absent from the note as an
  implementation detail with M2.3's RED list as its intended home. No
  semantic change.

## New findings in round 3

None.

## Round-3 verdict

All Blocks from all rounds are resolved. The M0 artifacts - specification,
API note, and critique, slug `continuous-streaming-runtime` - are
consistent, contradiction-free, and cleared for the wrap-up docs PR.

**BLOCKS REMAINING: 0**
