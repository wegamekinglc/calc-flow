# P1 Convenience-Stream Defaults - Critic Checklist

Target: `.codex/artifacts/specs/stream-convenience-defaults.md` (revision 1).
Baseline `main` `39cab2fb`.

## Verdict

**Proceed with caveats.** The route is sound. One item is blocking because
the spec's restore wording, followed literally, produces a failing job.

Verified OK, no change needed:

- **FIFO order.** A data batch and its watermark stay in order on the source
  edge, because each drain's emissions are dispatched in sequence and drains
  are serialized (`progress/driver.rs:3403-3520`). The checkpoint cut is
  serialized the same way (`driver.rs:3360-3401`).
- **No second emission path.** The direct path only admits `SourceProvided`
  (`runner.rs:3784-3796`).
- **Preflights.** The DISTINCT, ORDER, and ASOF preflights never read
  `emit_interval`.
- **Checkpointing off.** No convenience feature requires managed state
  (`runner.rs:1453-1471`).

## Implementation checklist

1. **BLOCKING: durable restore.** `restore_source` turns `Some(ZERO)` into
   `Some(origin)` (`progress/durable.rs:106-111`, `134`, `148-158`;
   `types.rs:92-100`). `driver.rs:1207-1214` then arms an immediate timer that
   fails in `next_phase_deadline` (`generated.rs:152-158`). In `restore_source`,
   map a zero interval to no delay and add a zero case to the `durable.rs` tests.
2. **Startup arming.** Skip zero intervals at `driver.rs:1389-1396`; otherwise
   `checked_add(ZERO)` arms an immediate timer. Keep the zero error in
   `next_phase_deadline` as a defensive invariant.
3. **Validation.** Accept zero only for `emit_interval` (`prepare.rs:397`).
   Keep the u64-nanosecond representability check from `validate_duration`
   (`prepare.rs:462-473`).
4. **Red-first Python test.** Test 4 is already green on `main`: the 100 ms
   timer fires while the gated iterable waits (`streaming-guide.md:1046`).
   Add a test that captures the default `SourceBinding.watermark_policy` and
   asserts `emit_interval == timedelta(0)` (`_stream_inputs.py:120-126`).
   Keep test 4 as a non-red end-to-end test.
5. **Existing tests that will break.** `test_stream_results.py:718` and
   `:1017` will hit their 5 s timeouts, and `:911` will miss its
   `"root-removed"` milestone. Pass
   `config=cf.StreamRuntimeConfig(checkpointing=True)` to all three.
   Extend `test_checkpointing.py:33` to cover `config=None` instead of adding
   test 5.
6. **Rust test 2 cases.** Add: two batches in one drain giving
   `[D1, W1, D2, W2]`; an equal-max batch and a null-only batch giving no
   watermark; the underflow error on the data path (whole drain fails,
   `driver.rs:1532-1578`); and `idle_timeout` with zero interval (idle timer
   only).
7. **Project-v3 side effect.** An omitted `emit_interval_ms` defaults to 0
   (`config.rs:351`, `schema.json:2182-2187`), so it now runs per batch
   instead of being rejected. Document this in `projects-guide.md` and
   `connectors/kafka.md`. Do not regenerate the schema.
8. **Explicit config re-enables checkpointing.** Any explicit config, e.g.
   `StreamRuntimeConfig(edge_budget=...)`, keeps `checkpointing=True`
   (`runtime.py:773`). State this in the `StreamResults` docstring
   (`stream.py:240-245`) and in `python-api.md`.
9. **Wider focused pytest set.** Add `test_checkpointing.py`,
   `test_snapshot_sql.py`, `test_symbolic_late_output.py`, `test_bounded_*.py`
   and the event-time operator suites. Their default paths switch from
   checkpoint-on to checkpoint-off operator branches.
10. **Docs sweep.** Have `cf-doc-writer` grep for
    `temporary checkpoint|temporary managed|100 ms`. Sites the spec misses
    include `introduction.md:115`, `README.md:66`, `api-reference.md:132`,
    `asof-join-guide.md:463`, `relational-frontends.md:43`,
    `runtime-envelope.md:754-755` and `streaming-guide.md:1046-1047`.
11. **Benchmark note.** The PR and CHANGELOG should note that
    `benchmarks/test_stream_operators.py:190` and
    `test_program_engine.py:63` change baseline: both defaults flip.

## Spec §7 follow-ups

Spec §7 adopts items 1, 4, 5, 7, 10 and 11. Its decision 4 supersedes
item 8. Decision 4 (tri-state `checkpointing`) adds three items:

12. **Decide how a bare `StreamingRunner` resolves the default.** Today,
    `StreamingRunner(plan, s, k, None, config=None)` raises "checkpoints must be
    a ManagedCheckpointRuntime" (`runtime.py:1295-1300`). Resolving `None`
    from "storage passed" makes that call run with checkpointing silently off.
    That contradicts spec §1, which says explicit runners keep their current
    defaults. Either resolve `None` to `True` for explicit runners, which
    keeps the error, or record the behavior change in §1 and CHANGELOG.
13. **Resolve `None` in one place, before the native boundary.**
    - Use one helper, shared by `stream.py:291-296` and `runtime.py:1295-1303`
      (including the project-plan branch at `runtime.py:1281-1292`).
    - `stream.py:317` and `runtime.py:1297,1301` read `config.checkpointing`
      directly. They must see the resolved bool, never `None`.
    - Decide what a bare `StreamRuntimeConfig()._native()` returns.
      `_native()` currently raises for anything that is not a bool
      (`runtime.py:779`), and `test_checkpointing.py:20` and `:157` call it on
      a default config.
    - The native `_StreamingRunner` must keep rejecting `None`
      (`test_checkpointing.py:154-160`).
14. **Tests and docs for the field type change.**
    - `test_checkpointing.py:14-24` asserts `config.checkpointing is True`;
      the default becomes `None`.
    - `test_checkpointing.py:26-30` expects `None` to raise `TypeError` from
      `_native()`.
    - Update both, and add cases for convenience `None → False` and the chosen
      runner resolution.
    - Document the field change (`bool` becomes `bool | None`) in
      `python-api.md` and CHANGELOG. Downstream calc-flow-studio may read the
      field.
