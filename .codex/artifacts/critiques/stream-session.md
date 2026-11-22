# P1 StreamSession - Critic Critique

## Target

- Spec: [stream-session.md](../specs/stream-session.md) (base `39cab2fb`).
- Plan: [streaming core re-architecture](../../../docs/plans/2026-11-15-streaming-core-rearchitecture.md).
- The plan's approved route and goals (decisions 1-6 and the P0-P7 gates) are
  fixed inputs. This critique does not change them.

## Verdict

**Proceed.** The spec matches the approved P1 route. The core design (reuse
`dispatch_message`/`finish_operator`, session edges, one topological pass)
needs no runner-only dependency for the admitted operators. It is a legal
runner schedule, and its budget semantics conflict with no documented
contract. There are no blocking issues. The items below are corrected during
implementation, each with a focused test where relevant, without another
review round.

## Implementation checklist

1. **Ordered + explicit watermark.** After `push(watermark=T)`, a row at `T`
   passes the ordering check, but rolling under the default policy treats it
   as late, which fails the session (`docs/streaming-guide.md:1073-1077`).
   Under `Ordered`, also reject rows at or below the current watermark as a
   validation error. Test: after `watermark=T`, a push at `T` is rejected and
   the session stays usable.
2. **Input metadata.** Stamp `BatchMetadata(binding_id, sequence,
   attributes)` the way `sequenced_source_message` does
   (`source_task.rs:1906-1922`); otherwise the late-output columns
   `_cf_late_source/_cf_late_sequence` disagree with the runner. Test: late
   output matches the runner.
3. **Measurement harness.** On `39cab2fb`, `warm_stream.py` still runs with
   checkpointing on and has no exact projection. When measuring, port the
   #616 harness changes (exact projection, checkpointing off, TA-Lib
   history) from `feature/compiled-stream-region`, then run the acceptance as
   planned. The existing `b9a46e12` baseline wheel can be reused. Add phase
   timers to the candidate samples (input conversion, native call, output
   assembly) so the cost can be located if the ≤20 µs target is missed.
4. **Rust async contract.** Rustdoc states that `push` must be polled inside
   a Tokio runtime; rolling indicator and SQL paths use
   `spawn_blocking`/`yield_now`. Mark the call in flight; if a `push` future
   is dropped midway, the session becomes failed.
5. **Calling `push` inside an event loop.** Implement it as the spec says.
   In the same PR, add a matching exception to the event-loop rule in
   `.claude/rules/code-style.md` and AGENTS.md so the rules and the
   implementation agree.
6. **Python binding state.** Use a state enum (idle, busy, closed, failed) so
   `close()` during a push cannot bring the session back. Raise
   `RuntimeError` instead of panicking when called inside a Tokio context.
   Map session lifecycle errors to `RuntimeError` in the binding, and write
   the actual exception types into the docs.
7. **Panic text.** Reuse `failure::panic_message` (at most 1,024 bytes,
   `runtime-envelope.md:788-793`).
8. **Inputs.** Enqueue on every source route of the binding. The ordering
   check covers every record batch.
9. **Test wording.** Compare against the runner on per-output concatenated
   rows. Compare Union output by branch or after sorting, because the
   runner's cross-ingress order is nondeterministic. Compare the Python
   default-policy test after `finish`.
10. **Wording.** §1/§3 say "no Pending / no thread switch", which does not
    hold for rolling indicators and SQL. Fix the wording to match the actual
    admission scope.
