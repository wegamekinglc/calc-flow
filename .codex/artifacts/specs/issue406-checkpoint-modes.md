# Issue 406: explicit checkpoint modes

This increment continues PR #407 after `3588cbd6`. It adds an explicit disabled
lifecycle and complete finite-lifecycle measurements. Default behavior remains
checkpoint-enabled. It does not change project-v3, Studio, snapshot formats, or
semantic fingerprints.

## Contract

- Rust and Python `StreamRuntimeConfig.checkpointing` defaults to `true`.
  Python accepts only `bool` and appends the field after existing positional
  parameters. Runtime configuration remains distinct from semantic identity.
- Rust keeps `StreamingRunner::new` and adds `without_checkpoints`. The latter
  owns no checkpoint backend and starts with `checkpointing=false`.
  Configuration/backend mismatches fail explicitly. Python chooses the same
  mode through config: enabled requires a managed backend; disabled requires
  `checkpoints=None`. The native binding validates this boundary independently.
- Expression and Program streams accept the same config and do not create a
  temporary checkpoint directory when disabled. Python connector-owned project
  plans keep their existing prohibition on runtime overrides.
- Disabled jobs open no checkpoint storage, restore no state, run no checkpoint
  coordinator or barriers, and create neither manual nor terminal snapshots.
  Manual requests return the existing explicit unsupported-mode error.
- Disabled jobs reject exactly-once requests, transactional/epoch-idempotent
  sinks, and immutable source history before connector lifecycle calls. Ordinary
  output's effective delivery is best effort. Requested delivery remains visible.
- Mode is fixed before tasks start. Live state, immutable inputs, V1 charges,
  budgets, watermarks, eviction, cancellation, and output order retain their
  contracts. Join/ASOF recovery journals, SQL dirty records, and Window snapshot
  retention are skipped or released when checkpointing is disabled.
- Natural EOF drains final operator output and closes owned resources; it ends
  as completed/natural_end with no completed epoch in disabled mode. Cancellation
  remains cancellation. Successful enabled EOF must not report checkpoint failure.
- Keep the existing checkpoint-status shape. Zero completed epochs is not proof
  that checkpointing is disabled; configuration records the selected mode.

## Verification

Use focused failing tests before implementation for public API validation,
ordinary EOF/cleanup, unsupported delivery/history preflight, preserved SQL
budget, and checkpoint-only operator bookkeeping. Cover Python explicit,
expression, and Program entrypoints, no temporary directory, and early exit.
Retain focused enabled checkpoint/recovery controls. Full CI remains separate.

## Measurement plan

Fix one interval Join workload at 200k rows per side and 8,192-row batches,
retaining the maintained workload, event order, Arrow oracle, and paired
statistics. Five cases: baseline low (24h), candidate low, baseline on (100ms),
candidate on, and candidate disabled. Each case has two rounds of ten samples
and one warmup per round, for 110 jobs. Run fixed alternating order. The total
budget is 600 seconds including fixtures, correctness, startup, warmup, and
cleanup; record builds separately. Stop at the budget without extending cases
or retrying for favorable results.

Baseline is `3588cbd6` plus the isolated successful-EOF status correction;
candidate contains the same correction and the disabled-mode implementation.
Freeze source/patch/native hashes. There is no historical disabled baseline.

Time from sources ready before first data through output collection, EOF,
natural completion, settled checkpoint status, final Arrow concatenation, and
owned cleanup. Report output and drain components separately. Validate output
outside that timer but inside the total budget. Record actual nonterminal epochs
and evidence of nonempty state; never count a terminal epoch as periodic work.
Do not use pauses, manual checkpoints, or sleeps to manufacture coverage.

These are complete finite-lifecycle diagnostics. They do not establish the
separate 20-epoch steady-state target or the Polars target. Preserve insufficient
checkpoint coverage as an inconclusive on-mode result. Keep disabled, low, and
on results separately labelled, and retain the previous investigation's goals:
disabled P50 at most 1.25x same-window Polars 1T; sustained on throughput at
least 90% of disabled at 1s and 75% at 100ms. These remain unverified goals.
