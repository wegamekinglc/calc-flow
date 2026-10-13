# Issue 406: compaction report timing scope

## Finding

The original fifth acceptance target is Join handler latency during compaction,
at most twice steady latency. The maintained native component completes
`prepare_checkpoint_async`, including worker completion, rebase installation and
release cleanup, before starting its `process_data` clock. Base/delta witnesses
prove that rebase occurred, but not that the handler overlapped it. Preparation,
handler and combined clocks remain useful separate diagnostics.

The acceptance report nevertheless compared the post-compaction handler ratio
with the original 2× goal. It had also changed the target title to "after
compaction". Independent source review confirms this boundary mismatch. PR #420
does not measure or claim completion of this target; its complete benchmark and
ordinary-release results remain separate, with their original identities.

## Correction

Restore the original during-compaction target title. Leave its target ratio null
and status unmeasured. Preserve baseline and candidate post-compaction ratios,
case identities, paired revision statistics and all raw phase observations as
explicit diagnostics in JSON and Markdown. Neither combined time, a rebase
witness nor an invented overlap flag establishes the missing original metric.
The remaining seven targets and the historical baseline OFF protocol are
unchanged. A real during-compaction measurement remains required before final
eight-row acceptance.

## Verification

The focused regression fails on the original report: a 1.5× diagnostic is marked
within target and a 3× diagnostic above target. After correction both remain
diagnostic, and the original target is unmeasured. Additional coverage preserves
missing-control behavior, all eight rows, input immutability, both revision arms
and the other seven target results. All 35 focused report, fresh-report and native
component tests pass. Scoped Ruff lint and formatting pass. No native build or
performance measurement is needed for this report-only correction. Final
implementation review is Approve, with no findings. Required CI remains pending.
