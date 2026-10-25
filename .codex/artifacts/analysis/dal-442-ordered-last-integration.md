# DAL-442 / T5: account example and compact SQL documentation

Baseline: `ff9f049d481df98da2a6ba22de78175f9dfdc3e6` (merged T4 / #499).
Fetching main confirmed this commit is an ancestor of `origin/main`; the fresh
task branch and main were identical and clean. This stage changes the account
example's runtime configuration and documentation, with no Rust, binding,
checkpoint format, dependency or generated contract changes.

## Delivered scope

Example 31 keeps its original three SQL queries and adds
`sql_compact_state_budget=StateBudget(100_000, 64 << 20)` with checkpointing
disabled. Each of the positions and quotes aggregate nodes independently pays
for its live groups and owned bytes. Custom inputs exceeding those limits now
fail rather than retaining unlimited state. No cumulative input quota is added.
Python still owns two latest Arrow snapshots and executes the reusable batch
valuation on updates; no native multi-input stream SQL is introduced.

The streaming guide and Python API now explain the actual ordered MVP shape,
physical admission and empty-first deferral, fixed mode, parsed sort/null
options, first-winner ties, explicit business sequence, independent budgets,
candidate/capture ownership, scratch/RSS exclusions, complete snapshot output,
and layout-3/4/5 recovery. The introduction, symbolic API and repository
guidance no longer describe every aggregate as retaining/recomputing history.
The `StateBudget` docstring now covers both uses of that existing type.

Runtime evidence is limited to the existing retained-fallback Rust tracing and
explicit operator checkpoint metadata. No new diagnostics, public API or
product execution mode is claimed. No additional runtime gap was found in
the example's existing lowering/configuration path.

## Acceptance and reused evidence

Read before changing this stage: `dal-439-ordered-last-state.md`,
`dal-440-compact-state-budget.md`, and `dal-441-ordered-last-recovery.md` in
this directory. Their exact commands/results remain the detailed records.

| Acceptance area                       | Evidence reused without rerunning                                                                                                                    |
|---------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| Admission, DF55 ordering and state    | T2 `7777b0599570676a04c10a25a0c9403dfc86989d`, merged as `926c4f3`: `cargo test --locked -p calc-flow --lib ordered_last -- --nocapture`, 22 passed. |
| Layout-3 and grouped seeding controls | T2 existing full/delta SUM/COUNT roundtrip, 1 passed; `incremental::grouped_sum::tests`, 2 passed.                                                   |
| Budgets and Python/PyO3 propagation   | T3 `c0b5546250a1bd9c767b829055a3a9d76d084aa9`: 12 Rust cases, 1 PyO3 validation case and 6 installed-wheel Python cases passed.                      |
| Recovery, corruption and faults       | T4 `41f23060c87559f7451b3975b76050ea19e89b60`: 26 targeted cases and layout-3 control passed; unaffected cases reused after correction below.        |
| Null-state correction and final gates | T4 `ba118c4c237dd5918d218879a9ac69f299e11faa`: 11 restore/budget cases and DF55 prefix oracle passed. Final head `df5e3d9` merged as `ff9f049`.      |

T3's six Python cases used `python/tests/test_stream_results.py` with
`-k 'compact or preserves_retained_epochs_position or uses_configured_state_budget'`
(five cases), then the corrected existing
`-k test_stream_compact_budget_applies_independently_to_each_aggregate`
(one case). They cover configuration pairing, actual native rejection, old
quota and independent per-node limits. No need to repeat these unchanged tests.

The coordinator's accepted T4 handoff records all 33 final-head checks passing,
including native Windows/Linux gates, wheel, Codacy and combined Rust 90%
coverage. This is prior-stage CI evidence, not a claim about T5 CI. The ASOF
resource-contention explanation remains a working diagnosis; this stage makes
no related product change or expanded documentation claim.

## New focused validation

The example now supplies an explicit budget, so its existing small-input
checks were run again. CPython 3.13.9 reused the T3 installed wheel from
`dal-440-7a65d58701e8/workdir/calc-flow/target/python-current`.
Both `calc_flow.__file__` and `calc_flow._native.__file__` resolved there,
rather than the unrelated editable installation in the interpreter's default
path. All 43 installed Python/native files matched the wheel byte-for-byte.

- Wheel SHA256: `bda1d56b953ed9b4ddbd375f5edbf2316fb3773cba8cbb8f202e42907c9bec84`.
- Native SHA256: `75a00246901e8066c6a3a0586bc27dbee033cf02825f571a0065b608a0e36f1c`.
- `git diff c0b5546250a1bd9c767b829055a3a9d76d084aa9 ff9f049d481df98da2a6ba22de78175f9dfdc3e6 -- python/calc_flow crates/calc-flow-python`:
  empty. This stage's Python library edit is a docstring only. The T3 native
  wheel predates T4's impossible-unset restore correction; the example disables
  checkpoints and never exercises that path. Final recovery relies on the T4
  evidence above, not this wheel run.
- `PYTHONPATH=/home/wegamekinglc/multica_workspaces/dal-4ee34dfed184/dal-440-7a65d58701e8/workdir/calc-flow/target/python-current JAX_PLATFORMS=cpu python -m pytest python/tests/test_account_market_values_example.py -q -k 'not feeds_use_requested_default_scale'`:
  4 passed, 1 deselected in 2.13 seconds. Covers live repricing, stale quotes,
  continued quotes after trade EOF, zero positions, final CLI result and awaited
  source cleanup. The CLI's 80-trade/20-quote case still yields two accounts at
  23,400 each. The unchanged ten-million-row feed generator case was not run.
- `uv tool run --from ruff==0.16.0 ruff check examples/31_streaming_account_market_values.py python/calc_flow/runtime.py`:
  passed.
- `uv tool run --from ruff==0.16.0 ruff format --check examples/31_streaming_account_market_values.py python/calc_flow/runtime.py`:
  passed, both files already formatted.
- A one-off `python` AST/content check passed: all three example SQL constants
  are identical to the baseline; changed Python syntax, Markdown final newlines
  and whitespace are valid; all six added relative links/anchors resolve. The
  acceptance table was aligned to the repository's Markdown convention.
- `git diff --check` and
  `git diff --exit-code -- Cargo.lock schemas/project-v3.schema.json`: passed.

No new placeholder tests, wheel rebuild, full regression, benchmark, coverage
run, Studio/REST/OpenAPI work or release verification was performed. Full T5
regression and platform gates remain with CI; no merge readiness is asserted.

## Preserved work

The original DAL-431 checkout remains on `agent/cf-implementer/5ceb4eb9b8b4`
with its 21 tracked modifications and four untracked Rust files. It was only
read, not edited, reset or cleaned. Stash
`32edcee9716fcd7a4d0cb595d69b86967fdb7c37` is still present; prior patch/archive
backups were left untouched. No old WIP enters the T5 source or evidence.
