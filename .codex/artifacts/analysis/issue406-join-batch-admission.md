# Issue 406: Join batch admission evidence

Base: `bb142320062995aeb318ad406c901288537fd766`. This is phase 1.1 work
on `feature/issue406-join-batch-admission`; all eight targets remain active.

The [spec](../specs/issue406-join-batch-admission.md) incorporates the bounded
inline, funded spill, workspace-error priority, and detach-or-decline ownership
rules from the [critique](../critiques/issue406-join-batch-admission.md).
The design verdict is Proceed with caveats; it approves no implementation.

## First structural RED

`cargo test -p calc-flow --lib
operator::join::tests::optimization_tests::test_batch_admission_does_not_stage_owned_rows
-- --exact --nocapture` fails for the expected reason: the two-record, 130-row
input creates 130 `AdmittedRow` records, while the required count is zero.
The harness returns 101 after 418.320 seconds (compilation 6m57s, test 0.01s).
The core test executable reports 2039 filtered tests; only this focused control
runs. Source hashes, environment and command are retained in
`target/issue406-join-batch-admission-20261014/dense-admission-red.json`, with
full compiler/test output in `dense-admission-red.log` beside it.

This counter observes calls to the existing owning row append boundary. It is
not an allocator count, RSS result or performance verdict. GREEN must also
satisfy independent segment/parent ownership checks and genuine sparse/generic
SQL behavior controls; merely removing the counter is insufficient.

Implementation, remaining RED/GREEN controls, scoped lint and final specialist
review are pending. No stopped performance acquisition is resumed here.

## Reviewed delivery publication

The separate reviewed branch `feature/issue406-second-stage` remains frozen at
`bb142320`. The user explicitly authorized push, PR creation, and merge once
final review and all required CI pass. The attempted `git push --set-upstream
origin feature/issue406-second-stage` was rejected before process creation:
`approval required by policy, but AskForApproval is set to Never`.
No push, PR or new branch CI was created by that attempt. Exact publication
state is recorded in `target/issue406-integration-20261014/publication-state.json`.
PR #421 remains already merged. This tool restriction does not complete issue406.
