# Sign CASE comparison: separately authorized retest

The user explicitly requests a new test after the stopped acquisition in
PR #553. This is one new attempt with an independent 600-second acquisition
budget, prior debit zero. Preserve every byte of the previous experiment,
protocol, manifests, products and receipts. Do not reuse its unused budget.

## Controlled products

Baseline is latest main at preparation,
`5a66b6d5cb3f9fa2f6c5d6c8d7ce03db0cea565f`. Restore exactly the three Rust
prototype files from public `b0a9633f44356022fd1aa169b4efa6b077bc2f7b`
on that baseline. The rewrite, fallback guards and tests remain unchanged;
the [original source specification](sum-case-projection-20261106.md) applies.
Retain the actual original RED/GREEN, lint and component evidence, explicitly
identified as historical rather than rerun results. Changed surrounding
semantics require focused diagnosis before admission.

Build both current products with the same common657 instrumentation from
public `84186554b078e25e71003a74fedc91c63b2b7f70`. Publish the exact candidate
source before building it. Ordinary copies of compatible dependency caches
may seed new target directories; never modify the old caches or relabel old
binaries. Require fresh core/example records, actual Cargo/rustc arguments
and environment, source and dependency inventories, copied binary hashes,
pre/post seals and independently reviewed portable proof. Persist actual
Cargo and helper outcomes immediately, before metadata postprocessing.
Estimate builds separately at approximately 15 minutes per product.

## Fixed acquisition

Use the existing maintained paired collector and scientific helpers.
The thin `sum_case_retest` module only replaces the study identity, baseline
and corresponding report description; it preserves every workload and
statistical rule and is itself included in the scientific source seals.

| Case                     | Connection to the rewrite                         | Fixed formal pairs |
|--------------------------|---------------------------------------------------|--------------------|
| stock-stream-1cpu        | Dense mixed CASE, one physical core               | 2 rounds of 10     |
| stock-stream-4cpu        | Same mixed CASE, four physical cores              | 2 rounds of 10     |
| sparse-stream-1cpu       | Sparse updates and alternating uniform masks      | 2 rounds of 10     |
| uniform-buy-stream-1cpu  | Separate control for lost shared-column fast path | 2 rounds of 10     |
| uniform-sell-stream-1cpu | Separate control for extra factor allocation      | 2 rounds of 10     |

Baseline is the above latest main; candidate is its exact published prototype
commit, frozen in the final protocol. Each pair has one observation per
product: 200 formal observations total, plus 20 warmups. Both dense one-CPU
products require six profile jobs, excluding the first; retain five admitted
processing windows per product with complete stacks and loss/unwind evidence.
Each job validates all 21 complete snapshot prefixes using independent
arithmetic/schema/input oracles. Preserve CPU affinity, target partitions,
thread pools, fixture sizes, resource checks and the maintained paired 5%
rule from the original contract.

One 600-second clock includes live quiet guard, fixtures, preflight, startup,
warmups, full oracles, resource collection, profiles and decoding, statistics,
publication, all postseals and owned cleanup. Admission and builds precede
the acquisition; no workload or profiling probe may run before its clock.
Stop and settle owned processes if the environment or evidence is unsuitable
or the deadline is reached. No automatic retry, additional samples, cases,
threshold changes or residual-time campaign. Do not kill competing processes.

Require independent source/build/protocol review and parent GO before the
single acquisition. Accept only complete evidence, at least one exact dense
improved verdict and no confirmed regression among the five cases. Otherwise
withdraw runtime changes, preserving experiment source and actual results.
Inconclusive intervals establish neither equivalence nor confirmed regression.

## Delivery

Keep this experiment under the distinct `sum-case-retest-20261106` slug.
Report completed counts, stopped or missing paths, clocks and uncertainty
accurately, alongside the prior zero-workload attempt. Integrate current main
for delivery and disclose any later unmeasured integration. Final specialist
review and green final-head CI remain prerequisites to merging under the
user's existing authorization.
