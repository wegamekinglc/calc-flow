# One-row public session phase diagnostic

The public dual-SMA Program processed 4,096 history rows followed by 64
individual pushes with explicit watermarks. Both observation modes used the
same official main wheel at `37dd0ab7` on a dedicated hosted VM. With the
observer off, the complete push took 59.621 µs at p50 and 80.661 µs at p95.
The steady-state goals are 20 and 40 µs. [Fixture and public clock](https://github.com/wegamekinglc/calc-flow/blob/b1efe76a675e7ec3da269f59ba0b781451e4cb19/benchmarks/stream-session/20261123-public-steady-push/protocol.md) · [Registered observer source](https://github.com/wegamekinglc/calc-flow/blob/0da2427bc49f1041b18f0232bb9bfa10baaefbab/benchmarks/steady-phase-protocol.md)

| Public push under observation                        | p50, µs | p95, µs |
| ---------------------------------------------------- | ------- | ------- |
| Input mapping, watermark and prepared batch handling | 8.646   | 10.088  |
| PyO3 call, native execution and Arrow record export  | 48.230  | 66.329  |
| Python table and named result construction           | 5.450   | 6.132   |
| Whole call with the observer enabled                 | 62.326  | 81.992  |

The observer records exactly one matching native call for every push. Each
raw triple sums to its individual whole-call clock. The percentiles in the
table summarize the respective distributions separately. The held input
already contains a native table Batch. Python input handling checks that
envelope's kind. The PyO3 call then takes an owning clone for native execution
and exports emitted Arrow records before returning. [Worker](https://github.com/wegamekinglc/calc-flow/blob/0da2427bc49f1041b18f0232bb9bfa10baaefbab/benchmarks/steady_phases.py)

The observer added 4.32% [3.62%, 5.14%] and 4.70% [2.95%, 6.02%] in the
two rounds. These exact 97.85% intervals compare the two observation modes.
The separate repository calibration measured 0.436 µs per callback pair.
All 44 fresh processes checked the history, the 64 complete returned rows,
input immutability and the empty flush. Each mode contributed 1,280 formal
public calls. All 42 CPU, memory and paging intervals passed. The hosted job
took 63 seconds within the registered 150-second job and 180-second total
allowances. [Worker observations](phase-workers.jsonl) · [Percentiles, source and wheel identities](phase-result.json) · [Hosted job](https://github.com/wegamekinglc/calc-flow/actions/runs/45187056499/job/137735305230)

## Reading the native batch kind

For a prepared Python table Batch, the public input adapter reads
`Batch.kind` before the binding takes the owned envelope used by the native
session. The earlier getter cloned the envelope under its read lock to
inspect its payload tag. Each table envelope clone retains the Arrow schema,
record-batch backing storage, byte-estimate cache and metadata. The getter
now reads the tag under the same lock. This keeps table and array results
and the cleared-batch error while avoiding temporary shared owners.

A binding test reads the actual Python property 64 times on a live table
Batch. Its thread-local test counter recorded 64 requests for an owning
clone before the change and zero afterward. The subsequent owning clone for
each native push remains necessary. [Uninstrumented public comparison protocol](protocol.md)

## Complete push without instrumentation

| Pooled single-row latency | Baseline | Borrowed kind | Target |
| ------------------------- | -------- | ------------- | ------ |
| p50, µs                   | 60.243   | 60.032        | ≤20    |
| p95, µs                   | 79.9105  | 79.6515       | ≤40    |

The worker-level paired medians changed by −0.39% [−2.00%, +1.95%] and
−0.62% [−2.13%, +0.28%]. The intervals include zero and the maintained
rule returned `no-confirmed-regression`. The structural reduction did not
establish a public latency improvement. The candidate exceeded the p50
and p95 targets by 40.032 µs and 39.6515 µs. [All 44 worker records](workers.jsonl) · [Paired intervals and complete resource samples](result.json)

The baseline used the official main `37dd0ab7` release. The candidate used
`8d4f913f` on top of `79ff363f`. The intervening commits changed the grouped
SUM benchmark example and repository reports. Both wheels used the same
runtime sources and package build inputs apart from the binding getter and
its test. Their verified build image, Rust 1.94.0, linker and GCC
identities matched. The candidate wheel job took 23 minutes 23 seconds.
[Baseline wheel](https://github.com/wegamekinglc/calc-flow/actions/runs/45182357543/job/137719754356) · [Candidate wheel](https://github.com/wegamekinglc/calc-flow/actions/runs/45189825472/job/137744913093)

Each arm contributed 1,280 formal public push clocks. All 44 processes,
including four warmups, validated the complete 4,096-row history and 64
additional rows for a total of 183,040 checked rows. All 42 resource
intervals qualified. The full hosted job took 74 seconds within the
240-second job and 300-second total budgets. The two same-wheel observation
modes above and this uninstrumented product comparison used separate hosted
VMs. [Registered source](https://github.com/wegamekinglc/calc-flow/commit/e93d5f5398f243acc6522b02b10ce650e77b2a44) · [Hosted job](https://github.com/wegamekinglc/calc-flow/actions/runs/45194346131/job/137760691293)
