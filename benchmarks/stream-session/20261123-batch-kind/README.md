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
