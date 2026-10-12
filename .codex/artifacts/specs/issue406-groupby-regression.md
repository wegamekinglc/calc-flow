# Original-wheel 10M group_by diagnosis

The October 12 morning suite and one sealed-wheel recovery both find a 10–12%
regression between `89e51141` and `8ee5b71b`. Correctness passes. Unchanged window
source and a local WSL non-reproduction do not explain or dismiss the failure.

Before collecting more evidence, fix this narrow scope:

| Case   | Product inputs                         | Connection to the failure                   | Fixed observations               |
|--------|----------------------------------------|---------------------------------------------|----------------------------------|
| A/A    | Original baseline wheel twice          | Check process/host bias                     | 2 rounds × 10 pairs + warmups    |
| B/B    | Original candidate wheel twice         | Check process/host bias                     | 2 rounds × 10 pairs + warmups    |
| A/B    | Original baseline / candidate wheels   | Reproduce the failing binary comparison     | 2 rounds × 10 pairs + warmups    |
| A      | Original baseline native module        | Locate baseline instruction-address costs   | One 30s CPU sampling window      |
| B      | Original candidate native module       | Locate added instruction-address costs      | One 30s CPU sampling window      |

All five cases use only the historical 10M `calc-flow-stream/group_by` workload.
They reuse the common harness, inputs, oracles, bounded subprocess ownership,
AB/BA order and paired regression rule. Keep two rounds, ten pairs, one warmup
per side per round, and the existing +5% threshold; never filter or retry samples
to obtain a favorable verdict. A completed diagnostic can report a regression.

The collection budget is 600 seconds, including wheel hashing/installation,
fixture creation, preflight, warmup, oracle checks, startup and cleanup. Work is
cancelled at 550 seconds, reserving 50 seconds to settle owned tasks. Native build
estimate: zero; original artifacts are reused. Workflow bootstrap, dependency/tool
installation and artifact transfer are separate setup, outside measured workloads.
Retain partial evidence on timeout/failure. Dispatch once after focused validation
and specialist review; do not dispatch another full shard or complete suite.

Pin the original run `38178694560`, artifact IDs `11716974975` and `11716848795`,
full source/wheel/native seals, Python 3.13.16, unchanged dependency lock, and
common harness SHA256 `e30672098c018ef1a650492403bf5ae4813d0182e2b4921cb9233b4d26b8b2e6`.
Require actual GitHub-hosted Linux with four logical CPUs and full affinity.
Preserve the existing Tokio/Polars 32-thread settings and reject native-build
overlap. Record CPU model/cache details, environment, all IPC, input Arrow hashes,
process exits, worker completion and 250ms resource observations.

This workload enables checkpoints with a 24-hour interval. Its ready-to-output
timer excludes the terminal checkpoint, but successful final job completion is
still required. Label it LOW, never OFF. There is no new runtime memory pool cap.

Sample exact original stripped modules with Linux perf at 49Hz, `cpu-clock:u`,
strict frequency and DWARF stack unwinding with 16KiB stack dumps. Never rebuild
with debug/frame-pointer flags and call it the original binary. The isolated CI
runner temporarily sets `perf_event_paranoid=1`, records its previous value and
restores it on shell exit. No fallback frequency or profiler is selected after a
failure. Keep raw `perf.data`, raw/decoded events, native maps, tool/version and
command logs. Reject lost/throttled traces or missing native instruction addresses.

Original wheels have no static symbol table. These profiles locate addresses;
they cannot claim resolved Rust functions or causal attribution yet. Any later
symbol mapping must prove matching executable bytes/layout, or identify a new
build and reproduce its own comparison before attributing the original regression.
Profile workloads include work outside the paired timer; their timing samples
never enter A/A, B/B or A/B. Perf does not supply a direct sampling-lag metric.
Arrow IPC fixture files are local collection intermediates; artifact upload keeps
their hashes, deterministic generator provenance and IPC requests/responses.

Required validation: focused RED/GREEN tests for seals/scope, loss detection,
permission refusal, preflight failure and cleanup; existing diagnostic tests;
Ruff and workflow validation; final specialist review. Production code and the
benchmark gate remain unchanged until evidence supports a specific fix.
