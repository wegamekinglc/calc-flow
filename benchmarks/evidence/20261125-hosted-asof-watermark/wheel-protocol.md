# Installed Python ASOF release and watermark attribution

Freeze each input at 10,000,000 rows and 64 keys, using the scheduled default
one-thread managed ASOF workload. Compare release 7012b5b4 with merged
de20f821. The ordinary pair uses exact sealed Linux portable wheels from suite
45623973262. The second pair applies only the registered exclusive-clock patch
to each original parent. Retain original artifacts, process identities, build
flags, dependency and binary hashes. Setup and compilation have a separate
120-minute limit.

Run each pair once on this fresh hosted VM. Each group uses the maintained two
rounds of ten alternating formal pairs plus one warmup per arm per round. Select
one suitable physical core with the registered controller and reuse it for the
second group when the first preflight succeeds. Resource failure or timeout stops
the affected group. Preserve all completed requests. Each cohort has 240 seconds,
with 60 seconds shared allowance for cleanup and coordination. Keep the native
runner experiment and historical local four-thread stops separate.

Demand exact physical output, full expected payload and post-EOF state checks on
both installed arms. The public timer ends when the expected Arrow output is
available. Worker callback sums include EOF and must be compared separately. For
each traced request, verify time closure, counters, PID and exclusive byte ranges.
Report product p50/p95 and paired-rule verdict, trace calibration on comparable
arms, and mean paired candidate-minus-release milliseconds by matching/readiness,
Arrow charge, locate/gather, assembly, other output, left retirement, right
retirement, emission and residual for completed groups.
