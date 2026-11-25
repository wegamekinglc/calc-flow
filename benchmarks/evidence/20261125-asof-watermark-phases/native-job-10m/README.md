# Rust StreamingRunner backward ASOF at 10M rows per side

Release 7012b5b4 and #733 dce77f04 ran the same compiled project in the public Rust StreamingRunner on one logical CPU. Each of the ten chunks per input contains one million rows and 64 UTF-8 keys. The managed job has two ready bounded inputs, source watermarks and a collecting Arrow sink. The clock spans ready publication through the last assembled two-column table. The full output oracle and terminal state checks follow that boundary.

Both rounds completed. Four warmups were excluded. Every one of the 44 fresh worker processes checked all ten million rows in order, exact prices and sequences, both source completions, ten million matches and right evictions, and zero remaining rows or budget failures.

- base p50 1041.101 ms and p95 1143.021 ms.
- head p50 1842.026 ms and p95 1976.173 ms.

Maintained paired verdict: regression. Paired round evidence: `[{"median": 75.46376447711425, "low": 69.32306121650547, "high": 78.44852542200253, "coverage": 0.978515625}, {"median": 77.86184060368863, "low": 75.55345886210067, "high": 83.63189597081993, "coverage": 0.978515625}]`.

Each executable also passed one full qualification before the formal clock. Group acquisition wall time 79.49 seconds. The separate public Python four-thread result and the native 100K callback use different boundaries and workloads.

[Replayable sources, checksum manifest, build receipts, observations, oracle and host logs](asof-native-job-10m.tar.gz) (SHA256 `b999b58666bc1456ab02961555957c14352eafbcd19cc1975f4f178dcc5986fc`). Extract in a fresh directory. With the repository on PYTHONPATH and the locked benchmark environment active, run `python verify.py obs-job-*` to recompute the result.
