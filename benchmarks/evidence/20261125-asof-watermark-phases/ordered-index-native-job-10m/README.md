# Rust StreamingRunner backward ASOF at 10M rows per side

Release 7012b5b4 and #737 f8780771 ran the same compiled project in the public Rust StreamingRunner on one logical CPU. Each of the ten chunks per input contains one million rows and 64 UTF-8 keys. The managed job has two ready bounded inputs, source watermarks and a collecting Arrow sink. The clock spans ready publication through the last assembled two-column table. The full output oracle and terminal state checks follow that boundary.

Acquisition stopped after 0 worker observations. The group has no qualified two-round estimate. The reason was `ExceptionGroup('unhandled errors in a TaskGroup', [ValueError("sparse changelog resource gate: {'cpu': 10, 'cpu_busy': 97.0, 'siblings': [11], 'sibling_busy': 0.0, 'host_busy': 3.125, 'available_bytes': 10648334336, 'major_faults': 0, 'swap_pages': 0, 'swap_in_pages': 0, 'swap_out_pages': 0, 'suitable': False}")])`.

Each executable also passed one full qualification before the formal clock. Group acquisition wall time 8.02 seconds. The separate public Python four-thread result and the native 100K callback use different boundaries and workloads.

[Replayable sources, checksum manifest, build receipts, observations, oracle and host logs](asof-native-job-10m.tar.gz) (SHA256 `d911db04ff8a3121321ad01617066c64f824c24e95f2a4c88ad0d930fbb8048d`). Extract in a fresh directory. With the repository on PYTHONPATH and the locked benchmark environment active, run `python verify.py obs-job-*` to recompute the result.
