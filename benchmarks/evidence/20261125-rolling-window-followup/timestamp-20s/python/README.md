# Public twenty-second timestamp rolling mean at ten million rows

Installed 11.14 7012b5b409553df04e29e6d7f9a11abf919cbf93 and merged #740 main e1d9ccd3fd72cb20668e06891cb49e1fbfd4d75a ran the same single-threaded StreamingRunner with checkpointing disabled, 64 interleaved string keys, ten one-million-row input batches, a watermark after every batch and a final EOF. The public ts.mean uses duration(20_000_000) and min_periods=1 over (t−20s,t]. The input timestamp is tick+floor(tick/7) seconds after 2026-01-01T00:00:00Z. Time starts at ready source enqueue and ends when the final Arrow table has been delivered and concatenated. The last native watermark is one microsecond before the last input time, so the timed feed includes source EOF. The exact product Cargo.lock is preserved for each revision. [Wheel identities](wheel-identities.json)

| Clock | Baseline   | Candidate  |
|-------|------------|------------|
| P50   | 414.299 ms | 425.506 ms |
| P95   | 433.306 ms | 455.683 ms |

| Round | Paired median | Exact interval   |
|-------|---------------|------------------|
| 1     | +5.37%        | [+2.72%, +6.76%] |
| 2     | +3.72%        | [-1.15%, +5.57%] |

The maintained verdict was **inconclusive**. Two rounds each completed ten alternating AB/BA pairs after four excluded warmups. Every invocation validated all ten million ordered sequence identities and rolling means outside timing against the independent per-key NumPy oracle. The reference uses integer eighths and per-key timestamp boundary search, with 10,000,000 finite values. The ordered five-column schema, all input columns, every rolling mean, ten watermarks, the sink closure, its physical row sequence and post-EOF total are checked in every invocation. The largest observed absolute error was 5.684e-14.

The Linux WSL2 host exposed 32 logical CPUs and 16 SMT groups. Each worker used one Tokio thread and one physical core, chosen during the three-second preflight. Each one-second interval checked every selected core's SMT sibling, host utilization, memory, swap and major faults. This run recorded 70 intervals and took 76.207 seconds within the controller and 76.29 seconds including interpreter startup and exit, against the 180-second limit. [Result and resources](result.json)

Each independent comparison reserved 180 seconds including installation, identity qualification, each worker startup, fixture construction, full correctness verification, sampling and shutdown. Release builds were accounted separately. The [original evidence archive](raw-evidence.tar.gz) contains every recorded worker response, condition, qualification record, build manifest and build log. Its [receipt](receipt.json) includes a hash and byte count for each original file.

To reproduce the verdict from the archive on the harness checkout, extract it at the checkout root, then run

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.benchmark_suite.report import comparison
print(comparison(json.loads(Path('observation/samples.json').read_text())))
PY
```
