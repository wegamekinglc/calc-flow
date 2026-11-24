# Native Join and ASOF regression measurements

Each comparison uses two independently built Rust executables on the same machine, with default product features and identical fixture code. The baseline is main at `92230f68`. The Join candidate is [#726](https://github.com/wegamekinglc/calc-flow/pull/726) at `0b6f2971`. The ASOF candidate is [#727](https://github.com/wegamekinglc/calc-flow/pull/727) at `b076010b`. Each case completed four excluded warmups and two rounds of ten formal alternating pairs. All 88 executions validated their complete output. [Registered workloads and build identities](registration.json)

| Native operator case                   | Baseline p50 | Candidate p50 | Baseline p95 | Candidate p95 | Paired verdict |
|----------------------------------------|--------------|---------------|--------------|---------------|----------------|
| Join with 10,000 one-to-one matches    | 4.706 ms     | 4.601 ms      | 5.009 ms     | 4.924 ms      | Inconclusive   |
| ASOF with 100,000 ordered rows per arm | 52.614 ms    | 35.106 ms     | 57.512 ms    | 37.382 ms     | Improved       |

Join timed the right-side handler with the left input already resident. The six output columns were checked after each call. The paired medians changed by −1.14% and −3.46%, with respective intervals [−6.41%, +7.38%] and [−5.98%, +2.80%]. The unit-level reduction from 4,096 descriptors to one did not establish a handler-wide latency improvement. [Join raw observations](join/observations.jsonl) · [Join statistics](join/outcome.json)

ASOF timed admission on both inputs followed by watermark settlement, output and retirement for 64 interleaved keys at zero tolerance. Each execution verified the row order, matched sequence and value, count and retained-plus-evicted state. Paired medians fell 32.93% and 33.24%, with intervals [−40.56%, −29.69%] and [−35.69%, −31.23%]. The maintained fixture includes allocation accounting inside its timing window. [ASOF raw observations](asof/observations.jsonl) · [ASOF statistics](asof/outcome.json)

The registered local time budgets include fixture preparation, process startup, correctness, resource checks and cleanup. Join used 4.109 seconds and ASOF 6.081 seconds. Cold compilation for both baseline executables took 1,331 seconds, and candidate builds took 777 and 508 seconds, respectively. Build time was accounted separately.

From the repository root, run the offline verifier with `PYTHONPATH=. python3 benchmarks/evidence/20261125-native-join-asof-regressions/verify.py benchmarks/evidence/20261125-native-join-asof-regressions/{join,asof}` in a shell supporting brace expansion. The verifier reads the original worker and CPU logs and reconstructs the paired verdicts, p50 and p95. Each build receipt records the Cargo profile, runtime source and executable digest. [Protocol](protocol.md) · [Fixture adapters](prepare.py) · [Build qualification](build.py) · [Paired clocks](measure.py)
