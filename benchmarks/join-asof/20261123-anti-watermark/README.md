# Active-watermark latency for bounded left anti joins

An active right-hand watermark moved from 12 to 13 µs on a warm join.
This made 257 previously pending left rows eligible for output.
The Rust operator delivered nine batches to its in-memory collector in
54.594 µs at P50. The public Python stream returned nine Arrow tables
in 1.406 ms at P50. Each boundary met its registered absolute median
target. These single-arm measurements establish a first latency baseline
for LeftAnti, introduced in [#686](https://github.com/wegamekinglc/calc-flow/pull/686).

| Measured boundary                      | P50    | P95    | Round medians   | Rows per second at P50 |
| -------------------------------------- | ------ | ------ | --------------- | ---------------------- |
| Rust operator callback, µs             | 54.594 | 66.568 | 54.019 / 57.296 | 4,707,520              |
| Python source to final Arrow table, ms | 1.406  | 1.474  | 1.406 / 1.407   | 182,789                |

Throughput divides 257 by the median time for a single release. P95 has
few tail observations across 20 timed releases on each surface.
Each trial used two rounds of ten independent processes plus one excluded
warmup per round. The latency targets were 1 ms for Rust and 5 ms for
Python. The hosts, runtime setup, and clock boundaries differ.

Both trials used product tree `d22e0c76`, main commit `37dd0ab7`.
Each process observed the immediate null-key row before the first cutoff.
At equality an earlier row was released, then a second null-key input
acknowledged completion of that cutoff on the same output. The timed
advance emitted the exact 257 rows in order, in eight 32-row batches and
one one-row batch. An existing right witness suppressed its left match.
After the clock, a later watermark released the final pending row with
both sources still active. The oracle checked keys, times, original
sequence values, nulls, schema, and the absence of surplus output.

The native worker used the public Rust operator, one Tokio thread and
CPU 0 on WSL2. Its clock includes the callback's state commit and
delivery to the in-memory collector. Acquisition used 4.02 of the
registered 240 seconds. The optimized build took 10 minutes 14 seconds.
Both resource intervals passed. The Python job used CPU 2 on a hosted
Linux runner with CPython 3.13 and the official release wheel. Its clock
includes source scheduling, managed edges and public Arrow results. The
whole hosted job took 44 of the registered 300 seconds, including final
artifact upload. All 17 resource intervals passed. [Native build and
observations](rust-native.json) · [Python wheel and observations](python-public.json)
· [Hosted job and raw artifact](https://github.com/wegamekinglc/calc-flow/actions/runs/45187528715)

The published observations used the exact workers in the [sampling
revision](https://github.com/wegamekinglc/calc-flow/tree/242972d76c0857e5f0f036d161617da5685d9fa4/benchmarks/join-asof/20261123-anti-watermark).
The maintained Python worker verifies outputs even under optimized Python
and the Rust example validates its command arguments through shared example
support. Each raw report preserves the identities used for timing.

## Reproduce

Use the [protocol](protocol.md) and the maintained resource gate from
the repository. To rebuild Rust, copy [worker.rs](worker.rs) into
`crates/calc-flow/examples/anti_watermark_sample.rs` in a clean checkout
of the measured commit. Run at that checkout's root.

```sh
cargo build --locked --release -p calc-flow --features benchmark-internals \
    --example anti_watermark_sample
target/release/examples/anti_watermark_sample --verify-only
```

The [collector](collect.py) takes `rust OUTPUT --worker EXECUTABLE --sha SHA256`
from the root of the study's checkout with `PYTHONPATH=.`. It creates a
new output directory, checks the sealed executable hash, and records all
raw timings and resource samples. The [public worker](python_worker.py)
uses two open source queues with explicitly supplied watermarks. Use
`--verify-only` for an untimed fixture check on the sealed wheel. The
[hosted registration and workflow](https://github.com/wegamekinglc/calc-flow/tree/242972d76c0857e5f0f036d161617da5685d9fa4)
record the installation, workload hashes, job deadline, and collection
command. The raw hosted zip has SHA-256
`d5371b2fbe66f21fc2d101eea2cdf0d703c84fb547cd686a2d9a31135501e577`.
