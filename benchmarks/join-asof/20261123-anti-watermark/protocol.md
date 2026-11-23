# Active right-watermark release for bounded left anti joins

Measure an already populated LeftAnti interval join on main
`37dd0ab7a61087f0bd799fc555e639ba55442fca`. Earlier controls measured
eligibility reads and EOF allocations. The parent before #686 has no
LeftAnti implementation. Report absolute latency for one product.

Both independent trials use columns key (nullable Int64), time
(non-null timestamp in UTC microseconds), and sequence (non-null UInt64).
Before=0 and after=2 microseconds. Both use output envelopes capped at
32 rows and 4,096 bytes, 512 rows and 8 MiB of state per side and 32
emitted matches per data input. Rust delivers to an in-memory collector
with the benchmark output budget. Python applies the same row and byte
limits to managed graph edges.
Prepare one right witness (key=8,time=10). In physical order, the left
input has key=7 at time=9 (sequence 0), 257 rows of key=7 at time=10
(sequences 1..257), key=8 at time=10 (sequence 258), key=7 at time=11
(sequence 259), and a null key at time=11 (sequence 260). Split it into
consecutive batches of at most 32. Keep both sources active.

1. Observe the immediate null-key output (260) before sending any
   progress. At right watermark 12, require precisely the earlier
   sentinel (0). Then submit a second left null-key row at time=11
   (sequence 261) and require it next. This public output acknowledgment
   proves that the equality cut left all 257 targets pending before timing.
2. Advance the right watermark to 13. Require sequences 1..257 in order
   with their original schema, key and time, in eight 32-row outputs and
   one single-row output. The witnessed left row must never appear.
3. Outside the clock, advance to 14 and require only the later sentinel
   (259). Complete both sources and check that the job emits no further
   rows. This verifies the timed release while the right source is live.

The native clock starts at the public Rust operator's
`on_ingress_progress_with_output` call and stops when it returns after
delivering the chunks to the in-memory collector. Configure a
single-thread Tokio runtime. The public direct Rust context uses its default
checkpointing flag and never captures an epoch. The public Python graph disables
checkpoints. The Python clock starts immediately before
publishing the prepared Watermark to an asyncio queue serving a public
`Program.stream` source under `SourceProvidedWatermarks`, and stops
on receipt of the last public Arrow table. Preserving the tables and
counting received rows are inside the Python clock. Schema, every
column value, chunk sizes, the later watermark, and cleanup are checked
outside both clocks. Each fresh process measures one release.

One independent single-arm trial per boundary, two rounds of ten fresh
processes, one excluded warmup per round. Report pooled P50/P95, round
medians, and 257 divided by the pooled median in seconds. The prospective P50 targets are 1 ms for the native callback and 5 ms for the
public Python operation, which includes source scheduling, queueing and Arrow
return. P95 is descriptive given only 20 observations and few tail samples. This
throughput describes one 257-row release. Print no speedup or comparison
between languages. Native uses a pinned local Linux CPU with its SMT
sibling idle. Python uses one assigned CPU on an isolated hosted Linux
runner with the official wheel produced by main's Linux CI run
45182357543 (artifact 14822848922) and locked dependencies. Use one Tokio worker. Include
individual times, product/worker identities, resource samples and stop
reasons in receipts.

Reserve 240 seconds for native and 300 seconds for Python, including
formal resource preflight (three seconds), setup, fixture generation,
worker startups, all verification, artifact acquisition/install on the
hosted runner, and cleanup. Build native separately using the locked
release profile; include build time separately. Stop a trial on budget
exhaustion, correctness failure, failure of the maintained
`scripts.sparse_changelog_ci._resource` check. It requires at least 10 GiB
available, host CPU below 50%, fewer than 256 system-wide major faults and
fewer than 1,024 swap pages per interval. Preflight caps both selected CPU
and sibling at 20%, and steady-state sibling utilization must stay below 50%.
The resource checks start with the three-second preflight after the installed
wheel and files have been verified, and continue until all workers exit. The
total time budget also includes activities before that preflight. Preserve
partial data and do not restart automatically.
