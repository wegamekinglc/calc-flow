# Public single-row session with a shared table envelope

Compare the Rust table batch storage change in PR #699 on the maintained
one-entity dual-SMA fixture under ordinary GC. Each worker opens a fresh
session, uses default edge limits and explicit watermarks, warms 4,096 rows
and times 64 consecutive append calls with one row each. The clock starts
just before calling the public method and ends on return of its dictionary
of Arrow tables. Input envelope construction, watermark preparation, oracle
validation and closing are outside the individual push clocks. Verify the
entire warm result and each append, including key and input columns, full
schema metadata, nulls, and both moving averages' spread. Check the original
input, zero-row flush and final cleanup. Keep all worker observations.

Use official Linux release wheels with the same build image, toolchain,
profile and ABI. Require matching runtime source except for the table envelope change and
record the exact two checkout trees, source equivalences, package hashes and
installed native-module hashes before timing. Use two rounds of ten adjacent
alternating A/B process pairs and one excluded warmup per arm per round.
Use the median of 64 calls as the paired unit, the maintained exact-median
intervals and five-percent regression rule. Report both pooled p50 and p95
against 20 and 40 microseconds. Use no tracing or phase probe.

Allow one 240-second hosted acquisition from job start through cleanup and
upload, within a 300-second feature measurement budget including local
fixture and source qualification. Compile time is separate. Pin every worker
to one CPU with one Tokio thread. Require the maintained three-second
admission check and per-second CPU, SMT, memory, paging and host checks.
Stop on any resource, provenance, correctness or deadline violation and
record partial results. The earlier same-wheel phase diagnostic has its
own budget and observer semantics.
