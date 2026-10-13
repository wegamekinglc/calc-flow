# Issue 406: borrow ordered right admission buckets once

The eight-target objective is active again after PR #412 merged. Its benchmark
repair completed; it did not complete the performance objective. The baseline
for this increment is exact main `8f890f287d44fa0c5ce8c79d3d23e595f8563dcd`.
Use the full inventory and timing contracts in
[performance acceptance](issue406-performance-acceptance.md). No default memory
pool limit is introduced.

## Evidence and selected change

Two fixed current-main CPU profiles cover ASOF 1M rows per input, 64,000-row
batches, checkpoint OFF, at one and 32 CPUs. Each has one warmup and five
processing windows, full-output validation and settled owned workers. The
ordinary release build takes 336.760 seconds separately. Collection including
the failed relative-path attempt takes 20.134 of the fixed 180 seconds.

The single-CPU profile has 2,788 processing samples. Admission is 52.08%
inclusive, preparation 37.02%, and installation 14.10%; installation itself has
7.78% self samples and unsigned bucket append 5.85%. These regions overlap.
The 32-CPU profile has 3,535 samples and installation is 10.16% inclusive.
The profile cannot establish a timing gain. The one-CPU profiler reports 243
lost events over its complete process lifetime; their window membership is
unknown. Keep this limitation with the profile rather than treating percentages
as exact CPU accounting.

The eligible columnar right path calls `Arc::get_mut` for each admitted row,
although preparation has already reserved and made each touched bucket unique.
Borrow each distinct bucket once and reuse those safe mutable references for
the synchronous commit. A bounded stack selection avoids extra heap allocation
and preserves existing explicit workspace refusal decisions. Larger key domains
continue through the existing per-row path; this is no state/key-count limit.

## Requirements

1. Apply only to the existing ordered, inline-string, UInt64-sequence right
   path. Preserve eligibility, duplicate/late/error precedence, canonical
   ordering and the retained-maximum check.
2. Resolve at most 64 distinct admitted handles into stack-held unique mutable
   bucket references. Resolve sparse handles through their current shard and
   position, without scanning untouched resident entries. Preserve mapping
   from batch-local IDs to actual dictionary handles, including moved slots.
3. Perform one exclusivity check per touched bucket, then append all rows in
   original record and row order. Keep the existing path above this stack
   selection size, with identical supported inputs and results.
4. Preserve capacity reservation and bucket COW before the infallible commit.
   Do not allocate extra heap storage, add an await or fallible operation inside
   that commit, mutate caller-owned columns, or add unsafe code.
5. Preserve exact state inventory, retained owners, checkpoint bytes/journals,
   cold restore, cancellation refunds and explicit limit decisions.
6. Record a focused failing work-count test before implementation. Reuse the
   existing differential test registration and extend its semantic controls;
   do not create a new libtest registration merely for the counter.

## Validation and measurement

Compare columnar and forced legacy paths for repeated keys, independent record
dictionaries, sliced/empty records, sparse resident handles, large key domains,
held COW owners, nullable projections, extreme sequences and fallbacks. Keep
the existing checkpoint/delta/restore, failure priority and pool-refund controls.
Run focused serial ASOF tests, core/test Clippy and formatting/contracts checks.
Final specialist review and green required CI remain merge gates.

After source validation, freeze four paired ASOF cases: 1M per input, 64,000-row
batches, OFF at one/32 CPUs and LOW/ON at one CPU. Use exact main/current
ordinary releases, two rounds of ten adjacent AB/BA pairs and one warmup per
arm per round. All setup, oracle, terminal and cleanup costs count toward one
600-second budget; builds are recorded separately. Keep the maintained paired
regression rule and actual advancing checkpoint coverage. This subset does
not replace the final eight-row acceptance inventory or measure row-7 overhead.

## Remaining full-plan work

Keep every original target visible in the accompanying analysis. After this
increment, use the remaining current profile to choose output matching and
parallel work; measure Join admission/gather and 1,024-row callback costs;
complete capture/restore and retained scaling optimizations; then acquire all
eight targets with supported matched OFF/LOW/ON controls. A historical result,
suite LOW result or profiling duration cannot certify a new OFF target. Missing,
invalid, unsupported or inconclusive acceptance remains unfinished.
