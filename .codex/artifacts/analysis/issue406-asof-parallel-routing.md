# Issue 406: parallel ASOF capture routing

This increment follows the [parallel-routing specification](../specs/issue406-asof-parallel-routing.md)
and uses reviewed parent `edb73649`. That parent's ordinary product is exactly
`6739fa2b`: subsequent parent commits change benchmark evidence and documents,
not Rust/Python package sources. Earlier timing binaries retain their recorded
identities. The eight-target objective, prior OFF/32 uncertainty and stopped ON
qualification remain active.

## Diagnostic evidence

Artifacts are under `target/issue406-routing-parallel-profile-20261013/`.
The frozen plan SHA-256 is
`97ca749aceebfd3b35c5a64623b6a1539bfa44a5b31d2da8d94e563f506c8fd9`.
One ASOF case uses 1M rows/input, 64k batches, true OFF and 32 CPUs, with one
warmup and five sampled windows. It reuses the attested ordinary `6739fa2b`
native `a131c569...340011`, without building another product. Acquisition,
symbolization, full oracles, owned cleanup and sealing take 10.614 / 180 seconds.

All six invocations pass oracle/lifecycle/cleanup and zero-state-file checks.
Independent validation verifies all 1,650 frozen inputs and 23 result seals.
The five windows contain 659/697/669/669/660 samples, totaling 3,354; 3,075
include native frames. No owned process group survives or requires forced kill.
Sixteen lost lifetime events have unknown processing-window membership, and
WSL2 host isolation remains unverified. Profiling does not establish a timing
improvement or a parallel regression.

Output matching accounts for 33.81% inclusive / 18.78% self samples;
`ChunkData::output_view` accounts for 5.28% self samples. Admission remains
material at 39.36% inclusive. Shares overlap. Source inspection identifies a
serial preparation pass that builds a view and hashes an encoded key per row
before submitting parallel probe work. The worker already receives integer
bucket coordinates, so routing preparation can reuse the existing chunk IDs.

## Implementation and focused verification

Parallel capture walks canonical output runs. Eligible runs resolve their
dictionary keys into a 64-slot stack array, then append worker inputs by integer
ID. A local `None` caches a missing bucket. The existing global route map still
deduplicates actual buckets across independently encoded chunks. Legacy rows,
more than 64 slots and runs with fewer than four selected rows per slot use the
generic route lookup. Shard compaction remains after all capture, preserving
route coordinates throughout preparation.

The first selected row's cancellation/progress check precedes route setup; the
remaining checks use the same global position and 1,024-row yield boundaries.
Missing keys still advance global position. Worker count, input/output credits,
right-bucket lifetime funding, matching, restore order, collector acceptance and
checkpoint contracts retain their existing boundaries. The route array adds
bounded stack scratch within the existing fixed scratch allowance and no
retained heap state. `OutputRun` is re-exported only within the ASOF module.

The focused regression first records 1,037 encoded-key hashes for 1,024 selected
rows and sixteen keys, including two missing keys. The first implementation
passes its bound of 128 while preserving exact binary-search matches, status
and reservation refund. Capture-specific differential controls cover dictionary
boundaries, short cuts, partial consumption, independent dictionaries, retired
slots, legacy rows and cancellation. They are invoked by the existing budget
regression, adding no libtest registrations.

The scoped finalization run passes 27 checks and exposes one new fixture error:
the all-missing-key control initially attempted to construct a batch without
any record batches. The helper now omits that empty reference input; the exact
affected registered test passes with all capture-specific boundary scenarios
in 0.13 seconds. The unchanged 27 passing checks are not repeated. They include
owned-work abandonment, cold checkpoint recovery and rejected collector
continuation. Default core registration remains 2,033.

Scoped `cargo clippy -p calc-flow --lib --tests --all-features -- -D warnings`
passes after the final refactor in 26.41 seconds. It initially identifies the
old `output_iter` helper's
last production caller as removed; that iterator is now explicitly test-only,
without suppressing a lint. The first-row refactor removes duplicate offset
branches; the exact affected regression passes again in 0.13 seconds, without
repeating the unchanged 27 passing checks. Lizard 1.24.1 recognizes all 36
functions in the complete probe/test modules, explicitly covers every changed
function, and reports maximum complexity eight. Formatting, generated-contract
drift and whitespace checks pass. Product-wide CI
and coverage are not certified by these focused local checks.

Final specialist source review approves implementation
`05c13fd983dd1ead625eef2278eb7127dac0c61c`. Review of the later measurement
report remains separate from source approval and required CI.

## Paired comparison scope

Before acquisition, freeze the exact ordinary parent product `6739fa2b`, the
new committed candidate, their wheel/native identities and the maintained
harness. Reuse the attested parent native; estimate the candidate release build
at 6–14 minutes, recorded separately. Both arms must match compiled dependency
features/profiles, toolchain, environment and normalized lock identity.

| Case                        | Connection to this change                                |
|-----------------------------|----------------------------------------------------------|
| ASOF 1M/input, 64k, OFF, 32 | Direct parallel probe capture and end-to-end output path |
| ASOF 1M/input, 64k, OFF, 1  | Necessary unchanged serial-path control                  |

Fix two rounds of ten adjacent AB/BA pairs and one warmup per arm per round,
with 600 seconds inclusive of preflight, fixtures, processing, full oracles,
CPU evidence, lifecycle settlement and seals. Use the same sealed Appendix A
CPU accounting at the maintained processing clock bounds in both arms. Stop on
failure, unsuitable resources or budget; do not retry, expand samples or infer
equivalence from uncertainty. A comparison with this parent isolates the new
parallel preparation change; it does not replace the earlier merged-main
comparison or certify the original eight-target set. OFF targets remain
100 ms / 30 ms at one / 32 CPUs, and source review is separate from performance.

## Completed paired release comparison

The exact parent ordinary product remains
`6739fa2b5573f4e184266793fd9ee4d4281a5b14`, native SHA-256
`a131c569adaa5fdb629e868be6af42ca0e5a3e31359694a37a39116930340011`.
The freshly built candidate is
`05c13fd983dd1ead625eef2278eb7127dac0c61c`, native SHA-256
`cbc1a85acd9d3b1d6baf0ec2e5fe0c888a5ec53353ee2176d1ae56281a6c600b`.
Its ordinary release build takes 428.449 seconds, recorded separately; parent
reuse requires no build. Both arms pass ordinary audits of 294 compiled
artifacts with equal normalized feature/profile, environment, toolchain and
lock identities. Benchmark-internal features remain absent.

Artifacts are under `target/issue406-parallel-routing-paired-20261013/`.
The sealed plan SHA-256 is
`bbd2edcb3f8028ba8bd3d75b8e23f97f2a6822ca5875717b2adda4096bc40b92`.
Both fixed cases complete two rounds of ten adjacent AB/BA pairs, with one
warmup per arm per round: 80 measured observations and eight warmups. All 88
raw results pass full oracles, completed lifecycle, owned cleanup and true OFF
zero-file checks. Acquisition, audit, preflight, fixtures, CPU bookkeeping,
oracles, settlement and seals take 39.019 / 600 seconds. Every process group
settles with no forced kills, surviving processes or cleanup errors. There are
no sample retries, exclusions or increases.

An independent read-only audit verifies all 3,109 frozen inputs and 198 output
seals, raw/normalized correspondence, actual AB/BA order, same processing-clock
bounds for CPU evidence and recomputed maintained statistics. The sealed
generic `results/performance.md` retains legacy `Merged main p50` / `CI95`
headers. This comparison's actual baseline is the exact parent above; the
following table corrects the labels. Each ten-pair order-statistic interval has
97.8515625% coverage. The sealed original report is preserved unchanged.

| Case              | Parent p50 ms | Candidate p50 ms | Change  | Maintained paired verdict |
|-------------------|---------------|------------------|---------|---------------------------|
| 64k, OFF, 32 CPUs | 178.517       | 154.690          | -13.35% | improved                  |
| 64k, OFF, 1 CPU   | 122.682       | 123.929          | +1.02%  | inconclusive              |

The 32-CPU round median changes are -12.17% / -13.89%, with intervals
[-15.59%, -10.68%] / [-14.90%, -10.26%]. Both upper bounds are below -5%,
satisfying the maintained `improved` rule. Single-CPU round changes are
-4.66% / +1.98%, with intervals [-11.55%, +4.20%] / [-3.28%, +9.25%].
That control remains inconclusive; it establishes neither equivalence nor a
confirmed regression. No blanket regression-free conclusion is justified.

| Processing CPU median, ms | Parent 32 CPUs | Candidate 32 CPUs | Parent 1 CPU | Candidate 1 CPU |
|---------------------------|----------------|-------------------|--------------|-----------------|
| Process CPU               | 205.404        | 180.801           | 122.466      | 122.577         |
| Busiest observed thread   | 84.447         | 66.361            | 112.012      | 113.022         |

Both arms use the same sealed Appendix A thread/process accounting wrapper.
Raw observations and the independent audit retain per-group run and runqueue
wait times. Read boundaries have uncertainty, and retired threads cannot be
reconstructed. These CPU values support diagnosis without establishing an exact
thread-accounting identity or replacing paired wall-clock statistics. WSL2 host
isolation remains unverified.

The measured candidate p50 values still exceed the ASOF OFF targets of
100 ms / 30 ms. This two-case parent comparison does not supersede the earlier
merged-main comparison, resolve either stopped ON attempt, measure the other
seven acceptance rows, or certify complete eight-target acceptance. Required
CI and coverage remain outside this focused local handoff.
