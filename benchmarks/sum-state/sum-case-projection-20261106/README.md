# sum-case-projection-20261106

Status: **stopped**.
Performance acceptance: **False**.

| Case | Baseline P50 ms | Candidate P50 ms | Change | Verdict |

| Case | Round | Paired median | Median interval | Coverage |

| Case | Side | P95 ms | CPU P50 ms | Cores P50 | Peak RSS P50 MiB |

Accepted profiles: 0.

Five fixed cases use200formal observations and20warmups in two rounds
of ten alternating pairs per case. Both dense1 profiles use six jobs
per product, excluding the first; five validated processing windows
are required. No case, sample, favorable retry or threshold is added.
One600-second budget includes controller setup and live global quiet
guard, fixtures, preflight, warmups, startup, full oracles, resources,
profiles/capture/decoding, statistics, publication, postseals and cleanup.
Builds are separate. At least one dense target needs the exact maintained
improved verdict, with complete evidence and no confirmed regression on
any case. Inconclusive is not equivalence, speedup or confirmed regression.
The fresh baseline is publicdc8393d2. Old1fc and experimental rankedP
products are incompatible; compatible dependency caches do not replace
fresh core/example compilation or public source and binary proof.
Both products use identical common657 instrumentation from public841.
Dense has10k zero seed rows plus20x100k source rows, selecting80k trades
per business batch. Sparse has10k seed groups plus20x100 source rows,
selecting80 trades/batch; those business batches alternate uniform sides.
Uniform-buy and uniform-sell reuse the maintained one-cycle all-trade
recipe:10k seeds plus20x10k business rows. Each forces side on the seed
AND all business records, preserving quantity, other column values/types,
exact side Field, schema metadata and caller-owned input buffers.
The two pure masks are separately paired, so one branch cannot hide the
other's regression in an alternating workload. Both retain21 complete
independent arithmetic/schema/input/snapshot checks per job.
DF target/input partitions remain1, min_rows_per_partition1, batch8192,
checkpoints off. Dense4 uses CPUs0/2/4/6 withTokio4; others CPU0/Tokio1.
The controller uses CPU31 and external helper pools stay1. Host load
and power outside the WSL guest remain unobservable.
The timer starts after IPC preload and covers graph planning, execution,
retained full outputs, natural end and cleanup. Oracles are outside that
timer but inside the total budget. Whole-process peak RSS includes input
preload and retained outputs, not operator-only heap or reservation fees.
Full CPU stacks and exact clock/loss evidence are retained. Inclusive
shares overlap and are not phase wall times; missing/inlined frames or
zero matched samples do not establish zero runtime cost or a route hit.
Mixed CASE with an all-trade filter, finite, nullable/UInt64/generic/UDF,
checkpoint and tight-budget latency remain unmeasured. Focused source
semantics and allocation tests are separate from these timings.
[Summary](summary.json), [formal samples](samples.csv),
[complete raw evidence](complete-evidence.json.gz),
[profiles](profile-summary.json), [targets](profile-targets.json) and
[manifest](evidence-manifest.json) retain receipts, intervals and seals.

Stopped reason: RuntimeError('competing build/measurement: [{\'status\': \'sleeping\', \'exe\': \'/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-mt-research/.venv/bin/maturin\', \'pid\': 54831, \'name\': \'maturin\', \'cmdline\': [\'/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-mt-research/.venv/bin/maturin\', \'build\', \'--release\', \'--locked\', \'--features\', \'pyo3/abi3-py313\', \'--out\', \'/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-op-analysis/target/op-analysis/extrema-candidate-release\']}, {\'status\': \'sleeping\', \'exe\': \'/home/wegamekinglc/.rustup/toolchains/1.94.0-x86_64-unknown-linux-gnu/bin/cargo\', \'pid\': 54863, \'name\': \'cargo\', \'cmdline\': [\'/home/wegamekinglc/.rustup/toolchains/1.94.0-x86_64-unknown-linux-gnu/bin/cargo\', \'rustc\', \'--profile\', \'release\', \'--features\', \'pyo3/abi3-py313\', \'--message-format\', \'json-render-diagnostics\', \'--locked\', \'--manifest-path\', \'/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-op-analysis/crates/calc-flow-python/Cargo.toml\', \'--lib\', \'--\', \'-C\', \'strip=symbols\']}, {\'status\': \'sleeping\', \'exe\': \'/home/wegamekinglc/.rustup/toolchains/1.94.0-x86_64-unknown-linux-gnu/bin/rustc\', \'pid\': 54875, \'name\': \'rustc\', \'cmdline\': [\'/home/wegamekinglc/.rustup/toolchains/1.94.0-x86_64-unknown-linux-gnu/bin/rustc\', \'--crate-name\', \'calc_flow\', \'--edition=2024\', \'crates/calc-flow/src/lib.rs\', \'--error-format=json\', \'--json=diagnostic-rendered-ansi,artifacts,future-incompat\', \'--crate-type\', \'lib\', \'--emit=dep-info,metadata,link\', \'-C\', \'opt-level=3\', \'-C\', \'linker-plugin-lto\', \'-C\', \'codegen-units=1\', \'--deny=clippy::pedantic\', \'--deny=clippy::all\', \'--deny=unused_qualifications\', \'--forbid=unsafe_code\', \'--allow=clippy::must_use_candidate\', \'--allow=clippy::module_name_repetitions\', \'--check-cfg\', \'cfg(docsrs,test)\', \'--check-cfg\', \'cfg(feature, values("benchmark-checkpoint-diagnostic", "benchmark-checkpoint-off", "benchmark-internals"))\', \'-C\', \'metadata=d32504bfeaaaeed8\', \'-C\', \'extra-filename=-703eaa4e8f7e1c90\', \'--out-dir\', \'/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps\', \'-C\', \'strip=debuginfo\', \'-L\', \'dependency=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps\', \'--extern\', \'ahash=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libahash-7909130cd344b7c9.rmeta\', \'--extern\', \'arrow_data=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libarrow_data-ffd5037cfe0e9632.rmeta\', \'--extern\', \'async_trait=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libasync_trait-c2de5d4165255b73.so\', \'--extern\', \'chrono=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libchrono-c7ce180c7f203684.rmeta\', \'--extern\', \'datafusion=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libdatafusion-bc082ddffbb8e722.rmeta\', \'--extern\', \'datafusion_functions_nested=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libdatafusion_functions_nested-6fcdf9f1dfc23340.rmeta\', \'--extern\', \'flatbuffers=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libflatbuffers-ca75cc0f41474ca1.rmeta\', \'--extern\', \'fs2=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libfs2-fba75d915dffe19f.rmeta\', \'--extern\', \'futures=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libfutures-63c2af29a66f7e47.rmeta\', \'--extern\', \'hashbrown=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libhashbrown-9e1cb5b97e206f63.rmeta\', \'--extern\', \'hex=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libhex-ec5953145966f7b7.rmeta\', \'--extern\', \'parking_lot=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libparking_lot-04c62e5de5b3616a.rmeta\', \'--extern\', \'regex=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libregex-bce63cf6c6206fa8.rmeta\', \'--extern\', \'schemars=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libschemars-c3f1e78ac026f1b9.rmeta\', \'--extern\', \'serde=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libserde-7b32adaf9a497220.rmeta\', \'--extern\', \'serde_saphyr=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libserde_saphyr-ba37d9fed64e7103.rmeta\', \'--extern\', \'serde_json=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libserde_json-4b69bed66b4bf2a1.rmeta\', \'--extern\', \'sha2=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libsha2-c6fd1740f2fee4f2.rmeta\', \'--extern\', \'tempfile=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libtempfile-cbfb8291c0967686.rmeta\', \'--extern\', \'thiserror=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libthiserror-2c02b97d0bd2e314.rmeta\', \'--extern\', \'tokio=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libtokio-24061613211293e5.rmeta\', \'--extern\', \'tokio_util=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libtokio_util-b51f3b475617b21a.rmeta\', \'--extern\', \'tracing=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libtracing-5e2ace97b66033b5.rmeta\', \'--extern\', \'uuid=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/deps/libuuid-64d2679abd8fe7b2.rmeta\', \'-L\', \'native=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow-batch-sql-admission/target/pr3-plan/baseline-release/cargo/release/build/zstd-sys-b2be3af772e6728d/out\']}]')
