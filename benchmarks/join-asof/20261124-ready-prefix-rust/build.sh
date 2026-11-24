set -eu
arm="$1"
product="$2"
test "$(git rev-parse HEAD)" = "$product"
git diff --exit-code HEAD --
export CARGO_TARGET_DIR="$3"
export CARGO_BUILD_JOBS=2
touch crates/calc-flow/src/lib.rs crates/calc-flow/benches/stream_asof_perf.rs
git rev-parse HEAD >"target/observations/$arm.sha"
rustc -vV >"target/observations/$arm.rustc"
start=$(date +%s)
cargo bench --locked -p calc-flow --features benchmark-internals --bench stream_asof_perf --no-run --message-format=json >"target/observations/$arm.build.jsonl" 2>"target/observations/$arm.build.log"
stop=$(date +%s)
python3 - "$arm" "$((stop - start))" <<'PY'
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('target/observations')
arm=sys.argv[1]
events=[json.loads(line) for line in (root/f'{arm}.build.jsonl').read_text().splitlines()]
kept={}
for name in ('calc_flow','stream_asof_perf'):
    (ev,) = [e for e in events if e.get('reason')=='compiler-artifact' and e['target']['name']==name and (name!='calc_flow' or e['target']['kind']==['lib'])]
    if ev['fresh'] or Path(ev['manifest_path']) != Path.cwd()/'crates/calc-flow/Cargo.toml' or ev['profile']['opt_level']!='3' or ev['profile']['debug_assertions'] or 'benchmark-internals' not in ev['features']:
        raise SystemExit(f'wrong product provenance: {name}')
    kept[name]=ev
exe=Path(kept['stream_asof_perf']['executable']); shutil.copy2(exe,root/arm)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
digest=sha(root/arm); assert digest==sha(exe)
(root/f'{arm}.digest').write_text(digest+'  '+arm+'\n')
files=['Cargo.lock','crates/calc-flow/benches/stream_asof_perf.rs','scripts/benchmark_suite/asof.py','scripts/benchmark_suite/report.py','scripts/benchmark_suite/statistics.py','scripts/sparse_changelog_ci.py']
git=lambda *args:subprocess.check_output(['git',*args],text=True).strip()
meta=dict(revision=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'),source_tree=git('rev-parse','HEAD:crates/calc-flow/src'),source_directory=str(Path.cwd()),build_seconds=int(sys.argv[2]),binary_sha256=digest,rustc=(root/f'{arm}.rustc').read_text(),files={f:sha(Path(f)) for f in files},environment={k:os.environ.get(k) for k in ('CARGO_TARGET_DIR','CARGO_BUILD_JOBS','CARGO_INCREMENTAL','CARGO_ENCODED_RUSTFLAGS','RUSTFLAGS','RUSTC_WRAPPER','CARGO_PROFILE_BENCH_LTO','CARGO_PROFILE_BENCH_CODEGEN_UNITS')},cargo_targets=kept)
(root/f'{arm}.receipt.json').write_text(json.dumps(meta,indent=2)+'\n')
PY
