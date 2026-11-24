set -eu
arm="$1"
git diff --exit-code HEAD --
git rev-parse HEAD > "target/observations/$arm.sha"
rustc -vV > "target/observations/$arm.rustc"
CARGO_TARGET_DIR="$PWD/target/cargo" cargo bench --locked -p calc-flow --features benchmark-internals --bench stream_asof_perf --no-run --message-format=json > "target/observations/$arm.build.jsonl" 2> "target/observations/$arm.build.log"
python3 - "$arm" <<'PY'
import json,shutil,sys
from pathlib import Path
root=Path("target/observations")
executables=[ev["executable"] for line in (root / (sys.argv[1]+".build.jsonl")).read_text().splitlines() if (ev:=json.loads(line)).get("executable") and ev["target"]["name"]=="stream_asof_perf"]
assert len(executables)==1
shutil.copy2(executables[0],root/sys.argv[1])
PY
sha256sum "target/observations/$arm" > "target/observations/$arm.digest"
