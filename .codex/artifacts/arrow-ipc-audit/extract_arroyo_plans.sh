#!/usr/bin/env bash
set -euo pipefail
task_python="${DAL417_PYTHON:-target/arroyo-attribution/venv/bin/python}"
evidence_root="${1:-target/arrow-ipc-extension}"
"$task_python" - "$evidence_root" <<'PY'
import sqlite3
import sys
from pathlib import Path

root = Path(sys.argv[1])
for relative in ("results/transform-full-0-arroyo", "results/transform-512-0-arroyo",
                 "window-probe-v2/arroyo"):
    directory = root / relative
    database = directory / "state/state.sqlite"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT program FROM pipelines").fetchall()
    if len(rows) != 1:
        raise ValueError("expected exactly one real executed pipeline")
    (directory / "executed-program.pb").write_bytes(rows[0][0])
PY
for relative in results/transform-full-0-arroyo results/transform-512-0-arroyo window-probe-v2/arroyo; do
  protoc --decode=api.ArrowProgram \
    -I target/arroyo-arrow-source/crates/arroyo-rpc/proto api.proto \
    < "$evidence_root/$relative/executed-program.pb" \
    > "$evidence_root/$relative/executed-program.txt"
done
