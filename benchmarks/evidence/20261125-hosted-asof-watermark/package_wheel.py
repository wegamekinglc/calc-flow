"""Retain measurement records without copying installed environments or wheel binaries."""
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
target = Path("target")
target.mkdir(exist_ok=True)
with tarfile.open(target / "wheel-study-raw.tar.gz", "w:gz") as output:
    for path in sorted(root.glob("wheel-*")):
        if path.is_file():
            output.add(path, arcname=path.name)
        elif path.is_dir():
            for record in sorted(path.rglob("*")):
                relative = record.relative_to(root)
                if "site" not in relative.parts and record.is_file() and record.suffix in {".json", ".jsonl", ".log", ".stderr"}:
                    output.add(record, arcname=str(relative))
    for path in sorted((target / "asof-wheel-products").glob("*-*/*")):
        if path.name in {"release.json", "build.log"}:
            output.add(path, arcname=f"products/{path.parent.name}/{path.name}")
