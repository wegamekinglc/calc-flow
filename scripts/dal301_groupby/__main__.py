from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from scripts.toolkit import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed DAL-301 group_by CI diagnostic")
    parser.add_argument("operation", choices=("plan", "run", "build", "worker"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--releases", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--side", choices=("A", "B"))
    args = parser.parse_args()
    match args.operation:
        case "plan":
            from scripts.dal301_groupby.contract import plan

            write_json(args.output / "plan.json", plan())
        case "worker":
            from scripts.dal301_groupby.runtime import worker

            worker(args.output)
        case "build":
            from scripts.dal301_groupby.profile import build

            if args.source is None or args.side is None:
                parser.error("build requires --source and --side")
            return asyncio.run(
                build(args.side, args.source.resolve(), args.output.resolve())
            )
        case "run":
            from scripts.dal301_groupby.controller import run

            if args.releases is None:
                parser.error("run requires --releases")
            return asyncio.run(
                run(args.output.resolve(), args.releases.resolve(), args.profiles)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
