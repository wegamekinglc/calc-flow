from __future__ import annotations

import json
from pathlib import Path

from calc_flow_studio.app import create_app
from calc_flow_studio.run_manager import RunManager


def main() -> None:
    output = Path(__file__).parents[1] / "openapi.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    app = create_app(run_manager=RunManager(use_processes=False))
    output.write_text(
        json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
