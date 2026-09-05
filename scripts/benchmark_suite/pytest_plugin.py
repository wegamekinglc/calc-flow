"""Capture the independently collected benchmark inventory, including skips."""

from __future__ import annotations

import json
import os
from pathlib import Path


def pytest_collection_finish(session) -> None:
    destination = os.environ.get("CALC_FLOW_SUITE_INVENTORY")
    if destination:
        items = [
            item.nodeid for item in session.items if "benchmark" in item.fixturenames
        ]
        Path(destination).write_text(
            json.dumps(items, indent=2) + "\n", encoding="utf-8"
        )
