"""Verify retained state outside every registered interval join observation."""

from __future__ import annotations

import json
from pathlib import Path

ROW_COUNT = 10998080
SIDES = ("baseline", "candidate")


def retained_state(sample: dict) -> dict:
    expected_proof = {
        "passed": True,
        "rows": ROW_COUNT,
        "finite_rows": ROW_COUNT,
        "max_abs_error": 0.0,
    }
    if sample["correctness"] != expected_proof:
        raise ValueError("interval output failed its full numeric oracle")
    (node,) = sample["stream_evidence"]["interval_join"].values()
    if any(
        node[key] != expected
        for key, expected in (
            ("emitted_match_rows", ROW_COUNT),
            ("match_limit_failures", 0),
            ("state_limit_failures", 0),
        )
    ):
        raise ValueError("interval join emitted rows or limit failures changed")
    for side in ("left", "right"):
        retained = node[side]
        if retained["retained_bytes"] <= 0 or any(
            retained[key] != expected
            for key, expected in (
                ("evicted_rows", 999680),
                ("retained_rows", 320),
                ("watermark_micros", 1767241224000001),
                ("idle", False),
                ("ended", False),
                ("late_rows", 0),
                ("null_key_rows", 0),
                ("null_event_time_rows", 0),
            )
        ):
            raise ValueError(f"{side} interval join post-eviction state changed")
    return {"seconds": sample["seconds"], "left": node["left"], "right": node["right"]}


def all_retained_states(evidence: list, measurement: Path) -> list:
    observations = []
    if len(evidence) != 2:
        raise ValueError("expected two complete independent rounds")
    for index, round_ in enumerate(evidence):
        for arm in SIDES:
            warmup = json.loads(
                (
                    measurement / f"round-{index}" / arm / "request-0002.completed.json"
                ).read_text()
            )["response"]["warmup"]
            samples = round_["samples"][arm]
            if len(samples) != 10:
                raise ValueError("expected ten complete pairs per round")
            for pair, sample in enumerate([warmup, *samples]):
                observations.append(
                    {
                        "round": index + 1,
                        "arm": arm,
                        "pair": pair,
                        "warmup": pair == 0,
                        **retained_state(sample),
                    }
                )
    return observations
