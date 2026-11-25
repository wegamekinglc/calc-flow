"""Replay one complete or stopped diagnostic from the registered wall boundaries."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from diagnostic_validate import FIELDS, counts


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return list(map(json.loads, p.read_bytes().splitlines()))


def observe(outer, seconds, records, candidate):
    lo, hi, pid = outer["start"], outer["end"], outer["pid"]
    assert 0 <= abs((hi - lo) - round(1e9 * seconds)) < 1000000
    assert records and all(
        r["pid"] == pid and r["end"] - r["start"] == r["wall_ns"] for r in records
    )
    a = [r for r in records if r["kind"] == "admit"]
    w = [r for r in records if r["kind"] == "watermark"]
    counts(w, pid)
    assert len(a) == 20 and len(w) + len(a) == len(records)
    rows = Counter()
    for r in a:
        assert r["rows"] == r["accepted"] == 1_000_000
        assert r["wall_ns"] == sum(r["phases_ns"].values())
        rows[r["side"]] += r["accepted"]
    assert sorted(rows.values()) == [10_000_000, 10_000_000]
    values = Counter()
    inv = Counter()
    intervals = []
    for r in records:
        begin, end = r["start"], r["end"]
        overlap = max(0, min(hi, end) - max(lo, begin))
        role = r["kind"] if r["kind"] == "watermark" else r["side"]
        if overlap == 0:
            inv["outside_" + role] += 1
            continue
        intervals.append((max(begin, lo), min(end, hi)))
        if begin < lo or end > hi:
            inv["crossing_" + role] += 1
            values["crossing_callback_ns"] += overlap
            continue
        inv["inside_" + role] += 1
        if r["kind"] == "admit":
            values["admit_" + role] += r["wall_ns"]
            for key, t in r["phases_ns"].items():
                values["admit_" + role + "." + key] += t
            if candidate:
                for i, n in enumerate(r["selection_charge_visits"]):
                    inv[
                        [
                            "cardinality_mask_rows",
                            "variable_value_length_rows",
                            "record_mask_rows",
                        ][i]
                    ] += n
        else:
            values["watermark_ns"] += r["wall_ns"]
            for key in (
                *FIELDS,
                "arrow_bookkeeping_ns",
                "arrow_charge_ns",
                "arrow_columns_ns",
                "arrow_assembly_ns",
            ):
                values["watermark." + key] += r[key]
    assert (
        sum(
            v
            for k, v in inv.items()
            if k.startswith("inside_") and k != "inside_watermark"
        )
        == 20
    )
    busy = 0
    last = lo
    for start, end in sorted(intervals):
        busy += max(0, end - max(start, last))
        last = max(end, last)
    values["other_inside_window_ns"] = hi - lo - busy
    values["native_union_ns"] = busy
    values["outer_ns"] = hi - lo
    values["overlapped_native_ns"] = sum(end - start for start, end in intervals) - busy
    spans = outer.get("spans", [])
    for label, start, end in spans:
        assert lo <= start <= end <= hi
        values["python." + label] += end - start
        inv["python_" + label] += 1
    if spans:
        assert inv["python_publish_events"] == inv["python_wait_for_chunk_output"] == 10
        assert inv["python_concat"] == 1 and inv["python_arrow_bridge"] > 0
        arrivals = [(b, e) for k, b, e in spans if k == "arrow_bridge"]
        values["python.time_to_first_bridge_ns"] = arrivals[0][0] - lo
        values["python.last_bridge_to_end_ns"] = hi - arrivals[-1][1]
    return dict(ns=dict(values), inventory=dict(inv))


def native(d):
    (probe,) = d.glob("obs-job-*")
    raw = defaultdict(list)
    for event in lines(d / "phases.jsonl"):
        raw[event["pid"]].append(event)
    for r in lines(probe / "observations.jsonl"):
        side = r["side"]
        x = r["result"]
        out = x["trace_outer"]
        yield (
            side,
            r["round"] - 1,
            r["pair"] - 1,
            observe(out, x["seconds"], raw.pop(out["pid"]), side == "head"),
        )
    assert not raw


def python(d):
    for round_index in range(2):
        q = d / "python/measurement" / f"round-{round_index}"
        for arm, side in (("baseline", "base"), ("candidate", "head")):
            worker = q / arm
            data = (worker / "phases.jsonl").read_bytes()
            offset = 0
            for f in sorted(worker.glob("trace-request-*.json")):
                link = read(f)
                rid = f.stem.split("-")[-1]
                msg = read(worker / f"request-{rid}.completed.json")
                assert link["start"] == offset and link["end"] > offset
                records = list(map(json.loads, data[offset : link["end"]].splitlines()))
                offset = link["end"]
                x = msg["response"].get("warmup", msg["response"])
                out = x["stream_evidence"]["trace_outer"]
                assert out["pid"] == link["pid"] and msg["context"]["side"] == arm
                yield (
                    side,
                    round_index,
                    int(rid) - 3,
                    observe(out, x["seconds"], records, side == "head"),
                )
            assert offset == len(data)


def report(d, surface):
    results = list((native if surface == "rust" else python)(d))
    assert len(results) == 44
    groups = {
        side: [r for arm, _, pair, r in results if arm == side and pair >= 0]
        for side in ("base", "head")
    }
    keys = set(k for arm in groups.values() for r in arm for k in r["ns"])
    summary = {
        k: {
            s: statistics.mean(r["ns"].get(k, 0) for r in arm) / 1e6
            for s, arm in groups.items()
        }
        for k in sorted(keys)
    }
    for _k, v in summary.items():
        v["delta_ms"] = v["head"] - v["base"]
    for _side, rows in groups.items():
        assert len(rows) == 20
    (d / "phase-observations.json").write_text(json.dumps(results, indent=2) + "\n")
    (d / "phase-means-ms.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(report(Path(sys.argv[1]), sys.argv[2]), indent=2))
