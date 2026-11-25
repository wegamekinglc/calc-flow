"""Attribute the existing 1M installed ASOF trace by observed callback role."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import statistics
import sys
import tarfile

archive=Path(sys.argv[1])
with archive.open("rb") as stream:
    assert hashlib.file_digest(stream,"sha256").hexdigest()=="e06a81bc3bdd63a0e085407c15efa3d9a3b0e03b9bd7cca00a3bcd0ac7737bce"
fields=("wall_ns","matching_ns","arrow_output_ns","arrow_charge_ns","arrow_columns_ns","arrow_assembly_ns","arrow_bookkeeping_ns","left_commit_ns","right_eviction_ns","emit_ns","residual_ns")
groups=("emitting","empty_no_frontier","empty_frontier","empty_ended")
series={side:[] for side in ("baseline","candidate")}
no_frontier={side:{} for side in series}
with tarfile.open(archive) as tar:
    read=lambda path:tar.extractfile(path).read()
    case=json.loads(read("py-trace/samples.json"))
    assert case["status"]=="ok" and len(case["evidence"])==2
    for rnd,rd in enumerate(case["evidence"]):
        for side,results in series.items():
            directory=f"py-trace/measurement/round-{rnd}/{side}"
            raw=read(directory+"/asof-finalize.jsonl")
            index={r["request_id"]:r for line in read(directory+"/trace-request-ranges.jsonl").splitlines() if (r:=json.loads(line))}
            for pos,sample in enumerate(rd["samples"][side]):
                trace=sample["trace_range"];idx=index[trace["request_id"]]
                assert all(idx[k]==trace[k] for k in idx)
                journal=json.loads(read(directory+f"/request-{idx['request_id']:04}.completed.json"))
                assert journal["response"]==sample and journal["context"]["sample"]==pos and journal["context"]["side"]==side
                assert sample["correctness"]==dict(passed=True,rows=1_000_000,finite_rows=1_000_000,max_abs_error=0.)
                records=[json.loads(x) for x in raw[idx["byte_start"]:idx["byte_end"]].splitlines()]
                assert len(records)==idx["records"] and records[-1]["ended"] and records[-1]["after"]["state_rows"]==0
                assert all(records[-1]["after"][k]==1_000_000 for k in ("emitted_left_rows","matched_rows","evicted_right_rows"))
                totals={g:Counter() for g in groups}
                for r in records:
                    assert r["pid"]==idx["pid"] and r["success"] and r["schema"]=="calc-flow.asof-watermark-exclusive.v1"
                    assert r["wall_ns"]==sum(r[k] for k in ("matching_ns","arrow_output_ns","left_commit_ns","right_eviction_ns","emit_ns","residual_ns"))
                    assert r["arrow_output_ns"]==sum(r[k] for k in fields[3:7])
                    group=("emitting" if r["emitted_delta"] else "empty_ended" if r["ended"] else "empty_frontier" if r["frontier"] is not None else "empty_no_frontier")
                    totals[group].update({k:r[k] for k in fields})
                    totals[group].update(callbacks=1,evicted=r["evicted_delta"])
                    if group=="empty_no_frontier":
                        delta=r["after"]["state_bytes"]-r["before"]["state_bytes"]
                        bucket=no_frontier[side].setdefault(str(delta),[])
                        bucket.append({"round":rnd+1,"sample":pos+1,"right_ms":r["right_eviction_ns"]*1e-6,"matched":r["matched_delta"],"evicted_payloads":r["evicted_delta"]})
                assert all(sum(m[k] for m in totals.values())==trace["stages"]["all"][k] for k in fields)
                results.append(totals)
assert all(len(x)==20 for x in series.values())
output={}
for group in groups:
    data={}
    for k in (*fields,"callbacks","evicted"):
        factor=1e-6 if k.endswith("_ns") else 1
        a,b=([r[group][k]*factor for r in arm] for arm in series.values())
        diff=[y-x for x,y in zip(a,b)]
        data[k]={"release_mean":statistics.mean(a),"main_mean":statistics.mean(b),"paired_mean_delta":statistics.mean(diff),"round_means":[statistics.mean(diff[:10]),statistics.mean(diff[10:])]}
    output[group]=data
output["empty_no_frontier_by_state_bytes"] = no_frontier
Path(sys.argv[2]).write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
