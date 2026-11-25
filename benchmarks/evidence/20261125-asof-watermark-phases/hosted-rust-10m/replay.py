"""Replay hosted ASOF product clocks, request oracles and exclusive trace closure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import tarfile
import zipfile
import sys

from diagnostic_validate import counts
from scripts.benchmark_suite.report import _paired_verdict
from scripts.benchmark_suite.statistics import paired_round
from scripts.toolkit import linear_percentile
from verify import verify

PHASES = ["wall_ns","matching_ns","arrow_output_ns","arrow_charge_ns","arrow_columns_ns","arrow_assembly_ns","arrow_bookkeeping_ns","left_commit_ns","right_eviction_ns","emit_ns","residual_ns"]
PRODUCTS = ["7012b5b409553df04e29e6d7f9a11abf919cbf93","de20f8216c8dcb1bfce3e3811c3884a3080157df"]
TRACE_ARMS=["d5b81630f0c2e3e75063e9be0a905fa0f7b93761","d4f39c6d988fc982c34686cc5d8e90690cde611b"]
TRACE_TREES=["ff284ee0b01495313ce31754999314295eba59bb","8711e80beb786bfa7a4d41d2134a9f70b6ac7951"]
ROOT = Path(sys.argv[1]).resolve()
RAW = ROOT / "raw"


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def authentic_raw(surface):
    # Verify a one-to-one mapping between every measured input and archive bytes.
    mapped={}
    with zipfile.ZipFile(ROOT/"original.zip") as outer:
        for entry in outer.infolist():
            if entry.is_dir():continue
            path=Path(entry.filename)
            assert not path.is_absolute() and ".." not in path.parts and path not in mapped
            with outer.open(entry) as f:mapped[path]=hashlib.file_digest(f,"sha256").hexdigest()
    if surface=="wheel":
        arc=Path("wheel-study-raw.tar.gz")
        assert set(mapped)=={arc} and digest(RAW/arc)==mapped[arc]
        with tarfile.open(RAW/arc) as packed:
            for entry in packed:
                path=Path(entry.name)
                assert entry.isfile() and not path.is_absolute() and ".." not in path.parts and path not in mapped
                with packed.extractfile(entry) as f:mapped[path]=hashlib.file_digest(f,"sha256").hexdigest()
    actual={p.relative_to(RAW) for p in RAW.rglob("*") if p.is_file()}
    assert actual==set(mapped)
    for path,expected in mapped.items():assert digest(RAW/path)==expected, path


def timeline(base, head):
    assert len(base)==len(head)==20
    diffs=[b-a for a,b in zip(base,head)]
    ci=[paired_round(base[i:i+10],head[i:i+10]) for i in (0,10)] if min(base)>0 and min(head)>0 else []
    return {"release_p50":statistics.median(base),"main_p50":statistics.median(head),
            "release_p95":linear_percentile(base,.95),"main_p95":linear_percentile(head,.95),
            "mean_delta_ms":statistics.mean(diffs),"round_mean_delta_ms":[statistics.mean(diffs[i:i+10]) for i in (0,10)],
            "round_intervals":ci,"verdict":_paired_verdict(ci) if ci else None}


def metrics(arms):
    return {group:{k:timeline(*[[s[group][k]*1e-6 for s in arm] for arm in arms]) for k in PHASES} for group in ("all","emitting","other")}


def verify_terminal(item):
    assert item["seconds"]>0 and item["correctness"]=={"passed":True,"rows":10_000_000,"finite_rows":10_000_000,"max_abs_error":0.}
    evidence=item["stream_evidence"];s=evidence["terminal_asof_status"]
    assert evidence["post_eof_output_rows"]==10_000_000
    for name in ("left","right"):
        a=s[name];assert a["ended"] and a["accepted_rows"]==10_000_000 and a["late_rows"]==a["duplicate_rows"]==0
    assert all(s[k]==10_000_000 for k in ("emitted_left_rows","matched_rows","evicted_right_rows"))
    assert all(s[k]==0 for k in ("state_rows","pending_left_rows","retained_right_rows","identity_only_rows","unmatched_rows","output_limit_failures","workspace_limit_failures","state_limit_failures"))


def terminal_trace(records, state):
    assert records and records[-1]["ended"] is True and records[-1]["frontier"] is None
    final=records[-1]["after"]
    assert all(final[k]==state[k] for k in final)
    assert final["evicted_right_rows"]==10_000_000
    by_role={}
    for r in records:
        role=("emitting" if r["emitted_delta"] else "empty_ended" if r["ended"] else "empty_no_frontier" if r["frontier"] is None else "empty_frontier")
        group=by_role.setdefault(role,{"count":0,"wall_ns":0,"right_eviction_ns":0,"state_bytes_delta":0,"payloads_retired":0})
        group["count"]+=1;group["wall_ns"]+=r["wall_ns"];group["right_eviction_ns"]+=r["right_eviction_ns"]
        group["state_bytes_delta"]+=r["after"]["state_bytes"]-r["before"]["state_bytes"]
        group["payloads_retired"]+=r["evicted_delta"]
    return {"callbacks":len(records),"emitting":sum(bool(r["emitted_delta"]) for r in records),
            "ended":sum(r["ended"] for r in records),"by_role":by_role}


def native(mode, arms):
    base,head=arms;d=RAW/f"obs-job-{base[:10]}-{head[:10]}"
    if not (d/"outcome.json").exists():return {"state":"no cohort artifact"}
    if mode=="trace":assert [read(d/(s+".receipt.json"))["tree"] for s in ("base","head")]==TRACE_TREES
    info=verify(d);out=read(d/"outcome.json"); obs=lines(d/"observations.jsonl")
    assert read(d/"identity.json")["base"]==base and read(d/"identity.json")["head"]==head
    result={"verification":info,"outcome":out,"accepted_warmups":sum(x["pair"]==0 for x in obs),"accepted_formal_arms":sum(x["pair"]>0 for x in obs)}
    if mode=="trace":
        result["trace_inventory"]=[]
        for row in obs:
            data=row["result"];records=lines(d/data["trace"]);assert counts(records,data["pid"])==data["phases"]
            result["trace_inventory"].append({"round":row["round"],"pair":row["pair"],"side":row["side"],**terminal_trace(records,data["state"])})
    if out["state"]!="complete":return result
    series=[[x["result"] for x in obs if x["side"]==side and x["pair"]>0] for side in ("base","head")]
    outer=timeline(*[[s["seconds"]*1000 for s in arm] for arm in series])
    assert outer["verdict"]==out["verdict"] and outer["round_intervals"]==out["rounds"]
    result.update(outer=outer,cpu=out["cpu"])
    if mode=="trace":result["stages"]=metrics([[s["phases"] for s in arm] for arm in series])
    return result


def wheel(mode, arms):
    d=RAW/("wheel-"+mode)
    if not (d/"outcome.json").exists():return {"state":"no cohort artifact"}
    outcome=read(d/"outcome.json");result={"outcome":outcome}
    trace_idx={}
    if mode=="trace":
        for p in sorted(d.rglob("trace-request-ranges.jsonl")):
            offset=0
            for entry in lines(p):
                assert entry["byte_start"]==offset
                offset=entry["byte_end"];key=(entry["pid"],entry["request_id"])
                assert key not in trace_idx;trace_idx[key]=(p.parent/entry["trace"],entry)
    def lookup(a, state):
        path,r=trace_idx[a["pid"],a["request_id"]]
        with path.open("rb") as f:
            f.seek(r["byte_start"]);payload=f.read(r["byte_end"]-r["byte_start"])
        rec=[json.loads(x) for x in payload.splitlines()]
        assert len(rec)==r["records"] and counts(rec,r["pid"])==a["stages"]
        return terminal_trace(rec,state)
    accepted={"prepare":0,"sample":0};journal={};traced=[]
    for p in sorted(d.rglob("*.completed.json")):
        record=read(p);response=record.get("response",{});item=response.get("warmup",response)
        if "seconds" in item and "correctness" in item:
            verify_terminal(item);accepted[record["operation"]]+=1
            ctx=record["context"];side=ctx["side"];rnd=ctx["round"]
            assert side==p.parent.name and p.parent.parent.name=="round-"+str(rnd)
            assert ctx["release"]["git_sha"]==arms[("baseline","candidate").index(side)]
            key=(rnd,side,ctx["sample"]);assert key not in journal
            journal[key]=item
            if mode=="trace":
                a=item["trace_range"]
                assert a["request_id"]==record["request_id"] and a["operation"]==record["operation"]
                assert trace_idx[a["pid"],a["request_id"]][0].parent==p.parent
                traced.append({"round":rnd,"side":side,"sample":ctx["sample"],**lookup(a,item["stream_evidence"]["terminal_asof_status"])})
    result["accepted"]=accepted
    if mode=="trace":result["trace_inventory"]=traced
    if outcome["state"]!="completed":return result
    assert accepted=={"prepare":4,"sample":40}
    q=read(d/"qualification.json");case=read(d/"samples.json");res=read(d/"result.json")
    if mode=="trace":assert [q["builds"][s]["tree"] for s in ("baseline","candidate")]==TRACE_TREES
    assert [q["builds"][s]["sha"] for s in ("baseline","candidate")]==arms
    assert all(all(c["suitable"] for c in r["cores"]) for r in lines(d/"resources.jsonl"))
    assert len(res["cpus"])==1
    series=[[s for rd in case["evidence"] for s in rd["samples"][side]] for side in ("baseline","candidate")]
    for rnd,rd in enumerate(case["evidence"]):
        for side in ("baseline","candidate"):
            for pos,sample in enumerate(rd["samples"][side]):
                assert sample==journal[(rnd,side,pos)]
                verify_terminal(sample)
    # Replay warmup traces from their worker boundary as well.
    if mode=="trace":
        requests=[(p,r) for p,r in trace_idx.values() if r["operation"] in ("prepare","sample")]
        assert len(requests)==44
        for p,r in requests:
            with p.open("rb") as f:
                f.seek(r["byte_start"]);payload=f.read(r["byte_end"]-r["byte_start"])
            rec=[json.loads(x) for x in payload.splitlines()]
            assert len(rec)==r["records"];counts(rec,r["pid"])
    outer=timeline(*[[s["seconds"]*1000 for s in arm] for arm in series])
    assert outer["verdict"]==res["verdict"]
    for k,j in [("base_p50","release_p50"),("head_p50","main_p50"),("base_p95","release_p95"),("head_p95","main_p95")]:
        assert abs(outer[j]-1000*res[k])<1e-7
    result.update(outer=outer,cpu=res["cpus"][0])
    if mode=="trace":result["stages"]=metrics([[s["trace_range"]["stages"] for s in arm] for arm in series])
    return result


meta=read(ROOT/"artifact.json")
surface="wheel" if meta["name"].startswith("hosted-asof-wheel-10m-") else "native"
assert surface=="wheel" or meta["name"].startswith("hosted-asof-watermark-10m-")
assert meta["digest"]=="sha256:"+digest(ROOT/"original.zip")
if not RAW.is_dir():
    scratch=tempfile.TemporaryDirectory()
    RAW=Path(scratch.name)
    with zipfile.ZipFile(ROOT/"original.zip") as z:
        assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())
        z.extractall(RAW)
    if surface=="wheel":
        with tarfile.open(RAW/"wheel-study-raw.tar.gz") as tf:
            tf.extractall(RAW,filter="data")
authentic_raw(surface)
obj={"source":read(ROOT/"run.json")["html_url"],"artifact_id":meta["id"],"digest":meta["digest"]}
arms_path=RAW/("arms.json" if surface=="native" else "wheel-arms.json")
if arms_path.exists():
    bindings=read(arms_path)
    assert [bindings["plain-"+s] for s in ("release","main")]==PRODUCTS
    assert [bindings["trace-"+s] for s in ("release","main")]==TRACE_ARMS
    for mode in ("plain","trace"):
        obj[mode]=(native if surface=="native" else wheel)(mode,[bindings[mode+"-"+s] for s in ("release","main")])
    if all("outer" in obj[m] for m in ("plain","trace")):
        assert obj["plain"]["cpu"]==obj["trace"]["cpu"]
        if surface=="wheel":
            for side in ("baseline","candidate"):
                a,b=[read(RAW/("wheel-"+m)/"qualification.json")["builds"][side]["release"] for m in ("plain","trace")]
                assert all(a[k]==b[k] for k in ("rustc","python","python_executable","build_profile","cargo_lock_sha256"))
                assert a["command"][:-1]==b["command"][:-1]
        obj["tracing_overhead_p50_percent"]={side:100*(obj["trace"]["outer"][side+"_p50"]/obj["plain"]["outer"][side+"_p50"]-1) for side in ("release","main")}
else:
    obj["qualification"]="No complete arm registry. See build logs."
(ROOT/"analysis.json").write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")
