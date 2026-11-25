"""Reproduce the release comparisons and stopped preflight from the archived run."""
import hashlib
import io
import json
from pathlib import Path
import statistics
import tarfile
import zipfile

from scripts.benchmark_suite.statistics import paired_round
from scripts.benchmark_suite.report import _paired_verdict
from scripts.toolkit import linear_percentile

base = Path(__file__).resolve().parent
packed = base / "original.zip"
art = json.loads((base / "artifact.json").read_text())
run = json.loads((base / "run.json").read_text())
assert art["id"] == 15035673616 and run["id"] == 45643368973
assert art["workflow_run"]["head_sha"] == run["head_sha"] == "7f3548e6b1b54da66cdb41f9978b89454d15f230"
assert "sha256:" + hashlib.sha256(packed.read_bytes()).hexdigest() == art["digest"]
with zipfile.ZipFile(packed) as z:
    bundle = io.BytesIO(z.read("wheel-study-raw.tar.gz"))
summary = {}
with tarfile.open(fileobj=bundle) as t:
    names=t.getnames()
    def read(n):
        return json.load(t.extractfile(n))
    for mode in ("runner","one"):
        d="wheel-"+mode+"/"
        qual=read(d+"qualification.json")
        builds=qual["builds"]
        ids=[builds[s]["sha"] for s in ("baseline","candidate")]
        assert ids==["7012b5b409553df04e29e6d7f9a11abf919cbf93","de20f8216c8dcb1bfce3e3811c3884a3080157df"]
        assert qual["scenario"]["id"]=="engines/1000000/calc-flow-stream-1t/asof_join"
        resources=[json.loads(b) for b in t.extractfile(d+"resources.jsonl")]
        mask=[c["cpu"] for c in resources[0]["cores"]]
        assert len(mask)==(4 if mode=="runner" else 1) and all([c["cpu"] for c in r["cores"]]==mask for r in resources)
        assert resources[0]["preflight"] and all(c["suitable"] for c in resources[0]["cores"])
        state=read(d+"outcome.json"); accepted={"prepare":0,"sample":0}
        def check(value):
            assert value["correctness"]=={"passed":True,"rows":1_000_000,"finite_rows":1_000_000,"max_abs_error":0.0}
            e=value["stream_evidence"]; s=e["terminal_asof_status"]
            assert e["post_eof_output_rows"]==1_000_000 and value["seconds"]>0
            for side in ("left","right"):
                assert s[side]["accepted_rows"]==1_000_000 and s[side]["ended"] and s[side]["late_rows"]==s[side]["duplicate_rows"]==0
            for k in ("emitted_left_rows","matched_rows","evicted_right_rows"):
                assert s[k]==1_000_000
            for k in ("state_rows","state_bytes","pending_left_rows","identity_only_rows","retained_right_rows","unmatched_rows","state_limit_failures","workspace_limit_failures","output_limit_failures"):
                assert s[k]==0
        for n in names:
            if n.startswith(d+"measurement/") and n.endswith(".completed.json"):
                r=read(n); v=r["response"]; v=v.get("warmup",v)
                if "seconds" in v and "correctness" in v:
                    check(v);accepted[r["operation"]]+=1
        summary[mode]={"mask":mask,"outcome":state,"accepted":accepted,"native_sha256":{s:v["release"]["native_sha256"] for s,v in builds.items()}}
        if state["state"]=="stopped":
            bad=[c for r in resources for c in r["cores"] if not c["suitable"]]
            assert bad and d+"result.json" not in names
            summary[mode]["rejected"]=bad
            continue
        assert all(c["suitable"] for r in resources for c in r["cores"]) and accepted=={"prepare":4,"sample":40}
        samples=read(d+"samples.json"); res=read(d+"result.json"); assert res["cpus"]==mask and res["oracle_checked_executions"]==44
        rounds=[];values={"baseline":[],"candidate":[]}
        for e in samples["evidence"]:
            assert e["environment"]["cpu_affinity"]==mask and e["environment"]["tokio_worker_threads"]=="1"
            groups=[]
            for arm in values:
                group=e["samples"][arm];assert len(group)==10
                for v in group:check(v)
                seconds=[v["seconds"] for v in group]
                values[arm].extend(seconds);groups.append(seconds)
            rounds.append(paired_round(*groups))
        assert len(rounds)==2 and rounds==res["round_intervals"] and _paired_verdict(rounds)==res["verdict"]
        for arm,key in [("baseline","base"),("candidate","head")]:
            assert statistics.median(values[arm])==res[key+"_p50"] and linear_percentile(values[arm],0.95)==res[key+"_p95"]
        summary[mode]["result"]=res
assert summary["runner"]["native_sha256"]==summary["one"]["native_sha256"]
print(json.dumps(summary,indent=2))
