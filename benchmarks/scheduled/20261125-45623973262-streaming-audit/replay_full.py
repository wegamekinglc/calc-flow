"""Reproduce the scoped inventory from the original scheduled report and 10M shard."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

from scripts.benchmark_suite.catalog import BATCH_ROWS, CHECKPOINT_ROW_SCALES, INTERVAL_MAX_ROWS, ROW_SCALES, STREAM_CASES, engine_cases
from scripts.benchmark_suite.report import comparison, table
from scripts.toolkit import linear_percentile

R=Path(sys.argv[1]).resolve(); dest=Path(sys.argv[2]).resolve();dest.mkdir(parents=True,exist_ok=True)
sealed=json.loads((R/"sources.json").read_text()) if (R/"original-complete-report.zip").is_file() else None
run=sealed["actions_run"] if sealed else json.loads((R/"run.json").read_text());assert run["event"] in {"schedule","workflow_dispatch"} and run["status"]=="completed"
rid=run["id"];repo="https://github.com/wegamekinglc/calc-flow";url=run["html_url"];head=run["head_sha"]
base="7012b5b409553df04e29e6d7f9a11abf919cbf93";native=sealed["native_comparison_baseline"] if sealed else (R/"first-parent.txt").read_text().strip()
artifacts=sealed["artifacts"] if sealed else json.loads((R/f"artifacts-{rid}.json").read_text())["artifacts"]
arts={a["name"]:a for a in artifacts}
def blob(n):return hashlib.sha256(n).hexdigest()
def load(p,name):
 b=(R/p).read_bytes();a=arts[name];assert len(b)==a["size_in_bytes"] and "sha256:"+blob(b)==a["digest"]
 return zipfile.ZipFile(R/p)
report=f"benchmark-complete-report-{rid}-{run['run_attempt']}";big=f"benchmark-suite-engines-10000000-{rid}-{run['run_attempt']}"
complete_name="original-complete-report.zip" if sealed else f"report-{rid}.zip"
z=load(complete_name,report)
raw=z.read("summary.json");summ=json.loads(raw);md=z.read("summary.md").decode(); assert base in md.splitlines()[2] and native in md.splitlines()[4]
if sealed:
 sr=gzip.decompress((R/"original-10m-shard-results.json.gz").read_bytes());assert blob(sr)==sealed["original_10m_shard_results_json_sha256"] and blob(raw)==sealed["original_complete_summary_json_sha256"];cpu=sealed["10m_shard_host_cpu_json"]
else:
 large=load(f"engines-10000000-{rid}.zip",big);sr=large.read("results.json");cpu=json.loads(large.read("host-cpu.json"))
shard=json.loads(sr);assert shard["releases"]["baseline"]["git_sha"]==base and shard["releases"]["candidate"]["git_sha"]==head
cases=summ["cases"];lookup={c["id"]:c for c in cases};assert len(lookup)==len(cases) and len({c["id"] for c in shard["cases"]})==len(shard["cases"])
assert all(lookup[c["id"]]==c for c in shard["cases"])
assert {c["id"] for c in engine_cases(10_000_000)}=={c["id"] for c in shard["cases"]}
def write(n,data): (dest/n).write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
def export_csv(n,data,fields=None):
 with (dest/n).open("w",newline="") as out:
  writer=csv.DictWriter(out,fieldnames=fields or list(data[0]),lineterminator="\n");writer.writeheader();writer.writerows(data)
def artifact_id(c):
 fam=c["family"]
 shard= ("engines-"+str(c["rows"]) if fam=="engines" else "warm-"+str(c["history_rows"]) if fam=="warm" else c["id"].split("/")[0] if fam=="python" else fam)
 return arts[f"benchmark-suite-{shard}-{rid}-{run['run_attempt']}"]["id"]
def rowscope(c):
 f=c["family"];b=c["backend"];s=c["scenario"]
 if f=="engines":return "python-public-managed-stream" if b.startswith("calc-flow-stream") else "python-public-finite-sql" if b.startswith("calc-flow-sql") else "third-party-reference"
 if f=="warm":return "python-public-warm-stream"
 if f=="rust":
  if s.startswith(("stream_","m4_state_window/")):return "rust-native-stream"
  if s.startswith(("core/rolling_grouped_lanes/", "core/rolling_state_history/")):return "rust-rolling-kernel"
  if s.startswith(("core/stream/", "edge_channel_")):return "rust-stream-infrastructure"
  if s.startswith(("sql_datafusion_performance/", "core/execute/sql_", "core/execute/datafusion_")):return "rust-finite-sql"
  return "rust-other"
 return "python-lifecycle" if f=="lifecycle" else "python-supplemental-suite"
def readme_row(c):
 a=c["result"];return f"{a['head_p50']*1000:,.2f}",f"{a['change_percent']:+.1f}%",a["verdict"]
rows=[]
for c in sorted(cases,key=lambda q:q["id"]):
 f=c["family"];b=c["backend"];k=c.get("comparison");meta=c.get("metadata",{});res=c.get("result") or {}; historical=k=="interleaved";e=c.get("evidence",[]);o=set();samples=0;verified=False
 ok=c.get("status")=="ok"
 if historical and ok:
  assert c["status"]=="ok" and c["correctness"] and len(e)==2
  assert comparison(c)["verdict"]==res["verdict"] and comparison(c)["round_intervals"]==res["round_intervals"]
  assert all(len(c[arm])==2 and all(len(rr)==10 for rr in c[arm]) for arm in ("baseline","candidate"))
  for i,rd in enumerate(e):
   for arm in ("baseline","candidate"):
    ss=rd["samples"][arm]
    assert rd["completion"][arm]["state"]=="completed" and [d["seconds"] for d in ss]==c[arm][i]
    assert all(d["correctness"]["passed"] is True for d in ss)
    samples+=len(ss)
    for d in ss:
     if "rows" in d["correctness"]:o.add(d["correctness"]["rows"])
  assert len(o)<=1 and samples==40
  if res["verdict"]=="regression":assert all(x["low"]>5 for x in res["round_intervals"])
  verified=True
 elif f=="engines" and ok and e:
  o.update(d["correctness"]["rows"] for rd in e for d in rd["samples"].get("candidate",[]) if "rows" in d.get("correctness",{}))
 feed=c.get("rows"); variants=c["id"].split("/")[4:] if f=="engines" else []
 oracle=meta.get("oracle") if isinstance(meta.get("oracle"),dict) else {}
 input_n = feed if f=="engines" else meta.get("input_rows",oracle.get("input_rows"))
 right=None
 if f=="engines" and b.startswith("calc-flow-stream"):
  if c["scenario"] in ("interval_join","asof_join"):right=feed
  if c["scenario"]=="join":right=min(64,max(1,feed//40))
 v=res.get("verdict")
 classif=("invalid-evidence" if not ok else v if verified else "metric-only" if c.get("kind")=="metric" else "external-reference" if k=="external" else "informational-only" if k=="suite-blocks" else v)
 ci=res.get("round_intervals",[]) if ok else [];res=res if ok else {}
 rec=dict(case_id=c["id"],surface=rowscope(c),family=f,backend=b,scenario=c["scenario"],variant="/".join(variants) or "default",threads=(1 if b.endswith("-1t") else 4) if b.startswith(("calc-flow-stream","calc-flow-sql")) else None,scope=c["scope"],dimension_rows=feed,primary_fixture_rows=input_n,right_fixture_rows=right,history_rows=c.get("history_rows"),timed_append_rows=feed if f=="warm" else None,batch_rows=c.get("batch_rows",BATCH_ROWS if f=="engines" and b.startswith("calc-flow-stream") else None),checkpoint_interval_millis=c.get("checkpoint_interval_millis"),oracle_output_rows=next(iter(o)) if len(o)==1 else oracle.get("output_rows"),baseline_source=native if f=="rust" else (None if k=="external" else base),candidate_source=head,baseline_p50_ms=1000*res["base_p50"] if res.get("base_p50") is not None else None,candidate_p50_ms=1000*res["head_p50"] if res.get("head_p50") is not None else None,baseline_p95_ms=1000*linear_percentile([t for rnd in c["baseline"] for t in rnd],0.95) if ok and c.get("baseline") else None,candidate_p95_ms=1000*res["head_p95"] if res.get("head_p95") is not None else None,change_percent=res.get("change_percent"),comparison=k,reported_verdict=v,audited_classification=classif,reported_status=c["status"],reported_correctness=c["correctness"],paired_full_sample_evidence_checked=verified,validated_formal_executions=samples if historical else None,round1_median_percent=ci[0]["median"] if ci else None,round1_lower_percent=ci[0]["low"] if ci else None,round1_upper_percent=ci[0]["high"] if len(ci)>0 else None,round2_median_percent=ci[1]["median"] if len(ci)>1 else None,round2_lower_percent=ci[1]["low"] if len(ci)>1 else None,round2_upper_percent=ci[1]["high"] if len(ci)>1 else None,artifact_url=f"{url}/artifacts/{artifact_id(c)}")
 rows.append(rec)
errors=set(summ["errors"]); regressions=[r for r in rows if r["audited_classification"]=="regression"]
assert all(any(error.startswith(c["id"]+":") for error in errors) for c in cases if c["status"]!="ok")
repeated={r["case_id"]+": repeated regression exceeds +5%" for r in regressions}
assert repeated=={error for error in errors if error.endswith(": repeated regression exceeds +5%")}
write("report-errors.json", {"regression_errors":sorted(repeated),"other_report_errors":sorted(errors-repeated)})
export_csv("all-scheduled-cases.csv",rows)
export_csv("confirmed-regressions.csv",sorted(regressions,key=lambda r:-r["change_percent"]),list(rows[0]))
coverage=[]
for b in ("calc-flow-stream","calc-flow-stream-1t"):
 for op in STREAM_CASES:
  rs=[r for r in rows if r["family"]=="engines" and r["backend"]==b and r["scenario"]==op]
  defaults=[r for r in rs if r["variant"]=="default"]; mx=max(r["primary_fixture_rows"] for r in defaults)
  ref=next(r for r in defaults if r["primary_fixture_rows"]==mx)
  coverage.append(dict(backend=b,scenario=op,reason= "catalog cap" if op=="interval_join" else "largest declared engine scale",catalog_cap_rows=INTERVAL_MAX_ROWS if op=="interval_join" else None,largest_tested_default_input=mx,measured_at_10m=mx==10000000,default_case_at_largest=ref["case_id"],largest_default_verdict=ref["audited_classification"],additional_10m_variants=",".join(r["variant"] for r in rs if r["primary_fixture_rows"]==10000000 and r["variant"]!="default")))
export_csv("stream-operator-coverage.csv",coverage)
gzipdata=gzip.compress(sr,compresslevel=9,mtime=0)
(dest/"original-10m-shard-results.json.gz").write_bytes(gzipdata)
(dest/"original-complete-report.zip").write_bytes((R/complete_name).read_bytes())
sources={"actions_run":{k:run[k] for k in ("id","run_attempt","status","conclusion","event","head_sha","created_at","updated_at","html_url")},"public_release_baseline":base,"native_comparison_baseline":native,"original_complete_summary_json_sha256":blob(raw),"original_10m_shard_results_json_sha256":blob(sr),"10m_shard_host_cpu_json":cpu,"sealed_10m_releases":shard["releases"],"catalog":"https://github.com/wegamekinglc/calc-flow/blob/"+head+"/scripts/benchmark_suite/catalog.py","fixture":"https://github.com/wegamekinglc/calc-flow/blob/"+head+"/benchmarks/engine_comparison.py","artifacts":[{k:a[k] for k in ("id","name","size_in_bytes","digest","created_at","expires_at")} for a in artifacts],"summary":{"scheduled_rows":len(rows),"independently_verified_paired_cases":sum(r["paired_full_sample_evidence_checked"] for r in rows),"10m_shard_rows_crossmatched":len(shard["cases"]),"qualifying_regressions":len(regressions),"invalid_evidence_rows":sum(r["audited_classification"]=="invalid-evidence" for r in rows),"other_report_errors":len(errors-repeated),"regressions_by_surface_backend":dict(Counter(r["surface"]+"/"+r["backend"] for r in regressions))}}
write("sources.json",sources)
print(sources["summary"])
