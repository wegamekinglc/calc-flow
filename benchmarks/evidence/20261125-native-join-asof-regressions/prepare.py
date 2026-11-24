"""Prepare two single-observation adapters for the unchanged repository fixtures."""

import subprocess
import sys
from pathlib import Path

checkout = Path(sys.argv[1]).resolve()
src = checkout / "crates/calc-flow/benches"
examples = checkout / "crates/calc-flow/examples"
out = Path(__file__).resolve().parent
base = "92230f68c9024a053f2e7c74185c3a98f5003917"


def original(name):
    rel = "crates/calc-flow/benches/" + name + ".rs"
    fixed = subprocess.check_output(["git", "show", f"{base}:{rel}"], cwd=checkout)
    if (src / (name + ".rs")).read_bytes() != fixed:
        raise ValueError(f"changed fixture {name}")
    return fixed.decode()


asof = original("stream_asof_e2e").split("    pub(super) fn main() {", 1)[0]
asof += """
    pub(super) fn main() {
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let case = Case::AdmitSettle;
        let batches = prepared_batches(case);
        println!("{}", sample(&runtime, case, &batches, true));
    }
}
fn main() { legacy::main(); }
"""
join = original("stream_join_perf")
join_helpers = join[join.index("const SECOND:") : join.index("fn read_rss()")]
join_oracle = join[
    join.index("fn check_pairs(") : join.index("async fn check_recovery(")
]
join = (
    join[join.index("use std::collections::BTreeMap;") : join.index("#[path = ")]
    + join_helpers
    + join_oracle
)
join += """
fn main() {
    let keys = unique_keys("U", ROWS);
    let left = valued_batch(&keys, BASE_TS, 3);
    let right = valued_batch(&keys, BASE_TS + SECOND, 7);
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let job = job();
        let mut operator = new_operator();
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        feed(&mut operator, &job, &mut collector, "left", &left).await;
        let started = Instant::now();
        operator.process_data("right", right.clone(), &plain_ctx(&job), &mut collector).await.unwrap();
        let seconds = started.elapsed().as_secs_f64();
        check_pairs(&mut collector, &left, &right, 1);
        println!("{}", serde_json::json!({"seconds": seconds, "output_rows": ROWS, "validated_all_rows": true}));
    });
}
"""
for name, body in (("asof", asof), ("join", join)):
    body = "#![allow(dead_code, unused_imports)]\n" + body
    dest = examples / ("stage_" + name + ".rs")
    if dest.exists() and dest.read_text() != body:
        raise ValueError(f"refusing to replace {dest}")
    dest.write_text(body)
    (out / ("stage_" + name + ".rs")).write_text(body)
