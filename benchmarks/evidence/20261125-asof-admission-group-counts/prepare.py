"""Use the same checked-in workload and complete output oracle for every revision."""
import subprocess
import sys
from pathlib import Path

checkout = Path(sys.argv[1])
out = Path(__file__).resolve().parent
path = "crates/calc-flow/benches/stream_asof_e2e.rs"
fixed = subprocess.check_output(["git", "show", "a35ad02a0b065a91505b94304abb6c67e3ac5644:" + path], cwd=checkout)
assert (checkout / path).read_bytes() == fixed
body = "#![allow(dead_code, unused_imports)]\n" + fixed.decode().split("    fn sample(", 1)[0]
body += """
    pub(super) fn main() {
        let case = Case::AdmitSettle;
        let batches = prepared_batches(case);
        let job = StreamJobContext::new(1, "asof-e2e", JsonMap::new(), None, CancellationToken::new());
        let mut operator = operator(case);
        let input = StreamOperatorContext::new(&job, "asof", None);
        let mut collector = Collector::default();
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        runtime.block_on(async {
            let started = Instant::now();
            for side in ["right", "left"] {
                for (batch, _) in &batches {
                    operator.process_data(side, batch.clone(), &input, &mut collector).await.unwrap();
                }
            }
            let seconds = started.elapsed().as_secs_f64();
            let admitted = operator.status();
            assert_eq!(admitted.pending_left_rows, ROWS as u64);
            assert_eq!(admitted.retained_right_rows, ROWS as u64);
            assert_eq!(admitted.emitted_left_rows, 0);
            assert!(collector.output.is_empty());
            let frontier = batches.last().unwrap().1;
            let context = progress(&job, frontier);
            let settled = Instant::now();
            operator.on_watermark(EventTime::from_micros(frontier), &context, &mut collector).await.unwrap();
            let watermark_seconds = settled.elapsed().as_secs_f64();
            let rows = validate_output(&collector, case);
            let status = operator.status();
            assert_eq!(status.emitted_left_rows, ROWS as u64);
            assert_eq!(status.matched_rows, ROWS as u64);
            assert_eq!(status.evicted_right_rows + status.retained_right_rows, ROWS as u64);
            println!("{}", json!({
                "seconds": seconds, "watermark_seconds": watermark_seconds,
                "output_rows": rows, "validated_all_rows": true,
                "evicted_right_rows": status.evicted_right_rows, "retained_right_rows": status.retained_right_rows,
                "pending_before_watermark": admitted.pending_left_rows,
                "retained_before_watermark": admitted.retained_right_rows
            }));
        });
    }
}
fn main() { legacy::main(); }
"""
target = checkout / "crates/calc-flow/examples/stage_admit.rs"
if target.exists() and target.read_text() != body:
    raise ValueError("unexpected existing adapter")
target.write_text(body)
(out / "stage_admit.rs").write_text(body)
