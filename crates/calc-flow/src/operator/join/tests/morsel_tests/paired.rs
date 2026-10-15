//! Fixed-scope diagnostic, not release acceptance or a normal timing test.
use super::{support::*, *};
use crate::benchmark_internals::PhaseClock;

#[test]
#[ignore = "explicit D7 paired wall/process/per-thread diagnostic"]
fn test_morsel_paired_resources() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    let budget = std::time::Instant::now();
    for (rows, projected) in [(ROWS, false), (1024, false), (ROWS, true)] {
        let source = input(true, rows, false);
        for round in 0..2 {
            for pair in 0..10 {
                let order = if pair % 2 == 0 {
                    [true, false]
                } else {
                    [false, true]
                };
                for split in order {
                    assert!(
                        budget.elapsed() < Duration::from_secs(600),
                        "fixed collection budget exhausted"
                    );
                    runtime.block_on(sample(
                        &service, &source, rows, projected, round, pair, split,
                    ));
                }
            }
        }
    }
    drop(runtime);
    service.shutdown();
}

async fn sample(
    service: &TestService,
    source: &Batch,
    rows: usize,
    projected: bool,
    round: usize,
    pair: usize,
    split: bool,
) {
    let _mode = data_work::configure_for_test(split);
    let mut operator = operator(projected);
    let job = job().with_gather_owner(service.owner("morsel-paired".into()));
    preload(&mut operator, &job, true).await;
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let clock = PhaseClock::start("join-process-data");
    operator
        .process_data("left", source.clone(), &context, &mut output)
        .await
        .unwrap();
    let resources = clock.finish();
    verify(&mut output, rows, true, projected);
    println!(
        "{}",
        serde_json::json!({"rows":rows,"projected":projected,"round":round,"pair":pair,"split":split,"resources":resources})
    );
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}
