use crate::operator::rolling_metrics::RollingMetricsStore;
use datafusion::arrow::array::StringArray;

use super::*;

#[tokio::test]
async fn test_incremental_accounting_warm_append_charges_only_new_history_rows() {
    let input = wide_input(513);
    let gap = float64_fast_record(&[(1_028, "a", 514, Some(514.0))]);
    let all = concat_batches(&input.schema(), &[input.clone(), gap.clone()]).unwrap();
    for resume in [false, true] {
        let spec = accounting_spec();
        let mut operator = RollingOperator::new("rolling", input.schema(), spec.clone()).unwrap();
        let expected = reference_output(&operator, &all);
        let (job, mut output) = job_output(&operator);
        let store = RollingMetricsStore::default();
        let mut emitted = finalize(
            &mut operator,
            &job,
            &store,
            &mut output,
            input.slice(0, 512),
            512,
        )
        .await;
        if resume {
            let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
            let mut restored = RollingOperator::new("rolling", input.schema(), spec).unwrap();
            StreamOperator::restore(&mut restored, &snapshot).unwrap();
            assert!(
                restored
                    .state
                    .histories
                    .by_entity
                    .values()
                    .all(|state| state.charge_cache.is_some())
            );
            operator = restored;
        }
        let before_append = store.snapshot().watermark.history_charge_scalar_rows;
        emitted.extend(
            finalize(
                &mut operator,
                &job,
                &store,
                &mut output,
                input.slice(512, 1),
                513,
            )
            .await,
        );
        let append_visits = store.snapshot().watermark.history_charge_scalar_rows - before_append;
        assert_live_charge(&operator);
        assert_exact_byte_boundary(&mut operator);
        let before_expiry = store.snapshot().watermark.history_charge_scalar_rows;
        emitted
            .extend(finalize(&mut operator, &job, &store, &mut output, gap.clone(), 1_028).await);
        let expiry_visits = store.snapshot().watermark.history_charge_scalar_rows - before_expiry;
        assert_live_charge(&operator);
        assert_eq!(
            concat_batches(&expected.schema(), &emitted).unwrap(),
            expected
        );
        eprintln!(
            "charge visits: resume={resume}, append={append_visits}, expired4_plus_new1={expiry_visits}"
        );
        assert!(
            append_visits <= 4,
            "one new row visited {append_visits} history rows"
        );
        assert!(
            expiry_visits <= 20,
            "four expired rows and one new row visited {expiry_visits} history rows"
        );
    }
}

#[test]
fn test_incremental_accounting_generic_replacements_ignore_and_invalidate_stale_cache() {
    let input = wide_input(3);
    let compiled = compile_spec(&accounting_spec(), input.schema().as_ref()).unwrap();
    let batch = Batch::table(vec![input], BatchMetadata::default()).unwrap();
    let rows = read_buffered_rows(batch.table_payload().unwrap(), &compiled, "rolling").unwrap();
    let first = compute_output_columns(
        &rows[..2],
        &RollingHistories::default(),
        &compiled,
        "rolling",
    )
    .unwrap();
    let mut histories = RollingHistories::default();
    histories.apply(first.touched);
    let (_, cache) = histories.prepare_charge_cache("rolling").unwrap();
    histories.install_charge_cache(cache);
    let previous = histories
        .by_entity
        .values()
        .next()
        .unwrap()
        .charge_cache
        .unwrap();
    let next = compute_output_columns(&rows[2..], &histories, &compiled, "rolling").unwrap();
    assert_eq!(next.touched[0].1.charge_cache, Some(previous));
    let expected = budget::changed_histories_charge(&histories, &next.touched, "rolling").unwrap();
    assert_eq!(expected.1.rows, 4);
    assert_ne!(expected.1, previous.total("rolling").unwrap());
    let mut prepared = PreparedHistoryUpdate::Replace(next.touched);
    assert_eq!(
        prepared.changed_charge(&histories, "rolling").unwrap(),
        expected
    );
    prepared.commit(&mut histories);
    assert!(
        histories
            .by_entity
            .values()
            .all(|state| state.charge_cache.is_none())
    );
    assert_eq!(
        budget::histories_charge(&histories, "rolling").unwrap(),
        expected.1
    );
}

fn accounting_spec() -> RollingSpec {
    kernel_spec(json!([
        duration_output("mean", "price", "time_mean", 1_024),
        aggregate_output("min", "price", "row_min", 64),
    ]))
}

fn wide_input(count: u32) -> RecordBatch {
    let rows = (1..=count)
        .map(|n| (i64::from(n), "a", u64::from(n), Some(f64::from(n))))
        .collect::<Vec<_>>();
    let input = float64_fast_record(&rows);
    let mut columns = input.columns().to_vec();
    columns[5] = Arc::new(StringArray::from(vec!["x".repeat(4_096); rows.len()]));
    RecordBatch::try_new(input.schema(), columns).unwrap()
}

fn assert_live_charge(operator: &RollingOperator) {
    assert_cached_rolling_charge(operator);
}

pub(super) fn assert_exact_byte_boundary(operator: &mut RollingOperator) {
    let charge = budget::full_state_charge(
        &operator.state.buffer,
        &operator.state.ordered,
        &operator.state.histories,
        &operator.compiled,
        "rolling",
    )
    .unwrap();
    let prior_budget = operator.state_budget;
    let prior_cache = operator
        .state
        .histories
        .by_entity
        .values()
        .map(|state| state.charge_cache)
        .collect::<Vec<_>>();
    for bytes in [charge.bytes - 1, charge.bytes, charge.bytes + 1] {
        let requested = StateBudget::new(charge.rows, bytes).unwrap();
        if bytes < charge.bytes {
            let error = operator.set_state_budget(requested).unwrap_err();
            assert!(
                matches!(error, CalcFlowError::InvalidArgument { field, .. } if field == "rolling.state_budget")
            );
            assert_eq!(operator.state_budget, prior_budget);
        } else {
            operator.set_state_budget(requested).unwrap();
            assert_eq!(operator.state_budget, requested);
        }
        assert_eq!(operator.state.charge, charge);
        assert_eq!(
            operator
                .state
                .histories
                .by_entity
                .values()
                .map(|state| state.charge_cache)
                .collect::<Vec<_>>(),
            prior_cache
        );
    }
    operator.set_state_budget(prior_budget).unwrap();
}

fn job_output(operator: &RollingOperator) -> (crate::StreamJobContext, crate::EdgeCollector) {
    (
        crate::StreamJobContext::new(
            7,
            TEST_FINGERPRINT,
            JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        ),
        crate::EdgeCollector::new(operator.output_ports().to_vec()),
    )
}

fn reference_output(operator: &RollingOperator, input: &RecordBatch) -> RecordBatch {
    let batch = Batch::table(vec![input.clone()], BatchMetadata::default()).unwrap();
    let rows = read_buffered_rows(
        batch.table_payload().unwrap(),
        &operator.compiled,
        "rolling",
    )
    .unwrap();
    let computed = compute_output_columns(
        &rows,
        &RollingHistories::default(),
        &operator.compiled,
        "rolling",
    )
    .unwrap();
    build_output_record(
        &rows,
        computed.columns,
        operator.output_ports()[0].schema().unwrap(),
        "rolling",
    )
    .unwrap()
}

async fn finalize(
    operator: &mut RollingOperator,
    job: &crate::StreamJobContext,
    store: &RollingMetricsStore,
    output: &mut crate::EdgeCollector,
    record: RecordBatch,
    watermark: i64,
) -> Vec<RecordBatch> {
    execution_diagnostics::observe(operator, job, store, output, record, watermark).await;
    output
        .drain("output")
        .into_iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .to_vec()
        })
        .collect()
}
