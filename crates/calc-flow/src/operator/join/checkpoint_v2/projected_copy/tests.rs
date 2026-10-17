use std::{
    cell::Cell,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use chrono::{DateTime, Utc};
use datafusion::{
    arrow::{
        array::{
            Array, ArrayRef, DictionaryArray, Float64Array, Int64Array, StringArray,
            StringViewArray, UInt64Array,
        },
        compute::take,
        datatypes::{Field, Int32Type, Schema},
        record_batch::RecordBatch,
    },
    execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool},
};

use super::{CHECK_STRIDE, SelectedRow, accounting, copy, reserve};
use crate::{
    CalcFlowError, CancellationToken, JsonMap, Result, StreamJobContext,
    runtime::streaming::gather_work::GatherStop,
};

// Row scans per column: run counting, run-end scans, and view or dictionary
// rebuild scans can each check once per stride inside one column.
const ROW_PASSES: usize = 6;
// Run slicing and admission check a bounded number of times per run, including
// the nested value copies a dictionary rebuild performs.
const RUN_CHECKS: usize = 32;

fn record(rows: usize) -> RecordBatch {
    let ids = (0..rows).map(|row| i64::try_from(row).unwrap());
    let views = (0..rows).map(|row| match row % 3 {
        0 => Some(format!("v{row}")),
        1 => Some(format!("long view value {row}")),
        _ => None,
    });
    let keys = (0..rows).map(|row| (row % 4 != 3).then(|| i32::try_from(row % 4).unwrap()));
    let dictionary = DictionaryArray::<Int32Type>::new(
        keys.collect(),
        Arc::new(StringArray::from(vec!["d0", "d1", "d2"])),
    );
    let values = (0..rows).map(|row| (!row.is_multiple_of(5)).then_some(1.5));
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from_iter_values(ids)),
        Arc::new(values.collect::<Float64Array>()),
        Arc::new(views.collect::<StringViewArray>()),
        Arc::new(dictionary),
    ];
    let fields = arrays
        .iter()
        .enumerate()
        .map(|(index, array)| Field::new(format!("c{index}"), array.data_type().clone(), true))
        .collect::<Vec<_>>();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn selected(sources: &[usize]) -> Vec<SelectedRow> {
    sources
        .iter()
        .map(|&source| SelectedRow {
            source,
            row_id: u64::try_from(source).unwrap(),
            time: crate::EventTime::from_micros(100),
        })
        .collect()
}

fn expected(record: &RecordBatch, sources: &[usize]) -> RecordBatch {
    let indices =
        UInt64Array::from_iter_values(sources.iter().map(|&source| u64::try_from(source).unwrap()));
    let columns = record
        .columns()
        .iter()
        .map(|array| take(array, &indices, None).unwrap())
        .collect();
    RecordBatch::try_new(record.schema(), columns).unwrap()
}

fn run_count(sources: &[usize]) -> usize {
    1 + sources
        .windows(2)
        .filter(|pair| pair[0].checked_add(1) != Some(pair[1]))
        .count()
}

/// Copies the selection with `check` and returns the result after all owners are dropped.
fn copied(
    record: &RecordBatch,
    sources: &[usize],
    check: &dyn Fn() -> Result<()>,
) -> Result<RecordBatch> {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(256 << 20));
    let credit = MemoryConsumer::new("projected-copy-stop").register(&pool);
    reserve(
        &credit,
        accounting::vector_peak::<SelectedRow>(sources.len()).unwrap(),
    )
    .unwrap();
    let rows = selected(sources);
    let result = copy(
        record,
        &rows,
        [record.schema(), record.schema()],
        &credit,
        None,
        check,
    )
    .map(|payload| payload.record().clone());
    let unpublished = result.is_err();
    drop((rows, credit));
    if unpublished {
        assert_eq!(pool.reserved(), 0, "a stopped copy left funded state");
    }
    result
}

fn counted(record: &RecordBatch, sources: &[usize]) -> usize {
    let checks = Cell::new(0);
    let copy = copied(record, sources, &|| {
        checks.set(checks.get() + 1);
        Ok(())
    })
    .unwrap();
    assert_eq!(copy, expected(record, sources));
    checks.get()
}

#[test]
fn test_selected_copy_checks_scale_with_strides_not_rows() {
    let fixed = counted(&record(1), &[0]);
    let rows = 3 * CHECK_STRIDE + 5;
    let strides = rows.div_ceil(CHECK_STRIDE);
    let record = record(rows);
    let columns = record.num_columns();
    let half = rows / 2;
    for sources in [
        (0..rows).collect::<Vec<_>>(),
        (half..rows).chain(0..half).collect(),
    ] {
        let runs = run_count(&sources);
        let checks = counted(&record, &sources);
        let allowed = fixed + columns * ROW_PASSES * strides + RUN_CHECKS * runs;
        assert!(
            checks <= allowed,
            "{checks} stop checks for {rows} rows in {runs} runs exceed {allowed}"
        );
        assert!(
            checks >= fixed + columns * (strides - 1),
            "{checks} stop checks leave a {rows}-row scan unchecked for over one stride"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum Raised {
    Cancellation,
    Deadline,
    Abandonment,
}

fn instant(seconds: i64) -> DateTime<Utc> {
    DateTime::from_timestamp(seconds, 0).unwrap()
}

fn raised_copy(record: &RecordBatch, sources: &[usize], raised: Raised, raise_at: usize) {
    let deadline = instant(1_000);
    let (clock, time) = tokio::sync::watch::channel(instant(999));
    let job = StreamJobContext::new(
        1,
        "projected-copy-stop",
        JsonMap::new(),
        Some(deadline),
        CancellationToken::new(),
    )
    .with_deadline_clock_for_test(time);
    let plain = GatherStop::from_job(&job);
    let abandon = plain.clone();
    let cancellation = job.cancellation().clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let stop = plain.with_check_hook(Arc::new(move |_| {
        if observed.fetch_add(1, Ordering::SeqCst) + 1 == raise_at {
            match raised {
                Raised::Cancellation => cancellation.cancel(),
                Raised::Deadline => {
                    clock.send_replace(deadline);
                }
                Raised::Abandonment => abandon.abandon_for_test(),
            }
        }
    }));
    let result = copied(record, sources, &|| stop.check());
    assert!(
        matches!(result, Err(CalcFlowError::Cancelled { .. })),
        "{raised:?} raised at check {raise_at} was not observed: {result:?}"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        raise_at,
        "{raised:?} raised at check {raise_at} did not stop the copy at that check"
    );
}

#[test]
fn test_selected_copy_observes_stop_raised_at_every_check() {
    let rows = 2 * CHECK_STRIDE + 3;
    let record = record(rows);
    let half = rows / 2;
    let sources = (half..rows).chain(0..half).collect::<Vec<_>>();
    let total = counted(&record, &sources);
    for raised in [Raised::Cancellation, Raised::Deadline, Raised::Abandonment] {
        for raise_at in 1..=total {
            raised_copy(&record, &sources, raised, raise_at);
        }
    }
}
