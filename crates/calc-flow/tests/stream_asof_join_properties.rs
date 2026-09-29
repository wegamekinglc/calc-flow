mod asof_support;
#[path = "asof_support/materialization.rs"]
mod materialization;
use asof_support::{batch, operator};
use calc_flow::{
    CalcFlowError, CancellationToken, EdgeCollector, Epoch, EventTime, IngressProgress,
    IngressProgressSnapshot, IngressState, JsonMap, OperatorMetadata, StreamJobContext,
    StreamOperator, StreamOperatorContext, StreamingFailureReason,
};
use datafusion::arrow::array::{Array, Int64Array};
use proptest::prelude::*;
use std::collections::BTreeMap;

#[tokio::test]
async fn backward_asof_is_inclusive_left_preserving_and_final() {
    for (tolerance, expected) in [(10, Some(100)), (4, None), (0, None)] {
        let mut op = operator(tolerance);
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let cx = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(op.output_ports().to_vec());
        op.process_data(
            "right",
            batch(&[("A", 90, 1, 90), ("A", 100, 2, 100), ("A", 110, 3, 110)]),
            &cx,
            &mut output,
        )
        .await
        .unwrap();
        op.process_data("left", batch(&[("A", 105, 1, 999)]), &cx, &mut output)
            .await
            .unwrap();
        assert!(output.drain("output").is_empty());
        op.on_end(&cx, &mut output).await.unwrap();
        let messages = output.drain("output");
        assert_eq!(messages.len(), 1);
        let table = messages[0].as_data().unwrap().table_payload().unwrap();
        let result = &table.batches()[0];
        assert_eq!(result.num_rows(), 1);
        let values = result
            .column(7)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(
            if values.is_null(0) {
                None
            } else {
                Some(values.value(0))
            },
            expected
        );
        assert!(result.schema().field(7).is_nullable());
        assert_eq!(op.status().emitted_left_rows, 1);
    }
}

#[tokio::test]
async fn typed_ties_and_legal_batch_interleavings_match_independent_oracle() {
    let right: Vec<_> = (0_i64..24)
        .map(|index| {
            (
                if index % 2 == 0 { "A" } else { "B" },
                90 + index % 5 * 5,
                index - 12,
                index * 10,
            )
        })
        .collect();
    let left: Vec<_> = (0_i64..18)
        .map(|index| {
            (
                if index % 3 == 0 { "A" } else { "B" },
                90 + index % 7 * 4,
                index - 9,
                index,
            )
        })
        .collect();
    let expected = oracle(&left, &right);
    for seed in 1..=8 {
        let mut left = left.clone();
        let mut right = right.clone();
        shuffle(&mut left, seed);
        shuffle(&mut right, seed + 23);
        let mut op = operator(10);
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let cx = StreamOperatorContext::new(&job, "asof", None);
        let mut out = EdgeCollector::new(op.output_ports().to_vec());
        for index in 0..24 {
            if let Some(row) = left.get(index) {
                op.process_data("left", batch(&[*row]), &cx, &mut out)
                    .await
                    .unwrap();
            }
            if index % 3 == 0 {
                op.process_data(
                    "right",
                    batch(&right[index..(index + 3).min(right.len())]),
                    &cx,
                    &mut out,
                )
                .await
                .unwrap();
            }
        }
        op.on_end(&cx, &mut out).await.unwrap();
        assert_eq!(output_rows(&mut out), expected, "seed {seed}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn shuffled_rows_across_watermark_and_restore_match_oracle(
        left_values in proptest::collection::vec(any::<u8>(), 1..24),
        right_values in proptest::collection::vec(any::<u8>(), 1..24),
        seed in any::<u64>(),
        restore in any::<bool>(),
    ) {
        let make_rows = |values: &[u8]| -> Vec<InputRow<'static>> {
            values.iter().enumerate().map(|(index, value)| {
                let key = ["A", "B", "C"][usize::from(value % 3)];
                let time = if value % 2 == 0 { 90 } else { 110 } + i64::from(value % 10);
                let sequence = i64::try_from(index).expect("bounded property input");
                (key, time, sequence, i64::from(*value) * 10 + sequence)
            }).collect()
        };
        let left = make_rows(&left_values);
        let right = make_rows(&right_values);
        let expected = oracle(&left, &right);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let actual = runtime.block_on(async {
            let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
            let cx = StreamOperatorContext::new(&job, "asof", None);
            let mut op = operator(10);
            let mut output = EdgeCollector::new(op.output_ports().to_vec());
            let mut emitted = Vec::new();
            for phase in 0..2 {
                for (ingress, source, salt) in [
                    ("right", &right, 37_u64),
                    ("left", &left, 71_u64),
                ] {
                    let mut rows: Vec<_> = source.iter().copied()
                        .filter(|row| (row.1 >= 100) == (phase == 1))
                        .collect();
                    shuffle(&mut rows, seed.wrapping_add(salt).wrapping_add(phase));
                    for chunk in rows.chunks(3) {
                        op.process_data(ingress, batch(chunk), &cx, &mut output)
                            .await
                            .unwrap();
                    }
                }
                if phase == 0 {
                    let ingress = IngressProgressSnapshot::new(BTreeMap::from([
                        ("left".into(), IngressProgress::new(
                            IngressState::Active, Some(EventTime::from_micros(100)))),
                        ("right".into(), IngressProgress::new(
                            IngressState::Active, Some(EventTime::from_micros(100)))),
                    ]));
                    let progress = StreamOperatorContext::with_ingress_progress(
                        &job, "asof", Some(EventTime::from_micros(100)), ingress);
                    op.on_watermark(EventTime::from_micros(100), &progress, &mut output)
                        .await
                        .unwrap();
                    emitted.extend(output_rows(&mut output));
                    if restore {
                        op.prepare_checkpoint_async(&cx).await.unwrap();
                        let snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
                        let mut recovered = operator(10);
                        recovered.restore(&snapshot).unwrap();
                        op = recovered;
                    }
                }
            }
            op.on_end(&cx, &mut output).await.unwrap();
            emitted.extend(output_rows(&mut output));
            assert_eq!(op.status().emitted_left_rows, left.len() as u64);
            assert_eq!(op.status().left.accepted_rows, left.len() as u64);
            assert_eq!(op.status().right.accepted_rows, right.len() as u64);
            emitted
        });
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn duplicate_or_cancelled_admission_leaves_no_partial_state(
        time in 0_i64..1_000,
        sequence in any::<i64>(),
        value in any::<i64>(),
    ) {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let row = ("A", time, sequence, value);
            let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
            let cx = StreamOperatorContext::new(&job, "asof", None);
            let mut op = operator(10);
            let mut output = EdgeCollector::new(op.output_ports().to_vec());
            let error = op.process_data("left", batch(&[row, row]), &cx, &mut output)
                .await
                .unwrap_err();
            assert!(matches!(error, CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofDuplicateIdentity, ..
            }));
            assert_eq!(op.status().left.duplicate_rows, 1);
            assert_eq!(op.status().left.accepted_rows, 0);
            assert_eq!(op.status().state_rows, 0);

            let token = CancellationToken::new();
            token.cancel();
            let cancelled = StreamJobContext::new(2, "asof", JsonMap::new(), None, token);
            let cancelled_cx = StreamOperatorContext::new(&cancelled, "asof", None);
            let mut op = operator(10);
            assert!(op.process_data("right", batch(&[row]), &cancelled_cx, &mut output)
                .await
                .is_err());
            assert_eq!(op.status().right.accepted_rows, 0);
            assert_eq!(op.status().state_rows, 0);
        });
    }
}

fn output_rows(output: &mut EdgeCollector) -> Vec<OutputRow> {
    use datafusion::arrow::array::StringArray;

    let mut actual = Vec::new();
    for message in output.drain("output") {
        for record in message
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()
        {
            let keys = record
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let times = record
                .column(1)
                .as_any()
                .downcast_ref::<datafusion::arrow::array::TimestampMicrosecondArray>()
                .unwrap();
            let sequences = record
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let values = record
                .column(7)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for row in 0..record.num_rows() {
                actual.push((
                    times.value(row),
                    keys.value(row).to_owned(),
                    sequences.value(row),
                    (!values.is_null(row)).then(|| values.value(row)),
                ));
            }
        }
    }
    actual
}

type InputRow<'a> = (&'a str, i64, i64, i64);
type OutputRow = (i64, String, i64, Option<i64>);

fn oracle(left: &[InputRow<'_>], right: &[InputRow<'_>]) -> Vec<OutputRow> {
    let mut expected: Vec<_> = left
        .iter()
        .map(|row| {
            let candidate = right
                .iter()
                .filter(|candidate| {
                    candidate.0 == row.0 && candidate.1 <= row.1 && candidate.1 >= row.1 - 10
                })
                .max_by_key(|candidate| (candidate.1, candidate.2));
            (
                row.1,
                row.0.to_owned(),
                row.2,
                candidate.map(|candidate| candidate.3),
            )
        })
        .collect();
    expected.sort();
    expected
}

fn shuffle<T>(values: &mut [T], mut seed: u64) {
    for index in (1..values.len()).rev() {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        values.swap(index, usize::try_from(seed % (index as u64 + 1)).unwrap());
    }
}
