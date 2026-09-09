mod asof_support;
#[path = "asof_support/materialization.rs"]
mod materialization;
use asof_support::{batch, operator};
use calc_flow::{
    CancellationToken, EdgeCollector, JsonMap, OperatorMetadata, StreamJobContext, StreamOperator,
    StreamOperatorContext,
};
use datafusion::arrow::array::{Array, Int64Array};

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
    use datafusion::arrow::array::StringArray;
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
        let mut actual = Vec::new();
        for message in out.drain("output") {
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
        assert_eq!(actual, expected, "seed {seed}");
    }
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
