use super::*;
use crate::EdgeBudget;
use crate::operator::join::{join_work, reset_join_work};

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn input(record: RecordBatch) -> Batch {
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

async fn assert_operand_gathers(
    mut operator: StreamJoinOperator,
    left: RecordBatch,
    right: RecordBatch,
    operands: usize,
) {
    let job = job();
    let context = hot_context(&job);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", input(right), &context, &mut collector)
        .await
        .unwrap();
    reset_join_work();
    operator
        .process_data("left", input(left), &context, &mut collector)
        .await
        .unwrap();
    let messages = collector.drain("output");
    assert_eq!(messages.len(), 1);
    let work = join_work();
    assert_eq!(
        work.output_column_takes + work.output_column_interleaves,
        operands
    );
}

#[tokio::test]
async fn test_static_product_uses_contiguous_inputs_and_owns_delivered_sequences() {
    use crate::operator::join::output_gather::COORDINATE_READS;

    let (left, right) = large_static_inputs(70, 4);
    let left = left.slice(2, 64);
    let sequence = left.column(1).to_data().buffers()[0].clone();
    let factors = [3.0, 5.0, 5.0, 7.0];
    let mut dimensions = right.columns().to_vec();
    dimensions[1] = Arc::new(Float64Array::from(factors.to_vec()));
    let right = RecordBatch::try_new(right.schema(), dimensions).unwrap();
    let mut operator = static_operator(true);
    let job = job();
    let context = hot_progress_context(&job, -1, EdgeBudget::new(64, 128).unwrap());
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", input(right), &context, &mut collector)
        .await
        .unwrap();
    COORDINATE_READS.set(0);
    reset_join_work();
    operator
        .process_data("left", input(left), &context, &mut collector)
        .await
        .unwrap();
    let reads = COORDINATE_READS.get();
    let takes = join_work().output_column_takes;
    let output = collector.drain("output");
    assert_eq!(
        static_pairs(&output, true),
        (2_u16..66)
            .map(|n| (u64::from(n), 2.0 * factors[usize::from(n % 4)]))
            .collect::<Vec<_>>()
    );
    assert_eq!(output.len(), 8);
    for (index, message) in output.iter().enumerate() {
        let batch = message.as_data().unwrap();
        assert_eq!(batch.metadata().sequence(), u64::try_from(index).unwrap());
        assert_eq!(batch.estimated_bytes().unwrap(), 128);
        let record = &batch.table_payload().unwrap().batches()[0];
        assert_ne!(
            record.column(0).to_data().buffers()[0].data_ptr(),
            sequence.data_ptr()
        );
    }
    assert_eq!((reads, takes), (64, 8));
}

#[test]
fn test_contiguous_product_borrows_operands_and_detaches_sliced_nullable_sequences() {
    let product = band_projection();
    let price: ArrayRef = Arc::new(Float64Array::from(vec![None, Some(-0.0), Some(3.0)]));
    let operand = price.slice(1, 2);
    for logical in [product.left_price, product.right_price] {
        assert!(Arc::ptr_eq(
            &product.contiguous_column(logical, Arc::clone(&operand)),
            &operand
        ));
    }
    let source = UInt64Array::from(vec![Some(42), None, Some(7), Some(8)]);
    for logical in [product.sequence, product.right_sequence.unwrap()] {
        let owned = product.contiguous_column(logical, Arc::new(source.slice(1, 2)));
        let owned = owned.as_primitive::<UInt64Type>();
        assert_eq!(owned.iter().collect::<Vec<_>>(), [None, Some(7)]);
        assert_ne!(
            owned.values().inner().data_ptr(),
            source.values().inner().data_ptr()
        );
        assert_ne!(
            owned.nulls().unwrap().buffer().data_ptr(),
            source.nulls().unwrap().buffer().data_ptr()
        );
        let all_valid = product.contiguous_column(logical, Arc::new(source.slice(2, 2)));
        assert!(all_valid.nulls().is_none());
    }
}

#[tokio::test]
async fn test_equi_gathers_only_its_three_operands() {
    let (left, right) = static_inputs();
    assert_operand_gathers(static_operator(true), left, right, 3).await;
}

#[tokio::test]
async fn test_band_gathers_only_its_four_operands() {
    let record = band_history([2.0, 3.0]);
    assert_operand_gathers(band_operator(true), record.clone(), record, 4).await;
}

#[tokio::test]
async fn test_queue_admission_reuses_measured_join_envelopes() {
    let job = job();
    let context = hot_progress_context(&job, -1, EdgeBudget::new(1, 1024).unwrap());
    let input = input(band_history([2.0, 3.0]));
    let mut allocations = Vec::new();
    for fused in [false, true] {
        let mut operator = band_operator(fused);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        for ingress in ["left", "right"] {
            operator
                .process_data(ingress, input.clone(), &context, &mut collector)
                .await
                .unwrap();
        }
        let messages = collector.drain("output");
        assert_eq!(messages.len(), 2);
        let mut costs = [crate::EnvelopeCost::ZERO; 2];
        let measured = allocation_counter::measure(|| {
            for (cost, message) in costs.iter_mut().zip(&messages) {
                *cost = crate::EnvelopeCost::of_message(message).unwrap();
            }
        });
        allocations.push(measured.count_total);
        for (sequence, (message, cost)) in messages.iter().zip(costs).enumerate() {
            let batch = message.as_data().unwrap();
            let record = &batch.table_payload().unwrap().batches()[0];
            let bytes = record
                .columns()
                .iter()
                .map(|column| column.to_data().get_slice_memory_size().unwrap())
                .sum();
            assert_eq!(cost, crate::EnvelopeCost::new(1, 1, bytes));
            assert_eq!(
                batch.metadata().sequence(),
                u64::try_from(sequence).unwrap()
            );
        }
    }
    assert_eq!(allocations, [0, 0]);
}

#[tokio::test]
async fn test_band_budget_charges_emitted_fields_and_retries_the_whole_input() {
    let mut operator = band_operator(true);
    let job = job();
    let context = hot_context(&job);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let record = band_history([2.0, 3.0]);
    let wide_symbol = "x".repeat(2048);
    let mut columns = record.columns().to_vec();
    columns[2] = Arc::new(StringArray::from(vec![wide_symbol.as_str(); 2]));
    let input = input(RecordBatch::try_new(record.schema(), columns).unwrap());
    operator
        .process_data("left", input.clone(), &context, &mut collector)
        .await
        .unwrap();
    let rejected = hot_progress_context(&job, -1, EdgeBudget::new(1, 23).unwrap());
    let error = operator
        .process_data("right", input.clone(), &rejected, &mut collector)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("byte"), "{error}");
    assert!(collector.drain("output").is_empty());
    assert_eq!(operator.state.next_output_sequence, 0);
    assert_eq!(operator.state.next_right_row_id, 0);
    let exact = hot_progress_context(&job, -1, EdgeBudget::new(1, 24).unwrap());
    operator
        .process_data("right", input, &exact, &mut collector)
        .await
        .unwrap();
    let messages = collector.drain("output");
    assert_eq!(band_pairs(&messages, true), [(0, 4.0), (1, 9.0)]);
    for (sequence, message) in messages.iter().enumerate() {
        let batch = message.as_data().unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.estimated_bytes().unwrap(), 24);
        assert_eq!(
            batch.metadata().sequence(),
            u64::try_from(sequence).unwrap()
        );
    }
    assert_eq!(operator.state.next_right_row_id, 2);
    assert_eq!(operator.state.next_output_sequence, 2);
    assert_eq!(operator.state.metrics.emitted_match_rows, 2);
}

#[tokio::test]
async fn test_cancelled_projected_delivery_replays_from_the_committed_sequence() {
    struct StopAfterOne<'a> {
        cancellation: &'a CancellationToken,
        output: &'a mut EdgeCollector,
    }

    #[async_trait::async_trait]
    impl crate::StreamCollector for StopAfterOne<'_> {
        async fn emit(&mut self, port: &str, batch: Batch) -> Result<()> {
            self.output.emit(port, batch).await?;
            self.cancellation.cancel();
            Ok(())
        }
    }

    let mut operator = band_operator(true);
    let active = job();
    let cancelled = job();
    let budget = EdgeBudget::new(1, 24).unwrap();
    let active_context = hot_progress_context(&active, -1, budget);
    let cancelled_context = hot_progress_context(&cancelled, -1, budget);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let data = input(band_history([2.0, 3.0]));
    operator
        .process_data("left", data.clone(), &active_context, &mut output)
        .await
        .unwrap();
    let state = operator.run_core.as_ref().unwrap().fingerprint();
    let mut stopping = StopAfterOne {
        cancellation: cancelled.cancellation(),
        output: &mut output,
    };
    let error = operator
        .process_data("right", data.clone(), &cancelled_context, &mut stopping)
        .await
        .unwrap_err();
    assert!(
        error.to_string().to_lowercase().contains("cancel"),
        "{error}"
    );
    let prefix = output.drain("output");
    assert_eq!(band_pairs(&prefix, true), [(0, 4.0)]);
    assert_eq!(prefix[0].as_data().unwrap().metadata().sequence(), 0);
    assert_eq!(operator.run_core.as_ref().unwrap().fingerprint(), state);
    assert_eq!(operator.state.metrics.emitted_match_rows, 0);
    let capture = native_snapshot(&mut operator).await;
    let mut restored = band_operator(true);
    restored.restore_native(&capture).unwrap();
    for join in [&mut operator, &mut restored] {
        join.process_data("right", data.clone(), &active_context, &mut output)
            .await
            .unwrap();
        let messages = output.drain("output");
        assert_eq!(band_pairs(&messages, true), [(0, 4.0), (1, 9.0)]);
        assert_eq!(
            messages
                .iter()
                .map(|message| message.as_data().unwrap().metadata().sequence())
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(join.state.metrics.emitted_match_rows, 2);
    }
}

fn nullable_quote(prices: Vec<Option<f64>>) -> RecordBatch {
    let schema = Arc::new(Schema::new(
        quote_schema()
            .fields()
            .iter()
            .map(|field| {
                field
                    .as_ref()
                    .clone()
                    .with_nullable(field.name() == "price")
            })
            .collect::<Vec<_>>(),
    ));
    let rows = prices.len();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![1; rows]).with_timezone("UTC")),
            Arc::new(UInt64Array::from_iter_values(
                (0..rows).map(|row| u64::try_from(row).unwrap()),
            )),
            Arc::new(StringArray::from(vec!["S000"; rows])),
            Arc::new(Float64Array::from(prices)),
        ],
    )
    .unwrap()
}

fn nullable_band_operator(left: &RecordBatch, right: &RecordBatch) -> StreamJoinOperator {
    let mut operator =
        StreamJoinOperator::new("hot-band", left.schema(), right.schema(), band_spec()).unwrap();
    operator
        .set_hot_projection(
            HotProjection {
                sequence: 1,
                right_sequence: Some(7),
                left_price: 3,
                right_price: 4,
            },
            Arc::new(Schema::new(vec![
                Field::new("sequence", DataType::UInt64, false),
                Field::new("right_sequence", DataType::UInt64, false),
                Field::new("value", DataType::Float64, true),
            ])),
        )
        .unwrap();
    operator.enable_native_checkpoint().unwrap();
    operator
}

#[tokio::test]
async fn test_nullable_products_preserve_bits_and_pair_order_under_the_final_byte_budget() {
    for (rows, bytes) in [(4, 97), (8, 192)] {
        assert_nullable_products(EdgeBudget::new(rows, bytes).unwrap()).await;
    }
}

async fn assert_nullable_products(budget: EdgeBudget) {
    let left_values = [
        Some(-0.0),
        Some(f64::from_bits(0x7ff8_0000_0000_0042)),
        None,
        Some(f64::INFINITY),
    ];
    let right_values = [Some(-2.0), None];
    let left = nullable_quote(left_values.to_vec());
    let right = nullable_quote(right_values.to_vec())
        .project(&[3, 2, 0, 1])
        .unwrap();
    let mut operator = nullable_band_operator(&left, &right);
    let job = job();
    let context = hot_progress_context(&job, -1, budget);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    for offset in [0, 2] {
        operator
            .process_data(
                "left",
                input(left.slice(offset, 2)),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
    }
    let right = input(right);
    let too_small_for_null = hot_progress_context(&job, -1, EdgeBudget::new(1, 24).unwrap());
    let error = operator
        .process_data("right", right.clone(), &too_small_for_null, &mut collector)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("byte"), "{error}");
    assert!(collector.drain("output").is_empty());
    assert_eq!(operator.state.next_right_row_id, 0);
    assert_eq!(operator.state.next_output_sequence, 0);
    reset_join_work();
    operator
        .process_data("right", right, &context, &mut collector)
        .await
        .unwrap();
    assert!(join_work().output_column_interleaves > 0);
    let messages = collector.drain("output");
    assert_eq!(messages.len(), 2);
    for (right_row, message) in messages.iter().enumerate() {
        let batch = message.as_data().unwrap();
        assert_eq!(batch.estimated_bytes().unwrap(), 97);
        assert_eq!(
            batch.metadata().sequence(),
            u64::try_from(right_row).unwrap()
        );
        let record = &batch.table_payload().unwrap().batches()[0];
        let sequence = |column: usize| {
            record
                .column(column)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
        };
        assert_eq!(sequence(0).as_ref(), &[0, 1, 2, 3]);
        assert_eq!(
            sequence(1).as_ref(),
            &[u64::try_from(right_row).unwrap(); 4]
        );
        let values = record
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let expected = left_values
            .iter()
            .map(|left| {
                left.zip(right_values[right_row])
                    .map(|(a, b)| (a * b).to_bits())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            values
                .iter()
                .map(|value| value.map(f64::to_bits))
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn test_dense_product_on_sliced_nullable_operands_preserves_ieee_values() {
    let nan = f64::from_bits(0x7ff8_0000_0000_0042);
    let left = Float64Array::from(vec![None, Some(-0.0), Some(nan), Some(f64::INFINITY), None]);
    let right = Float64Array::from(vec![Some(2.0), Some(2.0), Some(-2.0), None]);
    let sequence: ArrayRef = Arc::new(UInt64Array::from(vec![7, 8, 9]));
    let columns: [ArrayRef; 3] = [
        sequence.clone(),
        Arc::new(left.slice(1, 3)),
        Arc::new(right.slice(0, 3)),
    ];
    let projection = HotProjection {
        sequence: 0,
        right_sequence: None,
        left_price: 1,
        right_price: 2,
    };
    let output = projection
        .gather(|index, _| Ok(columns[index].clone()))
        .unwrap();
    assert!(Arc::ptr_eq(&output[0], &sequence));
    let products = output[1].as_primitive::<Float64Type>();
    assert!(products.nulls().is_none());
    assert_eq!(products.to_data().get_slice_memory_size().unwrap(), 24);
    for (actual, expected) in products
        .values()
        .iter()
        .zip([-0.0, nan * 2.0, f64::NEG_INFINITY])
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn test_dense_product_keeps_the_nullable_path_nan_bits_across_chunk_sizes() {
    let left_nan = f64::from_bits(std::hint::black_box(0x7ff8_0000_0000_0042));
    let right_nan = f64::from_bits(std::hint::black_box(0x7ff8_0000_0000_0a11));
    let projection = HotProjection {
        sequence: 0,
        right_sequence: None,
        left_price: 1,
        right_price: 2,
    };
    for rows in [1, 17, 4096] {
        let columns: [ArrayRef; 3] = [
            Arc::new(UInt64Array::from(vec![0; rows + 1])),
            Arc::new(Float64Array::from(vec![left_nan; rows + 1])),
            Arc::new(
                std::iter::repeat_n(Some(right_nan), rows)
                    .chain([None])
                    .collect::<Float64Array>(),
            ),
        ];
        let products = |length| {
            let output = projection
                .gather(|index, _| Ok(columns[index].slice(0, length)))
                .unwrap();
            output[1]
                .as_primitive::<Float64Type>()
                .iter()
                .map(|value| value.map(f64::to_bits))
                .collect::<Vec<_>>()
        };
        let nullable = products(rows + 1);
        assert_eq!(nullable[rows], None);
        let dense = products(rows);
        let difference = dense.iter().zip(&nullable).position(|(a, b)| a != b);
        assert_eq!(difference, None, "{rows} valid rows");
    }
}
