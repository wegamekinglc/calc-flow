use super::*;
use crate::CalcFlowError;
use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema};
use std::{collections::HashMap, sync::Arc};

fn values() -> Metadata<'static> {
    Metadata {
        pipeline: "pipeline",
        node: "operator",
        configuration: "configuration",
        state_schema: "state-schema",
        kernel: "kernel",
        numerical: "stateful_numeric_v1",
    }
}
fn construct(input: &Schema, values: Metadata<'_>) -> Arc<Schema> {
    let mut fields = vec![
        Field::new("_state_kind", DataType::UInt8, false),
        Field::new("_entity_id", DataType::UInt64, false),
        Field::new("_entity_position", DataType::UInt64, true),
    ];
    fields.extend(
        input
            .fields()
            .iter()
            .map(|field| Field::new(field.name(), field.data_type().clone(), true)),
    );
    fields.extend([
        Field::new("_group", DataType::UInt64, true),
        Field::new("_valid_count", DataType::UInt64, true),
        Field::new("_value", DataType::Float64, true),
    ]);
    fields.push(Field::new("_cumulative_summary", DataType::Binary, true));
    fields.push(Field::new("_indicator_state", DataType::Binary, true));
    let pairs = values.pairs();
    let initial: [(&str, &str); 5] = pairs[..5].try_into().unwrap();
    let mut metadata =
        HashMap::from(initial.map(|(key, value): (&str, &str)| (key.to_owned(), value.to_owned())));
    for (key, value) in &pairs[5..] {
        metadata.insert((*key).to_owned(), (*value).to_owned());
    }
    Arc::new(Schema::new_with_metadata(fields, metadata))
}

#[test]
fn test_fresh_schema_quote_covers_outer_owned_metadata_and_field_vector_peak() {
    let shared = Arc::new(
        Field::new("child", DataType::Float64, true)
            .with_metadata(HashMap::from([("large".into(), "x".repeat(8_192))])),
    );
    let nested = DataType::Dictionary(
        Box::new(DataType::Int16),
        Box::new(DataType::Struct(Fields::from(vec![Arc::clone(&shared)]))),
    );
    for width in [0, 1, 3, 4, 7, 17] {
        let input = Schema::new(
            (0..width)
                .map(|index| Field::new(format!("input_{index}"), nested.clone(), false))
                .collect::<Vec<_>>(),
        );
        let mut quote = None;
        let scan = allocation_counter::measure(|| {
            quote = Some(plan(&input, values(), &mut || Ok(())).unwrap());
        });
        assert_eq!(scan.count_total, 0);
        let quote = quote.unwrap();
        let mut output = None;
        let observed = allocation_counter::measure(|| {
            output = Some(construct(&input, values()));
        });
        assert!(
            quote.construction_peak >= observed.bytes_max,
            "width={width}, quote={}, observed={observed:?}",
            quote.construction_peak
        );
        assert!(quote.retained >= u64::try_from(observed.bytes_current).unwrap());
        let output = output.unwrap();
        assert_eq!(output.fields().len(), width + 8);
        assert_eq!(output.metadata().len(), 7);
        assert_eq!(output.metadata()["calc_flow.operator_id"], "operator");
    }
}

#[test]
fn test_fresh_schema_quote_cannot_exclude_large_fresh_names_or_cancel() {
    let input = Schema::new(vec![Field::new(
        "x".repeat(16_384),
        DataType::Float64,
        true,
    )]);
    let quote = plan(&input, values(), &mut || Ok(())).unwrap();
    assert!(quote.retained > 16_384);
    let error = plan(&input, values(), &mut || {
        Err(CalcFlowError::Cancelled {
            run_id: "schema".into(),
        })
    })
    .err()
    .unwrap();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}
