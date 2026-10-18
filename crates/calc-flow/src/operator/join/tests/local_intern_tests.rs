use super::*;
use crate::operator::join::{
    AdmittedRow, AdmittedRows, admitted::AdmittedGroup, columnar::RowPayload, join_work,
    reset_join_work,
};
use crate::{CancellationToken, EventTime, JsonMap, StreamJobContext, StreamOperatorContext};
use datafusion::arrow::{array::ArrayRef, datatypes::Schema, record_batch::RecordBatch};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, UnboundedMemoryPool};
use std::sync::Arc;

fn intern(rows: usize) -> LocalIntern {
    let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
    let credit = MemoryConsumer::new("local-intern-test").register(&pool);
    credit
        .try_grow(super::super::local_table_bytes(rows).unwrap())
        .unwrap();
    LocalIntern::new(rows, credit)
}

#[test]
fn test_single_strings_compare_bytes_after_a_plain_hash_collision() {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec!["other", "right", "right"])),
        Arc::new(LargeStringArray::from(vec!["other", "right", "right"])),
    ];
    for column in columns {
        let bound = LocalColumn::bind(column.as_ref()).unwrap();
        let mut local = intern(3);
        let collision = hash_bytes(HASH_SEED, b"right");
        // Seed a different equal-width representative in the same full-hash bucket.
        assert_eq!(
            local.intern_value(collision, 0, "test", |_| false).unwrap(),
            0
        );
        let actual = match bound {
            LocalColumn::Utf8(values) => local.intern_string(values, 1, "test").unwrap(),
            LocalColumn::LargeUtf8(values) => local.intern_string(values, 1, "test").unwrap(),
            _ => unreachable!(),
        };
        assert_eq!(actual, (1, 5));
        assert_eq!(local.intern_row(&[bound], 2, "test").unwrap(), (1, 5));
        assert_eq!(local.distinct, 2);
    }
}

#[tokio::test]
async fn test_single_string_blocks_keep_generic_work_and_cancellation_boundaries() {
    let wide = "x".repeat(COPY_BYTES + 1);
    let mut values = vec![""; COPY_VISITS + 1];
    values.extend(["é\0", &wide, "tail"]);
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(values.clone())),
        Arc::new(LargeStringArray::from(values)),
    ];
    for column in columns {
        for cancelled in [false, true] {
            let token = CancellationToken::new();
            if cancelled {
                token.cancel();
            }
            let job = StreamJobContext::new(1, "intern", JsonMap::new(), None, token);
            let context = StreamOperatorContext::new(&job, "intern", None);
            let schema = Arc::new(Schema::new(vec![datafusion::arrow::datatypes::Field::new(
                "key",
                column.data_type().clone(),
                false,
            )]));
            let parent = Arc::new(RecordBatch::try_new(schema, vec![Arc::clone(&column)]).unwrap());
            let rows = (0..parent.num_rows())
                .map(|row| AdmittedRow {
                    record: RowPayload::Rowed {
                        parent: Arc::clone(&parent),
                        row,
                    },
                    event_time: EventTime::from_micros(0),
                    row_id: row as u64,
                    retain: row % 3 == 0,
                })
                .collect::<Vec<_>>();
            let bound = bind_columns(rows.row(0).record, &[0]).unwrap();
            let mut outcomes = Vec::new();
            for specialized in [false, true] {
                let mut local = intern(rows.len());
                let mut flow = KeyFlow::new(Some(&context));
                let group = AdmittedGroup::new(&rows, 0..rows.len());
                reset_join_work();
                let result = if specialized {
                    local.intern_group(&bound, group, "test", &mut flow).await
                } else {
                    local
                        .intern_group_with(group, "test", &mut flow, |local, offset| {
                            local.intern_row(&bound, offset, "test")
                        })
                        .await
                };
                assert_eq!(result.is_err(), cancelled);
                let work = join_work();
                outcomes.push((
                    local.locals,
                    local.rows,
                    local.hashes,
                    local.first_retained,
                    local.retains,
                    local.appearances,
                    work.quantum_steps,
                    work.quantum_boundaries,
                ));
            }
            assert_eq!(outcomes[0], outcomes[1]);
        }
    }
}
