use super::super::{borrowed_key::framed_hash, encode_join_key_columns_v1};
use super::*;
use datafusion::arrow::array::{
    ArrayRef, BooleanArray, Date32Array, Int64Array, LargeStringArray, StringArray,
    TimestampMicrosecondArray, TimestampNanosecondArray,
};
use futures::executor::block_on;
use std::sync::Arc;

fn key_columns() -> Vec<ArrayRef> {
    let strings = [
        Some(String::new()),
        None,
        Some("a".to_owned()),
        Some("é\0".to_owned()),
        Some("x".repeat(55)),
        Some("y".repeat(64)),
        Some("z".repeat(4087)),
        Some("v".repeat(4088)),
        Some("w".repeat(5000)),
    ];
    vec![
        Arc::new(StringArray::from(strings.to_vec())),
        Arc::new(LargeStringArray::from(strings.to_vec())),
        Arc::new(Int64Array::from(vec![
            Some(0),
            None,
            Some(-1),
            Some(i64::MAX),
            Some(i64::MIN),
            Some(42),
            Some(7),
            Some(1),
            Some(0),
        ])),
        Arc::new(Date32Array::from(vec![
            Some(0),
            None,
            Some(-719_162),
            Some(19_000),
            Some(1),
            Some(i32::MAX),
            Some(i32::MIN),
            Some(2),
            Some(0),
        ])),
        Arc::new(
            TimestampNanosecondArray::from(vec![
                Some(0),
                None,
                Some(-1),
                Some(1_700_000_000_000_000_000),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                Some(0),
            ])
            .with_timezone("Europe/Paris"),
        ),
        Arc::new(BooleanArray::from(vec![
            Some(true),
            None,
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
            Some(true),
            Some(false),
        ])),
        Arc::new(TimestampMicrosecondArray::from(vec![
            Some(0),
            None,
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            Some(0),
        ])),
    ]
}

#[test]
fn test_small_key_identity_matches_chunked_hash_and_equality() {
    let columns = key_columns();
    let state = KeyHashState::default();
    let index_sets = [
        vec![0],
        vec![1],
        vec![2],
        vec![3],
        vec![4],
        vec![5],
        vec![6],
        vec![0, 2, 4, 5],
        vec![5, 4, 3, 2, 1, 0],
        vec![0, 1],
        vec![2; 63],
        vec![2; 64],
        vec![2; 65],
    ];
    let mut small_rows = 0;
    let mut chunked_rows = 0;
    for indices in &index_sets {
        let writers = indices
            .iter()
            .map(|&index| KeyWriter::bind(columns[index].as_ref()).unwrap())
            .collect::<Vec<_>>();
        let rows = columns[0].len();
        let frames = (0..rows)
            .map(|row| encode_join_key_columns_v1(&columns, row, indices).unwrap())
            .collect::<Vec<_>>();
        for (row, frame) in frames.iter().enumerate() {
            let key = TypedKey {
                columns: &writers,
                row,
            };
            let mut flow = KeyFlow::new(None);
            let chunked = block_on(key.hash(&state, &mut flow)).unwrap();
            assert_eq!(chunked, framed_hash(&state, frame), "{indices:?} row {row}");
            let fits = frame.len() <= 4096 && indices.len() < 64;
            let Some(small) = key.small() else {
                assert!(!fits, "{indices:?} row {row} fits one quantum");
                chunked_rows += 1;
                continue;
            };
            assert!(fits, "{indices:?} row {row} exceeds one quantum");
            small_rows += 1;
            assert_eq!(small.charge(), (indices.len() + 1, frame.len()));
            assert_eq!(
                small.hash(&state).unwrap(),
                chunked,
                "{indices:?} row {row}"
            );
            let shorter = &frame[..frame.len() - 1];
            let longer = [frame.as_slice(), &[0]].concat();
            let flipped = {
                let mut bytes = frame.clone();
                *bytes.last_mut().unwrap() ^= 1;
                bytes
            };
            let candidates = frames.iter().map(Vec::as_slice).chain([
                shorter,
                longer.as_slice(),
                flipped.as_slice(),
                &[][..],
            ]);
            for candidate in candidates {
                let expected = block_on(key.equals(candidate, &mut flow)).unwrap();
                assert_eq!(expected, candidate == frame.as_slice());
            }
        }
    }
    assert!(small_rows > 0 && chunked_rows > 0);
}
