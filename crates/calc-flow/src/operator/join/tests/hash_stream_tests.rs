use super::*;
use datafusion::arrow::array::TimestampNanosecondArray;
use std::sync::Arc;

#[test]
fn test_borrowed_hash_visits_fixed_canonical_blocks_independent_of_fields() {
    for length in [0, 54, 55, 56, 63, 64, 65, 129, 4097] {
        let value = format!("{}é\0", "x".repeat(length));
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec![value])),
            Arc::new(TimestampNanosecondArray::from(vec![-1]).with_timezone("Europe/Paris")),
            Arc::new(LargeStringArray::from(vec![""])),
        ];
        for indices in [&[0][..], &[0, 1, 2][..], &[2, 1, 0][..]] {
            let canonical = super::super::encode_join_key_columns_v1(&columns, 0, indices).unwrap();
            let key = BorrowedKey {
                columns: &columns,
                row: 0,
                indices,
            };
            let mut parts = Vec::new();
            key.visit_hash_blocks(|part| parts.push(part.to_vec()))
                .unwrap();
            assert_eq!(
                parts,
                canonical.chunks(64).map(<[u8]>::to_vec).collect::<Vec<_>>(),
                "length={length}, indices={indices:?}"
            );
            let state = KeyHashState::default();
            assert_eq!(key.hash(&state).unwrap(), framed_hash(&state, &canonical));
            assert!(key.equals(&canonical));
        }
    }
}
