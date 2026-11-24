use super::*;

#[test]
fn test_asof_charge_shares_grouped_rows_across_output_columns() {
    let declaration = spec(10, 2048, 1 << 24);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 24));
    let mut state = state(&declaration, &pool);
    for (key, value) in [(0, "a"), (1, "bbb")] {
        admit(&mut state, 1, &record(&[(key, 0, 0, Some(value))])).unwrap();
    }
    for start in [0, 256] {
        let rows = (start..start + 256)
            .map(|row| (row % 4, 10, row, Some("x")))
            .collect::<Vec<_>>();
        admit(&mut state, 0, &record(&rows)).unwrap();
    }
    let selection = state.select(None, true, 512).unwrap();
    assert_eq!((selection.len(), selection.matched()), (512, 256));

    // Two 256-row left runs. Each right run supplies 128 matches, followed
    // in the charge by the validity bitmap and null data for 256 misses.
    let left_integer = 256 + 2 * (128 + 256 * 8 + 32);
    let left_string = 256 + 2 * (128 + 257 * 4 + 256 + 32);
    let right_integer = 256 + 2 * (128 + 128 * 8 + 16) + 64 + 256 * 8;
    let right_string = 256 + 2 * (128 + 129 * 4 + 16) + 128 + 384 + 64 + 257 * 4;
    let expected = 3 * (left_integer + right_integer) + left_string + right_string;
    let bytes = allocation_counter::measure(|| {
        assert_eq!(state.output_charge(&selection, None).unwrap(), expected);
    });
    let projection = [(1, 3), (0, 1), (1, 3)];
    assert_eq!(
        state.output_charge(&selection, Some(&projection)).unwrap(),
        2 * right_string + left_integer
    );
    assert_eq!(state.output_charge(&selection, Some(&[])).unwrap(), 0);
    for (field, message) in [
        ((2, 0), "projection side is invalid"),
        ((1, 4), "projection column is out of bounds"),
    ] {
        let error = state.output_charge(&selection, Some(&[field])).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }
    eprintln!("ASOF output reservation allocations: {bytes:?}; charge={expected}");
    assert!(bytes.bytes_total < 18 * 1024, "{bytes:?}");
}
