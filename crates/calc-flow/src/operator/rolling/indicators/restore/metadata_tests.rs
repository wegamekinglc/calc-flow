use super::{metadata, state_tests};

#[test]
fn test_metadata_quote_covers_actual_borrowed_parse_and_inventory_validation() {
    let operator = state_tests::operator();
    let snapshot = state_tests::snapshot(&operator);
    let quote = metadata::quote(&snapshot, &mut || Ok(())).unwrap();
    let actual = allocation_counter::measure(|| {
        let typed = metadata::parse(&snapshot.inline_metadata).unwrap();
        metadata::validate(&typed, &operator.compiled, &snapshot).unwrap();
    });
    assert!(
        actual.bytes_max <= quote.peak,
        "quote={} actual={actual:?}",
        quote.peak
    );
}
