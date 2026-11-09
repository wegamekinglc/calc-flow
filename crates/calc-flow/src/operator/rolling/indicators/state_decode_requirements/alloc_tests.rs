use super::*;
use std::collections::VecDeque;

#[test]
fn test_fresh_vector_and_deque_bounds_cover_actual_growth_and_overlap() {
    for count in [0, 1, 4, 5, 8, 17, 65] {
        let quote = growing::<[usize; 3]>(count).unwrap();
        let mut vector = None;
        let actual = allocation_counter::measure(|| {
            let mut values = Vec::new();
            for _ in 0..count {
                values.push([0_usize; 3]);
            }
            vector = Some(values);
        });
        assert!(quote.retained >= u64::try_from(actual.bytes_current).unwrap());
        assert!(quote.retained + quote.overlap >= actual.bytes_max);
        let mut deque = None;
        let actual = allocation_counter::measure(|| {
            let mut values = VecDeque::new();
            for _ in 0..count {
                values.push_back([0_usize; 3]);
            }
            deque = Some(values);
        });
        assert!(quote.retained >= u64::try_from(actual.bytes_current).unwrap());
        assert!(quote.retained + quote.overlap >= actual.bytes_max);
    }
}
#[test]
fn test_independent_valid_allocations_may_sum_above_isize() {
    let each = exact::<u8>(usize::try_from(isize::MAX).unwrap()).unwrap();
    assert!(add(each, each).is_ok());
    assert!(exact::<u64>(usize::MAX).is_err());
    assert!(growing::<u64>(usize::MAX).is_err());
}
