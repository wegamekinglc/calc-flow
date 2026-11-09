use super::*;
use std::collections::{BTreeMap, HashMap, VecDeque};

#[test]
fn growth_quotes_cover_live_container_allocations() {
    for count in [0, 1, 3, 4, 5, 7, 8, 17, 129, 4097] {
        let mut deque = VecDeque::<(usize, f64)>::new();
        let mut vector = Vec::<usize>::new();
        let mut map = HashMap::<u64, usize>::new();
        let mut tree = BTreeMap::<usize, Vec<usize>>::new();
        let observed = allocation_counter::measure(|| {
            for index in 0..count {
                deque.push_back((index, 1.0));
                vector.push(index);
                map.insert(u64::try_from(index).unwrap(), index);
                tree.insert(index, Vec::new());
            }
        });
        let quote = add(
            add(
                growing::<(usize, f64)>(count).unwrap(),
                growing::<usize>(count).unwrap(),
            )
            .unwrap(),
            add(
                hash_table(count).unwrap(),
                tree_nodes::<usize, Vec<usize>>(count).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(
            observed.bytes_max <= quote,
            "count={count}, {observed:?}, quote={quote}"
        );
    }
}

#[test]
fn container_costs_reject_overflow() {
    assert!(growing::<usize>(usize::MAX).is_err());
    assert!(repeat(Ok(u64::MAX), 2).is_err());
    assert!(total([Ok(u64::MAX), Ok(1)]).is_err());
    assert!(hash_table(usize::MAX).is_err());
    assert!(tree_nodes::<usize, Vec<usize>>(usize::MAX).is_err());
}
