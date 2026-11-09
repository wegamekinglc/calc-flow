use super::*;
use datafusion::arrow::{array::ListArray, datatypes::Int32Type};
use datafusion::common::ScalarValue;
use std::sync::Arc;

#[test]
fn test_checkpoint_input_roles_borrow_only_their_layout_values() {
    let partition = [Some(ScalarValue::Utf8(Some("key".into()))), None];
    let values = [
        ScalarValue::Float64(Some(f64::from_bits(0x7ff8_0000_0000_0042))),
        ScalarValue::List(Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
            [Some(vec![Some(17)])],
        ))),
    ];
    let mut selected = None;
    let allocations = allocation_counter::measure(|| {
        selected = Data::History(&values).cell(Column::History(1));
        assert!(Data::History(&values).cell(Column::Input(0)).is_none());
        assert!(Data::History(&values).cell(Column::Partition(0)).is_none());
        assert!(Data::Buffer(&values).cell(Column::Partition(0)).is_none());
        assert!(std::ptr::eq(
            Data::Buffer(&values).cell(Column::Input(0)).unwrap(),
            &raw const values[0]
        ));
        assert!(std::ptr::eq(
            Data::Buffer(&values).cell(Column::History(1)).unwrap(),
            &raw const values[1]
        ));
        assert!(std::ptr::eq(
            Data::Entity(&partition).cell(Column::Partition(0)).unwrap(),
            partition[0].as_ref().unwrap()
        ));
        assert!(
            Data::Entity(&partition)
                .cell(Column::Partition(1))
                .is_none()
        );
        assert!(Data::Entity(&partition).cell(Column::History(0)).is_none());
        assert!(Data::Empty.cell(Column::Input(usize::MAX)).is_none());
    });
    assert_eq!(allocations.count_total, 0);
    assert!(std::ptr::eq(selected.unwrap(), &raw const values[1]));
    let Some(ScalarValue::Float64(Some(value))) = Data::Buffer(&values).cell(Column::Input(0))
    else {
        panic!("expected original number");
    };
    assert_eq!(value.to_bits(), 0x7ff8_0000_0000_0042);
}
