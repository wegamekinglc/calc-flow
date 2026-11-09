use super::*;
use datafusion::arrow::array::{Array, BinaryArray};

#[test]
fn test_checkpoint_binary_column_prepays_one_payload_and_all_target_buffers() {
    let cells = [None, Some(vec![0_u8, 255, 7]), Some(vec![42; 129])];
    let geometry = Shape {
        rows: cells.len(),
        values: 132,
        nullable: true,
        payload_peak: 129,
    };
    let quote = requirement(geometry).unwrap();
    let mut output = None;
    let actual = allocation_counter::measure(|| {
        output = Some(
            column(
                geometry,
                &mut |index: usize| Ok(cells[index].clone()),
                &mut || Ok(()),
            )
            .unwrap(),
        );
    });
    assert!(
        quote.construction_peak >= actual.bytes_max,
        "quote={}, actual={actual:?}",
        quote.construction_peak
    );
    let output = output.unwrap();
    let array = output.as_any().downcast_ref::<BinaryArray>().unwrap();
    assert!(array.is_null(0));
    assert_eq!(array.value(1), [0, 255, 7]);
    assert_eq!(array.value(2), vec![42; 129]);
    assert_eq!(array.value_data().len(), 132);
    assert_eq!(array.values().capacity(), 132);
}

#[test]
fn test_checkpoint_binary_column_empty_and_unplanned_allocation_guards() {
    let empty = Shape {
        rows: 0,
        values: 0,
        nullable: false,
        payload_peak: 0,
    };
    assert_eq!(
        column(empty, &mut |_| unreachable!(), &mut || Ok(()))
            .unwrap()
            .len(),
        0
    );
    let one = Shape {
        rows: 1,
        values: 1,
        nullable: false,
        payload_peak: 1,
    };
    assert!(column(one, &mut |_| Ok(None), &mut || Ok(())).is_err());
    assert!(column(one, &mut |_| Ok(Some(vec![1, 2])), &mut || Ok(())).is_err());
    assert!(
        requirement(Shape {
            values: usize::MAX,
            ..one
        })
        .is_err()
    );
}
