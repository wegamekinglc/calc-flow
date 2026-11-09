use super::*;
use datafusion::arrow::{
    array::{
        ArrayRef, DictionaryArray, Int8Array, Int16Array, ListArray, RunArray, StringArray,
        UnionArray,
    },
    datatypes::{Field, Int8Type, Int16Type, Int32Type, UnionFields},
};
use std::sync::Arc;

#[test]
fn test_restore_scalar_costs_match_selected_materialization_and_clone_heap() {
    let union_strings: ArrayRef = Arc::new(StringArray::from(vec![Some("union value"), None]));
    let union_fields = UnionFields::try_new(
        vec![7],
        vec![Arc::new(Field::new("value", DataType::Utf8, true))],
    )
    .unwrap();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(vec![Some("one"), None])),
        Arc::new(
            DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(vec![Some(0), None]),
                Arc::new(StringArray::from(vec!["dictionary value"])),
            )
            .unwrap(),
        ),
        Arc::new(
            RunArray::<Int16Type>::try_new(
                &Int16Array::from(vec![2, 3]),
                &StringArray::from(vec![Some("run value"), None]),
            )
            .unwrap()
            .slice(1, 2),
        ),
        Arc::new(
            UnionArray::try_new(
                union_fields,
                vec![7_i8, 7].into(),
                Some(vec![0_i32, 1].into()),
                vec![union_strings],
            )
            .unwrap(),
        ),
        Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>([
            Some(vec![Some(7)]),
            None,
        ])),
    ];
    for array in arrays {
        for row in 0..array.len() {
            let mut cost = None;
            let scan = allocation_counter::measure(|| {
                cost = Some(scalar_requirement(array.as_ref(), row, &mut || Ok(())).unwrap());
            });
            assert_eq!(scan.count_total, 0);
            let (owned, temporary) = cost.unwrap();
            let mut value = None;
            let actual = allocation_counter::measure(|| {
                value = Some(
                    super::super::materialize::read_scalar_checked(
                        array.as_ref(),
                        row,
                        "decode",
                        &mut || Ok(()),
                    )
                    .unwrap(),
                );
            });
            assert!(owned + temporary >= actual.bytes_max);
            let value = value.unwrap();
            let mut copied = None;
            let actual = allocation_counter::measure(|| {
                copied = Some(value.clone());
            });
            let mut planned = None;
            let scan = allocation_counter::measure(|| {
                planned =
                    Some(scalar_clone_requirement(array.as_ref(), row, &mut || Ok(())).unwrap());
            });
            assert_eq!(scan.count_total, 0);
            let planned = planned.unwrap();
            assert!(
                planned >= actual.bytes_max,
                "planned={planned}, actual={actual:?}"
            );
            if matches!(array.data_type(), DataType::List(_)) {
                assert_eq!(planned, 0);
            }
            drop(copied);
        }
    }
}
#[test]
fn test_restore_scalar_requirement_checks_range_and_preserves_cancel() {
    let array = StringArray::from(vec!["x"]);
    assert!(scalar_requirement(&array, 1, &mut || Ok(())).is_err());
    assert!(scalar_clone_requirement(&array, 1, &mut || Ok(())).is_err());
    let error = scalar_clone_requirement(&array, 0, &mut || {
        Err(CalcFlowError::Cancelled {
            run_id: "decode".into(),
        })
    })
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}
