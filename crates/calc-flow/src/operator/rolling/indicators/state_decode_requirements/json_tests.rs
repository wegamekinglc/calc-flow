use super::*;
use serde::de::DeserializeSeed;

#[test]
fn test_summary_lexical_bound_covers_actual_pinned_serde_content_allocations() {
    for bytes in [
        br#"{"value":[1,2,3,4],"kind":"SignedTotal"}"#.as_slice(),
        br#"{"value":[{"a":"escaped\nvalue"},[1,2,3,4,5]],"kind":"wrong"}"#.as_slice(),
        b"[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]".as_slice(),
        b"[]".as_slice(),
    ] {
        let mut quote = None;
        let scan = allocation_counter::measure(|| {
            quote = Some(requirement(bytes, &mut || Ok(())).unwrap());
        });
        assert_eq!(scan.count_total, 0);
        let quote = quote.unwrap();
        let mut output = None;
        let actual = allocation_counter::measure(|| {
            let mut deserializer = serde_json::Deserializer::from_slice(bytes);
            output =
                Some(serde::__private228::de::ContentVisitor::new().deserialize(&mut deserializer));
        });
        assert!(
            quote >= actual.bytes_max,
            "quote={quote}, actual={actual:?}"
        );
    }
}
#[test]
fn test_summary_lexical_bound_preserves_whitespace_and_cancellation() {
    let compact = requirement(br#"{"value":5,"kind":"Float"}"#, &mut || Ok(())).unwrap();
    let spaced = requirement(
        b"  { \n \"value\" : 5 , \"kind\" : \"Float\" }   ",
        &mut || Ok(()),
    )
    .unwrap();
    assert_eq!(compact, spaced);
    let error = requirement(b"[]", &mut || {
        Err(CalcFlowError::Cancelled {
            run_id: "json".into(),
        })
    })
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}

#[test]
fn test_summary_content_layout_and_actual_error_formatting_are_covered() {
    assert!(
        content_layout().unwrap().size() >= size_of::<serde::__private228::de::Content<'static>>()
    );
    let input = serde_json::to_vec(&"\n\"\\control".repeat(1_024)).unwrap();
    let quote = requirement(&input, &mut || Ok(())).unwrap();
    let mut message = None;
    let actual = allocation_counter::measure(|| {
        let error = serde_json::from_slice::<u64>(&input).unwrap_err();
        message = Some(format!("invalid cumulative summary: {error}"));
    });
    assert!(
        quote >= actual.bytes_max,
        "quote={quote}, actual={actual:?}"
    );
    assert!(allocation(usize::MAX).is_err());
    assert!(growing_bytes(usize::MAX).is_err());
}
