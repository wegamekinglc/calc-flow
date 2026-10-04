use super::{Callback, Record, codec};
use crate::{EventTime, IngressProgress, IngressState};

fn record() -> Record {
    Record {
        callback: Callback::Data {
            side: 1,
            sequence: u64::MAX - 1,
        },
        input_watermark: Some(EventTime::from_micros(i64::MIN)),
        progress: [
            IngressProgress::new(IngressState::Idle, Some(EventTime::from_micros(-1))),
            IngressProgress::new(IngressState::Ended, None),
        ],
        max_rows: 1000,
        max_bytes: 1024 * 1024,
    }
}

#[test]
fn callback_frame_preserves_full_coordinates_and_progress() {
    let records = [
        record(),
        Record {
            callback: Callback::Progress,
            ..record()
        },
        Record {
            callback: Callback::End,
            input_watermark: None,
            ..record()
        },
    ];
    let bytes = codec::encode(&records);
    let mut decoded = Vec::with_capacity(3);
    codec::decode_into(&bytes, 3, &mut decoded).unwrap();
    assert_eq!(decoded, records);
}

#[test]
fn callback_frame_rejects_corrupt_tags_lengths_and_unfunded_records() {
    let original = codec::encode(&[record()]);
    for (offset, value) in [(0, 0), (8, 2), (16, 3), (17, 2), (26, 2), (35, 3)] {
        let mut bytes = original.clone();
        bytes[offset] = value;
        assert!(codec::decode_into(&bytes, 1, &mut Vec::with_capacity(1)).is_err());
    }
    assert!(
        codec::decode_into(
            &original[..original.len() - 1],
            1,
            &mut Vec::with_capacity(1)
        )
        .is_err()
    );
    assert!(codec::decode_into(&original, 1, &mut Vec::new()).is_err());
    let mut extra = original;
    extra.push(0);
    assert!(codec::decode_into(&extra, 1, &mut Vec::with_capacity(1)).is_err());
}
