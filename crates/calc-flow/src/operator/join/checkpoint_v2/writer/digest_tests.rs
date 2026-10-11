use super::*;

thread_local! {
    static HASHED_PAYLOAD_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(in super::super) fn record_payload_hash(bytes: &[u8]) {
    if bytes.starts_with(PAYLOAD_MAGIC) {
        HASHED_PAYLOAD_BYTES.set(HASHED_PAYLOAD_BYTES.get() + bytes.len());
    }
}

#[test]
fn test_checkpoint_payload_hashes_each_byte_once() {
    let (rows, record) = fixture();
    let mut observations = Vec::new();
    for delta in [false, true] {
        let (pool, workspace) = reservation(16 << 20);
        HASHED_PAYLOAD_BYTES.set(0);
        let output = if delta {
            pending_owned(
                &[
                    pending_upsert(JoinSide::Left, &rows[0]),
                    pending_upsert(JoinSide::Right, &rows[2]),
                ],
                &[record.schema(), record.schema()],
                Epoch::INITIAL,
                &workspace,
                &|| Ok(()),
            )
        } else {
            base(
                [&rows, &rows],
                &[record.schema(), record.schema()],
                &workspace,
                &|| Ok(()),
            )
        }
        .unwrap();
        let hashed = HASHED_PAYLOAD_BYTES.replace(0);
        let expected = output
            .segments
            .iter()
            .filter(|(name, _)| name.contains("-payload-"))
            .map(|(name, segment)| {
                let sha256 = hex::encode(Sha256::digest(segment.bytes()));
                assert_eq!(segment.sha256(), sha256);
                assert!(name.ends_with(&sha256));
                let entry = output
                    .payloads
                    .iter()
                    .find(|entry| entry.sha256 == sha256)
                    .unwrap();
                assert_eq!(entry.bytes, segment.bytes().len() as u64);
                segment.bytes().len()
            })
            .sum::<usize>();
        assert_eq!(output.payloads.len(), 2);
        assert_index_digests(&output, delta);
        drop((output, workspace));
        assert_eq!(pool.reserved(), 0);
        observations.push((delta, hashed, expected));
    }
    for (delta, hashed, expected) in observations {
        assert_eq!(
            hashed, expected,
            "checkpoint payload bytes were hashed more than once; delta={delta}"
        );
    }
}

fn assert_index_digests(output: &Base, delta: bool) {
    use crate::operator::join::checkpoint_v2::frame::{Index, Payload, Record};

    for side in [JoinSide::Left, JoinSide::Right] {
        let entry = output
            .payloads
            .iter()
            .find(|entry| entry.side == side.as_str())
            .unwrap();
        let payload = &output.segments[&format!("{}-payload-{}", side.as_str(), entry.sha256)];
        let decoded = Payload::decode(payload.bytes(), side).unwrap();
        assert_eq!(entry.rows, if delta { 1 } else { 3 });
        assert_eq!(decoded.rows as u64, entry.rows);
        let name = if delta {
            format!("{}-delta-{}", side.as_str(), Epoch::INITIAL.as_u64())
        } else {
            format!("{}-base", side.as_str())
        };
        let bytes = output.segments[&name].bytes();
        let index = if delta {
            Index::delta(bytes, side, Epoch::INITIAL.as_u64())
        } else {
            Index::base(bytes, side)
        }
        .unwrap();
        assert_eq!(index.upserts as u64, entry.rows);
        assert_eq!(index.tombstones, 0);
        let mut records = index.records();
        let mut count = 0;
        while let Some(record) = records.next().unwrap() {
            let Record::Upsert(row) = record else {
                panic!("unexpected tombstone")
            };
            assert_eq!(hex::encode(row.digest), payload.sha256());
            assert_eq!(
                decoded
                    .row_id(usize::try_from(row.payload_row).unwrap())
                    .unwrap(),
                row.row_id
            );
            count += 1;
        }
        assert_eq!(count, index.upserts);
    }
}
