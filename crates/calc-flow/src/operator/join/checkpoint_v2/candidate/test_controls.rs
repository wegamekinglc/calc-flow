use super::*;
use crate::operator::join::StreamJoinOperator;

fn config(operator: &StreamJoinOperator) -> Config<'_> {
    Config {
        name: &operator.name,
        spec: &operator.spec,
        schemas: [
            Arc::clone(operator.input_schema(0)),
            Arc::clone(operator.input_schema(1)),
        ],
        keys: [
            &operator.compiled.left_key_indices,
            &operator.compiled.right_key_indices,
        ],
        time_indices: [
            operator.compiled.left_event_time_index,
            operator.compiled.right_event_time_index,
        ],
        metadata_hook: operator.metadata_test_hook.as_ref(),
        decoded_hook: operator.decoded_row_test_hook.as_ref(),
        owned_work: false,
    }
}

impl StreamJoinOperator {
    pub(in crate::operator::join) fn restore_v2_checked_for_test(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let credit = self
            .runtime
            .runtime()?
            .incremental_reservation("stream-join-v2-restore");
        let config = config(self);
        let prepared = Decoder::new(snapshot, &config, &credit, Vec::new(), check)?.decode()?;
        self.install_v2(prepared, check)
    }

    pub(in crate::operator::join) fn assert_v2_scratch_retirement_for_test(
        &mut self,
        snapshot: &OperatorStateSnapshot,
    ) {
        let runtime = self.runtime.runtime().unwrap();
        let pool = runtime.incremental_memory_pool();
        let baseline = pool.reserved();
        let credit = runtime.incremental_reservation("stream-join-v2-restore");
        let config = config(self);
        let check = || Ok(());
        let mut decoder = Decoder::new(snapshot, &config, &credit, Vec::new(), &check).unwrap();
        let inventory = Inventory::decode(snapshot, &check).unwrap();
        let geometry = Geometry::inspect(snapshot, &inventory, &check).unwrap();
        let metadata = decoder.decode_metadata().unwrap();
        let history = History::new(snapshot, &geometry, &decoder.workspace, &check).unwrap();
        let summaries = decoder.summaries(&history, &metadata).unwrap();
        let before = pool.reserved();
        let workspace = decoder.workspace.size();
        let payloads = decoder.decode_payloads(geometry.payloads).unwrap();
        let payload_credit = payloads
            .values()
            .map(|payload| payload.owner.funded_owner().1)
            .sum::<usize>();
        let map_credit = decoder.workspace.size() - workspace;
        assert_eq!(
            map_credit,
            budget::tree::<(JoinSide, [u8; 32]), Decoded<'_>>(geometry.payloads).unwrap()
        );
        assert_eq!(
            pool.reserved(),
            before + map_credit + payload_credit,
            "all IPC scratch retired while all decoded owners remain alive"
        );
        let before = pool.reserved();
        let container_credit = decoder.containers.credit.size();
        let rows = decoder
            .validate_histories(&history, &metadata, summaries, &payloads)
            .unwrap();
        let containers = decoder.containers.credit.size();
        assert_eq!(
            pool.reserved(),
            before + containers - container_credit,
            "all row and scan scratch retired while stored rows remain alive"
        );
        let key = Arc::clone(&rows[0][0].encoded_key);
        let expected_key = key.as_slice().to_vec();
        let record = rows[0][0].record.view();
        let buffer = record.column(0).to_data().buffers()[0].clone();
        let expected_buffer = buffer.as_slice().to_vec();
        drop(record);
        let columnar::RowPayload::RestoredV2 { chunk, .. } = &rows[0][0].record else {
            panic!("restored V2 payload");
        };
        let payload_credit = chunk.funded_owner().1;
        drop((rows, payloads, history, decoder, credit));
        assert_eq!(pool.reserved(), baseline + containers + payload_credit);
        assert_eq!(key.as_slice(), expected_key);
        drop(key);
        assert_eq!(pool.reserved(), baseline + payload_credit);
        assert_eq!(buffer.as_slice(), expected_buffer);
        drop(buffer);
        assert_eq!(pool.reserved(), baseline);
    }
}
