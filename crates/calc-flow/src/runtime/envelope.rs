use crate::Batch;

#[derive(Clone, Debug)]
pub(crate) enum RuntimeEnvelope {
    Data(Batch),
}

impl RuntimeEnvelope {
    pub(crate) const fn data(&self) -> &Batch {
        match self {
            Self::Data(batch) => batch,
        }
    }

    pub(crate) fn into_data(self) -> Batch {
        match self {
            Self::Data(batch) => batch,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

    use super::RuntimeEnvelope;
    use crate::{Batch, BatchMetadata};

    #[test]
    fn runtime_envelope_data_preserves_batch_sharing() {
        let record_batch = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as _,
        )])
        .unwrap();
        let batch = Batch::table(vec![record_batch], BatchMetadata::default()).unwrap();
        let schema = Arc::clone(batch.table_payload().unwrap().schema());

        let envelope = RuntimeEnvelope::Data(batch.clone());

        assert!(Arc::ptr_eq(
            envelope.data().table_payload().unwrap().schema(),
            &schema
        ));
        assert!(Arc::ptr_eq(
            envelope.into_data().table_payload().unwrap().schema(),
            &schema
        ));
    }
}
