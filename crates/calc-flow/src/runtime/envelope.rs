use std::sync::Arc;

use uuid::Uuid;

use crate::{Batch, CalcFlowError, Result};

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct ControlMarker {
    kind: MarkerKind,
    occurrence: ControlOccurrence,
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub(crate) struct ControlOccurrence(Uuid);

#[derive(Clone, Debug)]
pub(crate) struct SharedControlMarker(Arc<ControlMarker>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MarkerKind {
    Watermark,
    Epoch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ControlKind {
    Watermark,
    Epoch,
}

impl ControlMarker {
    pub(crate) fn watermark() -> Self {
        Self::new(MarkerKind::Watermark)
    }

    pub(crate) fn epoch() -> Self {
        Self::new(MarkerKind::Epoch)
    }

    fn new(kind: MarkerKind) -> Self {
        Self {
            kind,
            occurrence: ControlOccurrence(Uuid::new_v4()),
        }
    }

    pub(crate) const fn kind(&self) -> ControlKind {
        match self.kind {
            MarkerKind::Watermark => ControlKind::Watermark,
            MarkerKind::Epoch => ControlKind::Epoch,
        }
    }

    pub(crate) const fn occurrence(&self) -> &ControlOccurrence {
        &self.occurrence
    }

    pub(crate) fn into_shared(self) -> SharedControlMarker {
        SharedControlMarker(Arc::new(self))
    }
}

impl SharedControlMarker {
    pub(crate) fn marker(&self) -> &ControlMarker {
        &self.0
    }

    #[cfg(test)]
    pub(crate) fn shares_allocation(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

#[derive(Clone, Debug)]
pub(crate) enum RuntimeEnvelope {
    Data(Batch),
    Control(SharedControlMarker),
}

impl RuntimeEnvelope {
    pub(crate) fn data(&self) -> Result<&Batch> {
        match self {
            Self::Data(batch) => Ok(batch),
            Self::Control(_) => Err(CalcFlowError::Internal {
                message: "control envelope reached the data path".into(),
            }),
        }
    }

    pub(crate) fn into_data(self) -> Result<Batch> {
        match self {
            Self::Data(batch) => Ok(batch),
            Self::Control(_) => Err(CalcFlowError::Internal {
                message: "control envelope reached a data output".into(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};

    use super::{ControlKind, ControlMarker, RuntimeEnvelope};
    use crate::{Batch, BatchMetadata};

    #[test]
    fn data_envelope_preserves_batch_sharing() {
        let record_batch = RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![1, 2, 3])) as _,
        )])
        .unwrap();
        let batch = Batch::table(vec![record_batch], BatchMetadata::default()).unwrap();
        let schema = Arc::clone(batch.table_payload().unwrap().schema());

        let envelope = RuntimeEnvelope::Data(batch.clone());

        assert!(Arc::ptr_eq(
            envelope.data().unwrap().table_payload().unwrap().schema(),
            &schema
        ));
        assert!(Arc::ptr_eq(
            envelope
                .into_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .schema(),
            &schema
        ));
    }

    #[test]
    fn runtime_envelope_marker_single_submission_shares_one_occurrence() {
        let shared = ControlMarker::watermark().into_shared();
        let sibling = shared.clone();

        assert_eq!(shared.marker().kind(), ControlKind::Watermark);
        assert_eq!(shared.marker().occurrence(), sibling.marker().occurrence());
        assert!(shared.shares_allocation(&sibling));
    }

    #[test]
    fn runtime_envelope_marker_retry_mints_new_occurrence() {
        let failed = ControlMarker::watermark();
        let retry = ControlMarker::watermark();

        assert_ne!(failed.occurrence(), retry.occurrence());
    }
}
