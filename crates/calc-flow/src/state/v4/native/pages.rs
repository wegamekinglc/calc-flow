use super::{NativePage, mismatch};
use crate::Result;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Clone, Debug)]
pub(crate) enum NativePageChange {
    Upsert {
        page: NativePage,
        previous_bytes: Option<u64>,
    },
    Remove {
        column: String,
        previous_bytes: u64,
    },
}

impl NativePageChange {
    pub(crate) fn upsert(page: NativePage, previous: Option<&NativePage>) -> Result<Self> {
        let previous_bytes = previous.map(NativePage::reservation_bytes).transpose()?;
        Ok(Self::Upsert {
            page,
            previous_bytes,
        })
    }

    pub(crate) fn remove(previous: &NativePage) -> Result<Self> {
        Ok(Self::Remove {
            column: previous.column.clone(),
            previous_bytes: previous.reservation_bytes()?,
        })
    }

    pub(crate) fn column(&self) -> &str {
        match self {
            Self::Upsert { page, .. } => &page.column,
            Self::Remove { column, .. } => column,
        }
    }

    fn page(&self) -> Option<&NativePage> {
        match self {
            Self::Upsert { page, .. } => Some(page),
            Self::Remove { .. } => None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NativePageSet {
    parent: Option<Arc<NativePageSet>>,
    changes: Vec<NativePageChange>,
    len: usize,
    reservation_bytes: u64,
}

impl NativePageSet {
    #[cfg(test)]
    pub(crate) fn full(pages: Vec<NativePage>) -> Result<Self> {
        let changes = pages
            .into_iter()
            .map(|page| NativePageChange::upsert(page, None))
            .collect::<Result<Vec<_>>>()?;
        Self::new(None, changes)
    }

    pub(crate) fn new(
        parent: Option<Arc<NativePageSet>>,
        changes: Vec<NativePageChange>,
    ) -> Result<Self> {
        let mut columns = BTreeSet::new();
        for change in &changes {
            if !columns.insert(change.column()) {
                return Err(mismatch("native state page names are duplicated".into()));
            }
            if let Some(page) = change.page()
                && (page.range.start > page.range.end || page.column == super::METADATA_COLUMN)
            {
                return Err(mismatch(
                    "native state page name or range is invalid".into(),
                ));
            }
        }
        let (mut len, mut reservation_bytes) = parent
            .as_ref()
            .map_or((0, 0), |parent| (parent.len, parent.reservation_bytes));
        for change in &changes {
            match change {
                NativePageChange::Upsert {
                    page,
                    previous_bytes,
                } => {
                    reservation_bytes = reservation_bytes
                        .checked_add(page.reservation_bytes()?)
                        .ok_or_else(|| mismatch("native capture reservation overflow".into()))?;
                    if let Some(previous) = previous_bytes {
                        len = len.checked_sub(1).ok_or_else(|| {
                            mismatch("native page replacement count underflow".into())
                        })?;
                        reservation_bytes =
                            reservation_bytes.checked_sub(*previous).ok_or_else(|| {
                                mismatch("native page replacement reservation underflow".into())
                            })?;
                    }
                    len = len
                        .checked_add(1)
                        .ok_or_else(|| mismatch("native page count overflow".into()))?;
                }
                NativePageChange::Remove { previous_bytes, .. } => {
                    len = len
                        .checked_sub(1)
                        .ok_or_else(|| mismatch("native page removal count underflow".into()))?;
                    reservation_bytes =
                        reservation_bytes
                            .checked_sub(*previous_bytes)
                            .ok_or_else(|| {
                                mismatch("native page removal reservation underflow".into())
                            })?;
                }
            }
        }
        Ok(Self {
            parent,
            changes,
            len,
            reservation_bytes,
        })
    }

    pub(crate) fn extend(self, additional: Vec<NativePageChange>) -> Result<Self> {
        let Self {
            parent,
            mut changes,
            ..
        } = self;
        changes.extend(additional);
        Self::new(parent, changes)
    }

    pub(crate) fn changes(&self) -> &[NativePageChange] {
        &self.changes
    }

    pub(crate) const fn reservation_bytes(&self) -> u64 {
        self.reservation_bytes
    }

    #[cfg(test)]
    pub(crate) fn change_count(&self) -> usize {
        self.changes.len()
    }

    pub(crate) fn materialize(&self) -> Vec<NativePage> {
        let mut lineage = Vec::new();
        let mut current = self;
        loop {
            lineage.push(&current.changes);
            let Some(parent) = &current.parent else {
                break;
            };
            current = parent;
        }
        let mut pages = BTreeMap::new();
        for changes in lineage.into_iter().rev() {
            for change in changes {
                match change {
                    NativePageChange::Upsert { page, .. } => {
                        pages.insert(page.column.clone(), page.clone());
                    }
                    NativePageChange::Remove { column, .. } => {
                        pages.remove(column);
                    }
                }
            }
        }
        pages.into_values().collect()
    }
}
