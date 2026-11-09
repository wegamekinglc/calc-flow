use std::{ops::Deref, sync::Arc};

use super::super::{Result, RollingStreamState, internal_error};

pub(in super::super) enum StateOwner {
    Owned(Box<RollingStreamState>),
    Shared(Arc<RollingStreamState>),
}

impl StateOwner {
    pub(in super::super) fn new(state: RollingStreamState, shared: bool) -> Self {
        if shared {
            Self::Shared(Arc::new(state))
        } else {
            Self::Owned(Box::new(state))
        }
    }

    pub(in super::super) fn packet(&self) -> Option<Arc<RollingStreamState>> {
        match self {
            Self::Owned(_) => None,
            Self::Shared(state) => Some(Arc::clone(state)),
        }
    }

    pub(in super::super) fn owned_mut(&mut self) -> Result<&mut RollingStreamState> {
        match self {
            Self::Owned(state) => Ok(state),
            Self::Shared(_) => Err(internal_error(
                "indicator state requires prepared replacement",
            )),
        }
    }
}

impl Deref for StateOwner {
    type Target = RollingStreamState;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(state) => state,
            Self::Shared(state) => state,
        }
    }
}
