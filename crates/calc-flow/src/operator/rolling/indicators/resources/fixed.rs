use super::super::footprint_state;
use super::{Arc, Charges, Commit, Footprint, Resources, StateBudget, replace_pins};
use crate::{Result, operator::rolling::RollingStreamState};

pub(super) struct Empty {
    state: Arc<RollingStreamState>,
    footprint: Arc<Footprint>,
    charges: Arc<Charges>,
}

impl Empty {
    pub(super) fn charges(&self) -> &Charges {
        &self.charges
    }
}

impl Resources {
    pub(in crate::operator::rolling) fn for_indicator(
        node: &str,
    ) -> Result<(Self, Arc<RollingStreamState>)> {
        let resources = Self::default();
        let budget = StateBudget::default();
        let mut credit = resources.begin_work();
        let mut quote = Footprint::default();
        quote.add_owned(footprint_state::packet_allocation_bytes(node)?, node)?;
        credit.adjust(quote, 0, budget, node)?;
        let (state, footprint) =
            footprint_state::prepare_checked(RollingStreamState::default(), node, &mut || Ok(()))?;
        credit.adjust(footprint.clone(), 0, budget, node)?;
        let commit = Commit::new(footprint);
        resources.0.lock().fixed = Some(Empty {
            state: Arc::clone(&state),
            footprint: Arc::clone(&commit.footprint),
            charges: Arc::clone(&commit.charges),
        });
        credit.commit(commit);
        Ok((resources, state))
    }

    pub(in crate::operator::rolling) fn empty_state(&self) -> Arc<RollingStreamState> {
        Arc::clone(
            &self
                .0
                .lock()
                .fixed
                .as_ref()
                .expect("indicator empty packet")
                .state,
        )
    }

    pub(in crate::operator::rolling) fn reset_committed(&self) {
        let (footprint, charges) = {
            let pool = self.0.lock();
            let fixed = pool.fixed.as_ref().expect("indicator empty packet");
            (Arc::clone(&fixed.footprint), Arc::clone(&fixed.charges))
        };
        replace_pins(&self.2, footprint);
        self.0.lock().committed = charges;
    }
}
