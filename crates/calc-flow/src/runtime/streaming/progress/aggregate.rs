use std::collections::BTreeMap;

use crate::EventTime;

use super::{
    prepare::BindingOrdinal,
    types::{CheckedSemanticAllocator, GlobalSequence, IdleEpoch, ProgressFailure},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IngressActivity {
    Active { watermark: Option<EventTime> },
    Idle { watermark: Option<EventTime> },
    Ended { final_watermark: Option<EventTime> },
}

impl IngressActivity {
    const fn watermark(self) -> Option<EventTime> {
        match self {
            Self::Active { watermark } | Self::Idle { watermark } => watermark,
            Self::Ended { final_watermark } => final_watermark,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AggregateInput {
    Data,
    Watermark(EventTime),
    Idle,
    End,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProgressEmissionKind {
    Watermark(EventTime),
    Idle,
    EndOfInput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProgressEmission {
    pub(crate) sequence: GlobalSequence,
    pub(crate) kind: ProgressEmissionKind,
}

#[derive(Clone, Debug)]
pub(crate) struct MultiInputProgress {
    ingresses: BTreeMap<BindingOrdinal, IngressActivity>,
    last_emitted_watermark: Option<EventTime>,
    idle_latched: bool,
    idle_epoch: IdleEpoch,
    terminal: bool,
    next_global_sequence: CheckedSemanticAllocator,
    next_idle_epoch: CheckedSemanticAllocator,
}

struct EvaluationRollback {
    activity: IngressActivity,
    last_emitted_watermark: Option<EventTime>,
    idle_latched: bool,
    idle_epoch: IdleEpoch,
    terminal: bool,
    next_global_sequence: CheckedSemanticAllocator,
    next_idle_epoch: CheckedSemanticAllocator,
}

impl MultiInputProgress {
    pub(crate) fn new(ordinals: impl IntoIterator<Item = BindingOrdinal>) -> Self {
        Self {
            ingresses: ordinals
                .into_iter()
                .map(|ordinal| (ordinal, IngressActivity::Active { watermark: None }))
                .collect(),
            last_emitted_watermark: None,
            idle_latched: false,
            idle_epoch: IdleEpoch(0),
            terminal: false,
            next_global_sequence: CheckedSemanticAllocator::new(
                0,
                "runtime.progress.counters.global_sequence",
            ),
            next_idle_epoch: CheckedSemanticAllocator::new(
                1,
                "runtime.progress.counters.idle_epoch",
            ),
        }
    }

    pub(crate) fn restore(
        activities: impl IntoIterator<Item = (BindingOrdinal, IngressActivity)>,
    ) -> Self {
        let ingresses = activities.into_iter().collect::<BTreeMap<_, _>>();
        // The emitted frontier is the S5.2 input watermark: the minimum over
        // primed Active ingresses. A cut can persist skewed per-ingress
        // watermarks (one source ahead of another); seeding from the maximum
        // would re-emit a watermark the operator never sent and let downstream
        // windows close ahead of replayed rows. With no primed-active minimum
        // (all idle/ended, or an unprimed active ingress) no such minimum
        // exists: keep the conservative maximum so emissions never retreat.
        let active_watermarks = ingresses
            .values()
            .filter_map(|activity| match activity {
                IngressActivity::Active { watermark } => Some(*watermark),
                IngressActivity::Idle { .. } | IngressActivity::Ended { .. } => None,
            })
            .collect::<Vec<_>>();
        let last_emitted_watermark = if active_watermarks.is_empty() {
            ingresses
                .values()
                .filter_map(|activity| activity.watermark())
                .max()
        } else if active_watermarks.iter().any(Option::is_none) {
            None
        } else {
            active_watermarks.into_iter().min().expect("non-empty")
        };
        let terminal = ingresses
            .values()
            .all(|activity| matches!(activity, IngressActivity::Ended { .. }));
        let idle_latched = !terminal
            && ingresses.values().all(|activity| {
                matches!(
                    activity,
                    IngressActivity::Idle { .. } | IngressActivity::Ended { .. }
                )
            });
        Self {
            ingresses,
            last_emitted_watermark,
            idle_latched,
            idle_epoch: IdleEpoch(0),
            terminal,
            next_global_sequence: CheckedSemanticAllocator::new(
                0,
                "runtime.progress.counters.global_sequence",
            ),
            next_idle_epoch: CheckedSemanticAllocator::new(
                1,
                "runtime.progress.counters.idle_epoch",
            ),
        }
    }

    pub(crate) fn evaluate(
        &mut self,
        ordinal: BindingOrdinal,
        input: AggregateInput,
    ) -> Result<Vec<ProgressEmission>, ProgressFailure> {
        let rollback = self.evaluation_rollback(ordinal)?;
        if let Err(error) = self.apply(ordinal, input) {
            self.restore_evaluation(ordinal, rollback);
            return Err(error);
        }
        match self.derive_emissions() {
            Ok(emissions) => Ok(emissions),
            Err(error) => {
                self.restore_evaluation(ordinal, rollback);
                Err(error)
            }
        }
    }

    fn evaluation_rollback(
        &self,
        ordinal: BindingOrdinal,
    ) -> Result<EvaluationRollback, ProgressFailure> {
        let activity = self.ingresses.get(&ordinal).copied().ok_or_else(|| {
            ProgressFailure::protocol(
                "runtime.progress.aggregate.binding",
                "unknown binding ordinal",
            )
        })?;
        Ok(EvaluationRollback {
            activity,
            last_emitted_watermark: self.last_emitted_watermark,
            idle_latched: self.idle_latched,
            idle_epoch: self.idle_epoch,
            terminal: self.terminal,
            next_global_sequence: self.next_global_sequence.clone(),
            next_idle_epoch: self.next_idle_epoch.clone(),
        })
    }

    fn restore_evaluation(&mut self, ordinal: BindingOrdinal, rollback: EvaluationRollback) {
        self.ingresses.insert(ordinal, rollback.activity);
        self.last_emitted_watermark = rollback.last_emitted_watermark;
        self.idle_latched = rollback.idle_latched;
        self.idle_epoch = rollback.idle_epoch;
        self.terminal = rollback.terminal;
        self.next_global_sequence = rollback.next_global_sequence;
        self.next_idle_epoch = rollback.next_idle_epoch;
    }

    fn apply(
        &mut self,
        ordinal: BindingOrdinal,
        input: AggregateInput,
    ) -> Result<(), ProgressFailure> {
        let activity = self.ingresses.get_mut(&ordinal).ok_or_else(|| {
            ProgressFailure::protocol(
                "runtime.progress.aggregate.binding",
                "unknown binding ordinal",
            )
        })?;
        if matches!(activity, IngressActivity::Ended { .. }) {
            return Err(ProgressFailure::protocol(
                "runtime.progress.aggregate.post_end",
                "input/control after EndOfInput is forbidden",
            ));
        }
        let watermark = activity.watermark();
        *activity = match input {
            AggregateInput::Data => {
                self.idle_latched = false;
                IngressActivity::Active { watermark }
            }
            AggregateInput::Watermark(next) => {
                self.idle_latched = false;
                IngressActivity::Active {
                    watermark: Some(next),
                }
            }
            AggregateInput::Idle => IngressActivity::Idle { watermark },
            AggregateInput::End => IngressActivity::Ended {
                final_watermark: watermark,
            },
        };
        Ok(())
    }

    fn derive_emissions(&mut self) -> Result<Vec<ProgressEmission>, ProgressFailure> {
        if self.terminal {
            return Ok(Vec::new());
        }
        self.pending_emission_kinds()?
            .into_iter()
            .map(|kind| self.allocate_emission(kind))
            .collect()
    }

    fn pending_emission_kinds(&mut self) -> Result<Vec<ProgressEmissionKind>, ProgressFailure> {
        let mut kinds = Vec::new();
        if let Some(minimum) = self.next_watermark() {
            self.last_emitted_watermark = Some(minimum);
            kinds.push(ProgressEmissionKind::Watermark(minimum));
        }
        if self.all_ended() {
            self.terminal = true;
            self.idle_latched = false;
            kinds.push(ProgressEmissionKind::EndOfInput);
        } else if self.all_live_idle() && !self.idle_latched {
            self.advance_idle_epoch()?;
            kinds.push(ProgressEmissionKind::Idle);
        }
        Ok(kinds)
    }

    fn next_watermark(&self) -> Option<EventTime> {
        let active = self
            .ingresses
            .values()
            .filter_map(|activity| match activity {
                IngressActivity::Active { watermark } => Some(*watermark),
                IngressActivity::Idle { .. } | IngressActivity::Ended { .. } => None,
            })
            .collect::<Vec<_>>();
        if active.is_empty() || active.iter().any(Option::is_none) {
            return None;
        }
        active.into_iter().flatten().min().filter(|minimum| {
            self.last_emitted_watermark
                .is_none_or(|previous| *minimum > previous)
        })
    }

    fn all_ended(&self) -> bool {
        self.ingresses
            .values()
            .all(|activity| matches!(activity, IngressActivity::Ended { .. }))
    }

    fn all_live_idle(&self) -> bool {
        self.ingresses
            .values()
            .any(|activity| !matches!(activity, IngressActivity::Ended { .. }))
            && self.ingresses.values().all(|activity| {
                matches!(
                    activity,
                    IngressActivity::Idle { .. } | IngressActivity::Ended { .. }
                )
            })
    }

    fn advance_idle_epoch(&mut self) -> Result<(), ProgressFailure> {
        let epoch = self
            .next_idle_epoch
            .allocate()
            .map_err(|_| ProgressFailure::counter("runtime.progress.counters.idle_epoch"))?;
        self.idle_epoch = IdleEpoch(epoch);
        self.idle_latched = true;
        Ok(())
    }

    fn allocate_emission(
        &mut self,
        kind: ProgressEmissionKind,
    ) -> Result<ProgressEmission, ProgressFailure> {
        self.next_global_sequence
            .allocate()
            .map(GlobalSequence)
            .map(|sequence| ProgressEmission { sequence, kind })
            .map_err(|_| ProgressFailure::counter("runtime.progress.counters.global_sequence"))
    }

    pub(crate) fn activity(&self, ordinal: BindingOrdinal) -> Option<IngressActivity> {
        self.ingresses.get(&ordinal).copied()
    }

    pub(crate) const fn last_emitted_watermark(&self) -> Option<EventTime> {
        self.last_emitted_watermark
    }

    pub(crate) const fn idle_epoch(&self) -> IdleEpoch {
        self.idle_epoch
    }

    pub(crate) const fn idle_latched(&self) -> bool {
        self.idle_latched
    }

    pub(crate) const fn terminal(&self) -> bool {
        self.terminal
    }

    pub(crate) fn next_global_sequence(&self) -> u64 {
        self.next_global_sequence.next()
    }

    pub(crate) fn next_idle_epoch(&self) -> u64 {
        self.next_idle_epoch.next()
    }

    pub(crate) fn set_next_global_for_test(&mut self, next: u64) {
        self.next_global_sequence.set_next_for_restore(next);
    }

    pub(crate) fn set_next_idle_epoch_for_test(&mut self, next: u64) {
        self.next_idle_epoch.set_next_for_restore(next);
    }
}

#[cfg(test)]
mod tests {
    use super::{AggregateInput, MultiInputProgress, ProgressEmissionKind};
    use crate::{EventTime, runtime::streaming::progress::prepare::BindingOrdinal};

    fn wm(value: i64) -> EventTime {
        EventTime::from_micros(value)
    }

    fn kinds(
        progress: &mut MultiInputProgress,
        ordinal: u64,
        input: AggregateInput,
    ) -> Vec<ProgressEmissionKind> {
        progress
            .evaluate(BindingOrdinal::new(ordinal), input)
            .unwrap()
            .into_iter()
            .map(|emission| emission.kind)
            .collect()
    }

    #[test]
    fn active_ingress_without_watermark_holds_progress() {
        let mut progress =
            MultiInputProgress::new([BindingOrdinal::new(0), BindingOrdinal::new(1)]);
        assert!(kinds(&mut progress, 0, AggregateInput::Watermark(wm(9))).is_empty());
    }

    #[test]
    fn multi_input_watermark_is_active_minimum() {
        let mut progress =
            MultiInputProgress::new([BindingOrdinal::new(0), BindingOrdinal::new(1)]);
        kinds(&mut progress, 0, AggregateInput::Watermark(wm(9)));
        assert_eq!(
            kinds(&mut progress, 1, AggregateInput::Watermark(wm(7))),
            [ProgressEmissionKind::Watermark(wm(7))]
        );
        assert_eq!(
            kinds(&mut progress, 1, AggregateInput::Watermark(wm(11))),
            [ProgressEmissionKind::Watermark(wm(9))]
        );
    }

    #[test]
    fn idle_and_ended_ingresses_are_excluded_from_minimum() {
        let mut progress =
            MultiInputProgress::new([BindingOrdinal::new(0), BindingOrdinal::new(1)]);
        kinds(&mut progress, 0, AggregateInput::Watermark(wm(3)));
        kinds(&mut progress, 1, AggregateInput::Watermark(wm(8)));
        assert_eq!(
            kinds(&mut progress, 0, AggregateInput::Idle),
            [ProgressEmissionKind::Watermark(wm(8))]
        );
        assert!(kinds(&mut progress, 0, AggregateInput::End).is_empty());
    }

    #[test]
    fn restore_seeds_emitted_frontier_at_the_active_minimum() {
        // A checkpoint cut can persist skewed per-ingress watermarks when one
        // source runs ahead of another; restore must resume emission from the
        // S5.2 input watermark (the primed-active minimum), never the maximum,
        // or the first post-restore transition re-emits a watermark the
        // operator never sent and downstream windows close ahead of replayed
        // rows.
        let mut restored = MultiInputProgress::restore([
            (
                BindingOrdinal::new(0),
                super::IngressActivity::Active {
                    watermark: Some(wm(640)),
                },
            ),
            (
                BindingOrdinal::new(1),
                super::IngressActivity::Active {
                    watermark: Some(wm(638)),
                },
            ),
        ]);
        assert_eq!(restored.last_emitted_watermark(), Some(wm(638)));
        assert_eq!(
            kinds(&mut restored, 1, AggregateInput::Watermark(wm(639))),
            [ProgressEmissionKind::Watermark(wm(639))]
        );
    }

    #[test]
    fn restore_with_unprimed_active_ingress_has_no_emitted_frontier() {
        // S5.2: an Active ingress that has delivered no watermark leaves the
        // input watermark undefined, so nothing was ever emitted and restore
        // must not invent a frontier from the other ingress.
        let mut restored = MultiInputProgress::restore([
            (
                BindingOrdinal::new(0),
                super::IngressActivity::Active {
                    watermark: Some(wm(640)),
                },
            ),
            (
                BindingOrdinal::new(1),
                super::IngressActivity::Active { watermark: None },
            ),
        ]);
        assert_eq!(restored.last_emitted_watermark(), None);
        assert_eq!(
            kinds(&mut restored, 1, AggregateInput::Watermark(wm(100))),
            [ProgressEmissionKind::Watermark(wm(100))]
        );
    }

    #[test]
    fn restore_without_active_ingresses_keeps_the_conservative_maximum() {
        // With no active ingress the pre-cut emission may have advanced to any
        // ingress watermark (idle/ended exclusion), so the frontier stays at
        // the conservative maximum to keep emissions monotone.
        let restored = MultiInputProgress::restore([
            (
                BindingOrdinal::new(0),
                super::IngressActivity::Idle {
                    watermark: Some(wm(640)),
                },
            ),
            (
                BindingOrdinal::new(1),
                super::IngressActivity::Ended {
                    final_watermark: Some(wm(638)),
                },
            ),
        ]);
        assert_eq!(restored.last_emitted_watermark(), Some(wm(640)));
    }

    #[test]
    fn multi_input_emits_idle_once_per_idle_epoch() {
        let mut progress =
            MultiInputProgress::new([BindingOrdinal::new(0), BindingOrdinal::new(1)]);
        assert!(kinds(&mut progress, 0, AggregateInput::Idle).is_empty());
        assert_eq!(
            kinds(&mut progress, 1, AggregateInput::Idle),
            [ProgressEmissionKind::Idle]
        );
        assert_eq!(progress.idle_epoch().0, 1);
    }

    #[test]
    fn repeated_idle_is_idempotent_within_epoch() {
        let mut progress = MultiInputProgress::new([BindingOrdinal::new(0)]);
        assert_eq!(
            kinds(&mut progress, 0, AggregateInput::Idle),
            [ProgressEmissionKind::Idle]
        );
        assert!(kinds(&mut progress, 0, AggregateInput::Idle).is_empty());
    }

    #[test]
    fn data_reactivation_precedes_processing() {
        let mut progress = MultiInputProgress::new([BindingOrdinal::new(0)]);
        kinds(&mut progress, 0, AggregateInput::Idle);
        assert!(progress.idle_latched());
        kinds(&mut progress, 0, AggregateInput::Data);
        assert!(!progress.idle_latched());
        assert!(matches!(
            progress.activity(BindingOrdinal::new(0)),
            Some(super::IngressActivity::Active { .. })
        ));
    }

    #[test]
    fn legal_watermark_reactivation_precedes_aggregation() {
        let mut progress = MultiInputProgress::new([BindingOrdinal::new(0)]);
        kinds(&mut progress, 0, AggregateInput::Idle);
        assert_eq!(
            kinds(&mut progress, 0, AggregateInput::Watermark(wm(5))),
            [ProgressEmissionKind::Watermark(wm(5))]
        );
        assert!(!progress.idle_latched());
    }

    #[test]
    fn reactivation_starts_a_new_idle_epoch() {
        let mut progress = MultiInputProgress::new([BindingOrdinal::new(0)]);
        kinds(&mut progress, 0, AggregateInput::Idle);
        kinds(&mut progress, 0, AggregateInput::Data);
        assert_eq!(
            kinds(&mut progress, 0, AggregateInput::Idle),
            [ProgressEmissionKind::Idle]
        );
        assert_eq!(progress.idle_epoch().0, 2);
    }

    #[test]
    fn reactivation_cannot_regress_aggregate_watermark() {
        let mut progress = MultiInputProgress::new([BindingOrdinal::new(0)]);
        kinds(&mut progress, 0, AggregateInput::Watermark(wm(10)));
        kinds(&mut progress, 0, AggregateInput::Idle);
        assert!(kinds(&mut progress, 0, AggregateInput::Data).is_empty());
        assert_eq!(progress.last_emitted_watermark(), Some(wm(10)));
    }

    #[test]
    fn final_watermark_advancement_precedes_end() {
        let mut progress =
            MultiInputProgress::new([BindingOrdinal::new(0), BindingOrdinal::new(1)]);
        kinds(&mut progress, 0, AggregateInput::Watermark(wm(2)));
        kinds(&mut progress, 1, AggregateInput::Watermark(wm(9)));
        assert_eq!(
            kinds(&mut progress, 0, AggregateInput::End),
            [ProgressEmissionKind::Watermark(wm(9))]
        );
        assert_eq!(
            kinds(&mut progress, 1, AggregateInput::End),
            [ProgressEmissionKind::EndOfInput]
        );
    }

    #[test]
    fn all_ended_emits_no_idle_or_sentinel_watermark() {
        let mut progress = MultiInputProgress::new([BindingOrdinal::new(0)]);
        assert_eq!(
            kinds(&mut progress, 0, AggregateInput::End),
            [ProgressEmissionKind::EndOfInput]
        );
        assert!(progress.terminal());
    }

    #[test]
    fn idle_epoch_exhaustion_aborts_without_mutation() {
        let mut progress = MultiInputProgress::new([BindingOrdinal::new(0)]);
        progress.set_next_idle_epoch_for_test(u64::MAX);
        assert!(
            progress
                .evaluate(BindingOrdinal::new(0), AggregateInput::Idle)
                .is_err()
        );
        assert!(!progress.idle_latched());
    }

    #[test]
    fn global_sequence_exhaustion_aborts_without_mutation() {
        let ordinal = BindingOrdinal::new(0);
        let mut progress = MultiInputProgress::new([ordinal]);
        progress.set_next_global_for_test(u64::MAX);
        assert!(
            progress
                .evaluate(ordinal, AggregateInput::Watermark(wm(5)))
                .is_err()
        );
        assert_eq!(
            progress.activity(ordinal),
            Some(super::IngressActivity::Active { watermark: None })
        );
        assert_eq!(progress.last_emitted_watermark(), None);
        assert_eq!(progress.next_global_sequence(), u64::MAX);
    }
}
