use super::super::pairs::PairOutput;
use super::{
    Arc, Counts, GatherStop, MemoryReservation, ParallelCpuWork, PreparedJoinBatch, ProbeInputs,
    Result, StreamJoinOperator, StreamOperatorContext, worker,
};
use parking_lot::Mutex;

struct FallbackWork {
    inputs: Arc<ProbeInputs>,
    count: usize,
    credit: Mutex<Option<MemoryReservation>>,
}

impl ParallelCpuWork for FallbackWork {
    type Output = PairOutput;

    fn unit_count(&self) -> usize {
        1
    }

    fn run(&self, _: usize, stop: &GatherStop) -> Result<PairOutput> {
        let credit = self
            .credit
            .lock()
            .take()
            .expect("prepaid serial pair lease");
        let pairs = worker::fill_all(&self.inputs, self.count, stop)?;
        Ok(PairOutput { pairs, credit })
    }
}

impl StreamJoinOperator {
    pub(super) async fn fallback_morsels(
        &mut self,
        inputs: &Arc<ProbeInputs>,
        counts: &Counts,
        credit: MemoryReservation,
        prepared: &mut PreparedJoinBatch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let work = Arc::new(FallbackWork {
            inputs: Arc::clone(inputs),
            count: counts.total,
            credit: Mutex::new(Some(credit)),
        });
        let output = self
            .run_probe_units(
                Arc::clone(&work),
                context,
                |mut values| Ok(values.pop().expect("single serial fill result")),
                #[cfg(test)]
                false,
            )
            .await?;
        // Only a refusal of native-worker control admission uses the inherited
        // inline fallback. Pair storage itself needs no second grant.
        let output = match output {
            Some(output) => output,
            None => work.run(0, &GatherStop::from_job(context.job()))?,
        };
        let PreparedJoinBatch {
            output: pairs,
            _native_scratch: scratch,
            ..
        } = prepared;
        *pairs = output.pairs;
        *scratch = Some(output.credit);
        Ok(())
    }
}
