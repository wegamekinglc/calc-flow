struct ContinuousJobSpec;
struct OneShotStartObserver;

struct OneShotContinuousRunner;

impl OneShotContinuousRunner {
    fn start(self, _spec: ContinuousJobSpec) -> OneShotStartObserver {
        OneShotStartObserver
    }
}

fn main() {
    let runner = OneShotContinuousRunner;
    let _first = runner.start(ContinuousJobSpec);
    let _second = runner.start(ContinuousJobSpec);
}
