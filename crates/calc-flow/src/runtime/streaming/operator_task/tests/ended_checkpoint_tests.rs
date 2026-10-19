use super::super::{OperatorCheckpointCommand, OperatorTerminalPort};
use super::*;
use crate::Epoch;

type CheckpointHarness = (
    Harness,
    mpsc::Receiver<OperatorCheckpointAck>,
    mpsc::Sender<OperatorCheckpointCommand>,
    mpsc::Receiver<String>,
);

fn checkpoint_harness() -> CheckpointHarness {
    let (acks, ack_rx) = mpsc::channel(4);
    let (ready, ready_rx) = mpsc::channel(4);
    let (commands, command_rx) = mpsc::channel(4);
    let output = Port::new("output", BatchKind::Table, false, None).unwrap();
    let probe = ProbeOperator {
        input_ports: vec![Port::new("input", BatchKind::Table, true, None).unwrap()],
        output_ports: vec![output.clone()],
        behavior: Behavior::Forward,
        watermarks: Arc::new(AtomicUsize::new(0)),
        ends: Arc::new(AtomicUsize::new(0)),
        observed: Arc::new(Mutex::new(Vec::new())),
    };
    let operator = harness_with_operator(
        &["input"],
        1,
        CompiledStreamOperator::External(Box::new(probe)),
        output,
        Some(OperatorCheckpointPort {
            acks,
            transaction: None,
            terminal: Some(OperatorTerminalPort {
                ready,
                commands: command_rx,
            }),
            alignment_fault: None,
        }),
        None,
    );
    (operator, ack_rx, commands, ready_rx)
}

#[tokio::test]
async fn ended_operator_preserves_inflight_ack_then_captures_each_later_epoch() {
    let (mut harness, mut acks, commands, mut ready) = checkpoint_harness();
    start(&mut harness).await;
    harness
        .inputs
        .get_mut("input")
        .unwrap()
        .send(StreamMessage::barrier(Epoch::INITIAL))
        .await
        .unwrap();
    assert_eq!(
        harness.outputs[0].recv().await.unwrap().unwrap().kind(),
        StreamMessageKind::Barrier
    );
    // Leave the original ACK in flight while readiness overtakes it.
    harness
        .inputs
        .get_mut("input")
        .unwrap()
        .send(StreamMessage::end_of_input())
        .await
        .unwrap();
    assert_eq!(
        harness.outputs[0].recv().await.unwrap().unwrap().kind(),
        StreamMessageKind::EndOfInput
    );
    assert_eq!(ready.recv().await.as_deref(), Some("node"));
    commands
        .send(OperatorCheckpointCommand::Capture(Epoch::INITIAL))
        .await
        .unwrap();
    let next = Epoch::INITIAL.next().unwrap();
    commands
        .send(OperatorCheckpointCommand::Capture(next))
        .await
        .unwrap();
    let original = acks.recv().await.unwrap();
    assert_eq!(original.epoch, Epoch::INITIAL);
    assert_eq!(
        original.state.progress["input"].state,
        ManifestIngressState::Active
    );
    let captured = acks.recv().await.unwrap();
    assert_eq!(captured.epoch, next);
    assert_eq!(
        captured.state.progress["input"].state,
        ManifestIngressState::Ended
    );
    assert!(acks.try_recv().is_err());
    assert_eq!(harness.supervisor.task_count(), 1);
    commands
        .send(OperatorCheckpointCommand::Terminal(next.next().unwrap()))
        .await
        .unwrap();
    assert_eq!(acks.recv().await.unwrap().epoch, next.next().unwrap());
    assert!(
        harness
            .supervisor
            .join_all()
            .await
            .primary_errors()
            .is_empty()
    );
}

#[tokio::test]
async fn ended_operator_rejects_regressed_capture_without_acknowledging_it() {
    let (mut harness, mut acks, commands, mut ready) = checkpoint_harness();
    start(&mut harness).await;
    harness
        .inputs
        .get_mut("input")
        .unwrap()
        .send(StreamMessage::end_of_input())
        .await
        .unwrap();
    harness.outputs[0].recv().await.unwrap().unwrap();
    ready.recv().await.unwrap();
    let next = Epoch::INITIAL.next().unwrap();
    commands
        .send(OperatorCheckpointCommand::Capture(next))
        .await
        .unwrap();
    assert_eq!(acks.recv().await.unwrap().epoch, next);
    commands
        .send(OperatorCheckpointCommand::Capture(Epoch::INITIAL))
        .await
        .unwrap();
    let report = harness.supervisor.join_all().await;
    assert!(matches!(
        &report.primary_errors()[0].error,
        CalcFlowError::CheckpointMismatch { .. }
    ));
    assert!(acks.try_recv().is_err());
}
