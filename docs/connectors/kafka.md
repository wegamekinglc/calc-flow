# Kafka connector

[Documentation](../README.md) / [Connectors](README.md) / Kafka

Consume assigned partitions with exact offset positioning. The sink supports
Kafka transactions with a dedicated compacted epoch ledger.

Identity: `calc-flow-connectors/kafka/2.0.0`.

## Python example

Run [16_kafka_source.py](../../examples/16_kafka_source.py).
`build_project` assigns partition `0` of `calc-flow-example-orders` and
selects the JSON codec. `run` waits for two delivered rows before draining
the continuous job, then verifies Parquet totals `[20.0, 60.0]` and prints
at-least-once delivery.

Both the command-line entry point and imported functions default to
`127.0.0.1:9092`. Set `CALC_FLOW_KAFKA_BOOTSTRAP` to select another broker.
The broker and sample topic must be prepared even when using the default.

Build a Python wheel with `--features connector-kafka`, retaining the
default file connector for output. Follow the
[shared environment preparation](README.md#prepare-the-python-environment)
before the service commands below.

Start a local broker, then create a dedicated one-partition topic:

```bash
docker run --rm -d --name calc-flow-example-kafka -p 127.0.0.1:9092:9092 apache/kafka:3.9.0
```

Wait for broker readiness in `docker logs calc-flow-example-kafka`, then
prepare the topic and run the Python consumer:

```bash
docker exec calc-flow-example-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic calc-flow-example-orders --partitions 1 --replication-factor 1
docker exec -i calc-flow-example-kafka /opt/kafka/bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic calc-flow-example-orders < examples/data/orders.jsonl
export CALC_FLOW_KAFKA_BOOTSTRAP=127.0.0.1:9092
uv run --no-sync python examples/16_kafka_source.py
```

The source assigns partition `0`, starts at `earliest`, and decodes one JSON
object per message. The example requests at-least-once delivery. Run the
producer once; rerunning only the consumer uses a fresh checkpoint directory
and reads the same retained records. Do not add unrelated messages to this
demo topic. A transactional Kafka sink requires its own ledger setup; see
the [Kafka contract](#project-configuration).

After the example completes, stop the disposable demo service:

```bash
docker stop calc-flow-example-kafka
```

The `--rm` option removes its container data on stop; the setup above can
recreate it.

## Project configuration

These bindings also apply to graphs exported from Python expressions; follow
[expression-to-connector integration](README.md#connect-python-expressions-to-transports)
to supply physical graph bindings and explicit runtime/state settings.

```json
{
  "sources": [{
    "binding": "input",
    "connector": {
      "provider": "calc-flow-connectors",
      "name": "kafka",
      "version": "2.0.0"
    },
    "format": {"name": "json", "version": "1"},
    "options": {
      "bootstrap_servers": "127.0.0.1:9092",
      "topic": "orders",
      "partitions": [0, 1],
      "auto_offset_reset": "earliest",
      "format": "json"
    },
    "watermark": {
      "policy": "bounded_out_of_orderness",
      "column": "event_time",
      "delay_ms": 5000,
      "emit_interval_ms": 1000,
      "idle_timeout_ms": 30000
    }
  }],
  "sinks": [{
    "binding": "output",
    "connector": {
      "provider": "calc-flow-connectors",
      "name": "kafka",
      "version": "2.0.0"
    },
    "format": {"name": "json", "version": "1"},
    "options": {
      "bootstrap_servers": "127.0.0.1:9092",
      "topic": "totals",
      "ledger_topic": "calc-flow-totals-ledger",
      "pipeline": "orders",
      "output": "output",
      "format": "json"
    },
    "delivery": "exactly_once"
  }]
}
```

The ledger topic must be dedicated, have exactly one partition, and use only
`cleanup.policy=compact`. Calc Flow derives the transactional ID from pipeline
and output identity; a project cannot supply it. Recovery validates the exact
prepared record bytes and checks the committed epoch marker before replay.

See the [connector overview](README.md) for shared delivery, secret,
and recovery rules.
