# Kafka connector

[Documentation](../README.md) / [Connectors](README.md) / Kafka

Consume assigned partitions with exact offset positioning. The sink supports
Kafka transactions with a dedicated compacted epoch ledger.

Identity: `calc-flow-connectors/kafka/2.0.0`.

Use [16_kafka_source.py](../../examples/16_kafka_source.py) to read orders
into Parquet, or [22_kafka_sink.py](../../examples/22_kafka_sink.py) to write
calculated totals to Kafka. Both compose `pipe(order_totals)` with Python
operators and export a stream project before adding connector bindings.

## Python source example

Run [16_kafka_source.py](../../examples/16_kafka_source.py).
`build_project` assigns partition `0` of `calc-flow-example-orders` and
selects the JSON codec. `run` waits for two delivered rows before draining
the continuous job, then verifies Parquet totals `[20.0, 60.0]` and prints
at-least-once delivery.

Both the command-line entry point and imported functions default to
`127.0.0.1:9092`. Set `CALC_FLOW_KAFKA_BOOTSTRAP` to select another broker.
The broker and sample topic must be prepared even when using the default.

Build a Python wheel with `--features connector-kafka`, retaining the
default file connector for local input/output. Follow the
[shared environment preparation](README.md#prepare-the-python-environment)
before the service commands below.

Start a disposable single-broker KRaft service. The listener configuration
matches the examples' default bootstrap address; the replication settings
allow Kafka's transaction and consumer-offset topics on this one broker:

```bash
docker run --rm -d --name calc-flow-example-kafka \
  -p 127.0.0.1:9092:9092 \
  -e KAFKA_NODE_ID=1 \
  -e KAFKA_PROCESS_ROLES=broker,controller \
  -e KAFKA_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://127.0.0.1:9092 \
  -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT \
  -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER \
  -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 \
  -e KAFKA_INTER_BROKER_LISTENER_NAME=PLAINTEXT \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  -e KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=1 \
  -e KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=1 \
  apache/kafka:3.9.0
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
demo topic. The graph and JSON codec share an explicit `id`, `quantity`,
`price` schema; the calculation casts quantity to `float64` before multiplying.
The transactional write example below uses separate target and ledger topics.

## Python sink example

Run [22_kafka_sink.py](../../examples/22_kafka_sink.py) after starting the
broker above. It generates three local Parquet rows, composes order totals,
filters the zero-quantity third order, and writes two rows to Kafka. It does
not depend on the source example or its orders topic.

Create an empty destination topic and a dedicated, single-partition ledger
whose only cleanup policy is `compact`. Then run and read the committed output:

```bash
docker exec calc-flow-example-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic calc-flow-example-totals --partitions 1 --replication-factor 1
docker exec calc-flow-example-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --topic calc-flow-example-totals-ledger --partitions 1 --replication-factor 1 --config cleanup.policy=compact
export CALC_FLOW_KAFKA_BOOTSTRAP=127.0.0.1:9092
uv run --no-sync python examples/22_kafka_sink.py
docker exec calc-flow-example-kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic calc-flow-example-totals --from-beginning --max-messages 1 --timeout-ms 10000 --consumer-property isolation.level=read_committed
```

The sink encodes the two rows as one Kafka message containing two JSON Lines
objects. Expect `(id, total) = (1, 20.0), (2, 60.0)`; JSON field order is not
significant. `--max-messages 1` counts Kafka messages, not JSON rows.
`read_committed` reads the committed transaction. The Python script checks
two delivered rows and `exactly_once` effective delivery after ledger preflight.

The script leaves topic data intact and removes its local inputs/checkpoints.
Each invocation uses a new temporary lineage and pipeline identity, so running
it again appends another message; it does not resume the earlier job. Use an
empty destination for this readback check, or recreate the disposable broker
and topics for a fresh demo. Keep a stable state root and sink identity for
[durable recovery](README.md#recovery-ownership); retain the ledger for that
lineage and do not share it with unrelated sinks.

## Clean up

After the examples complete, stop the disposable demo service:

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
