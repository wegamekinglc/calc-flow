import type { NodeConfig, OperatorSpec } from '../types';

type StatefulOperator = Extract<OperatorSpec, { kind: 'rolling' | 'cross_section' }>;
type Policy = StatefulOperator['spec']['late_policy'];

interface Props {
  node: NodeConfig;
  operator: StatefulOperator;
  streamMode: boolean;
  supported: boolean;
  inUse: boolean;
  onChange: (node: NodeConfig) => void;
}

const policyFor = (kind: string): Policy => {
  if (kind === 'side_output') return { kind, metrics_version: 1, schema_version: 1 };
  if (kind === 'drop') return { kind, metrics_version: 1 };
  return { kind: 'error', scope: 'envelope' };
};

const withPolicy = (operator: StatefulOperator, late_policy: Policy): StatefulOperator =>
  operator.kind === 'rolling'
    ? { kind: 'rolling', spec: { ...operator.spec, late_policy } }
    : { kind: 'cross_section', spec: { ...operator.spec, late_policy } };

const diagnosticTypes: readonly (readonly [string, string])[] = [
  ['node', 'string'], ['input_port', 'string'],
  ['event_time_micros', 'int64'], ['closing_time_micros', 'int64'],
  ['watermark_micros', 'int64'], ['reason', 'string'], ['source', 'string'],
  ['sequence', 'uint64'], ['row_index', 'uint64'],
];

const diagnosticFields = () => diagnosticTypes.map(([name, data_type]) => ({
  name: `_cf_late_${name}`, data_type, nullable: false,
}));

const policyPorts = (node: NodeConfig, kind: string) => {
  if (node.output_ports.length === 0) return node.output_ports;
  const normal = node.output_ports.filter((port) => port.name !== 'late');
  if (kind !== 'side_output') return normal;
  return [...normal, {
    name: 'late', kind: 'table' as const, required: true,
    schema: [...(node.input_ports[0]?.schema ?? []), ...diagnosticFields()],
  }];
};

export function LatePolicyEditor({ node, operator, streamMode, supported, inUse, onChange }: Props) {
  const policy = operator.spec.late_policy;
  const bound = policy.kind === 'side_output' && inUse;
  return <section className="inspector-section">
    <label>
      Late row policy
      <select value={policy.kind} disabled={bound} onChange={(event) => {
        const kind = event.target.value;
        if (bound || (kind === 'side_output' && (!streamMode || !supported))) return;
        onChange({ ...node, output_ports: policyPorts(node, kind),
          operator: withPolicy(operator, policyFor(kind)),
        });
      }}>
        <option value="error">Error</option>
        <option value="drop">Drop</option>
        <option value="side_output" disabled={!streamMode || !supported}>Side output</option>
      </select>
    </label>
    {bound && <p>Disconnect late routes and remove its sink binding before changing policy.</p>}
    {!supported && <p>The runtime has not confirmed late output support; validate before starting a job.</p>}
    {policy.kind === 'side_output' && <>
      <h3>Late side output</h3>
      <p>late has no event-time ordering and no Watermark/Idle; Barrier/EOF still pass.</p>
      <p>Local boundary: {operator.kind === 'rolling' ? 't' : 'group closing time'} + {operator.spec.allowed_lateness_micros}µs ≤ W.</p>
      <p>64-bit diagnostic times and sequence values stay in Arrow; no row JSON preview.</p>
      <p>late_rows counts excluded normal rows. Delivery counts come from sink/edge metrics.</p>
    </>}
  </section>;
}
