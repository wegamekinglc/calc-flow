import type { ArrowFieldConfig, NodeConfig, ProjectDocument } from './types';

const diagnosticTypes: readonly (readonly [string, string])[] = [
  ['node', 'string'], ['input_port', 'string'],
  ['event_time_micros', 'int64'], ['closing_time_micros', 'int64'],
  ['watermark_micros', 'int64'], ['reason', 'string'], ['source', 'string'],
  ['sequence', 'uint64'], ['row_index', 'uint64'],
];

export const lateOutputSchema = (node: NodeConfig): ArrowFieldConfig[] => [
  ...(node.input_ports[0]?.schema ?? []),
  ...diagnosticTypes.map(([name, data_type]) => ({
    name: `_cf_late_${name}`, data_type, nullable: false,
  })),
];

export const hasLateOutput = (node: NodeConfig): boolean =>
  (node.operator.kind === 'rolling' || node.operator.kind === 'cross_section')
  && node.operator.spec.late_policy.kind === 'side_output';

export const derivedInputNames = (node: NodeConfig): string[] => {
  if (node.input_ports.length) return node.input_ports.map((port) => port.name);
  if (node.operator.kind === 'sql') return node.operator.aliases;
  if (node.operator.kind === 'expression') return ['input'];
  return [];
};

export const derivedOutputNames = (node: NodeConfig): string[] => {
  if (node.output_ports.length) return node.output_ports.map((port) => port.name);
  if (node.operator.kind === 'external') return [];
  if (hasLateOutput(node)) return ['output', 'late'];
  return ['output'];
};

export const externalOutputs = (graph: ProjectDocument['graph']) => {
  const connected = new Set(graph.edges.map((edge) => `${edge.source_node}.${edge.source_port}`));
  const outputs = graph.nodes.flatMap((node) => derivedOutputNames(node)
    .filter((port) => !connected.has(`${node.id}.${port}`))
    .map((port) => ({ nodeId: node.id, port })));
  const counts = new Map<string, number>();
  for (const { port } of outputs) counts.set(port, (counts.get(port) ?? 0) + 1);
  return outputs.map((output) => ({
    ...output,
    binding: counts.get(output.port) === 1 ? output.port : `${output.nodeId}.${output.port}`,
  }));
};

interface OutputIdentity {
  nodeId: string;
  port: string;
}

const qualifiedOutput = (output: OutputIdentity): string => `${output.nodeId}.${output.port}`;

export const withProjectGraph = (
  project: ProjectDocument,
  graph: ProjectDocument['graph'],
): ProjectDocument => {
  const previous = new Map(externalOutputs(project.graph).map((output) => [output.binding, output]));
  const current = new Map(externalOutputs(graph).map((output) => [qualifiedOutput(output), output.binding]));
  return {
    ...project,
    graph,
    sinks: project.sinks.map((sink) => {
      const output = previous.get(sink.binding);
      const identity = output ? qualifiedOutput(output) : sink.binding;
      // Retain disconnected consumers for explicit rebinding instead of assigning a different route.
      const binding = current.get(identity) ?? identity;
      return binding === sink.binding ? sink : { ...sink, binding };
    }),
  };
};

export const lateOutputInUse = (project: ProjectDocument, node: NodeConfig): boolean => {
  if (project.graph.edges.some((edge) => edge.source_node === node.id && edge.source_port === 'late')) return true;
  const output = externalOutputs(project.graph).find((item) => item.nodeId === node.id && item.port === 'late');
  return project.sinks.some((sink) =>
    sink.binding === output?.binding || sink.binding === `${node.id}.late`);
};
