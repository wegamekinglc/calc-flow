import type { NodeConfig, ProjectDocument } from './types';

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

export const lateOutputInUse = (project: ProjectDocument, node: NodeConfig): boolean => {
  if (project.graph.edges.some((edge) => edge.source_node === node.id && edge.source_port === 'late')) return true;
  const output = externalOutputs(project.graph).find((item) => item.nodeId === node.id && item.port === 'late');
  return project.sinks.some((sink) => sink.binding === output?.binding);
};
