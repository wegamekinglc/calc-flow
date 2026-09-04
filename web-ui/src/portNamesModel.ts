import type { NodeConfig } from './types';

export const derivedInputNames = (node: NodeConfig): string[] => {
  if (node.input_ports.length) return node.input_ports.map((port) => port.name);
  if (node.operator.kind === 'sql') return node.operator.aliases;
  if (node.operator.kind === 'expression') return ['input'];
  return [];
};

export const derivedOutputNames = (node: NodeConfig): string[] => {
  if (node.output_ports.length) return node.output_ports.map((port) => port.name);
  if (node.operator.kind === 'external') return [];
  return ['output'];
};
