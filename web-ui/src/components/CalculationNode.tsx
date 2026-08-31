import { Handle, Position, type Node, type NodeProps } from '@xyflow/react';

import type { NodeConfig } from '../types';

export interface FlowNodeData extends Record<string, unknown> {
  label: string;
  kind: NodeConfig['operator']['kind'];
  inputPorts: string[];
  outputPorts: string[];
}

export type CalculationFlowNode = Node<FlowNodeData, 'calculation'>;

const nodeKindLabel = (kind: FlowNodeData['kind']): string => {
  switch (kind) {
    case 'expression': return 'ƒx';
    case 'sql': return 'SQL';
    case 'rolling': return 'ROLL';
    case 'cross_section': return 'CS';
    case 'stream_join': return 'JOIN';
    case 'window': return 'WIN';
    case 'union': return 'UNION';
    default: return 'EXT';
  }
};

export function CalculationNode({ data, selected }: NodeProps<CalculationFlowNode>) {
  return (
    <div className={`calculation-node ${data.kind} ${selected ? 'selected' : ''}`}>
      {data.inputPorts.map((port, index) => (
        <Handle
          id={port}
          key={port}
          type="target"
          position={Position.Left}
          style={{ top: `${((index + 1) / (data.inputPorts.length + 1)) * 100}%` }}
        />
      ))}
      <span className="node-kind">
        {nodeKindLabel(data.kind)}
      </span>
      <strong>{data.label}</strong>
      <small>{data.inputPorts.join(' · ')} → {data.outputPorts.join(' · ')}</small>
      {data.outputPorts.map((port, index) => (
        <Handle
          id={port}
          key={port}
          type="source"
          position={Position.Right}
          style={{ top: `${((index + 1) / (data.outputPorts.length + 1)) * 100}%` }}
        />
      ))}
    </div>
  );
}
