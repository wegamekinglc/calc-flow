import { Handle, Position, type Node, type NodeProps } from '@xyflow/react';

import type { NodeConfig } from '../types';

export interface FlowNodeData extends Record<string, unknown> {
  label: string;
  kind: NodeConfig['kind'];
  inputPorts: string[];
  outputPorts: string[];
}

export type CalculationFlowNode = Node<FlowNodeData, 'calculation'>;

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
      <span className="node-kind">{data.kind === 'expression' ? 'ƒx' : data.kind === 'sql' ? 'SQL' : '[ ]'}</span>
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
