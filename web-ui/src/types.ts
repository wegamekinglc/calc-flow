import type { components } from './api/schema';

export type ProjectConfig = components['schemas']['ProjectConfig-Input'];
export type ProjectCreateRequest = components['schemas']['ProjectCreateRequest'];
export type EditableProject = ProjectConfig | ProjectCreateRequest;
export type ProjectSummary = components['schemas']['ProjectSummary'];
export type NodeConfig = components['schemas']['NodeConfig'];
export type EdgeConfig = components['schemas']['EdgeConfig'];
export type PortConfig = components['schemas']['PortConfig'];
export type ArrowFieldConfig = components['schemas']['ArrowFieldConfig'];
export type CatalogResponse = components['schemas']['CatalogResponse'];
export type CheckpointSummary = components['schemas']['CheckpointSummary'];
export type RunRequest = components['schemas']['RunRequest'];
export type RunResponse = components['schemas']['RunResponse'];
export type ValidationReport = components['schemas']['ValidationReport'];
export type JSONValue = components['schemas']['JSONValue-Input'];

export interface OutputPreview {
  kind: 'table' | 'array';
  total_rows: number;
  truncated?: boolean;
  schema?: ArrowFieldConfig[];
  rows?: Record<string, JSONValue>[];
  data?: JSONValue;
}

export interface NodeTimingPreview {
  duration_ns: number;
  input_rows: Record<string, number>;
  output_rows: Record<string, number>;
}

export interface DataFusionMetricPreview {
  node_id: string | null;
  planning_ns: number;
  execution_ns: number;
  output_rows: number;
  logical_plan: string;
  physical_plan: string;
}

export interface RunResultPreview {
  outputs: Record<string, OutputPreview>;
  warnings: string[];
  node_timings: Record<string, NodeTimingPreview>;
  datafusion_metrics: DataFusionMetricPreview[];
  metadata: Record<string, JSONValue>;
}

export const blankProject = (): ProjectCreateRequest => ({
  format_version: '1',
  name: 'Untitled flow',
  description: '',
  pipeline: {
    id: 'main',
    name: 'Main pipeline',
    nodes: [
      {
        id: 'calculate',
        kind: 'expression',
        expression: 'total = a + b',
        select: [],
        filter_expression: null,
        query: null,
        inputs: [],
        backend: null,
        udfs: [],
        input_ports: [],
        output_ports: [],
        position: { x: 80, y: 100 },
      },
    ],
    edges: [],
    datafusion: {
      batch_size: 8192,
      target_partitions: 1,
      repartition_aggregations: true,
      repartition_joins: true,
      repartition_sorts: true,
      repartition_windows: true,
    },
  },
  data_sources: [],
  run_options: {
    max_input_bytes: 10 * 1024 * 1024,
    max_rows: 100_000,
    timeout_seconds: 30,
    memory_limit_mb: 512,
    output_rows: 1000,
  },
});
