import type { components } from './api/schema';

type GeneratedProjectCreateRequest = components['schemas']['ProjectCreateRequest'];
type GeneratedProjectDocument = components['schemas']['ProjectDocument'];

export type ProjectCreateRequest = Omit<GeneratedProjectCreateRequest, '$defs'>;
export type ProjectDocument = Omit<GeneratedProjectDocument, '$defs'>;
export type ProjectConfig = ProjectDocument;
export type EditableProject = ProjectDocument;
export type DataSourceSpec = GeneratedProjectDocument['$defs']['DataSourceSpec'];
export type ProjectSummary = components['schemas']['ProjectSummary'];
export type NodeConfig = GeneratedProjectDocument['$defs']['NodeSpec'];
export type OperatorSpec = GeneratedProjectDocument['$defs']['OperatorSpec'];
export type EdgeConfig = GeneratedProjectDocument['$defs']['EdgeSpec'];
export type PortConfig = GeneratedProjectDocument['$defs']['PortSpec'];
export type ArrowFieldConfig = GeneratedProjectDocument['$defs']['ArrowFieldSpec'];
export type UdfReference = GeneratedProjectDocument['$defs']['UdfReference'];
export type CheckpointSummary = components['schemas']['CheckpointSummary'];
export type RunRequest = components['schemas']['RunRequest'];
export type RunResponse = components['schemas']['RunResponse'];
export type JSONValue = components['schemas']['calc_flow_studio__models__JSONValue-Input'];

export interface UdfCatalogEntry {
  provider: string;
  name: string;
  version: string;
  kind: 'data_fusion_scalar';
  signature: {
    input_types: string[];
    return_type: string;
  };
  volatility: string;
}

export type CatalogResponse = UdfCatalogEntry[];

export interface ValidationIssue {
  path: string;
  code: string;
  message: string;
}

export interface ValidationReport {
  valid: boolean;
  issues: ValidationIssue[];
  fingerprint: string | null;
}

export interface OutputFieldPreview {
  name: string;
  type: string;
  nullable: boolean;
}

export interface OutputPreview {
  kind: 'table' | 'array';
  total_rows: number;
  truncated?: boolean;
  schema?: OutputFieldPreview[];
  rows?: Record<string, JSONValue>[];
  data?: JSONValue;
}

export interface NodeTimingPreview {
  duration_ns: number;
  input_rows: Record<string, number>;
  output_rows: Record<string, number>;
}

export interface DataFusionMetricPreview {
  query_id: number;
  node_id: string | null;
  planning_ns: number;
  execution_ns: number;
  output_rows: number;
  logical_plan: string;
  physical_plan: string;
}

export interface RunResultPreview {
  outputs: Record<string, OutputPreview>;
  node_timings: Record<string, NodeTimingPreview>;
  datafusion_metrics: DataFusionMetricPreview[];
  metadata: Record<string, JSONValue>;
}

export const blankProject = (): ProjectCreateRequest => ({
  format_version: 2,
  id: `project_${crypto.randomUUID().replaceAll('-', '')}`,
  name: 'Untitled flow',
  description: '',
  pipeline: {
    name: 'Main pipeline',
    nodes: [
      {
        id: 'calculate',
        operator: {
          kind: 'expression',
          expression: 'total = a + b',
          select: [],
          filter: null,
          udfs: [],
        },
        input_ports: [],
        output_ports: [],
        position: { x: 80, y: 100 },
      },
    ],
    edges: [],
    datafusion: { batch_size: 8192, target_partitions: 1 },
  },
  data_sources: [
    {
      id: 'sample',
      input: 'input',
      format: 'inline_json',
      data: [
        { a: 1, b: 2 },
        { a: 3, b: 4 },
      ],
    },
  ],
  run_options: {
    max_input_bytes: 10 * 1024 * 1024,
    max_rows: 100_000,
    timeout_seconds: 30,
    memory_limit_mb: 512,
    output_rows: 1000,
  },
});

export const updateNodeOperator = (
  project: EditableProject,
  nodeId: string,
  update: (operator: OperatorSpec) => OperatorSpec,
): EditableProject => ({
  ...project,
  pipeline: {
    ...project.pipeline,
    nodes: project.pipeline.nodes.map((node) =>
      node.id === nodeId
        ? { ...node, operator: update(node.operator) }
        : node,
    ),
  },
});
