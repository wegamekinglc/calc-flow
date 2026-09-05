import type { components } from './api/schema';

type GeneratedProjectCreateRequest = components['schemas']['ProjectCreateRequest'];
type GeneratedProjectDocument = components['schemas']['ProjectDocument'];

export type ProjectCreateRequest = Omit<GeneratedProjectCreateRequest, '$defs'>;
export type ProjectDocument = Omit<GeneratedProjectDocument, '$defs'>;
export type EditableProject = ProjectDocument;
export type DataSourceSpec = GeneratedProjectDocument['$defs']['DataSourceSpec'];
export type ProjectSummary = components['schemas']['ProjectSummary'];
export type NodeConfig = GeneratedProjectDocument['$defs']['NodeSpec'];
export type OperatorSpec = GeneratedProjectDocument['$defs']['OperatorSpec'];
export type EdgeConfig = GeneratedProjectDocument['$defs']['EdgeSpec'];
export type PortConfig = GeneratedProjectDocument['$defs']['PortSpec'];
export type ArrowFieldConfig = GeneratedProjectDocument['$defs']['ArrowFieldSpec'];
export type UdfReference = GeneratedProjectDocument['$defs']['UdfReference'];
export type ProjectSourceBinding = GeneratedProjectDocument['$defs']['ProjectSourceBinding'];
export type ProjectSinkBinding = GeneratedProjectDocument['$defs']['ProjectSinkBinding'];
export type ConnectorCapability = components['schemas']['ConnectorCapabilityResponse'];
export type JobResponse = components['schemas']['JobResponse'];
export type ValidationIssue = components['schemas']['ValidationIssue'];
export type ValidationReport = components['schemas']['ValidationReport'];
export type CapabilitiesResponse = components['schemas']['CapabilitiesResponse'];
export type JSONValue = components['schemas']['JSONValue-Input'];

export interface StreamJoinSideMetrics {
  retained_rows: number;
  retained_bytes: number;
  evicted_rows: number;
  late_rows: number;
  late_affected_batches: number;
  max_lateness_micros: number | null;
  null_event_time_rows: number;
  null_key_rows: number;
}

export interface StreamJoinMetrics {
  node_id: string;
  left: StreamJoinSideMetrics;
  right: StreamJoinSideMetrics;
  emitted_match_rows: number;
  state_limit_failures: number;
  match_limit_failures: number;
}

export interface JobEvent {
  sequence: number;
  timestamp: string;
  type: 'state' | 'progress' | 'checkpoint' | 'terminal';
  message: string;
  state?: string;
  epoch?: number;
  watermark?: string;
  throughput_rows?: number;
  queue_envelopes?: number;
  queue_rows?: number;
  queue_bytes?: number;
  backpressure_events?: number;
  late_rows?: number;
  stream_joins?: StreamJoinMetrics[] | null;
}

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

export const blankProject = (): ProjectCreateRequest => ({
  format_version: 3,
  id: `project_${crypto.randomUUID().replaceAll('-', '')}`,
  name: 'Untitled flow',
  description: '',
  runtime: {
    mode: 'batch',
    options: {
      max_input_bytes: 10 * 1024 * 1024,
      max_rows: 100_000,
      timeout_seconds: 30,
      memory_limit_mb: 512,
      output_rows: 1000,
    },
  },
  graph: {
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
  sources: [],
  sinks: [],
  state: { root: '.calc-flow-state', retention: 3 },
});

/** A fixture accessor that fails loudly instead of asserting presence. */
export const at = <T>(items: readonly T[], index = 0): T => {
  const item = items.at(index);
  if (item === undefined) {
    throw new Error(`fixture item ${index} is missing`);
  }
  return item;
};

/** The first element of a sequence, explicitly undefined when empty. */
export const firstOf = <T>(items: readonly T[]): T | undefined => items.at(0);
