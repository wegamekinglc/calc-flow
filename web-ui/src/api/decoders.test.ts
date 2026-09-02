import { describe, expect, it } from 'vitest';

import {
  ApiContractError,
  decodeCapabilitiesResponse,
  decodeJobResponse,
} from './decoders';

const connectorFixture = () => ({
  provider: 'builtin',
  name: 'file',
  version: '1',
  kind: 'both',
  capabilities: {
    delivery: 'exactly_once',
    replay: 'replayable_exact',
    watermark: 'generated_only',
    transaction: 'pre_commit_commit',
    snapshot: true,
    polling: true,
    cdc: false,
    lookup: false,
  },
  formats: ['json'],
  optionsSchema: { path: { type: 'string' } },
});

const operatorFixture = () => ({
  kind: 'expression',
  version: '1',
  inputPorts: [{ name: 'input', kind: 'table', required: true }],
  outputPorts: [{ name: 'output', kind: 'table', required: true }],
  modes: ['batch', 'stream'],
  finality: 'per_row_final',
  requiresDatafusion: true,
  stateful: false,
  microbatchInvariant: true,
  requiresWatermark: false,
  checkpointSupport: 'stateless',
  stateVersion: null,
  deterministic: true,
  replaySafe: true,
  stateLayouts: [],
});

const providerFixture = () => ({
  provider: 'numpy',
  name: 'table_matmul',
  version: '1',
  inputPorts: [
    { name: 'table', kind: 'table', required: true },
    { name: 'weights', kind: 'array', required: true },
  ],
  outputPorts: [{ name: 'output', kind: 'array', required: true }],
  optionsSchema: null,
  modes: ['batch'],
  finality: 'unproven',
  stateful: false,
  microbatchInvariant: false,
  requiresWatermark: false,
  checkpointSupport: 'stateless',
  stateVersion: null,
  deterministic: false,
  replaySafe: false,
  supportsStaticInputs: false,
  partitionContract: 'none',
  arrayRules: null,
});

const capabilitiesFixture = () => ({
  schemaVersion: 3,
  runtime: {
    scope: { kind: 'runtimeSession', sessionId: 'session', revision: 0 },
    packageVersion: '4.0.0',
    projectFormatVersions: [3],
    batchKinds: ['table', 'array'],
    portableArrowTypes: ['int64'],
    operators: [operatorFixture()],
    udfs: [],
    providers: [providerFixture()],
    connectors: [connectorFixture()],
  },
  preview: {
    inputBatchKinds: ['table'],
    requestInputFormats: ['arrow_ipc', 'columns', 'records'],
    projectInputFormats: ['arrow_ipc', 'csv', 'inline_json', 'json'],
    workerRegistrations: [],
    limits: {
      maxInputBytes: { default: 10, minimum: 1, maximum: 10 },
      maxRows: { default: 10, minimum: 1, maximum: 10 },
      timeoutSeconds: { default: 30, minimum: 1, maximum: 300 },
      memoryLimitMb: { default: 512, minimum: 64, maximum: 4096 },
      outputRows: { default: 1000, minimum: 1, maximum: 10_000 },
    },
  },
});

const job = (status: string) => ({
  id: 'job-1',
  project_id: 'project-1',
  status,
  created_at: '2026-01-01T00:00:00Z',
  started_at: status === 'pending' ? null : '2026-01-01T00:00:01Z',
  finished_at: ['completed', 'failed', 'cancelled'].includes(status)
    ? '2026-01-01T00:00:02Z'
    : null,
  error_code: status === 'failed' ? 'worker_failed' : null,
  reason_code: null,
  error: status === 'failed' ? 'worker exited' : null,
});

describe('capabilities decoder', () => {
  it('accepts the closed runtime connector capability axes', () => {
    const document = capabilitiesFixture();

    expect(decodeCapabilitiesResponse(document)).toEqual(document);
  });

  it('rejects the retired schema version 1 envelope', () => {
    const document = capabilitiesFixture();

    expect(() => decodeCapabilitiesResponse({
      ...document,
      schemaVersion: 1,
    })).toThrowError(new ApiContractError(
      'capabilities schema version 1 is unsupported; expected 3',
    ));
  });

  it('rejects the retired schema version 2 envelope', () => {
    const document = capabilitiesFixture();

    expect(() => decodeCapabilitiesResponse({
      ...document,
      schemaVersion: 2,
    })).toThrowError(new ApiContractError(
      'capabilities schema version 2 is unsupported; expected 3',
    ));
  });

  it('accepts every durable rolling state layout', () => {
    const document = capabilitiesFixture();
    const rolling = {
      ...operatorFixture(),
      kind: 'rolling',
      requiresDatafusion: false,
      stateful: true,
      requiresWatermark: true,
      checkpointSupport: 'checkpointed_stateful',
      stateVersion: 1,
      stateLayouts: [1, 2],
    };

    expect(decodeCapabilitiesResponse({
      ...document,
      runtime: { ...document.runtime, operators: [rolling] },
    })).toEqual({
      ...document,
      runtime: { ...document.runtime, operators: [rolling] },
    });
  });

  it('rejects hostile operator state layouts', () => {
    const stateful = {
      ...operatorFixture(),
      kind: 'rolling',
      requiresDatafusion: false,
      stateful: true,
      requiresWatermark: true,
      checkpointSupport: 'checkpointed_stateful',
      stateVersion: 1,
      stateLayouts: [1],
    };
    const document = capabilitiesFixture();
    const withOperator = (operator: Record<string, unknown>) => ({
      ...document,
      runtime: { ...document.runtime, operators: [operator] },
    });

    expect(() => decodeCapabilitiesResponse(
      withOperator({ ...stateful, stateLayouts: [] }),
    )).toThrowError(/requires at least one state layout/);
    expect(() => decodeCapabilitiesResponse(
      withOperator({ ...stateful, stateLayouts: [2] }),
    )).toThrowError(/must contain stateVersion/);
    expect(() => decodeCapabilitiesResponse(
      withOperator({ ...stateful, stateLayouts: [2, 1] }),
    )).toThrowError(/strictly ascending/);
    expect(() => decodeCapabilitiesResponse(
      withOperator({ ...stateful, stateLayouts: [0, 1] }),
    )).toThrowError(/expected a positive state layout/);
    expect(() => decodeCapabilitiesResponse(
      withOperator({ ...operatorFixture(), stateLayouts: [1] }),
    )).toThrowError(/must be empty unless checkpointSupport is checkpointed_stateful/);
  });

  it('rejects the legacy v1 operator shape', () => {
    const document = capabilitiesFixture();

    expect(() => decodeCapabilitiesResponse({
      ...document,
      runtime: {
        ...document.runtime,
        operators: [{
          kind: 'expression',
          inputKinds: ['table'],
          outputKinds: ['table'],
          requiresDatafusion: true,
        }],
      },
    })).toThrowError(new ApiContractError(
      'capabilities.runtime.operators[0].inputKinds: extra fields are forbidden',
    ));
  });

  it('rejects an operator state version on a stateless checkpoint support', () => {
    const document = capabilitiesFixture();

    expect(() => decodeCapabilitiesResponse({
      ...document,
      runtime: {
        ...document.runtime,
        operators: [{ ...operatorFixture(), stateVersion: 1 }],
      },
    })).toThrowError(new ApiContractError(
      'capabilities.runtime.operators[0].stateVersion: must be null unless checkpointSupport is checkpointed_stateful',
    ));
  });

  it('rejects an unknown operator finality', () => {
    const document = capabilitiesFixture();

    expect(() => decodeCapabilitiesResponse({
      ...document,
      runtime: {
        ...document.runtime,
        operators: [{ ...operatorFixture(), finality: 'append_only' }],
      },
    })).toThrowError(new ApiContractError(
      "capabilities.runtime.operators[0].finality: expected 'per_row_final' or 'group_final_append_only' or 'unproven'",
    ));
  });

  it('rejects an unknown capability rule identity in the safe dtype rule', () => {
    const document = capabilitiesFixture();

    expect(() => decodeCapabilitiesResponse({
      ...document,
      runtime: {
        ...document.runtime,
        providers: [{
          ...providerFixture(),
          arrayRules: {
            supportedDtypes: ['float64'],
            safeDtypeRule: { name: 'unrestricted_matmul', version: '1' },
            shapeRules: [],
          },
        }],
      },
    })).toThrowError(new ApiContractError(
      'capabilities.runtime.providers[0].arrayRules.safeDtypeRule: unknown capability rule unrestricted_matmul@1',
    ));
  });

  it('rejects an unknown capability rule identity inside shape rules', () => {
    const document = capabilitiesFixture();

    expect(() => decodeCapabilitiesResponse({
      ...document,
      runtime: {
        ...document.runtime,
        providers: [{
          ...providerFixture(),
          arrayRules: {
            supportedDtypes: ['float64'],
            safeDtypeRule: { name: 'array_api_safe_dtype', version: '1' },
            shapeRules: [{ name: 'reduce_along_any_axis', version: '2' }],
          },
        }],
      },
    })).toThrowError(new ApiContractError(
      'capabilities.runtime.providers[0].arrayRules.shapeRules[0]: unknown capability rule reduce_along_any_axis@2',
    ));
  });

  it('rejects extra connector capability axes', () => {
    const document = capabilitiesFixture();
    const connector = connectorFixture();

    expect(() => decodeCapabilitiesResponse({
      ...document,
      runtime: {
        ...document.runtime,
        connectors: [{
          ...connector,
          capabilities: { ...connector.capabilities, future: true },
        }],
      },
    })).toThrowError(new ApiContractError(
      'capabilities.runtime.connectors[0].capabilities.future: extra fields are forbidden',
    ));
  });

  it('rejects an unknown connector transaction semantic', () => {
    const document = capabilitiesFixture();
    const connector = connectorFixture();

    expect(() => decodeCapabilitiesResponse({
      ...document,
      runtime: {
        ...document.runtime,
        connectors: [{
          ...connector,
          capabilities: { ...connector.capabilities, transaction: 'deduplicated' },
        }],
      },
    })).toThrowError(new ApiContractError(
      "capabilities.runtime.connectors[0].capabilities.transaction: expected 'none' or 'pre_commit_commit' or 'ledger_idempotent' or 'retry_deduplicated'",
    ));
  });

  it('rejects extra fields inside the nested runtime scope', () => {
    const document = capabilitiesFixture();

    expect(() => decodeCapabilitiesResponse({
      ...document,
      runtime: {
        ...document.runtime,
        scope: { ...document.runtime.scope, unexpected: true },
      },
    })).toThrowError(new ApiContractError(
      'capabilities.runtime.scope.unexpected: extra fields are forbidden',
    ));
  });
});

describe('job decoder', () => {
  it('accepts every public continuous job state', () => {
    ['pending', 'running', 'completed', 'failed', 'cancelled'].forEach((status) => {
      expect(decodeJobResponse(job(status))).toEqual(job(status));
    });
  });

  it('rejects preview-only terminal states', () => {
    expect(() => decodeJobResponse(job('timed_out'))).toThrowError(
      new ApiContractError(
        "job.status: expected 'pending' or 'running' or 'completed' or 'failed' or 'cancelled'",
      ),
    );
  });

  it('rejects inconsistent running failures', () => {
    expect(() => decodeJobResponse({
      ...job('running'),
      error_code: 'worker_failed',
      error: 'worker exited',
    })).toThrowError(new ApiContractError('job: running job payload is inconsistent'));
  });

  it('requires typed failure codes', () => {
    expect(() => decodeJobResponse({
      ...job('failed'),
      error_code: 'unknown_failure',
    })).toThrowError(new ApiContractError(
      "job.error_code: expected 'job_limit_exceeded' or 'worker_failed'",
    ));
  });

  it('preserves a Join reason while retaining a future-string fallback', () => {
    for (const reasonCode of ['join_match_limit_exceeded', 'future_join_reason']) {
      const failed = { ...job('failed'), reason_code: reasonCode };
      expect(decodeJobResponse(failed)).toEqual(failed);
    }
  });
});
