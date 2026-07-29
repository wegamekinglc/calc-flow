import { describe, expect, it } from 'vitest';

import {
  ApiContractError,
  decodeCapabilitiesResponse,
  decodeRunResponse,
} from './decoders';

const capabilitiesFixture = () => ({
  schemaVersion: 1,
  runtime: {
    scope: { kind: 'runtimeSession', sessionId: 'session', revision: 0 },
    packageVersion: '2.0.0',
    projectFormatVersions: [2],
    batchKinds: ['table', 'array'],
    portableArrowTypes: ['int64'],
    operators: [],
    udfs: [],
    providers: [],
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

const providerFixture = () => ({
  provider: 'numpy',
  name: 'expression',
  version: '1',
  inputPorts: [{ name: 'input', kind: 'array', required: true }],
  outputPorts: [{ name: 'output', kind: 'array', required: true }],
  optionsSchema: null,
});

const registrationFixture = (
  reconstruction: string,
  extra: Record<string, unknown> = {},
) => ({
  reconstruction,
  registrationKind: 'provider',
  provider: 'numpy',
  name: 'expression',
  version: '1',
  ...extra,
});

const capabilitiesWithRegistrations = (registrations: unknown[]) => ({
  ...capabilitiesFixture(),
  preview: { ...capabilitiesFixture().preview, workerRegistrations: registrations },
});

const completedRunWithArrayData = (data: unknown) => ({
  id: 'run',
  project_id: 'project',
  status: 'completed',
  created_at: '2026-01-01T00:00:00Z',
  started_at: '2026-01-01T00:00:00Z',
  finished_at: '2026-01-01T00:00:01Z',
  error: null,
  result: {
    outputs: {
      output: {
        kind: 'array',
        backend: 'numpy',
        total_rows: 1,
        truncated: false,
        data,
        metadata: {},
      },
    },
    node_timings: {},
    datafusion_metrics: [],
    metadata: {},
  },
});

const nestArrays = (depth: number): unknown => {
  let value: unknown = 1;
  for (let level = 0; level < depth; level += 1) value = [value];
  return value;
};

describe('capabilities decoder nested object validation', () => {
  it('accepts a fully populated nested provider document', () => {
    const document = {
      ...capabilitiesFixture(),
      runtime: { ...capabilitiesFixture().runtime, providers: [providerFixture()] },
    };

    expect(decodeCapabilitiesResponse(document)).toEqual(document);
  });

  it('rejects extra fields inside the nested runtime scope', () => {
    const document = capabilitiesFixture();

    expect(() =>
      decodeCapabilitiesResponse({
        ...document,
        runtime: {
          ...document.runtime,
          scope: { ...document.runtime.scope, unexpected: true },
        },
      }),
    ).toThrowError(
      new ApiContractError(
        'capabilities.runtime.scope.unexpected: extra fields are forbidden',
      ),
    );
  });

  it('rejects extra fields inside a nested provider port', () => {
    const document = capabilitiesFixture();

    expect(() =>
      decodeCapabilitiesResponse({
        ...document,
        runtime: {
          ...document.runtime,
          providers: [
            {
              ...providerFixture(),
              inputPorts: [
                { name: 'input', kind: 'array', required: true, futureFlag: true },
              ],
            },
          ],
        },
      }),
    ).toThrowError(
      new ApiContractError(
        'capabilities.runtime.providers[0].inputPorts[0].futureFlag: '
        + 'extra fields are forbidden',
      ),
    );
  });

  it('rejects extra fields inside a nested preview limit entry', () => {
    const document = capabilitiesFixture();

    expect(() =>
      decodeCapabilitiesResponse({
        ...document,
        preview: {
          ...document.preview,
          limits: {
            ...document.preview.limits,
            maxRows: { default: 10, minimum: 1, maximum: 10, soft: true },
          },
        },
      }),
    ).toThrowError(
      new ApiContractError(
        'capabilities.preview.limits.maxRows.soft: extra fields are forbidden',
      ),
    );
  });
});

describe('worker registration discriminator validation', () => {
  it('accepts every defined reconstruction discriminator', () => {
    const document = capabilitiesWithRegistrations([
      registrationFixture('serialized'),
      registrationFixture('lazyBuiltin'),
      registrationFixture('unavailable', { reasonCode: 'serializationFailed' }),
    ]);

    expect(decodeCapabilitiesResponse(document)).toEqual(document);
  });

  it('rejects an unknown reconstruction discriminator', () => {
    expect(() =>
      decodeCapabilitiesResponse(
        capabilitiesWithRegistrations([registrationFixture('embedded')]),
      ),
    ).toThrowError(
      new ApiContractError(
        'capabilities.preview.workerRegistrations[0].reconstruction: '
        + "expected 'serialized' or 'lazyBuiltin' or 'unavailable'",
      ),
    );
  });

  it('rejects an unknown reason code on unavailable registrations', () => {
    expect(() =>
      decodeCapabilitiesResponse(
        capabilitiesWithRegistrations([
          registrationFixture('unavailable', { reasonCode: 'unsupportedBackend' }),
        ]),
      ),
    ).toThrowError(
      new ApiContractError(
        'capabilities.preview.workerRegistrations[0].reasonCode: '
        + "expected 'serializationFailed'",
      ),
    );
  });

  it('requires a reason code on unavailable registrations', () => {
    expect(() =>
      decodeCapabilitiesResponse(
        capabilitiesWithRegistrations([registrationFixture('unavailable')]),
      ),
    ).toThrowError(
      new ApiContractError(
        'capabilities.preview.workerRegistrations[0].reasonCode: field is required',
      ),
    );
  });

  it('forbids a reason code on serializable registrations', () => {
    expect(() =>
      decodeCapabilitiesResponse(
        capabilitiesWithRegistrations([
          registrationFixture('serialized', { reasonCode: 'serializationFailed' }),
        ]),
      ),
    ).toThrowError(
      new ApiContractError(
        'capabilities.preview.workerRegistrations[0].reasonCode: '
        + 'extra fields are forbidden',
      ),
    );
  });
});

describe('run decoder JSON depth limit', () => {
  it('accepts JSON values nested exactly to the depth limit', () => {
    const run = completedRunWithArrayData(nestArrays(32));

    expect(decodeRunResponse(run)).toEqual(run);
  });

  it('rejects JSON values nested beyond the depth limit', () => {
    expect(() =>
      decodeRunResponse(completedRunWithArrayData(nestArrays(33))),
    ).toThrowError(
      new ApiContractError(
        'run.result.outputs.output.data'
        + `${'[0]'.repeat(33)}: JSON value exceeds maximum depth 32`,
      ),
    );
  });

  it('rejects non-finite JSON numbers in run outputs', () => {
    expect(() =>
      decodeRunResponse(completedRunWithArrayData([Number.NaN])),
    ).toThrowError(
      new ApiContractError(
        'run.result.outputs.output.data[0]: JSON numbers must be finite',
      ),
    );
  });
});
