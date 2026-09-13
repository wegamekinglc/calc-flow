// @vitest-environment node
import { describe, expect, it } from 'vitest';
import openapiTS, { astToString } from 'openapi-typescript';

import { typeSchema } from './generate_api.mjs';

describe('project API type generation', () => {
  it('preserves optional null defaults and required fields without mutating OpenAPI', async () => {
    const schema = {
      openapi: '3.1.0', info: { title: 'Contract', version: '1' }, paths: {},
      components: { schemas: { Example: {
        type: 'object', required: ['explicit'], properties: {
          optional: { type: ['string', 'null'], default: null },
          explicit: { type: ['string', 'null'], default: null },
          limit: { type: 'integer', default: 1 },
        },
      } } },
    };
    const before = JSON.parse(JSON.stringify(schema));
    const output = astToString(await openapiTS(typeSchema(schema)));
    expect(output).toContain('optional?: string | null;');
    expect(output).toContain('explicit: string | null;');
    expect(output).toContain('limit: number;');
    expect(schema).toEqual(before);
  });
});
