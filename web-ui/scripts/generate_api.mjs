import { readFile, writeFile } from 'node:fs/promises';
import process from 'node:process';
import { fileURLToPath } from 'node:url';
import openapiTS, { astToString, COMMENT_HEADER } from 'openapi-typescript';

// A nullable default does not make an optional request property required.
export const typeSchema = (value) => {
  if (Array.isArray(value)) return value.map(typeSchema);
  if (value === null || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value)
    .filter(([key, item]) => key !== 'default' || item !== null)
    .map(([key, item]) => [key, typeSchema(item)]));
};

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const schema = JSON.parse(await readFile('openapi.json', 'utf8'));
  const ast = await openapiTS(typeSchema(schema));
  await writeFile('src/api/schema.d.ts', COMMENT_HEADER + astToString(ast));
}
