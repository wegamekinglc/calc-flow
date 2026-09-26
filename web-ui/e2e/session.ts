import type { APIRequestContext, APIResponse } from '@playwright/test';

const sessionUrl = 'http://127.0.0.1:8765/api/v3/session';

async function launchTokenHeaders(
  request: APIRequestContext,
): Promise<Record<string, string>> {
  const response = await request.get(sessionUrl);
  if (!response.ok()) {
    throw new Error(`Studio session request failed: ${response.status()}`);
  }
  const body: unknown = await response.json();
  if (!body || typeof body !== 'object' || !('token' in body)
      || typeof body.token !== 'string' || body.token.length === 0) {
    throw new Error('Studio session response has no launch token');
  }
  return { 'X-Calc-Flow-Token': body.token };
}

export async function postWithLaunchToken(
  request: APIRequestContext,
  url: string,
  options?: Parameters<APIRequestContext['post']>[1],
): Promise<APIResponse> {
  return request.post(url, {
    ...options,
    headers: { ...options?.headers, ...await launchTokenHeaders(request) },
  });
}

export async function deleteWithLaunchToken(
  request: APIRequestContext,
  url: string,
  options?: Parameters<APIRequestContext['delete']>[1],
): Promise<APIResponse> {
  return request.delete(url, {
    ...options,
    headers: { ...options?.headers, ...await launchTokenHeaders(request) },
  });
}
