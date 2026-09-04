import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { useBusyAction } from './useBusyAction';

describe('useBusyAction', () => {
  it('clears the busy flag after a successful action', async () => {
    const setMessage = vi.fn();
    const action = vi.fn().mockResolvedValue(undefined);
    const { result } = renderHook(() => useBusyAction(setMessage));

    expect(result.current.busy).toBe(false);
    await act(async () => {
      await result.current.run(action);
    });

    expect(action).toHaveBeenCalledOnce();
    expect(setMessage).not.toHaveBeenCalled();
    expect(result.current.busy).toBe(false);
  });

  it('surfaces thrown error messages and always clears busy', async () => {
    const setMessage = vi.fn();
    const { result } = renderHook(() => useBusyAction(setMessage));

    await act(async () => {
      await result.current.run(async () => {
        throw new Error('job rejected');
      });
    });

    expect(setMessage).toHaveBeenCalledWith('job rejected');
    expect(result.current.busy).toBe(false);
  });
});
