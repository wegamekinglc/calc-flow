import {
  useCallback,
  useState,
  type Dispatch,
  type SetStateAction,
} from 'react';

/**
 * One busy-flag shell for user-triggered actions.
 *
 * Wraps an async action with the message/busy contract every button in the
 * Studio shares: set busy, surface thrown error messages, always clear busy.
 */
export const useBusyAction = (setMessage: Dispatch<SetStateAction<string>>) => {
  const [busy, setBusy] = useState(false);
  const run = useCallback(
    async (action: () => Promise<void>) => {
      setBusy(true);
      try {
        await action();
      } catch (error) {
        setMessage((error as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [setMessage],
  );
  return { busy, run };
};
