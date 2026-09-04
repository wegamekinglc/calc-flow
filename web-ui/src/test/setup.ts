import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

afterEach(() => cleanup());

class ResizeObserverStub {
  observe() {
    // jsdom has no ResizeObserver; observing is a no-op.
    return;
  }
  unobserve() {
    return;
  }
  disconnect() {
    return;
  }
}

Object.defineProperty(globalThis, 'ResizeObserver', { value: ResizeObserverStub });
Object.defineProperty(globalThis, 'matchMedia', {
  value: () => ({
    matches: false,
    addEventListener() {
      return;
    },
    removeEventListener() {
      return;
    },
  }),
});
