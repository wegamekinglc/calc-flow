import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  DEFAULT_PANEL_LAYOUT,
  PANEL_LAYOUT_STORAGE_KEY,
  clampResultsLayout,
  clampWorkspaceLayout,
  maxMetricsWidth,
  parsePanelLayout,
  readPanelLayout,
  useElementWidth,
  usePanelLayout,
  writePanelLayout,
} from './panelLayout';

beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe('panel layout', () => {
  it('falls back when stored layout data is missing, malformed, or incompatible', () => {
    expect(parsePanelLayout(null)).toEqual(DEFAULT_PANEL_LAYOUT);
    expect(parsePanelLayout('{bad json')).toEqual(DEFAULT_PANEL_LAYOUT);
    expect(parsePanelLayout(JSON.stringify({
      version: 2,
      toolbox: 300,
      inspector: 400,
      metrics: 350,
    }))).toEqual(DEFAULT_PANEL_LAYOUT);
    expect(parsePanelLayout(JSON.stringify({
      version: 1,
      toolbox: null,
      inspector: 400,
      metrics: 350,
    }))).toEqual(DEFAULT_PANEL_LAYOUT);
  });

  it('clamps individual and combined workspace widths around the canvas minimum', () => {
    expect(clampWorkspaceLayout({
      toolbox: 420,
      inspector: 640,
      metrics: 330,
    }, 1180)).toEqual({
      toolbox: 408,
      inspector: 280,
      metrics: 330,
    });
    expect(clampWorkspaceLayout({
      toolbox: 10,
      inspector: 900,
      metrics: 330,
    }, 1600)).toEqual({
      toolbox: 200,
      inspector: 640,
      metrics: 330,
    });
  });

  it('keeps the output readable when calculating the metrics bound', () => {
    expect(maxMetricsWidth(900)).toBe(414);
    expect(maxMetricsWidth(600)).toBe(260);
    expect(clampResultsLayout({
      toolbox: 235,
      inspector: 335,
      metrics: 500,
    }, 900).metrics).toBe(414);
  });

  it('treats unavailable browser storage as best effort', () => {
    const storage = {
      getItem: vi.fn(() => { throw new Error('unavailable'); }),
      setItem: vi.fn(() => { throw new Error('unavailable'); }),
    };

    expect(readPanelLayout(storage)).toEqual(DEFAULT_PANEL_LAYOUT);
    expect(() => writePanelLayout(storage, DEFAULT_PANEL_LAYOUT)).not.toThrow();
  });

  it('restores and persists panel widths through the layout hook', async () => {
    localStorage.setItem(PANEL_LAYOUT_STORAGE_KEY, JSON.stringify({
      version: 1,
      toolbox: 300,
      inspector: 410,
      metrics: 360,
    }));
    const { result } = renderHook(() => usePanelLayout());

    expect(result.current.layout).toEqual({
      toolbox: 300,
      inspector: 410,
      metrics: 360,
    });

    act(() => result.current.setPanelWidth('toolbox', 316));
    await waitFor(() => expect(
      JSON.parse(localStorage.getItem(PANEL_LAYOUT_STORAGE_KEY) ?? '{}'),
    ).toMatchObject({ version: 1, toolbox: 316 }));

    act(() => result.current.resetPanelWidth('toolbox'));
    expect(result.current.layout.toolbox).toBe(DEFAULT_PANEL_LAYOUT.toolbox);
  });

  it('tracks element width and disconnects its observer', () => {
    const observe = vi.spyOn(ResizeObserver.prototype, 'observe');
    const disconnect = vi.spyOn(ResizeObserver.prototype, 'disconnect');
    const element = document.createElement('div');
    vi.spyOn(element, 'getBoundingClientRect').mockReturnValue({
      width: 800,
    } as DOMRect);
    const { result, unmount } = renderHook(() => useElementWidth<HTMLDivElement>());

    act(() => result.current.ref(element));
    expect(result.current.width).toBe(800);
    expect(observe).toHaveBeenCalledWith(element);

    unmount();
    expect(disconnect).toHaveBeenCalledOnce();
  });
});
