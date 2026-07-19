import { useCallback, useEffect, useState } from 'react';

export interface PanelLayout {
  toolbox: number;
  inspector: number;
  metrics: number;
}

export type PanelName = keyof PanelLayout;

interface StorageReader {
  getItem: (key: string) => string | null;
}

interface StorageWriter {
  setItem: (key: string, value: string) => void;
}

export const PANEL_LAYOUT_STORAGE_KEY = 'calc-flow-studio:panel-layout:v1';
export const PANEL_RESIZE_HANDLE_WIDTH = 6;
export const DEFAULT_PANEL_LAYOUT: PanelLayout = {
  toolbox: 235,
  inspector: 335,
  metrics: 330,
};
export const PANEL_LIMITS = {
  toolbox: { min: 200, max: 420 },
  inspector: { min: 280, max: 640 },
  metrics: { min: 260 },
  canvasMin: 480,
  outputMin: 480,
} as const;

const defaultPanelLayout = (): PanelLayout => ({ ...DEFAULT_PANEL_LAYOUT });
const clamp = (value: number, minimum: number, maximum: number): number =>
  Math.min(maximum, Math.max(minimum, value));

const clampIndividualWidths = (layout: PanelLayout): PanelLayout => ({
  toolbox: clamp(
    layout.toolbox,
    PANEL_LIMITS.toolbox.min,
    PANEL_LIMITS.toolbox.max,
  ),
  inspector: clamp(
    layout.inspector,
    PANEL_LIMITS.inspector.min,
    PANEL_LIMITS.inspector.max,
  ),
  metrics: Math.max(PANEL_LIMITS.metrics.min, layout.metrics),
});

export const parsePanelLayout = (raw: string | null): PanelLayout => {
  if (raw === null) return defaultPanelLayout();
  try {
    const parsed: unknown = JSON.parse(raw);
    if (
      typeof parsed !== 'object'
      || parsed === null
      || !('version' in parsed)
      || parsed.version !== 1
      || !('toolbox' in parsed)
      || !('inspector' in parsed)
      || !('metrics' in parsed)
      || typeof parsed.toolbox !== 'number'
      || typeof parsed.inspector !== 'number'
      || typeof parsed.metrics !== 'number'
      || !Number.isFinite(parsed.toolbox)
      || !Number.isFinite(parsed.inspector)
      || !Number.isFinite(parsed.metrics)
    ) return defaultPanelLayout();
    return clampIndividualWidths({
      toolbox: parsed.toolbox,
      inspector: parsed.inspector,
      metrics: parsed.metrics,
    });
  } catch {
    return defaultPanelLayout();
  }
};

export const clampWorkspaceLayout = (
  layout: PanelLayout,
  containerWidth: number,
): PanelLayout => {
  const next = clampIndividualWidths(layout);
  const minimumSides = PANEL_LIMITS.toolbox.min + PANEL_LIMITS.inspector.min;
  const availableSides = Math.max(
    minimumSides,
    containerWidth - PANEL_LIMITS.canvasMin - 2 * PANEL_RESIZE_HANDLE_WIDTH,
  );
  let overflow = Math.max(0, next.toolbox + next.inspector - availableSides);
  const inspectorReduction = Math.min(
    overflow,
    next.inspector - PANEL_LIMITS.inspector.min,
  );
  const inspector = next.inspector - inspectorReduction;
  overflow -= inspectorReduction;
  const toolbox = next.toolbox - Math.min(
    overflow,
    next.toolbox - PANEL_LIMITS.toolbox.min,
  );
  return { ...next, toolbox, inspector };
};

export const maxMetricsWidth = (containerWidth: number): number => Math.max(
  PANEL_LIMITS.metrics.min,
  containerWidth - PANEL_LIMITS.outputMin - PANEL_RESIZE_HANDLE_WIDTH,
);

export const clampResultsLayout = (
  layout: PanelLayout,
  containerWidth: number,
): PanelLayout => ({
  ...layout,
  metrics: clamp(
    layout.metrics,
    PANEL_LIMITS.metrics.min,
    maxMetricsWidth(containerWidth),
  ),
});

export const readPanelLayout = (storage?: StorageReader): PanelLayout => {
  try {
    const source = storage ?? window.localStorage;
    return parsePanelLayout(source.getItem(PANEL_LAYOUT_STORAGE_KEY));
  } catch {
    return defaultPanelLayout();
  }
};

export const writePanelLayout = (
  storage: StorageWriter | undefined,
  layout: PanelLayout,
) => {
  try {
    const target = storage ?? window.localStorage;
    target.setItem(PANEL_LAYOUT_STORAGE_KEY, JSON.stringify({
      version: 1,
      ...layout,
    }));
  } catch {
    // Layout preferences are best effort and never block the Studio.
  }
};

const clampNamedWidth = (name: PanelName, value: number): number => {
  if (name === 'toolbox') {
    return clamp(value, PANEL_LIMITS.toolbox.min, PANEL_LIMITS.toolbox.max);
  }
  if (name === 'inspector') {
    return clamp(value, PANEL_LIMITS.inspector.min, PANEL_LIMITS.inspector.max);
  }
  return Math.max(PANEL_LIMITS.metrics.min, value);
};

export const usePanelLayout = () => {
  const [layout, setLayout] = useState<PanelLayout>(() => readPanelLayout());

  const setPanelWidth = useCallback((name: PanelName, value: number) => {
    if (!Number.isFinite(value)) return;
    setLayout((current) => ({
      ...current,
      [name]: clampNamedWidth(name, value),
    }));
  }, []);

  const resetPanelWidth = useCallback((name: PanelName) => {
    setLayout((current) => ({
      ...current,
      [name]: DEFAULT_PANEL_LAYOUT[name],
    }));
  }, []);

  useEffect(() => writePanelLayout(undefined, layout), [layout]);

  return { layout, setPanelWidth, resetPanelWidth };
};

export const useElementWidth = <T extends HTMLElement>() => {
  const [element, setElement] = useState<T | null>(null);
  const [width, setWidth] = useState(0);
  const ref = useCallback((next: T | null) => setElement(next), []);

  useEffect(() => {
    if (element === null) {
      setWidth(0);
      return undefined;
    }
    setWidth(element.getBoundingClientRect().width);
    if (typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver((entries) => {
      const nextWidth = entries[0]?.contentRect.width;
      if (typeof nextWidth === 'number' && Number.isFinite(nextWidth)) {
        setWidth(nextWidth);
      }
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [element]);

  return { ref, width };
};
