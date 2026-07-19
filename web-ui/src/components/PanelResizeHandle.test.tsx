import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { PanelResizeHandle } from './PanelResizeHandle';

describe('PanelResizeHandle', () => {
  it('resizes a start panel with pointer movement and clamps to its bounds', () => {
    const onChange = vi.fn();
    render(
      <PanelResizeHandle
        label="Resize Toolbox"
        value={300}
        min={200}
        max={340}
        grow="start"
        onChange={onChange}
        onReset={vi.fn()}
      />,
    );
    const separator = screen.getByRole('separator', { name: 'Resize Toolbox' });

    fireEvent.pointerDown(separator, { pointerId: 1, clientX: 100 });
    fireEvent.pointerMove(separator, { pointerId: 1, clientX: 140 });
    expect(onChange).toHaveBeenLastCalledWith(340);

    fireEvent.pointerMove(separator, { pointerId: 1, clientX: 180 });
    expect(onChange).toHaveBeenLastCalledWith(340);
    fireEvent.pointerUp(separator, { pointerId: 1 });
  });

  it('inverts physical movement for a right-hand panel', () => {
    const onChange = vi.fn();
    render(
      <PanelResizeHandle
        label="Resize Inspector"
        value={300}
        min={280}
        max={500}
        grow="end"
        onChange={onChange}
        onReset={vi.fn()}
      />,
    );
    const separator = screen.getByRole('separator', { name: 'Resize Inspector' });

    fireEvent.pointerDown(separator, { pointerId: 2, clientX: 100 });
    fireEvent.pointerMove(separator, { pointerId: 2, clientX: 60 });
    expect(onChange).toHaveBeenLastCalledWith(340);
  });

  it('supports keyboard bounds, accelerated steps, reset, and separator metadata', () => {
    const onChange = vi.fn();
    const onReset = vi.fn();
    render(
      <PanelResizeHandle
        label="Resize Toolbox"
        value={300}
        min={200}
        max={400}
        grow="start"
        onChange={onChange}
        onReset={onReset}
      />,
    );
    const separator = screen.getByRole('separator', { name: 'Resize Toolbox' });

    expect(separator).toHaveAttribute('aria-orientation', 'vertical');
    expect(separator).toHaveAttribute('aria-valuemin', '200');
    expect(separator).toHaveAttribute('aria-valuemax', '400');
    expect(separator).toHaveAttribute('aria-valuenow', '300');

    fireEvent.keyDown(separator, { key: 'ArrowRight' });
    expect(onChange).toHaveBeenLastCalledWith(316);
    fireEvent.keyDown(separator, { key: 'ArrowRight', shiftKey: true });
    expect(onChange).toHaveBeenLastCalledWith(348);
    fireEvent.keyDown(separator, { key: 'Home' });
    expect(onChange).toHaveBeenLastCalledWith(200);
    fireEvent.keyDown(separator, { key: 'End' });
    expect(onChange).toHaveBeenLastCalledWith(400);

    fireEvent.doubleClick(separator);
    expect(onReset).toHaveBeenCalledOnce();
  });

  it('uses physical arrow direction for an end-growing panel', () => {
    const onChange = vi.fn();
    render(
      <PanelResizeHandle
        label="Resize Metrics"
        value={300}
        min={260}
        max={500}
        grow="end"
        onChange={onChange}
        onReset={vi.fn()}
      />,
    );
    const separator = screen.getByRole('separator', { name: 'Resize Metrics' });

    fireEvent.keyDown(separator, { key: 'ArrowLeft' });
    expect(onChange).toHaveBeenLastCalledWith(316);
    fireEvent.keyDown(separator, { key: 'ArrowRight', shiftKey: true });
    expect(onChange).toHaveBeenLastCalledWith(260);
  });
});
