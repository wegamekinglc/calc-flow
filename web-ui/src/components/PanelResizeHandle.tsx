import { useRef } from 'react';

interface PanelResizeHandleProps {
  label: string;
  value: number;
  min: number;
  max: number;
  grow: 'start' | 'end';
  onChange: (value: number) => void;
  onReset: () => void;
}

interface DragState {
  pointerId: number;
  startX: number;
  startValue: number;
}

const clamp = (value: number, minimum: number, maximum: number): number =>
  Math.min(maximum, Math.max(minimum, value));

export function PanelResizeHandle({
  label,
  value,
  min,
  max,
  grow,
  onChange,
  onReset,
}: PanelResizeHandleProps) {
  const drag = useRef<DragState | null>(null);
  const maximum = Math.max(min, max);
  const emit = (next: number) => onChange(clamp(next, min, maximum));

  return (
    <div
      className="panel-resize-handle"
      role="separator"
      aria-label={label}
      aria-orientation="vertical"
      aria-valuemin={min}
      aria-valuemax={maximum}
      aria-valuenow={Math.round(value)}
      tabIndex={0}
      onDoubleClick={onReset}
      onKeyDown={(event) => {
        const step = event.shiftKey ? 48 : 16;
        if (event.key === 'Home') {
          event.preventDefault();
          emit(min);
        } else if (event.key === 'End') {
          event.preventDefault();
          emit(maximum);
        } else if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
          event.preventDefault();
          const physicalDirection = event.key === 'ArrowRight' ? 1 : -1;
          const valueDirection = grow === 'start'
            ? physicalDirection
            : -physicalDirection;
          emit(value + valueDirection * step);
        }
      }}
      onPointerDown={(event) => {
        drag.current = {
          pointerId: event.pointerId,
          startX: event.clientX,
          startValue: value,
        };
        event.currentTarget.setPointerCapture?.(event.pointerId);
      }}
      onPointerMove={(event) => {
        const active = drag.current;
        if (active?.pointerId !== event.pointerId) return;
        const physicalDelta = event.clientX - active.startX;
        const valueDelta = grow === 'start' ? physicalDelta : -physicalDelta;
        emit(active.startValue + valueDelta);
      }}
      onPointerUp={(event) => {
        if (drag.current?.pointerId !== event.pointerId) return;
        drag.current = null;
        if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
            event.currentTarget.releasePointerCapture?.(event.pointerId);
        }
      }}
      onPointerCancel={() => {
        drag.current = null;
      }}
      onLostPointerCapture={() => {
        drag.current = null;
      }}
    />
  );
}
