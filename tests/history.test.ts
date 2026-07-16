import { describe, expect, it } from 'vitest';
import {
  ActionHistory,
  areHistorySnapshotsEquivalent,
  compressDlaSnapshot,
  decompressDlaSnapshot,
  estimateSnapshotBytes,
} from '../src/history';
import {
  DEFAULT_DISPLAY_SETTINGS,
  DEFAULT_DLA_SETTINGS,
  DEFAULT_SIMULATION_SETTINGS,
  type DlaSnapshot,
} from '../src/types';

describe('ActionHistory', () => {
  it('undoes and redoes complete actions', () => {
    const history = new ActionHistory<number>();
    history.push(1, 2);
    history.push(2, 3);
    expect(history.undo()).toBe(2);
    expect(history.undo()).toBe(1);
    expect(history.redo()).toBe(2);
  });

  it('evicts old entries using action and byte limits', () => {
    const history = new ActionHistory<number>({ maxActions: 2, maxBytes: 8 });
    history.push(0, 1, 4);
    history.push(1, 2, 4);
    history.push(2, 3, 4);
    expect(history.size).toBe(2);
    expect(history.undo()).toBe(2);
    expect(history.undo()).toBe(1);
    expect(history.undo()).toBeNull();
  });

  it('counts typed-array snapshot memory', () => {
    expect(estimateSnapshotBytes({ a: new Int32Array(3), b: new Uint8Array(4) })).toBe(16);
  });

  it('losslessly compacts lattice positions, enclosed flags, and exact GPU continuation state', () => {
    const live = {
      positions: new Int32Array([-512, 0, 511, 12, -4, 7]),
      enclosed: new Uint8Array([0, 1]),
      seedCount: 1,
      currentCount: 2,
      latestCount: 2,
      maxRadiusSq: 262_144,
      rngState: 123,
      branchSerial: 4,
      walkerState: new Int32Array([10, 20, 30, -40]),
      walkerCount: 1,
      epoch: 17,
      epochCredit: 0.625,
    } satisfies DlaSnapshot & {
      walkerState: Int32Array;
      walkerCount: number;
      epoch: number;
      epochCredit: number;
    };

    const compressed = compressDlaSnapshot(live);
    live.positions.fill(99);
    live.enclosed.fill(0);
    live.walkerState.fill(99);
    const restored = decompressDlaSnapshot(compressed) as typeof live;

    expect(compressed.positionEncoding).toBe('i16');
    expect(compressed.positions).toBeInstanceOf(Int16Array);
    expect(compressed.enclosedBits.byteLength).toBe(1);
    expect(Array.from(restored.positions)).toEqual([-512, 0, 511, 12, -4, 7]);
    expect(Array.from(restored.enclosed)).toEqual([0, 1]);
    expect(Array.from(restored.walkerState)).toEqual([10, 20, 30, -40]);
    expect(restored.walkerCount).toBe(1);
    expect(restored.epoch).toBe(17);
    expect(restored.epochCredit).toBe(0.625);
    expect(estimateSnapshotBytes(compressed)).toBeLessThan(estimateSnapshotBytes(restored));
  });

  it('falls back to 32-bit coordinates when a snapshot exceeds the compact lattice range', () => {
    const snapshot: DlaSnapshot = {
      positions: new Int32Array([40_000, -40_000, 0]),
      enclosed: new Uint8Array([1]),
      seedCount: 1,
      currentCount: 1,
      latestCount: 1,
      maxRadiusSq: 1_600_000_000,
      rngState: 1,
      branchSerial: 0,
    };
    const compressed = compressDlaSnapshot(snapshot);
    expect(compressed.positionEncoding).toBe('i32');
    expect(Array.from(decompressDlaSnapshot(compressed).positions)).toEqual([40_000, -40_000, 0]);
  });

  it('ignores autonomous timeline drift for no-op gestures but preserves deliberate simulation changes', () => {
    const snapshot = (
      timeline: number,
      latestTimeline: number,
      actionLabel: string,
      running = false,
      rate = 1,
    ) => ({
      simulation: {
        ...DEFAULT_SIMULATION_SETTINGS,
        timeline,
        latestTimeline,
        running,
        rate,
      },
      dla: { ...DEFAULT_DLA_SETTINGS },
      display: { ...DEFAULT_DISPLAY_SETTINGS },
      actionLabel,
    });

    const before = snapshot(10, 10, 'Brightness');
    expect(areHistorySnapshotsEquivalent(before, snapshot(25, 25, 'Brightness'))).toBe(true);
    expect(areHistorySnapshotsEquivalent(before, snapshot(25, 25, 'Brightness', true))).toBe(false);
    expect(areHistorySnapshotsEquivalent(before, snapshot(25, 25, 'Brightness', false, 2))).toBe(false);
    expect(
      areHistorySnapshotsEquivalent(
        snapshot(10, 25, 'Simulation Timeline'),
        snapshot(12, 25, 'Simulation Timeline'),
      ),
    ).toBe(false);

    const structuralBefore = {
      ...snapshot(10, 25, 'Reset Simulation'),
      aggregate: { currentCount: 11, latestCount: 26, branchSerial: 0, rngState: 1 },
    };
    const structuralAfter = {
      ...snapshot(0, 0, 'Reset Simulation'),
      aggregate: { currentCount: 1, latestCount: 1, branchSerial: 0, rngState: 1 },
    };
    expect(areHistorySnapshotsEquivalent(structuralBefore, structuralAfter)).toBe(false);
  });

  it('enforces the byte budget even when one heavy action is oversized', () => {
    const history = new ActionHistory<number>({ maxActions: 10, maxBytes: 4 });
    history.push(0, 1, 8);
    expect(history.size).toBe(0);
    expect(history.canUndo).toBe(false);
  });

  it('evicts through the oldest heavy checkpoint without dropping later light actions', () => {
    const history = new ActionHistory<number>({ maxActions: 10, maxBytes: 4 });
    history.push(0, 1, 4);
    history.push(1, 2, 0);
    history.push(2, 3, 4);

    expect(history.size).toBe(2);
    expect(history.undo()).toBe(2);
    expect(history.undo()).toBe(1);
  });
});
