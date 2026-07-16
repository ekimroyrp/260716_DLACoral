import { describe, expect, it } from 'vitest';
import { createAppSnapshot, createInitialAppState } from '../src/state';

describe('app state snapshots', () => {
  it('copies settings and aggregate arrays', () => {
    const state = createInitialAppState();
    const positions = new Int32Array([0, 0, 0]);
    const enclosed = new Uint8Array([0]);
    const snapshot = createAppSnapshot(state, {
      positions,
      enclosed,
      seedCount: 1,
      currentCount: 1,
      latestCount: 1,
      maxRadiusSq: 0,
      rngState: 1,
      branchSerial: 0,
    });
    positions[0] = 9;
    state.display.innerColor = '#000000';
    expect(snapshot.aggregate?.positions[0]).toBe(0);
    expect(snapshot.display.innerColor).toBe('#6b2f24');
  });
});
