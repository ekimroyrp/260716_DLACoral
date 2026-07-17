import { describe, expect, it } from 'vitest';
import { createAppSnapshot, createInitialAppState } from '../src/state';

describe('app state snapshots', () => {
  it('copies settings and aggregate arrays', () => {
    const state = createInitialAppState();
    expect(state.dla.particleSize).toBe(1);
    expect(state.dla.particleScale).toBe(1);
    expect(state.dla.particleResolution).toBe(2);
    expect(state.dla.adaptiveStickNeighbors).toBe(true);
    expect(state.dla.contactHits).toBe(1);
    expect(state.dla.bootstrapParticles).toBe(50);
    expect(state.display.gradientContrast).toBe(1.37);
    expect(state.display.gradientBias).toBe(-0.74);
    expect(state.display.gradientBlur).toBe(0.45);
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
    expect(snapshot.display.innerColor).toBe('#ac2a4a');
    expect(snapshot.display.outerColor).toBe('#ffffff');
    expect(snapshot.display.gradientContrast).toBe(1.37);
  });

  it('copies optional GPU walker state used by exact history restores', () => {
    const state = createInitialAppState();
    const walkerState = new Int32Array([1, 2, 3, 4]);
    const aggregate = {
      positions: new Int32Array([0, 0, 0]),
      enclosed: new Uint8Array([0]),
      seedCount: 1,
      currentCount: 1,
      latestCount: 1,
      maxRadiusSq: 0,
      rngState: 1,
      branchSerial: 0,
      walkerState,
    };

    const snapshot = createAppSnapshot(state, aggregate);
    walkerState[0] = 99;

    const clonedWalkerState = (snapshot.aggregate as typeof aggregate | undefined)?.walkerState;
    expect(clonedWalkerState?.[0]).toBe(1);
  });
});
