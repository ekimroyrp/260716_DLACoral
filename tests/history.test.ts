import { describe, expect, it } from 'vitest';
import { ActionHistory, estimateSnapshotBytes } from '../src/history';

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
});
