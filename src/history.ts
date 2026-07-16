import { MAX_HISTORY_ACTIONS, MAX_HISTORY_BYTES } from './types';
import type { DisplaySettings, DlaSettings, DlaSnapshot, SimulationSettings } from './types';

interface ExactGpuSnapshotFields {
  walkerState: Int32Array;
  walkerCount: number;
  epoch: number;
  epochCredit: number;
}

export interface CompressedDlaSnapshot {
  positions: Int16Array | Int32Array;
  positionEncoding: 'i16' | 'i32';
  enclosedBits: Uint8Array;
  enclosedLength: number;
  seedCount: number;
  currentCount: number;
  latestCount: number;
  maxRadiusSq: number;
  rngState: number;
  branchSerial: number;
  walkerState?: Int32Array;
  walkerCount?: number;
  epoch?: number;
  epochCredit?: number;
}

export interface ComparableHistorySnapshot {
  simulation: SimulationSettings;
  dla: DlaSettings;
  display: DisplaySettings;
  aggregate?: Pick<
    DlaSnapshot,
    'currentCount' | 'latestCount' | 'branchSerial' | 'rngState'
  >;
  actionLabel?: string;
}

export interface HistoryEntry<T> {
  before: T;
  after: T;
  bytes: number;
}

export interface HistoryLimits {
  maxActions?: number;
  maxBytes?: number;
}

export class ActionHistory<T> {
  private readonly maxActions: number;
  private readonly maxBytes: number;
  private undoEntries: HistoryEntry<T>[] = [];
  private redoEntries: HistoryEntry<T>[] = [];
  private undoBytes = 0;
  private redoBytes = 0;

  constructor(limits: HistoryLimits = {}) {
    this.maxActions = limits.maxActions ?? MAX_HISTORY_ACTIONS;
    this.maxBytes = limits.maxBytes ?? MAX_HISTORY_BYTES;
  }

  get canUndo(): boolean {
    return this.undoEntries.length > 0;
  }

  get canRedo(): boolean {
    return this.redoEntries.length > 0;
  }

  get size(): number {
    return this.undoEntries.length;
  }

  push(before: T, after: T, bytes = 0): void {
    const normalizedBytes = Math.max(0, Math.floor(bytes));
    this.undoEntries.push({ before, after, bytes: normalizedBytes });
    this.undoBytes += normalizedBytes;
    this.redoEntries = [];
    this.redoBytes = 0;
    this.trimUndo();
  }

  undo(): T | null {
    const entry = this.undoEntries.pop();
    if (!entry) return null;
    this.undoBytes -= entry.bytes;
    this.redoEntries.push(entry);
    this.redoBytes += entry.bytes;
    return entry.before;
  }

  redo(): T | null {
    const entry = this.redoEntries.pop();
    if (!entry) return null;
    this.redoBytes -= entry.bytes;
    this.undoEntries.push(entry);
    this.undoBytes += entry.bytes;
    this.trimUndo();
    return entry.after;
  }

  clear(): void {
    this.undoEntries = [];
    this.redoEntries = [];
    this.undoBytes = 0;
    this.redoBytes = 0;
  }

  private trimUndo(): void {
    while (this.undoEntries.length > this.maxActions) {
      this.removeOldestUndoEntry();
    }

    // Removing a middle action would make the remaining undo chain skip a
    // state transition. Drop the oldest prefix through the first heavy entry
    // instead, preserving every lightweight action after that checkpoint.
    while (this.undoBytes > this.maxBytes) {
      const heavyIndex = this.undoEntries.findIndex((entry) => entry.bytes > 0);
      if (heavyIndex < 0) {
        this.undoBytes = 0;
        break;
      }
      for (let index = 0; index <= heavyIndex; index += 1) {
        this.removeOldestUndoEntry();
      }
    }
  }

  private removeOldestUndoEntry(): void {
    const removed = this.undoEntries.shift();
    if (removed) {
      this.undoBytes -= removed.bytes;
    }
  }
}

export function compressDlaSnapshot(snapshot: DlaSnapshot): CompressedDlaSnapshot {
  const canUseInt16 = snapshot.positions.every(
    (coordinate) => coordinate >= -32_768 && coordinate <= 32_767,
  );
  const positions = canUseInt16
    ? new Int16Array(snapshot.positions)
    : snapshot.positions.slice();
  const enclosedBits = new Uint8Array(Math.ceil(snapshot.enclosed.length / 8));
  for (let index = 0; index < snapshot.enclosed.length; index += 1) {
    if (snapshot.enclosed[index]) {
      enclosedBits[index >> 3] |= 1 << (index & 7);
    }
  }

  const compressed: CompressedDlaSnapshot = {
    positions,
    positionEncoding: canUseInt16 ? 'i16' : 'i32',
    enclosedBits,
    enclosedLength: snapshot.enclosed.length,
    seedCount: snapshot.seedCount,
    currentCount: snapshot.currentCount,
    latestCount: snapshot.latestCount,
    maxRadiusSq: snapshot.maxRadiusSq,
    rngState: snapshot.rngState,
    branchSerial: snapshot.branchSerial,
  };
  const gpuSnapshot = snapshot as DlaSnapshot & Partial<ExactGpuSnapshotFields>;
  if (gpuSnapshot.walkerState instanceof Int32Array) {
    compressed.walkerState = gpuSnapshot.walkerState.slice();
    compressed.walkerCount = gpuSnapshot.walkerCount;
    compressed.epoch = gpuSnapshot.epoch;
    compressed.epochCredit = gpuSnapshot.epochCredit;
  }
  return compressed;
}

export function decompressDlaSnapshot(compressed: CompressedDlaSnapshot): DlaSnapshot {
  const positions = new Int32Array(compressed.positions.length);
  positions.set(compressed.positions);
  const enclosed = new Uint8Array(compressed.enclosedLength);
  for (let index = 0; index < enclosed.length; index += 1) {
    enclosed[index] = (compressed.enclosedBits[index >> 3] & (1 << (index & 7))) !== 0 ? 1 : 0;
  }

  const snapshot: DlaSnapshot & Partial<ExactGpuSnapshotFields> = {
    positions,
    enclosed,
    seedCount: compressed.seedCount,
    currentCount: compressed.currentCount,
    latestCount: compressed.latestCount,
    maxRadiusSq: compressed.maxRadiusSq,
    rngState: compressed.rngState,
    branchSerial: compressed.branchSerial,
  };
  if (compressed.walkerState instanceof Int32Array) {
    snapshot.walkerState = compressed.walkerState.slice();
    snapshot.walkerCount = compressed.walkerCount;
    snapshot.epoch = compressed.epoch;
    snapshot.epochCredit = compressed.epochCredit;
  }
  return snapshot;
}

export function areHistorySnapshotsEquivalent(
  a: ComparableHistorySnapshot,
  b: ComparableHistorySnapshot,
): boolean {
  const timelineSensitive =
    a.actionLabel === 'Simulation Timeline' || b.actionLabel === 'Simulation Timeline';
  const simulationA = timelineSensitive
    ? a.simulation
    : { running: a.simulation.running, rate: a.simulation.rate };
  const simulationB = timelineSensitive
    ? b.simulation
    : { running: b.simulation.running, rate: b.simulation.rate };
  const settingsA = JSON.stringify({ simulation: simulationA, dla: a.dla, display: a.display });
  const settingsB = JSON.stringify({ simulation: simulationB, dla: b.dla, display: b.display });
  if (settingsA !== settingsB) {
    return false;
  }
  if (!a.aggregate && !b.aggregate) {
    return true;
  }
  if (!a.aggregate || !b.aggregate) {
    return false;
  }
  return (
    a.aggregate.currentCount === b.aggregate.currentCount &&
    a.aggregate.latestCount === b.aggregate.latestCount &&
    a.aggregate.branchSerial === b.aggregate.branchSerial &&
    a.aggregate.rngState === b.aggregate.rngState
  );
}

export function estimateSnapshotBytes(value: unknown): number {
  if (!value || typeof value !== 'object') return 0;
  const visited = new Set<object>();

  const visit = (entry: unknown): number => {
    if (!entry || typeof entry !== 'object') return 0;
    if (visited.has(entry)) return 0;
    visited.add(entry);
    if (ArrayBuffer.isView(entry)) return entry.byteLength;
    if (entry instanceof ArrayBuffer) return entry.byteLength;
    if (Array.isArray(entry)) return entry.reduce((sum, item) => sum + visit(item), 0);
    return Object.values(entry).reduce((sum, item) => sum + visit(item), 0);
  };

  return visit(value);
}
