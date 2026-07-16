import { MAX_HISTORY_ACTIONS, MAX_HISTORY_BYTES } from './types';

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
    while (
      this.undoEntries.length > this.maxActions ||
      (this.undoBytes > this.maxBytes && this.undoEntries.length > 1)
    ) {
      const removed = this.undoEntries.shift();
      if (removed) this.undoBytes -= removed.bytes;
    }
  }
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
