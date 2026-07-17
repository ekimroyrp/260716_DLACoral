import './style.css';

import {
  countSeedPositions,
  GpuDlaSimulator,
  maxSeedRadiusForCapacity,
} from './dla';
import {
  ActionHistory,
  areHistorySnapshotsEquivalent,
  compressDlaSnapshot,
  decompressDlaSnapshot,
  estimateSnapshotBytes,
  type CompressedDlaSnapshot,
} from './history';
import {
  automaticParticleResolution,
  DlaRenderer,
  type SeedRotationPhase,
} from './render';
import {
  applySettingsSnapshot,
  createAppSnapshot,
  createInitialAppState,
  type MutableAppState,
} from './state';
import { attachmentNeighborhoodMaximum, type AppSnapshot } from './types';
import {
  createUiController,
  type DlaUiChangeMeta,
  type UiChangeMeta,
  type UiController,
} from './ui/controller';

type DlaStatus = ReturnType<GpuDlaSimulator['getStatus']>;

interface PendingTransaction {
  label: string;
  before: HistorySnapshot;
  beforeAggregate?: Promise<CompressedDlaSnapshot>;
  heavy: boolean;
}

interface HistorySnapshot extends Omit<AppSnapshot, 'aggregate'> {
  aggregate?: CompressedDlaSnapshot;
  actionLabel?: string;
}

const state = createInitialAppState();
const history = new ActionHistory<HistorySnapshot>();
const canvas = requireCanvas();

let ui: UiController;
let renderer: DlaRenderer | null = null;
let simulator: GpuDlaSimulator | null = null;
let status: DlaStatus | null = null;
let operationTail: Promise<void> = Promise.resolve();
let queuedOperations = 0;
let tickQueued = false;
let pendingSeek: number | null = null;
let seekLoopQueued = false;
let pendingTransaction: PendingTransaction | null = null;
let disposed = false;
let animationFrame = 0;
let rendererCapacity = 0;
let rendererResolution = -1;

ui = createUiController({
  onSimulationChange: handleSimulationChange,
  onDlaChange: handleDlaChange,
  onDisplayChange: handleDisplayChange,
  onStartPause: handleStartPause,
  onReset: handleReset,
  onTimelineInput: requestTimelineSeek,
  onTimelineCommit: requestTimelineSeek,
  onUndo: undo,
  onRedo: redo,
  onExportGlb: () => runExport((activeRenderer) => activeRenderer.exportGlb()),
  onExportObj: () => runExport((activeRenderer) => activeRenderer.exportObj()),
  onScreenshot: () => renderer?.exportScreenshot(),
  onTransactionStart: beginTransaction,
  onTransactionCommit: commitTransaction,
});

void initialize();

async function initialize(): Promise<void> {
  ui.setBusy();
  try {
    const activeRenderer = new DlaRenderer(canvas, {
      onSeedRotationChange: handleSeedRotation,
      onError: reportError,
    });
    renderer = activeRenderer;
    await activeRenderer.init();

    const activeSimulator = GpuDlaSimulator.fromDevice(activeRenderer.getWebGpuDevice());
    simulator = activeSimulator;
    enforceDeviceLimits(state);

    const targets = activeRenderer.prepareInstances(state.dla.targetParticles, state.dla.particleResolution);
    rendererCapacity = targets.capacity;
    rendererResolution = state.dla.particleResolution;
    status = await activeSimulator.initialize(state.dla, targets);

    activeRenderer.update(state.dla, state.display, renderState(status));
    activeRenderer.setSeedRotation(state.dla.seedRotation);
    syncStatus(status);
    ui.sync(state);
    ui.setReady();
    animationFrame = requestAnimationFrame(frame);
  } catch (error) {
    reportError(error);
  }
}

function frame(): void {
  if (disposed) {
    return;
  }

  renderer?.render();
  const activeStatus = status;
  if (
    state.simulation.running &&
    activeStatus &&
    activeStatus.currentCount < state.dla.targetParticles &&
    !tickQueued &&
    queuedOperations === 0
  ) {
    tickQueued = true;
    void serial(async () => {
      const activeSimulator = requireSimulator();
      const result = await activeSimulator.step(state.dla, state.simulation.rate);
      status = result;
      syncStatus(result);
      if (result.currentCount >= state.dla.targetParticles || result.overflowed) {
        state.simulation.running = false;
        ui.setRunning(false);
      }
    })
      .catch(() => undefined)
      .finally(() => {
        tickQueued = false;
      });
  }

  animationFrame = requestAnimationFrame(frame);
}

function handleSimulationChange(settings: MutableAppState['simulation'], _meta: UiChangeMeta): void {
  state.simulation.rate = Math.max(0.01, settings.rate);
}

function handleDlaChange(settings: MutableAppState['dla'], meta: DlaUiChangeMeta): void {
  const previousTarget = state.dla.targetParticles;
  const previousResolution = state.dla.particleResolution;
  Object.assign(state.dla, settings);
  enforceDeviceLimits(state);
  const visibleCount = status
    ? (state.dla.hideEnclosed ? status.visibleCount : status.currentCount)
    : 0;
  const resolutionReduced = reduceParticleResolution(visibleCount);

  if (
    state.dla.targetParticles !== settings.targetParticles
    || state.dla.walkerPool !== settings.walkerPool
    || state.dla.seedRadius !== settings.seedRadius
    || state.dla.particleSize !== settings.particleSize
    || resolutionReduced
  ) {
    ui.sync({ dla: state.dla });
  }

  renderer?.updateDlaSettings(state.dla);
  if (meta.source === 'seedRotation') {
    renderer?.setSeedRotation(state.dla.seedRotation);
    return;
  }
  if (meta.phase !== 'commit') {
    return;
  }

  if (meta.requiresReset) {
    state.simulation.timeline = 0;
    state.simulation.latestTimeline = 0;
    ui.setTimeline(0, 0);
    schedule(async () => resetAggregate());
    return;
  }

  if (state.dla.particleResolution !== previousResolution) {
    schedule(async () => rebindRendererTargets(true));
  } else if (state.dla.targetParticles > previousTarget && status && state.dla.targetParticles > status.particleCapacity) {
    schedule(async () => growParticleCapacity());
  } else if (meta.source === 'hideEnclosed') {
    schedule(async () => {
      const next = await requireSimulator().refreshRender(state.dla);
      status = next;
      syncStatus(next);
    });
  }

  if (status && status.currentCount >= state.dla.targetParticles) {
    state.simulation.running = false;
    ui.setRunning(false);
  }
}

function handleDisplayChange(settings: MutableAppState['display'], _meta: UiChangeMeta): void {
  Object.assign(state.display, settings);
  renderer?.updateDisplay(state.display);
}

function handleStartPause(running: boolean): void {
  if (running && status && status.currentCount >= state.dla.targetParticles) {
    state.simulation.running = false;
    ui.setRunning(false);
    return;
  }
  state.simulation.running = running;
  ui.setRunning(running);
  if (running && status && status.currentCount < status.latestCount) {
    schedule(async () => {
      const next = await requireSimulator().branch(state.simulation.timeline);
      status = next;
      syncStatus(next);
    });
  }
}

function handleReset(): void {
  state.simulation.timeline = 0;
  state.simulation.latestTimeline = 0;
  ui.setTimeline(0, 0);
  schedule(async () => resetAggregate());
}

function requestTimelineSeek(value: number): void {
  if (state.simulation.running || !simulator) {
    return;
  }
  const latest = status?.latestAttachedCount ?? state.simulation.latestTimeline;
  pendingSeek = Math.min(latest, Math.max(0, Math.round(value)));
  state.simulation.timeline = pendingSeek;
  if (seekLoopQueued) {
    return;
  }
  seekLoopQueued = true;
  schedule(async () => {
    try {
      while (pendingSeek !== null) {
        const nextValue = pendingSeek;
        pendingSeek = null;
        const next = await requireSimulator().seek(nextValue);
        status = next;
        syncStatus(next);
      }
    } finally {
      seekLoopQueued = false;
      if (pendingSeek !== null) {
        requestTimelineSeek(pendingSeek);
      }
    }
  });
}

function handleSeedRotation(rotation: number, phase: SeedRotationPhase): void {
  if (phase === 'begin') {
    ui.beginTransaction('Seed Rotation');
  }
  state.dla.seedRotation = normalizeSeedRotation(rotation);
  ui.setSeedRotation(state.dla.seedRotation);
  renderer?.setSeedRotation(state.dla.seedRotation);
  if (phase === 'end') {
    ui.commitTransaction('Seed Rotation');
  }
}

async function resetAggregate(): Promise<void> {
  const activeRenderer = requireRenderer();
  const activeSimulator = requireSimulator();
  ui.setBusy();
  try {
    enforceDeviceLimits(state);
    let targets = activeRenderer.getRenderTargets();
    if (!targets || targets.capacity < state.dla.targetParticles || rendererResolution !== state.dla.particleResolution) {
      targets = activeRenderer.prepareInstances(state.dla.targetParticles, state.dla.particleResolution);
      rendererCapacity = targets.capacity;
      rendererResolution = state.dla.particleResolution;
    }
    const next = await activeSimulator.initialize(state.dla, targets);
    status = next;
    syncStatus(next);
  } finally {
    ui.setReady();
  }
}

async function rebindRendererTargets(force: boolean): Promise<void> {
  const activeRenderer = requireRenderer();
  const activeSimulator = requireSimulator();
  const existing = activeRenderer.getRenderTargets();
  const capacity = Math.max(
    state.dla.targetParticles,
    status?.latestCount ?? 1,
    existing?.capacity ?? 1,
  );
  if (!force && existing && existing.capacity >= capacity && rendererResolution === state.dla.particleResolution) {
    return;
  }
  ui.setBusy();
  try {
    const targets = activeRenderer.prepareInstances(capacity, state.dla.particleResolution);
    rendererCapacity = targets.capacity;
    rendererResolution = state.dla.particleResolution;
    const next = await activeSimulator.rebindRenderTargets(targets);
    await activeSimulator.setVertexCount(targets.vertexCount);
    status = next;
    syncStatus(next);
  } finally {
    ui.setReady();
  }
}

async function growParticleCapacity(): Promise<void> {
  const activeRenderer = requireRenderer();
  const activeSimulator = requireSimulator();
  ui.setBusy();
  try {
    const aggregate = await activeSimulator.snapshot();
    const targets = activeRenderer.prepareInstances(state.dla.targetParticles, state.dla.particleResolution);
    rendererCapacity = targets.capacity;
    rendererResolution = state.dla.particleResolution;
    const next = await activeSimulator.restore(state.dla, aggregate, targets);
    status = next;
    syncStatus(next);
  } finally {
    ui.setReady();
  }
}

function syncStatus(next: DlaStatus): void {
  const resolutionReduced = reduceParticleResolution(next.visibleCount);
  state.simulation.timeline = next.attachedCount;
  state.simulation.latestTimeline = next.latestAttachedCount;
  ui.setTimeline(next.attachedCount, next.latestAttachedCount);
  ui.setParticleCount(next.visibleCount);
  if (resolutionReduced) {
    ui.sync({ dla: state.dla });
  }
  const activeRenderer = renderer;
  if (activeRenderer) {
    activeRenderer.update(state.dla, state.display, renderState(next));
  }
  if (resolutionReduced) {
    schedule(async () => rebindRendererTargets(true));
  }
}

function reduceParticleResolution(visibleCount: number): boolean {
  const nextResolution = automaticParticleResolution(
    visibleCount,
    state.dla.particleResolution,
  );
  if (nextResolution === state.dla.particleResolution) {
    return false;
  }
  state.dla.particleResolution = nextResolution;
  return true;
}

function renderState(next: DlaStatus) {
  return {
    displayedCount: next.visibleCount,
    totalCount: next.currentCount,
    seedCount: next.seedCount,
    newestVisibleBirth: next.newestVisibleBirth,
    maxRadiusSq: next.maxRadiusSq,
  };
}

function beginTransaction(label: string): void {
  if (pendingTransaction || disposed) {
    return;
  }
  const heavy = transactionNeedsAggregate(label);
  const transaction: PendingTransaction = {
    label,
    before: createHistorySnapshot(),
    heavy,
  };
  if (heavy && simulator) {
    transaction.beforeAggregate = serial(async () => compressDlaSnapshot(await requireSimulator().snapshot()));
  }
  pendingTransaction = transaction;
}

function commitTransaction(label: string): void {
  const transaction = pendingTransaction;
  if (!transaction || transaction.label !== label) {
    return;
  }
  pendingTransaction = null;
  const after = createHistorySnapshot();
  transaction.before.actionLabel = label;
  after.actionLabel = label;

  if (!transaction.heavy || !simulator) {
    pushHistory(transaction.before, after);
    return;
  }

  schedule(async () => {
    if (transaction.beforeAggregate) {
      transaction.before.aggregate = await transaction.beforeAggregate;
    }
    after.aggregate = compressDlaSnapshot(await requireSimulator().snapshot());
    pushHistory(transaction.before, after);
  });
}

function transactionNeedsAggregate(label: string): boolean {
  if (
    label === 'Reset Simulation'
    || label === 'Seed'
    || label === 'Seed Shape'
    || label === 'Seed Radius'
    || label === 'Particle Size'
  ) {
    return true;
  }
  return label === 'Start Simulation' && Boolean(status && status.currentCount < status.latestCount);
}

function pushHistory(before: HistorySnapshot, after: HistorySnapshot): void {
  if (areHistorySnapshotsEquivalent(before, after)) {
    return;
  }
  const bytes = estimateSnapshotBytes(before.aggregate) + estimateSnapshotBytes(after.aggregate);
  history.push(before, after, bytes);
}

function undo(): void {
  const snapshot = history.undo();
  if (snapshot) {
    restoreHistorySnapshot(snapshot);
  }
}

function redo(): void {
  const snapshot = history.redo();
  if (snapshot) {
    restoreHistorySnapshot(snapshot);
  }
}

function restoreHistorySnapshot(snapshot: HistorySnapshot): void {
  state.simulation.running = false;
  ui.setRunning(false);
  schedule(async () => {
    const activeRenderer = requireRenderer();
    const activeSimulator = requireSimulator();
    const liveTimeline = status?.attachedCount ?? state.simulation.timeline;
    const liveLatestTimeline = status?.latestAttachedCount ?? state.simulation.latestTimeline;
    const restoreTimeline = snapshot.actionLabel === 'Simulation Timeline';
    applySettingsSnapshot(state, {
      simulation: snapshot.simulation,
      dla: snapshot.dla,
      display: snapshot.display,
    });
    if (!snapshot.aggregate && !restoreTimeline) {
      state.simulation.timeline = liveTimeline;
      state.simulation.latestTimeline = liveLatestTimeline;
    }
    enforceDeviceLimits(state);

    const needsTargets =
      rendererCapacity < state.dla.targetParticles || rendererResolution !== state.dla.particleResolution;
    let targets = activeRenderer.getRenderTargets();
    if (needsTargets || !targets) {
      targets = activeRenderer.prepareInstances(
        Math.max(state.dla.targetParticles, snapshot.aggregate?.latestCount ?? 1),
        state.dla.particleResolution,
      );
      rendererCapacity = targets.capacity;
      rendererResolution = state.dla.particleResolution;
    }

    let next: DlaStatus;
    if (snapshot.aggregate) {
      next = await activeSimulator.restore(
        state.dla,
        decompressDlaSnapshot(snapshot.aggregate),
        targets,
      );
    } else if (state.dla.targetParticles > activeSimulator.getStatus().particleCapacity) {
      const aggregate = await activeSimulator.snapshot();
      next = await activeSimulator.restore(state.dla, aggregate, targets);
    } else {
      if (needsTargets) {
        await activeSimulator.rebindRenderTargets(targets);
      }
      if (restoreTimeline) {
        const latest = activeSimulator.getStatus().latestAttachedCount;
        next = await activeSimulator.seek(Math.min(latest, state.simulation.timeline));
      } else {
        next = activeSimulator.getStatus();
      }
      next = await activeSimulator.refreshRender(state.dla);
    }

    status = next;
    activeRenderer.updateDlaSettings(state.dla);
    activeRenderer.updateDisplay(state.display);
    activeRenderer.setSeedRotation(state.dla.seedRotation);
    ui.sync(state);
    syncStatus(next);
  });
}

function runExport(action: (activeRenderer: DlaRenderer) => Promise<void>): void {
  schedule(async () => {
    ui.setBusy();
    try {
      await action(requireRenderer());
    } finally {
      ui.setReady();
    }
  });
}

function enforceDeviceLimits(targetState: MutableAppState): void {
  const activeSimulator = simulator;
  const activeRenderer = renderer;
  if (!activeSimulator || !activeRenderer) {
    return;
  }
  const limits = activeSimulator.getLimits();
  const rawMaxParticles = Math.max(
    1,
    Math.min(limits.maxParticles, activeRenderer.getMaxSupportedCapacity()),
  );
  // Target Particles uses a 1,000-particle slider step. Keep a device-clamped
  // maximum on that lattice so the range and selectable numeric field show
  // the same exact value on devices whose byte limit is not step-aligned.
  const maxParticles = rawMaxParticles >= 1_000
    ? Math.max(1_000, Math.floor(rawMaxParticles / 1_000) * 1_000)
    : rawMaxParticles;
  targetState.dla.particleSize = Math.max(
    0.01,
    Number.isFinite(targetState.dla.particleSize) ? targetState.dla.particleSize : 1,
  );
  targetState.dla.seedRadius = clampSeedRadius(
    targetState.dla.seedShape,
    targetState.dla.seedRadius,
    maxParticles,
    limits.maxSeedRadius,
    targetState.dla.particleSize,
  );
  const seedCount = countSeedPositions(
    targetState.dla.seedShape,
    targetState.dla.seedRadius,
    targetState.dla.particleSize,
  );
  targetState.dla.targetParticles = Math.min(
    maxParticles,
    Math.max(seedCount, Math.round(targetState.dla.targetParticles)),
  );
  targetState.dla.adaptiveStickNeighbors = Boolean(targetState.dla.adaptiveStickNeighbors);
  targetState.dla.walkerPool = Math.min(
    limits.maxWalkers,
    Math.max(1, Math.round(targetState.dla.walkerPool)),
  );
  targetState.dla.particleScale = Math.max(0.01, targetState.dla.particleScale);
  targetState.dla.particleGap = Math.max(0, targetState.dla.particleGap);
  targetState.dla.particleResolution = Math.min(2, Math.max(0, Math.round(targetState.dla.particleResolution)));
  targetState.dla.stickChance = Math.min(1, Math.max(0.01, targetState.dla.stickChance));
  targetState.dla.stickNeighbors = Math.min(
    attachmentNeighborhoodMaximum(targetState.dla.attachmentNeighborhood),
    Math.max(1, Math.round(targetState.dla.stickNeighbors)),
  );
  targetState.dla.contactHits = Math.max(1, Math.round(targetState.dla.contactHits));
  targetState.dla.bootstrapParticles = Math.max(0, Math.round(targetState.dla.bootstrapParticles));
  targetState.dla.growthBatch = Math.max(1, Math.round(targetState.dla.growthBatch));
}

function clampSeedRadius(
  shape: MutableAppState['dla']['seedShape'],
  radius: number,
  maxParticles: number,
  latticeMaximum: number,
  particleSize: number,
): number {
  const requested = Math.max(1, Math.round(Number.isFinite(radius) ? radius : 1));
  if (shape === 'point') {
    return requested;
  }
  const latticeClamped = Math.min(
    requested,
    Math.max(1, Math.floor(latticeMaximum * Math.max(0.001, particleSize))),
  );
  if (countSeedPositions(shape, latticeClamped, particleSize) <= maxParticles) {
    return latticeClamped;
  }
  return maxSeedRadiusForCapacity(shape, maxParticles, latticeMaximum, particleSize);
}

function serial<T>(task: () => Promise<T> | T): Promise<T> {
  queuedOperations += 1;
  const result = operationTail.then(task);
  operationTail = result.then(
    () => {
      queuedOperations -= 1;
    },
    (error) => {
      queuedOperations -= 1;
      reportError(error);
    },
  );
  return result;
}

function schedule(task: () => Promise<void> | void): void {
  void serial(task).catch(() => undefined);
}

function createHistorySnapshot(): HistorySnapshot {
  const snapshot = createAppSnapshot(state);
  return {
    simulation: snapshot.simulation,
    dla: snapshot.dla,
    display: snapshot.display,
  };
}

function normalizeSeedRotation(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return ((value + 360) % 720 + 720) % 720 - 360;
}

function reportError(error: unknown): void {
  const resolved = error instanceof Error ? error : new Error(String(error));
  console.error(resolved);
  ui?.setError(resolved.message || 'The WebGPU application encountered an error.');
}

function requireCanvas(): HTMLCanvasElement {
  const element = document.getElementById('app-canvas');
  if (!(element instanceof HTMLCanvasElement)) {
    throw new Error('Missing #app-canvas.');
  }
  return element;
}

function requireRenderer(): DlaRenderer {
  if (!renderer) {
    throw new Error('The WebGPU renderer is not ready.');
  }
  return renderer;
}

function requireSimulator(): GpuDlaSimulator {
  if (!simulator) {
    throw new Error('The DLA simulator is not ready.');
  }
  return simulator;
}

function dispose(): void {
  if (disposed) {
    return;
  }
  disposed = true;
  cancelAnimationFrame(animationFrame);
  ui.dispose();
  simulator?.dispose();
  renderer?.dispose();
}

window.addEventListener('beforeunload', dispose, { once: true });
