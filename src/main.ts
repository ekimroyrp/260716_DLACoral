import './style.css';

import { generateSeedPositions, GpuDlaSimulator } from './dla';
import { ActionHistory, estimateSnapshotBytes } from './history';
import { DlaRenderer, type RotationPhase } from './render';
import {
  applySettingsSnapshot,
  createAppSnapshot,
  createInitialAppState,
  type MutableAppState,
} from './state';
import type { AppSnapshot, DlaSnapshot } from './types';
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
  beforeAggregate?: Promise<DlaSnapshot>;
  heavy: boolean;
}

interface HistorySnapshot extends AppSnapshot {
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
let rendererDetail = -1;

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
  onExportGlb: () => runExport('Preparing GLB…', (activeRenderer) => activeRenderer.exportGlb()),
  onExportObj: () => runExport('Preparing OBJ…', (activeRenderer) => activeRenderer.exportObj()),
  onScreenshot: () => renderer?.exportScreenshot(),
  onTransactionStart: beginTransaction,
  onTransactionCommit: commitTransaction,
});

void initialize();

async function initialize(): Promise<void> {
  ui.setBusy('Initializing native WebGPU…');
  try {
    const activeRenderer = new DlaRenderer(canvas, {
      onModelRotationChange: handleModelRotation,
      onError: reportError,
    });
    renderer = activeRenderer;
    await activeRenderer.init();

    const activeSimulator = GpuDlaSimulator.fromDevice(activeRenderer.getWebGpuDevice());
    simulator = activeSimulator;
    enforceDeviceLimits(state);

    const targets = activeRenderer.prepareInstances(state.dla.targetParticles, state.dla.sphereDetail);
    rendererCapacity = targets.capacity;
    rendererDetail = state.dla.sphereDetail;
    status = await activeSimulator.initialize(state.dla, targets);

    activeRenderer.update(state.dla, state.display, renderState(status));
    activeRenderer.setModelRotation(state.dla.rotation);
    activeRenderer.frameCamera(status.maxRadiusSq, true);
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
  const previousDetail = state.dla.sphereDetail;
  Object.assign(state.dla, settings);
  enforceDeviceLimits(state);

  if (
    state.dla.targetParticles !== settings.targetParticles
    || state.dla.walkerPool !== settings.walkerPool
    || state.dla.seedRadius !== settings.seedRadius
  ) {
    ui.sync({ dla: state.dla });
  }

  renderer?.updateDlaSettings(state.dla);
  if (meta.source === 'rotation') {
    renderer?.setModelRotation(state.dla.rotation);
    return;
  }
  if (meta.phase !== 'commit') {
    return;
  }

  if (meta.requiresReset) {
    state.simulation.timeline = 0;
    state.simulation.latestTimeline = 0;
    ui.setTimeline(0, 0);
    schedule(async () => resetAggregate('Rebuilding seed…'));
    return;
  }

  if (state.dla.sphereDetail !== previousDetail) {
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
  schedule(async () => resetAggregate('Resetting aggregate…'));
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

function handleModelRotation(rotation: number, phase: RotationPhase): void {
  if (phase === 'begin') {
    ui.beginTransaction('Rotation');
  }
  state.dla.rotation = normalizeRotation(rotation);
  ui.setRotation(state.dla.rotation);
  renderer?.setModelRotation(state.dla.rotation);
  if (phase === 'end') {
    ui.commitTransaction('Rotation');
  }
}

async function resetAggregate(message: string): Promise<void> {
  const activeRenderer = requireRenderer();
  const activeSimulator = requireSimulator();
  ui.setBusy(message);
  try {
    enforceDeviceLimits(state);
    let targets = activeRenderer.getRenderTargets();
    if (!targets || targets.capacity < state.dla.targetParticles || rendererDetail !== state.dla.sphereDetail) {
      targets = activeRenderer.prepareInstances(state.dla.targetParticles, state.dla.sphereDetail);
      rendererCapacity = targets.capacity;
      rendererDetail = state.dla.sphereDetail;
    }
    const next = await activeSimulator.initialize(state.dla, targets);
    status = next;
    syncStatus(next);
    activeRenderer.frameCamera(next.maxRadiusSq, true);
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
  if (!force && existing && existing.capacity >= capacity && rendererDetail === state.dla.sphereDetail) {
    return;
  }
  ui.setBusy('Updating sphere geometry…');
  try {
    const targets = activeRenderer.prepareInstances(capacity, state.dla.sphereDetail);
    rendererCapacity = targets.capacity;
    rendererDetail = state.dla.sphereDetail;
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
  ui.setBusy('Growing GPU buffers…');
  try {
    const aggregate = await activeSimulator.snapshot();
    const targets = activeRenderer.prepareInstances(state.dla.targetParticles, state.dla.sphereDetail);
    rendererCapacity = targets.capacity;
    rendererDetail = state.dla.sphereDetail;
    const next = await activeSimulator.restore(state.dla, aggregate, targets);
    status = next;
    syncStatus(next);
  } finally {
    ui.setReady();
  }
}

function syncStatus(next: DlaStatus): void {
  state.simulation.timeline = next.attachedCount;
  state.simulation.latestTimeline = next.latestAttachedCount;
  ui.setTimeline(next.attachedCount, next.latestAttachedCount);
  ui.setParticleCount(next.currentCount, state.dla.targetParticles);
  const activeRenderer = renderer;
  if (activeRenderer) {
    activeRenderer.update(state.dla, state.display, renderState(next));
    activeRenderer.frameCamera(next.maxRadiusSq);
  }
}

function renderState(next: DlaStatus) {
  return {
    displayedCount: next.visibleCount,
    totalCount: next.currentCount,
    seedCount: next.seedCount,
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
    before: createAppSnapshot(state),
    heavy,
  };
  if (heavy && simulator) {
    transaction.beforeAggregate = serial(() => requireSimulator().snapshot());
  }
  pendingTransaction = transaction;
}

function commitTransaction(label: string): void {
  const transaction = pendingTransaction;
  if (!transaction || transaction.label !== label) {
    return;
  }
  pendingTransaction = null;
  const after: HistorySnapshot = createAppSnapshot(state);
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
    after.aggregate = await requireSimulator().snapshot();
    pushHistory(transaction.before, after);
  });
}

function transactionNeedsAggregate(label: string): boolean {
  if (label === 'Reset Simulation' || label === 'Seed' || label === 'Seed Shape' || label === 'Seed Radius') {
    return true;
  }
  return label === 'Start Simulation' && Boolean(status && status.currentCount < status.latestCount);
}

function pushHistory(before: HistorySnapshot, after: HistorySnapshot): void {
  if (snapshotsEquivalent(before, after)) {
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
    applySettingsSnapshot(state, snapshot);
    if (!snapshot.aggregate && !restoreTimeline) {
      state.simulation.timeline = liveTimeline;
      state.simulation.latestTimeline = liveLatestTimeline;
    }
    enforceDeviceLimits(state);

    const needsTargets =
      rendererCapacity < state.dla.targetParticles || rendererDetail !== state.dla.sphereDetail;
    let targets = activeRenderer.getRenderTargets();
    if (needsTargets || !targets) {
      targets = activeRenderer.prepareInstances(
        Math.max(state.dla.targetParticles, snapshot.aggregate?.latestCount ?? 1),
        state.dla.sphereDetail,
      );
      rendererCapacity = targets.capacity;
      rendererDetail = state.dla.sphereDetail;
    }

    let next: DlaStatus;
    if (snapshot.aggregate) {
      next = await activeSimulator.restore(state.dla, snapshot.aggregate, targets);
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
    activeRenderer.setModelRotation(state.dla.rotation);
    ui.sync(state);
    syncStatus(next);
  });
}

function runExport(message: string, action: (activeRenderer: DlaRenderer) => Promise<void>): void {
  schedule(async () => {
    ui.setBusy(message);
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
  targetState.dla.seedRadius = clampSeedRadius(
    targetState.dla.seedShape,
    targetState.dla.seedRadius,
    maxParticles,
  );
  const seedCount = generateSeedPositions(targetState.dla.seedShape, targetState.dla.seedRadius).length;
  targetState.dla.targetParticles = Math.min(
    maxParticles,
    Math.max(seedCount, Math.round(targetState.dla.targetParticles)),
  );
  targetState.dla.walkerPool = Math.min(
    limits.maxWalkers,
    Math.max(1, Math.round(targetState.dla.walkerPool)),
  );
  targetState.dla.sphereDetail = Math.min(2, Math.max(0, Math.round(targetState.dla.sphereDetail)));
  targetState.dla.stickChance = Math.min(1, Math.max(0.01, targetState.dla.stickChance));
  targetState.dla.stickNeighbors = Math.min(
    targetState.dla.attachmentNeighborhood,
    Math.max(1, Math.round(targetState.dla.stickNeighbors)),
  );
  targetState.dla.growthBatch = Math.max(1, Math.round(targetState.dla.growthBatch));
}

function clampSeedRadius(
  shape: MutableAppState['dla']['seedShape'],
  radius: number,
  maxParticles: number,
): number {
  const requested = Math.max(1, Math.round(Number.isFinite(radius) ? radius : 1));
  if (shape === 'point') {
    return requested;
  }
  const estimatedLimit = shape === 'sphere'
    ? Math.floor(Math.sqrt(maxParticles / (4 * Math.PI)))
    : Math.min(46_000, Math.floor(maxParticles / (2 * Math.PI)));
  let safeRadius = Math.min(requested, Math.max(1, estimatedLimit));
  while (safeRadius > 1 && generateSeedPositions(shape, safeRadius).length > maxParticles) {
    safeRadius -= 1;
  }
  return safeRadius;
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

function snapshotsEquivalent(a: AppSnapshot, b: AppSnapshot): boolean {
  const settingsA = JSON.stringify({ simulation: a.simulation, dla: a.dla, display: a.display });
  const settingsB = JSON.stringify({ simulation: b.simulation, dla: b.dla, display: b.display });
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

function normalizeRotation(value: number): number {
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
