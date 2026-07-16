import type { DlaSettings, DlaSnapshot, Int3 } from '../types';
import { HASH_COORD_MAX, HASH_COORD_MIN, generateSeedPositions, nextPowerOfTwo } from './cpu';

const WORKGROUP_SIZE = 128;
const PARAM_WORDS = 32;
const COUNTER_WORDS = 8;
const HASH_ENTRY_BYTES = 16;
const WALKER_BYTES = 16;
const CANDIDATE_BYTES = 32;
const PARTICLE_BYTES = 32;
const MATRIX_BYTES = 64;
const BIRTH_BYTES = 16;
const INDIRECT_BYTES = 16;
const EMPTY_SLOT = 0xffff_ffff;
const DEFAULT_VERTEX_COUNT = 60;
const DEFAULT_WALKER_RESERVE = 131_072;
const MAX_HASH_PROBES = 256;
const MAX_EPOCHS_PER_STEP = 12;
const HASH_SLOTS_PER_PARTICLE = 8;

export interface GpuDlaInstanceTargets {
  instanceMatrix: GPUBuffer;
  instanceBirth: GPUBuffer;
  indirectArgs: GPUBuffer;
  capacity: number;
  vertexCount: number;
}

export interface GpuDlaLimits {
  maxParticles: number;
  maxWalkers: number;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
}

export interface DlaStatus {
  seedCount: number;
  currentCount: number;
  latestCount: number;
  attachedCount: number;
  latestAttachedCount: number;
  visibleCount: number;
  candidateCount: number;
  maxRadiusSq: number;
  hashCapacity: number;
  particleCapacity: number;
  branchSerial: number;
  overflowed: boolean;
  hashEntries: number;
  hashLoadFactor: number;
}

export interface DlaStepResult extends DlaStatus {
  attachedThisStep: number;
}

export interface GpuDlaSnapshot extends DlaSnapshot {
  walkerState: Int32Array;
  walkerCount: number;
  epoch: number;
  epochCredit: number;
}

interface Buffers {
  hash: GPUBuffer;
  walkers: GPUBuffer;
  candidates: GPUBuffer;
  particles: GPUBuffer;
  counters: GPUBuffer;
  statusReadback: GPUBuffer;
  params: GPUBuffer;
}

interface ActiveState {
  buffers: Buffers;
  targets: GpuDlaInstanceTargets;
  ownsTargets: boolean;
  settings: DlaSettings;
  hashCapacity: number;
  particleCapacity: number;
  walkerCapacity: number;
  seedCount: number;
  currentCount: number;
  latestCount: number;
  branchSerial: number;
  epoch: number;
  epochCredit: number;
  status: DlaStatus;
}

type PipelineName =
  | 'clearHash'
  | 'insertPrefix'
  | 'rebuildNeighbors'
  | 'buildFrontier'
  | 'beginCompact'
  | 'compactPrefix'
  | 'finishIndirect'
  | 'initWalkers'
  | 'clearCandidates'
  | 'advanceWalkers'
  | 'commitCandidates';

export class GpuDlaSimulator {
  private readonly device: GPUDevice;
  private readonly pipelines: Record<PipelineName, GPUComputePipeline>;
  private readonly bindGroupLayout: GPUBindGroupLayout;
  private readonly pipelineReady: Promise<void>;
  private active: ActiveState | null = null;
  private operation: Promise<void> = Promise.resolve();
  private lostMessage: string | null = null;
  private disposed = false;

  private constructor(
    device: GPUDevice,
    pipelines: Record<PipelineName, GPUComputePipeline>,
    bindGroupLayout: GPUBindGroupLayout,
    pipelineReady: Promise<void>,
  ) {
    this.device = device;
    this.pipelines = pipelines;
    this.bindGroupLayout = bindGroupLayout;
    this.pipelineReady = pipelineReady;
    void device.lost.then((info) => {
      this.lostMessage = info.message || `WebGPU device was lost (${info.reason}).`;
    });
  }

  static fromDevice(device: GPUDevice): GpuDlaSimulator {
    if (device.limits.maxStorageBuffersPerShaderStage < 8) {
      throw new Error('The WebGPU device must support at least 8 storage buffers per shader stage.');
    }
    const module = device.createShaderModule({ label: 'DLA compute module', code: COMPUTE_WGSL });
    const bindGroupLayout = device.createBindGroupLayout({
      label: 'DLA compute bindings',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const names: PipelineName[] = [
      'clearHash',
      'insertPrefix',
      'rebuildNeighbors',
      'buildFrontier',
      'beginCompact',
      'compactPrefix',
      'finishIndirect',
      'initWalkers',
      'clearCandidates',
      'advanceWalkers',
      'commitCandidates',
    ];
    const pipelines = Object.fromEntries(
      names.map((entryPoint) => [
        entryPoint,
        device.createComputePipeline({
          label: `DLA ${entryPoint}`,
          layout: pipelineLayout,
          compute: { module, entryPoint },
        }),
      ]),
    ) as Record<PipelineName, GPUComputePipeline>;
    const pipelineReady = module.getCompilationInfo().then((info) => {
      const errors = info.messages.filter((message) => message.type === 'error');
      if (errors.length > 0) {
        const detail = errors
          .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`)
          .join('\n');
        throw new Error(`DLA WGSL compilation failed:\n${detail}`);
      }
    });
    return new GpuDlaSimulator(device, pipelines, bindGroupLayout, pipelineReady);
  }

  static async create(): Promise<GpuDlaSimulator> {
    if (!('gpu' in navigator) || !navigator.gpu) {
      throw new Error('WebGPU is not available in this browser.');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('No WebGPU adapter was found.');
    }
    return GpuDlaSimulator.fromDevice(await adapter.requestDevice());
  }

  getLimits(): GpuDlaLimits {
    const storageLimit = Number(this.device.limits.maxStorageBufferBindingSize);
    const bufferLimit = Number(this.device.limits.maxBufferSize);
    const largestLimit = Math.min(storageLimit, bufferLimit);
    const dispatchItems = Number(this.device.limits.maxComputeWorkgroupsPerDimension) * WORKGROUP_SIZE;
    const largestHashCapacity = highestPowerOfTwo(Math.floor(largestLimit / HASH_ENTRY_BYTES));
    return {
      maxParticles: Math.max(
        1,
        Math.min(
          Math.floor(largestLimit / MATRIX_BYTES),
          Math.floor(largestLimit / (HASH_ENTRY_BYTES * HASH_SLOTS_PER_PARTICLE)),
          Math.floor(largestHashCapacity / HASH_SLOTS_PER_PARTICLE),
          dispatchItems,
        ),
      ),
      maxWalkers: Math.max(
        1,
        Math.min(Math.floor(largestLimit / Math.max(WALKER_BYTES, CANDIDATE_BYTES)), dispatchItems),
      ),
      maxStorageBufferBindingSize: storageLimit,
      maxBufferSize: bufferLimit,
    };
  }

  initialize(settings: DlaSettings, targets?: GpuDlaInstanceTargets): Promise<DlaStatus> {
    return this.enqueue(async () => {
      await this.pipelineReady;
      this.assertReady();
      if (settings.seedShape !== 'point' && Math.floor(settings.seedRadius) > HASH_COORD_MAX) {
        throw new Error(`Seed Radius cannot exceed ${HASH_COORD_MAX} on the compact WebGPU lattice.`);
      }
      this.disposeActive();
      const seedPositions = generateSeedPositions(settings.seedShape, settings.seedRadius);
      const particleCapacity = this.resolveParticleCapacity(settings, targets, seedPositions.length);
      const hashCapacity = this.resolveHashCapacity(particleCapacity);
      const walkerCapacity = this.resolveWalkerCapacity(settings.walkerPool);
      const renderTargets = targets ?? this.createInternalTargets(particleCapacity);
      const buffers = this.createBuffers(hashCapacity, particleCapacity, walkerCapacity);
      const normalized = normalizeSettings(settings, particleCapacity, walkerCapacity);
      const seedCount = Math.min(seedPositions.length, particleCapacity);
      const state: ActiveState = {
        buffers,
        targets: renderTargets,
        ownsTargets: !targets,
        settings: normalized,
        hashCapacity,
        particleCapacity,
        walkerCapacity,
        seedCount,
        currentCount: seedCount,
        latestCount: seedCount,
        branchSerial: 0,
        epoch: 0,
        epochCredit: 0,
        status: emptyStatus(seedCount, hashCapacity, particleCapacity),
      };
      this.active = state;
      this.writeParticlePositions(buffers.particles, seedPositions.slice(0, seedCount));
      await this.rebuildPrefix(state, seedCount, true);
      state.status = await this.readStatus(state);
      return state.status;
    });
  }

  restore(settings: DlaSettings, snapshot: DlaSnapshot, targets?: GpuDlaInstanceTargets): Promise<DlaStatus> {
    return this.enqueue(async () => {
      await this.pipelineReady;
      this.assertReady();
      const availablePositions = Math.floor(snapshot.positions.length / 3);
      const latestCount = Math.max(1, Math.min(snapshot.latestCount, availablePositions));
      for (let index = 0; index < latestCount * 3; index += 1) {
        const coordinate = snapshot.positions[index] ?? 0;
        if (coordinate < HASH_COORD_MIN || coordinate > HASH_COORD_MAX) {
          throw new Error(`Snapshot coordinate ${coordinate} exceeds the compact WebGPU lattice range.`);
        }
      }
      this.disposeActive();
      const seedCount = Math.max(1, Math.min(snapshot.seedCount, latestCount));
      const particleCapacity = this.resolveParticleCapacity(settings, targets, latestCount);
      const hashCapacity = this.resolveHashCapacity(particleCapacity);
      const walkerCapacity = this.resolveWalkerCapacity(settings.walkerPool);
      const renderTargets = targets ?? this.createInternalTargets(particleCapacity);
      const buffers = this.createBuffers(hashCapacity, particleCapacity, walkerCapacity);
      const normalized = normalizeSettings(settings, particleCapacity, walkerCapacity);
      const restoredCurrentCount = Math.max(seedCount, Math.min(snapshot.currentCount, latestCount));
      const state: ActiveState = {
        buffers,
        targets: renderTargets,
        ownsTargets: !targets,
        settings: normalized,
        hashCapacity,
        particleCapacity,
        walkerCapacity,
        seedCount,
        currentCount: restoredCurrentCount,
        latestCount,
        branchSerial: snapshot.branchSerial >>> 0,
        epoch: isGpuSnapshot(snapshot) ? snapshot.epoch >>> 0 : 0,
        epochCredit: isGpuSnapshot(snapshot) ? Math.max(0, snapshot.epochCredit ?? 0) : 0,
        status: emptyStatus(seedCount, hashCapacity, particleCapacity),
      };
      this.active = state;
      this.writeSnapshotParticles(buffers.particles, snapshot, latestCount);
      await this.rebuildPrefix(state, latestCount, true);
      if (isGpuSnapshot(snapshot)) {
        const restoredWalkers = Math.min(snapshot.walkerCount, state.settings.walkerPool, state.walkerCapacity);
        const availableWalkers = Math.floor(snapshot.walkerState.byteLength / WALKER_BYTES);
        const copyCount = Math.min(restoredWalkers, availableWalkers);
        if (copyCount > 0) {
          this.device.queue.writeBuffer(
            state.buffers.walkers,
            0,
            snapshot.walkerState,
            0,
            copyCount * 4,
          );
        }
      }
      if (restoredCurrentCount !== latestCount) {
        await this.compactPrefix(state, restoredCurrentCount);
        state.currentCount = restoredCurrentCount;
      }
      state.status = await this.readStatus(state);
      return state.status;
    });
  }

  step(settings: DlaSettings, simulationRate = 1): Promise<DlaStepResult> {
    return this.enqueue(async () => {
      const state = this.requireActive();
      await this.ensureWalkerCapacity(state, settings.walkerPool);
      state.settings = normalizeSettings(settings, state.particleCapacity, state.walkerCapacity);
      if (state.currentCount < state.latestCount) {
        state.branchSerial = (state.branchSerial + 1) >>> 0;
        await this.rebuildPrefix(state, state.currentCount, true);
        state.latestCount = state.currentCount;
      }
      if (state.latestCount >= state.settings.targetParticles) {
        const status = await this.readStatus(state);
        state.status = status;
        return { ...status, attachedThisStep: 0 };
      }

      const aggregateFactor = Math.min(
        4,
        Math.max(1, 1 + Math.floor(Math.log2(Math.max(1, state.latestCount / 10_000)) / 2)),
      );
      state.epochCredit += Math.max(0.01, simulationRate) * aggregateFactor;
      const epochs = Math.min(MAX_EPOCHS_PER_STEP, Math.floor(state.epochCredit));
      if (epochs < 1) {
        return { ...state.status, attachedThisStep: 0 };
      }
      state.epochCredit -= epochs;
      state.epoch = (state.epoch + epochs) >>> 0;
      const previousCount = state.latestCount;
      this.writeParams(state, state.latestCount, 16);
      const bindGroup = this.createBindGroup(state);
      const epochPasses: Array<[PipelineName, number]> = [];
      for (let epoch = 0; epoch < epochs; epoch += 1) {
        epochPasses.push(
          ['clearCandidates', 1],
          ['advanceWalkers', workgroups(state.settings.walkerPool)],
          ['commitCandidates', 1],
        );
      }
      this.submitPasses(state, bindGroup, epochPasses);
      const status = await this.readStatus(state);
      state.latestCount = status.latestCount;
      state.currentCount = status.currentCount;
      state.status = status;
      return { ...status, attachedThisStep: Math.max(0, status.latestCount - previousCount) };
    });
  }

  seek(attachedCount: number): Promise<DlaStatus> {
    return this.enqueue(async () => {
      const state = this.requireActive();
      const prefix = state.seedCount + clampInteger(attachedCount, 0, state.latestCount - state.seedCount);
      await this.compactPrefix(state, prefix);
      state.currentCount = prefix;
      state.status = await this.readStatus(state);
      return state.status;
    });
  }

  branch(attachedCount: number): Promise<DlaStatus> {
    return this.enqueue(async () => {
      const state = this.requireActive();
      const prefix = state.seedCount + clampInteger(attachedCount, 0, state.latestCount - state.seedCount);
      state.branchSerial = (state.branchSerial + 1) >>> 0;
      await this.rebuildPrefix(state, prefix, true);
      state.currentCount = prefix;
      state.latestCount = prefix;
      state.status = await this.readStatus(state);
      return state.status;
    });
  }

  refreshRender(settings: DlaSettings): Promise<DlaStatus> {
    return this.enqueue(async () => {
      const state = this.requireActive();
      state.settings = normalizeSettings(settings, state.particleCapacity, state.walkerCapacity);
      await this.compactPrefix(state, state.currentCount);
      state.status = await this.readStatus(state);
      return state.status;
    });
  }

  snapshot(): Promise<GpuDlaSnapshot> {
    return this.enqueue(async () => {
      const state = this.requireActive();
      const count = state.latestCount;
      const [raw, walkerRaw] = await Promise.all([
        this.readBuffer(state.buffers.particles, count * PARTICLE_BYTES),
        this.readBuffer(state.buffers.walkers, state.settings.walkerPool * WALKER_BYTES),
      ]);
      const view = new DataView(raw);
      const positions = new Int32Array(count * 3);
      const enclosed = new Uint8Array(count);
      for (let index = 0; index < count; index += 1) {
        const offset = index * PARTICLE_BYTES;
        positions[index * 3] = view.getInt32(offset, true);
        positions[index * 3 + 1] = view.getInt32(offset + 4, true);
        positions[index * 3 + 2] = view.getInt32(offset + 8, true);
        const enclosedAt = view.getUint32(offset + 16, true);
        enclosed[index] = enclosedAt !== EMPTY_SLOT && enclosedAt < count ? 1 : 0;
      }
      return {
        positions,
        enclosed,
        seedCount: state.seedCount,
        currentCount: state.currentCount,
        latestCount: state.latestCount,
        maxRadiusSq: state.status.maxRadiusSq,
        rngState: new Uint32Array(walkerRaw)[3] ?? (state.settings.seed >>> 0),
        branchSerial: state.branchSerial,
        walkerState: new Int32Array(walkerRaw),
        walkerCount: state.settings.walkerPool,
        epoch: state.epoch,
        epochCredit: state.epochCredit,
      };
    });
  }

  getStatus(): DlaStatus {
    return { ...this.requireActive().status };
  }

  getRenderTargets(): GpuDlaInstanceTargets {
    return this.requireActive().targets;
  }

  setVertexCount(vertexCount: number): Promise<DlaStatus> {
    return this.enqueue(async () => {
      const state = this.requireActive();
      state.targets.vertexCount = Math.max(0, Math.floor(vertexCount));
      await this.compactPrefix(state, state.currentCount);
      state.status = await this.readStatus(state);
      return state.status;
    });
  }

  rebindRenderTargets(targets: GpuDlaInstanceTargets): Promise<DlaStatus> {
    return this.enqueue(async () => {
      const state = this.requireActive();
      if (targets.capacity < state.currentCount || targets.capacity < state.particleCapacity) {
        throw new Error(
          `Replacement render targets need capacity ${state.particleCapacity}; received ${targets.capacity}.`,
        );
      }
      if (state.ownsTargets) {
        state.targets.instanceMatrix.destroy();
        state.targets.instanceBirth.destroy();
        state.targets.indirectArgs.destroy();
      }
      state.targets = targets;
      state.ownsTargets = false;
      await this.compactPrefix(state, state.currentCount);
      state.status = await this.readStatus(state);
      return state.status;
    });
  }

  dispose(): void {
    this.disposed = true;
    this.disposeActive();
  }

  private async rebuildPrefix(state: ActiveState, prefixCount: number, initializeWalkers: boolean): Promise<void> {
    state.currentCount = prefixCount;
    state.latestCount = prefixCount;
    this.writeParams(state, prefixCount, 16);
    const bindGroup = this.createBindGroup(state);
    const passes: Array<[PipelineName, number]> = [
      [
        'clearHash',
        Math.min(workgroups(state.hashCapacity), Number(this.device.limits.maxComputeWorkgroupsPerDimension)),
      ],
      ['insertPrefix', workgroups(prefixCount)],
      ['rebuildNeighbors', workgroups(prefixCount)],
      ['buildFrontier', workgroups(prefixCount)],
      ['beginCompact', 1],
      ['compactPrefix', workgroups(prefixCount)],
      ['finishIndirect', 1],
    ];
    if (initializeWalkers) {
      passes.push(['initWalkers', workgroups(state.settings.walkerPool)]);
    }
    this.submitPasses(state, bindGroup, passes);
    await this.device.queue.onSubmittedWorkDone();
  }

  private async compactPrefix(state: ActiveState, prefixCount: number): Promise<void> {
    this.writeParams(state, prefixCount, 16);
    const bindGroup = this.createBindGroup(state);
    this.submitPasses(state, bindGroup, [
      ['beginCompact', 1],
      ['compactPrefix', workgroups(prefixCount)],
      ['finishIndirect', 1],
    ]);
    await this.device.queue.onSubmittedWorkDone();
  }

  private writeParams(state: ActiveState, prefixCount: number, stepsPerWalker: number): void {
    const values = new ArrayBuffer(PARAM_WORDS * Uint32Array.BYTES_PER_ELEMENT);
    const u32 = new Uint32Array(values);
    const f32 = new Float32Array(values);
    const settings = state.settings;
    u32[0] = state.hashCapacity - 1;
    u32[1] = state.hashCapacity;
    u32[2] = state.particleCapacity;
    u32[3] = settings.walkerPool;
    u32[4] = state.seedCount;
    u32[5] = prefixCount;
    u32[6] = settings.attachmentNeighborhood;
    u32[7] = settings.stickNeighbors;
    u32[8] = Math.round(settings.stickChance * 1_000_000);
    u32[9] = settings.launchPadding;
    u32[10] = settings.killPadding;
    u32[11] = settings.growthBatch;
    u32[12] = stepsPerWalker;
    u32[13] = settings.seed >>> 0;
    u32[14] = state.branchSerial;
    u32[15] = state.targets.vertexCount;
    u32[16] = settings.hideEnclosed ? 1 : 0;
    u32[17] = settings.targetParticles;
    u32[18] = state.epoch;
    f32[19] = settings.sphereScale;
    f32[20] = settings.sphereGap;
    u32[21] = MAX_HASH_PROBES;
    u32[22] = Number(this.device.limits.maxComputeWorkgroupsPerDimension) * WORKGROUP_SIZE;
    this.device.queue.writeBuffer(state.buffers.params, 0, values);
  }

  private createBindGroup(state: ActiveState): GPUBindGroup {
    return this.device.createBindGroup({
      label: 'DLA active compute group',
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: state.buffers.hash } },
        { binding: 1, resource: { buffer: state.buffers.walkers } },
        { binding: 2, resource: { buffer: state.buffers.candidates } },
        { binding: 3, resource: { buffer: state.buffers.particles } },
        { binding: 4, resource: { buffer: state.targets.instanceMatrix } },
        { binding: 5, resource: { buffer: state.targets.instanceBirth } },
        { binding: 6, resource: { buffer: state.targets.indirectArgs } },
        { binding: 7, resource: { buffer: state.buffers.counters } },
        { binding: 8, resource: { buffer: state.buffers.params } },
      ],
    });
  }

  private submitPasses(
    state: ActiveState,
    bindGroup: GPUBindGroup,
    passes: ReadonlyArray<readonly [PipelineName, number]>,
  ): void {
    const encoder = this.device.createCommandEncoder({ label: 'DLA compute commands' });
    for (const [name, count] of passes) {
      if (count > this.device.limits.maxComputeWorkgroupsPerDimension) {
        throw new Error(
          `DLA ${name} needs ${count} workgroups, above this device's ${this.device.limits.maxComputeWorkgroupsPerDimension} limit.`,
        );
      }
      const pass = encoder.beginComputePass({ label: `DLA ${name}` });
      pass.setPipeline(this.pipelines[name]);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.max(1, count));
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
    void state;
  }

  private async readStatus(state: ActiveState): Promise<DlaStatus> {
    const byteLength = COUNTER_WORDS * Uint32Array.BYTES_PER_ELEMENT;
    const encoder = this.device.createCommandEncoder({ label: 'DLA counter readback' });
    encoder.copyBufferToBuffer(state.buffers.counters, 0, state.buffers.statusReadback, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);
    await state.buffers.statusReadback.mapAsync(GPUMapMode.READ, 0, byteLength);
    const raw = state.buffers.statusReadback.getMappedRange(0, byteLength).slice(0);
    state.buffers.statusReadback.unmap();
    const counters = new Uint32Array(raw);
    const latestCount = counters[0] ?? state.latestCount;
    const currentCount = state.currentCount < state.latestCount ? state.currentCount : latestCount;
    const status: DlaStatus = {
      seedCount: state.seedCount,
      currentCount,
      latestCount,
      attachedCount: Math.max(0, currentCount - state.seedCount),
      latestAttachedCount: Math.max(0, latestCount - state.seedCount),
      visibleCount: counters[1] ?? 0,
      candidateCount: counters[3] ?? 0,
      maxRadiusSq: counters[2] ?? 0,
      hashCapacity: state.hashCapacity,
      particleCapacity: state.particleCapacity,
      branchSerial: state.branchSerial,
      overflowed: (counters[6] ?? 0) !== 0,
      hashEntries: counters[7] ?? 0,
      hashLoadFactor: (counters[7] ?? 0) / state.hashCapacity,
    };
    return status;
  }

  private createBuffers(hashCapacity: number, particleCapacity: number, walkerCapacity: number): Buffers {
    return {
      hash: this.createBuffer(hashCapacity * HASH_ENTRY_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
      walkers: this.createBuffer(
        walkerCapacity * WALKER_BYTES,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      ),
      candidates: this.createBuffer(walkerCapacity * CANDIDATE_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
      particles: this.createBuffer(
        particleCapacity * PARTICLE_BYTES,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      ),
      counters: this.createBuffer(
        COUNTER_WORDS * Uint32Array.BYTES_PER_ELEMENT,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      ),
      statusReadback: this.createBuffer(
        COUNTER_WORDS * Uint32Array.BYTES_PER_ELEMENT,
        GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      ),
      params: this.createBuffer(PARAM_WORDS * Uint32Array.BYTES_PER_ELEMENT, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
    };
  }

  private createInternalTargets(capacity: number): GpuDlaInstanceTargets {
    return {
      instanceMatrix: this.createBuffer(
        capacity * MATRIX_BYTES,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      ),
      instanceBirth: this.createBuffer(
        capacity * BIRTH_BYTES,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      ),
      indirectArgs: this.createBuffer(
        INDIRECT_BYTES,
        GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      ),
      capacity,
      vertexCount: DEFAULT_VERTEX_COUNT,
    };
  }

  private createBuffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const aligned = Math.max(16, Math.ceil(size / 16) * 16);
    const limits = this.getLimits();
    if (aligned > limits.maxBufferSize) {
      throw new Error(`Requested WebGPU buffer (${aligned} bytes) exceeds the device maxBufferSize.`);
    }
    return this.device.createBuffer({ size: aligned, usage });
  }

  private writeParticlePositions(buffer: GPUBuffer, positions: readonly Int3[]): void {
    const bytes = new ArrayBuffer(Math.max(1, positions.length) * PARTICLE_BYTES);
    const view = new DataView(bytes);
    positions.forEach((position, index) => {
      const offset = index * PARTICLE_BYTES;
      view.setInt32(offset, position.x, true);
      view.setInt32(offset + 4, position.y, true);
      view.setInt32(offset + 8, position.z, true);
      view.setUint32(offset + 16, EMPTY_SLOT, true);
      view.setUint32(offset + 20, EMPTY_SLOT, true);
    });
    this.device.queue.writeBuffer(buffer, 0, bytes, 0, positions.length * PARTICLE_BYTES);
  }

  private writeSnapshotParticles(buffer: GPUBuffer, snapshot: DlaSnapshot, count: number): void {
    const bytes = new ArrayBuffer(count * PARTICLE_BYTES);
    const view = new DataView(bytes);
    for (let index = 0; index < count; index += 1) {
      const offset = index * PARTICLE_BYTES;
      view.setInt32(offset, snapshot.positions[index * 3] ?? 0, true);
      view.setInt32(offset + 4, snapshot.positions[index * 3 + 1] ?? 0, true);
      view.setInt32(offset + 8, snapshot.positions[index * 3 + 2] ?? 0, true);
      view.setUint32(offset + 16, snapshot.enclosed[index] ? Math.max(0, count - 1) : EMPTY_SLOT, true);
      view.setUint32(offset + 20, EMPTY_SLOT, true);
    }
    this.device.queue.writeBuffer(buffer, 0, bytes);
  }

  private async readBuffer(source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    if (byteLength <= 0) {
      return new ArrayBuffer(0);
    }
    const aligned = Math.ceil(byteLength / 4) * 4;
    const staging = this.device.createBuffer({
      size: aligned,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, staging, 0, aligned);
    this.device.queue.submit([encoder.finish()]);
    try {
      await staging.mapAsync(GPUMapMode.READ, 0, aligned);
      return staging.getMappedRange(0, byteLength).slice(0);
    } finally {
      if (staging.mapState === 'mapped') {
        staging.unmap();
      }
      staging.destroy();
    }
  }

  private resolveParticleCapacity(settings: DlaSettings, targets: GpuDlaInstanceTargets | undefined, minimum: number): number {
    const deviceMaximum = this.getLimits().maxParticles;
    const requested = Math.max(minimum, Math.floor(settings.targetParticles));
    const targetMaximum = targets?.capacity ?? deviceMaximum;
    const capacity = Math.min(requested, deviceMaximum, targetMaximum);
    if (capacity < minimum) {
      throw new Error(`The selected seed requires ${minimum} particles, but the renderer/device capacity is ${capacity}.`);
    }
    return Math.max(1, capacity);
  }

  private resolveHashCapacity(particleCapacity: number): number {
    const desired = nextPowerOfTwo(Math.max(16, particleCapacity * HASH_SLOTS_PER_PARTICLE));
    const storageLimit = this.getLimits().maxStorageBufferBindingSize;
    if (desired * HASH_ENTRY_BYTES > storageLimit) {
      throw new Error('The requested particle target cannot fit a half-full sparse hash on this WebGPU device.');
    }
    return desired;
  }

  private resolveWalkerCapacity(requested: number): number {
    return clampInteger(Math.max(requested, DEFAULT_WALKER_RESERVE), 1, this.getLimits().maxWalkers);
  }

  private async ensureWalkerCapacity(state: ActiveState, requested: number): Promise<void> {
    const desired = clampInteger(requested, 1, this.getLimits().maxWalkers);
    if (desired <= state.walkerCapacity) {
      return;
    }
    const nextCapacity = Math.min(this.getLimits().maxWalkers, nextPowerOfTwo(desired));
    const nextWalkers = this.createBuffer(
      nextCapacity * WALKER_BYTES,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
    const previousWalkers = state.buffers.walkers;
    const previousCandidates = state.buffers.candidates;
    const previousCapacity = state.walkerCapacity;
    let nextCandidates: GPUBuffer | null = null;
    try {
      nextCandidates = this.createBuffer(
        nextCapacity * CANDIDATE_BYTES,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      );
      state.buffers.walkers = nextWalkers;
      state.buffers.candidates = nextCandidates;
      state.walkerCapacity = nextCapacity;
      state.settings = normalizeSettings({ ...state.settings, walkerPool: desired }, state.particleCapacity, nextCapacity);
      this.writeParams(state, state.currentCount, 16);
      const bindGroup = this.createBindGroup(state);
      this.submitPasses(state, bindGroup, [['initWalkers', workgroups(desired)]]);
      await this.device.queue.onSubmittedWorkDone();
      previousWalkers.destroy();
      previousCandidates.destroy();
    } catch (error) {
      state.buffers.walkers = previousWalkers;
      state.buffers.candidates = previousCandidates;
      state.walkerCapacity = previousCapacity;
      nextWalkers.destroy();
      nextCandidates?.destroy();
      throw error;
    }
  }

  private assertReady(): void {
    if (this.disposed) {
      throw new Error('GpuDlaSimulator has been disposed.');
    }
    if (this.lostMessage) {
      throw new Error(this.lostMessage);
    }
  }

  private requireActive(): ActiveState {
    this.assertReady();
    if (!this.active) {
      throw new Error('Initialize GpuDlaSimulator before using it.');
    }
    return this.active;
  }

  private disposeActive(): void {
    if (!this.active) {
      return;
    }
    Object.values(this.active.buffers).forEach((buffer) => buffer.destroy());
    if (this.active.ownsTargets) {
      this.active.targets.instanceMatrix.destroy();
      this.active.targets.instanceBirth.destroy();
      this.active.targets.indirectArgs.destroy();
    }
    this.active = null;
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.operation.then(operation, operation);
    this.operation = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }
}

function normalizeSettings(settings: DlaSettings, particleCapacity: number, walkerCapacity: number): DlaSettings {
  const neighborhood = settings.attachmentNeighborhood === 6 || settings.attachmentNeighborhood === 18 ? settings.attachmentNeighborhood : 26;
  return {
    ...settings,
    targetParticles: clampInteger(settings.targetParticles, 1, particleCapacity),
    attachmentNeighborhood: neighborhood,
    stickNeighbors: clampInteger(settings.stickNeighbors, 1, neighborhood),
    stickChance: Math.max(0, Math.min(1, settings.stickChance)),
    launchPadding: clampInteger(settings.launchPadding, 1, 1024),
    killPadding: clampInteger(settings.killPadding, 1, 4096),
    growthBatch: clampInteger(settings.growthBatch, 1, 65_536),
    walkerPool: clampInteger(settings.walkerPool, 1, walkerCapacity),
    sphereScale: Math.max(0.001, settings.sphereScale),
    sphereGap: Math.max(0, settings.sphereGap),
  };
}

function emptyStatus(seedCount: number, hashCapacity: number, particleCapacity: number): DlaStatus {
  return {
    seedCount,
    currentCount: seedCount,
    latestCount: seedCount,
    attachedCount: 0,
    latestAttachedCount: 0,
    visibleCount: seedCount,
    candidateCount: 0,
    maxRadiusSq: 0,
    hashCapacity,
    particleCapacity,
    branchSerial: 0,
    overflowed: false,
    hashEntries: seedCount,
    hashLoadFactor: seedCount / hashCapacity,
  };
}

function clampInteger(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, Math.floor(Number.isFinite(value) ? value : minimum)));
}

function workgroups(items: number): number {
  return Math.max(1, Math.ceil(items / WORKGROUP_SIZE));
}

function highestPowerOfTwo(value: number): number {
  if (!Number.isFinite(value) || value < 1) {
    return 1;
  }
  return 2 ** Math.floor(Math.log2(value));
}

function isGpuSnapshot(snapshot: DlaSnapshot): snapshot is GpuDlaSnapshot {
  const candidate = snapshot as Partial<GpuDlaSnapshot>;
  return candidate.walkerState instanceof Int32Array && Number.isFinite(candidate.walkerCount) && Number.isFinite(candidate.epoch);
}

const COMPUTE_WGSL = /* wgsl */ `
const EMPTY: u32 = 0u;
const OCCUPIED: u32 = 1u;
const FRONTIER: u32 = 2u;
const INVALID: u32 = 0xffffffffu;
const HASH_COORD_BIAS: i32 = 512;
const HASH_COORD_MAX: i32 = 511;
const META_NEIGHBOR_MASK: u32 = 0x1fffu;
const META_STATE_SHIFT: u32 = 30u;
const NEIGHBOR_OFFSETS: array<vec3<i32>, 26> = array<vec3<i32>, 26>(
  vec3<i32>(-1, -1, -1), vec3<i32>( 0, -1, -1), vec3<i32>( 1, -1, -1),
  vec3<i32>(-1,  0, -1), vec3<i32>( 0,  0, -1), vec3<i32>( 1,  0, -1),
  vec3<i32>(-1,  1, -1), vec3<i32>( 0,  1, -1), vec3<i32>( 1,  1, -1),
  vec3<i32>(-1, -1,  0), vec3<i32>( 0, -1,  0), vec3<i32>( 1, -1,  0),
  vec3<i32>(-1,  0,  0),                              vec3<i32>( 1,  0,  0),
  vec3<i32>(-1,  1,  0), vec3<i32>( 0,  1,  0), vec3<i32>( 1,  1,  0),
  vec3<i32>(-1, -1,  1), vec3<i32>( 0, -1,  1), vec3<i32>( 1, -1,  1),
  vec3<i32>(-1,  0,  1), vec3<i32>( 0,  0,  1), vec3<i32>( 1,  0,  1),
  vec3<i32>(-1,  1,  1), vec3<i32>( 0,  1,  1), vec3<i32>( 1,  1,  1)
);

struct HashEntry {
  key: atomic<u32>,
  birth: atomic<u32>,
  packedData: atomic<u32>,
  pad: u32,
};

struct Walker {
  data: vec4<i32>,
};

struct Candidate {
  position: vec4<i32>,
  valid: u32,
  neighborCount: u32,
  rank: u32,
  pad: u32,
};

struct Particle {
  position: vec4<i32>,
  enclosedAt: u32,
  slot: u32,
  neighborCount: u32,
  pad: u32,
};

struct Counters {
  particleCount: atomic<u32>,
  visibleCount: atomic<u32>,
  maxRadiusSq: atomic<u32>,
  candidateCount: atomic<u32>,
  epoch: atomic<u32>,
  attachedThisStep: atomic<u32>,
  overflow: atomic<u32>,
  hashEntries: atomic<u32>,
};

struct Params {
  hashMask: u32,
  hashCapacity: u32,
  particleCapacity: u32,
  walkerCount: u32,
  seedCount: u32,
  prefixCount: u32,
  neighborhood: u32,
  stickNeighbors: u32,
  stickChance: u32,
  launchPadding: u32,
  killPadding: u32,
  growthBatch: u32,
  stepsPerWalker: u32,
  seed: u32,
  branchSerial: u32,
  vertexCount: u32,
  hideEnclosed: u32,
  targetParticles: u32,
  epoch: u32,
  sphereScale: f32,
  sphereGap: f32,
  maxHashProbes: u32,
  clearThreads: u32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
  pad4: u32,
  pad5: u32,
  pad6: u32,
  pad7: u32,
  pad8: u32,
  pad9: u32,
};

@group(0) @binding(0) var<storage, read_write> hashTable: array<HashEntry>;
@group(0) @binding(1) var<storage, read_write> walkers: array<Walker>;
@group(0) @binding(2) var<storage, read_write> candidates: array<Candidate>;
@group(0) @binding(3) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(4) var<storage, read_write> instanceMatrices: array<mat4x4<f32>>;
@group(0) @binding(5) var<storage, read_write> instanceBirths: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> indirectArgs: array<u32>;
@group(0) @binding(7) var<storage, read_write> counters: Counters;
@group(0) @binding(8) var<uniform> params: Params;

fn hash32(value: u32) -> u32 {
  var next = value;
  next = next ^ (next >> 16u);
  next = next * 0x7feb352du;
  next = next ^ (next >> 15u);
  next = next * 0x846ca68bu;
  next = next ^ (next >> 16u);
  return next;
}

fn positionInHashRange(position: vec3<i32>) -> bool {
  return all(position >= vec3<i32>(-HASH_COORD_BIAS)) && all(position <= vec3<i32>(HASH_COORD_MAX));
}

fn positionKey(position: vec3<i32>) -> u32 {
  let biased = vec3<u32>(position + vec3<i32>(HASH_COORD_BIAS));
  return (biased.x | (biased.y << 10u) | (biased.z << 20u)) + 1u;
}

fn hashPosition(position: vec3<i32>) -> u32 {
  return hash32(positionKey(position));
}

fn entryState(index: u32) -> u32 {
  return atomicLoad(&hashTable[index].packedData) >> META_STATE_SHIFT;
}

fn entryNeighbors(index: u32) -> u32 {
  return atomicLoad(&hashTable[index].packedData) & META_NEIGHBOR_MASK;
}

fn entryMeta(state: u32, packedNeighbors: u32) -> u32 {
  return (state << META_STATE_SHIFT) | (packedNeighbors & META_NEIGHBOR_MASK);
}

fn findHash(position: vec3<i32>) -> u32 {
  if (!positionInHashRange(position)) {
    return INVALID;
  }
  let targetKey = positionKey(position);
  var index = hashPosition(position) & params.hashMask;
  for (var probe = 0u; probe < params.maxHashProbes; probe = probe + 1u) {
    let key = atomicLoad(&hashTable[index].key);
    if (key == 0u) {
      return INVALID;
    }
    if (key == targetKey && entryState(index) != EMPTY) {
      return index;
    }
    index = (index + 1u) & params.hashMask;
  }
  return INVALID;
}

fn findOccupied(position: vec3<i32>) -> u32 {
  if (!positionInHashRange(position)) {
    return INVALID;
  }
  let targetKey = positionKey(position);
  var index = hashPosition(position) & params.hashMask;
  for (var probe = 0u; probe < params.maxHashProbes; probe = probe + 1u) {
    let key = atomicLoad(&hashTable[index].key);
    if (key == 0u) {
      return INVALID;
    }
    // Frontier keys may still be published by another buildFrontier workgroup;
    // only stable occupied entries are inspected during that pass.
    if (key == targetKey && entryState(index) == OCCUPIED) {
      return index;
    }
    index = (index + 1u) & params.hashMask;
  }
  return INVALID;
}

fn insertUnchecked(position: vec3<i32>, stateValue: u32, birth: u32, packedNeighbors: u32) -> u32 {
  if (!positionInHashRange(position)) {
    atomicStore(&counters.overflow, 1u);
    return INVALID;
  }
  let targetKey = positionKey(position);
  var index = hashPosition(position) & params.hashMask;
  for (var probe = 0u; probe < params.maxHashProbes; probe = probe + 1u) {
    let key = atomicLoad(&hashTable[index].key);
    if (key == targetKey) {
      return INVALID;
    }
    if (key == 0u) {
      loop {
        let claim = atomicCompareExchangeWeak(&hashTable[index].key, 0u, targetKey);
        if (claim.exchanged) {
          atomicStore(&hashTable[index].birth, birth);
          atomicStore(&hashTable[index].packedData, entryMeta(stateValue, packedNeighbors));
          let used = atomicAdd(&counters.hashEntries, 1u) + 1u;
          if (used >= params.hashCapacity - params.hashCapacity / 10u) {
            atomicStore(&counters.overflow, 1u);
          }
          return index;
        }
        if (claim.old_value != 0u) {
          break;
        }
      }
    }
    index = (index + 1u) & params.hashMask;
  }
  atomicStore(&counters.overflow, 1u);
  return INVALID;
}

fn insertHash(position: vec3<i32>, birth: u32) -> u32 {
  let existing = findHash(position);
  if (existing != INVALID) {
    if (entryState(existing) == FRONTIER) {
      let packedNeighbors = entryNeighbors(existing);
      atomicStore(&hashTable[existing].birth, birth);
      atomicStore(&hashTable[existing].packedData, entryMeta(OCCUPIED, packedNeighbors));
      return existing;
    }
    return INVALID;
  }
  return insertUnchecked(position, OCCUPIED, birth, 0u);
}

fn packedIncrement(offset: vec3<i32>) -> u32 {
  let distance = abs(offset.x) + abs(offset.y) + abs(offset.z);
  if (distance == 1) { return 1u; }
  if (distance == 2) { return 16u; }
  return 512u;
}

fn packedCount(packed: u32, neighborhood: u32) -> u32 {
  let faces = packed & 15u;
  let edges = (packed >> 4u) & 31u;
  let corners = (packed >> 9u) & 15u;
  if (neighborhood == 6u) { return faces; }
  if (neighborhood == 18u) { return faces + edges; }
  return faces + edges + corners;
}

fn cachedNeighborCount(position: vec3<i32>, neighborhood: u32) -> u32 {
  let slot = findHash(position);
  if (slot == INVALID || entryState(slot) != FRONTIER) {
    return 0u;
  }
  return packedCount(entryNeighbors(slot), neighborhood);
}

fn scanPackedNeighbors(position: vec3<i32>) -> vec3<u32> {
  var packed = 0u;
  var earliestBirth = INVALID;
  var latestBirth = 0u;
  for (var index = 0u; index < 26u; index = index + 1u) {
    let offset = offsetFor(index);
    let slot = findOccupied(position + offset);
    if (slot != INVALID) {
      packed = packed + packedIncrement(offset);
      let neighborBirth = atomicLoad(&hashTable[slot].birth);
      earliestBirth = min(earliestBirth, neighborBirth);
      latestBirth = max(latestBirth, neighborBirth);
    }
  }
  return vec3<u32>(packed, earliestBirth, latestBirth);
}

fn offsetFor(index: u32) -> vec3<i32> {
  return NEIGHBOR_OFFSETS[min(index, 25u)];
}

fn radiusSq(position: vec3<i32>) -> u32 {
  let value = position.x * position.x + position.y * position.y + position.z * position.z;
  return u32(max(0, value));
}

fn makeMatrix(position: vec3<i32>) -> mat4x4<f32> {
  return mat4x4<f32>(
    vec4<f32>(1.0, 0.0, 0.0, 0.0),
    vec4<f32>(0.0, 1.0, 0.0, 0.0),
    vec4<f32>(0.0, 0.0, 1.0, 0.0),
    vec4<f32>(vec3<f32>(position), 1.0)
  );
}

fn addVisible(birth: u32) {
  let slot = atomicLoad(&counters.visibleCount);
  if (slot >= params.particleCapacity) {
    atomicStore(&counters.overflow, 1u);
    return;
  }
  atomicStore(&counters.visibleCount, slot + 1u);
  particles[birth].slot = slot;
  instanceMatrices[slot] = makeMatrix(particles[birth].position.xyz);
  instanceBirths[slot] = vec4<f32>(f32(birth), 0.0, 0.0, 0.0);
}

fn removeVisible(birth: u32) {
  let slot = particles[birth].slot;
  let count = atomicLoad(&counters.visibleCount);
  if (slot == INVALID || count == 0u || slot >= count) {
    return;
  }
  let last = count - 1u;
  if (slot != last) {
    let movedBirth = u32(instanceBirths[last].x + 0.5);
    instanceMatrices[slot] = instanceMatrices[last];
    instanceBirths[slot] = instanceBirths[last];
    particles[movedBirth].slot = slot;
  }
  particles[birth].slot = INVALID;
  atomicStore(&counters.visibleCount, last);
}

fn xorshift(value: u32) -> u32 {
  var next = select(value, 0xa341316cu, value == 0u);
  next = next ^ (next << 13u);
  next = next ^ (next >> 17u);
  next = next ^ (next << 5u);
  return next;
}

fn randomFloat(state: ptr<function, u32>) -> f32 {
  *state = xorshift(*state);
  return f32(*state) / 4294967296.0;
}

fn launchPosition(state: ptr<function, u32>) -> vec3<i32> {
  let u = randomFloat(state);
  let v = randomFloat(state);
  let theta = 6.28318530718 * u;
  let phi = acos(1.0 - 2.0 * v);
  let launchRadius = ceil(sqrt(f32(atomicLoad(&counters.maxRadiusSq)))) + f32(params.launchPadding);
  let direction = vec3<f32>(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta));
  return vec3<i32>(round(direction * max(1.0, launchRadius)));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn clearHash(@builtin(global_invocation_id) id: vec3<u32>) {
  var index = id.x;
  while (index < params.hashCapacity) {
    atomicStore(&hashTable[index].key, 0u);
    atomicStore(&hashTable[index].birth, INVALID);
    atomicStore(&hashTable[index].packedData, 0u);
    index = index + params.clearThreads;
  }
  if (id.x == 0u) {
    atomicStore(&counters.particleCount, params.prefixCount);
    atomicStore(&counters.visibleCount, 0u);
    atomicStore(&counters.maxRadiusSq, 0u);
    atomicStore(&counters.candidateCount, 0u);
    atomicStore(&counters.attachedThisStep, 0u);
    atomicStore(&counters.overflow, 0u);
    atomicStore(&counters.hashEntries, 0u);
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn insertPrefix(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.prefixCount) {
    return;
  }
  particles[id.x].enclosedAt = INVALID;
  particles[id.x].slot = INVALID;
  particles[id.x].neighborCount = 0u;
  _ = insertUnchecked(particles[id.x].position.xyz, OCCUPIED, id.x, 0u);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn rebuildNeighbors(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.prefixCount) {
    return;
  }
  let position = particles[id.x].position.xyz;
  let scanned = scanPackedNeighbors(position);
  let count = packedCount(scanned.x, 26u);
  particles[id.x].neighborCount = count;
  particles[id.x].enclosedAt = select(INVALID, scanned.z, count == 26u);
  let slot = findOccupied(position);
  if (slot != INVALID) {
    atomicStore(&hashTable[slot].packedData, entryMeta(OCCUPIED, scanned.x));
  }
  atomicMax(&counters.maxRadiusSq, radiusSq(position));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn buildFrontier(@builtin(global_invocation_id) id: vec3<u32>) {
  let birth = id.x;
  if (birth >= params.prefixCount) {
    return;
  }
  let position = particles[birth].position.xyz;
  for (var neighborIndex = 0u; neighborIndex < 26u; neighborIndex = neighborIndex + 1u) {
    let frontierPosition = position + offsetFor(neighborIndex);
    if (findOccupied(frontierPosition) != INVALID) {
      continue;
    }
    let scanned = scanPackedNeighbors(frontierPosition);
    // Exactly one stable occupied neighbor owns publication of this frontier
    // cell, avoiding cross-workgroup reads of a partially published key.
    if (scanned.y == birth) {
      _ = insertUnchecked(frontierPosition, FRONTIER, INVALID, scanned.x);
    }
  }
}

@compute @workgroup_size(1)
fn beginCompact() {
  atomicStore(&counters.visibleCount, 0u);
  indirectArgs[0] = params.vertexCount;
  indirectArgs[1] = 0u;
  indirectArgs[2] = 0u;
  indirectArgs[3] = 0u;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn compactPrefix(@builtin(global_invocation_id) id: vec3<u32>) {
  let birth = id.x;
  if (birth >= params.prefixCount) {
    return;
  }
  particles[birth].slot = INVALID;
  let enclosedAt = particles[birth].enclosedAt;
  if (params.hideEnclosed == 1u && enclosedAt != INVALID && enclosedAt < params.prefixCount) {
    return;
  }
  let slot = atomicAdd(&counters.visibleCount, 1u);
  if (slot >= params.particleCapacity) {
    atomicStore(&counters.overflow, 1u);
    return;
  }
  particles[birth].slot = slot;
  instanceMatrices[slot] = makeMatrix(particles[birth].position.xyz);
  instanceBirths[slot] = vec4<f32>(f32(birth), 0.0, 0.0, 0.0);
}

@compute @workgroup_size(1)
fn finishIndirect() {
  indirectArgs[0] = params.vertexCount;
  indirectArgs[1] = min(atomicLoad(&counters.visibleCount), params.particleCapacity);
  indirectArgs[2] = 0u;
  indirectArgs[3] = 0u;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn initWalkers(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.walkerCount) {
    return;
  }
  var rng = hash32(params.seed ^ ((id.x + 1u) * 0x9e3779b9u) ^ params.branchSerial);
  let position = launchPosition(&rng);
  walkers[id.x].data = vec4<i32>(position, bitcast<i32>(rng));
}

@compute @workgroup_size(1)
fn clearCandidates() {
  atomicStore(&counters.candidateCount, 0u);
  atomicStore(&counters.attachedThisStep, 0u);
  atomicStore(&counters.epoch, params.epoch);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn advanceWalkers(@builtin(global_invocation_id) id: vec3<u32>) {
  let walkerIndex = id.x;
  if (walkerIndex >= params.walkerCount) {
    return;
  }
  candidates[walkerIndex].valid = 0u;
  var position = walkers[walkerIndex].data.xyz;
  var rng = bitcast<u32>(walkers[walkerIndex].data.w);
  let launchRadius = ceil(sqrt(f32(atomicLoad(&counters.maxRadiusSq)))) + f32(params.launchPadding);
  let killRadius = launchRadius + f32(params.killPadding);
  let killRadiusSq = u32(killRadius * killRadius);

  for (var step = 0u; step < params.stepsPerWalker; step = step + 1u) {
    rng = xorshift(rng);
    let direction = rng % 6u;
    if (direction == 0u) { position.x = position.x + 1; }
    if (direction == 1u) { position.x = position.x - 1; }
    if (direction == 2u) { position.y = position.y + 1; }
    if (direction == 3u) { position.y = position.y - 1; }
    if (direction == 4u) { position.z = position.z + 1; }
    if (direction == 5u) { position.z = position.z - 1; }

    if (radiusSq(position) > killRadiusSq) {
      position = launchPosition(&rng);
      continue;
    }
    let frontierSlot = findHash(position);
    if (frontierSlot == INVALID || entryState(frontierSlot) != FRONTIER) {
      continue;
    }
    let count = packedCount(entryNeighbors(frontierSlot), params.neighborhood);
    if (count < params.stickNeighbors) {
      continue;
    }
    let roll = u32(randomFloat(&rng) * 1000000.0);
    if (roll >= params.stickChance) {
      continue;
    }
    candidates[walkerIndex].position = vec4<i32>(position, 0);
    candidates[walkerIndex].valid = 1u;
    candidates[walkerIndex].neighborCount = count;
    candidates[walkerIndex].rank = walkerIndex;
    atomicAdd(&counters.candidateCount, 1u);
    position = launchPosition(&rng);
    break;
  }
  walkers[walkerIndex].data = vec4<i32>(position, bitcast<i32>(rng));
}

@compute @workgroup_size(1)
fn commitCandidates() {
  var particleCount = atomicLoad(&counters.particleCount);
  var attached = 0u;
  let limit = min(params.growthBatch, params.targetParticles - min(params.targetParticles, particleCount));
  for (var walkerIndex = 0u; walkerIndex < params.walkerCount && attached < limit; walkerIndex = walkerIndex + 1u) {
    if (candidates[walkerIndex].valid == 0u || particleCount >= params.particleCapacity) {
      continue;
    }
    let position = candidates[walkerIndex].position.xyz;
    let candidateSlot = findHash(position);
    if (candidateSlot == INVALID || entryState(candidateSlot) != FRONTIER) {
      continue;
    }
    let packed = entryNeighbors(candidateSlot);
    let birth = particleCount;
    let hashIndex = insertHash(position, birth);
    if (hashIndex == INVALID) {
      continue;
    }
    let count = packedCount(packed, 26u);
    particles[birth].position = vec4<i32>(position, 0);
    particles[birth].enclosedAt = select(INVALID, birth, count == 26u);
    particles[birth].slot = INVALID;
    particles[birth].neighborCount = count;
    atomicStore(&hashTable[hashIndex].packedData, entryMeta(OCCUPIED, packed));

    for (var neighborIndex = 0u; neighborIndex < 26u; neighborIndex = neighborIndex + 1u) {
      let offset = offsetFor(neighborIndex);
      let neighborPosition = position + offset;
      let increment = packedIncrement(offset);
      let otherHash = findHash(neighborPosition);
      if (otherHash == INVALID) {
        _ = insertUnchecked(neighborPosition, FRONTIER, INVALID, increment);
      } else {
        let otherState = entryState(otherHash);
        if (otherState == FRONTIER) {
          atomicAdd(&hashTable[otherHash].packedData, increment);
        } else if (otherState == OCCUPIED) {
          let otherBirth = atomicLoad(&hashTable[otherHash].birth);
          if (otherBirth != birth) {
            let oldMeta = atomicAdd(&hashTable[otherHash].packedData, increment);
            let updated = packedCount((oldMeta & META_NEIGHBOR_MASK) + increment, 26u);
            particles[otherBirth].neighborCount = updated;
            if (updated == 26u && particles[otherBirth].enclosedAt == INVALID) {
              particles[otherBirth].enclosedAt = birth;
              if (params.hideEnclosed == 1u) {
                removeVisible(otherBirth);
              }
            }
          }
        }
      }
    }

    if (params.hideEnclosed == 0u || count < 26u) {
      addVisible(birth);
    }
    atomicMax(&counters.maxRadiusSq, radiusSq(position));
    particleCount = particleCount + 1u;
    attached = attached + 1u;
  }
  atomicStore(&counters.particleCount, particleCount);
  atomicStore(&counters.attachedThisStep, attached);
  indirectArgs[0] = params.vertexCount;
  indirectArgs[1] = min(atomicLoad(&counters.visibleCount), params.particleCapacity);
  indirectArgs[2] = 0u;
  indirectArgs[3] = 0u;
}
`;
