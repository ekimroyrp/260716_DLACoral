import {
  ACESFilmicToneMapping,
  BufferAttribute,
  BufferGeometry,
  Color,
  DirectionalLight,
  HemisphereLight,
  IcosahedronGeometry,
  IndirectStorageBufferAttribute,
  InstancedMesh,
  MeshStandardNodeMaterial,
  MOUSE,
  PCFSoftShadowMap,
  PerspectiveCamera,
  RenderPipeline,
  Scene,
  SRGBColorSpace,
  StorageInstancedBufferAttribute,
  WebGPURenderer,
} from 'three/webgpu';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { bloom } from 'three/addons/tsl/display/BloomNode.js';
import { add, attribute, clamp, max, mix, pass, uniform } from 'three/tsl';
import {
  DEFAULT_DISPLAY_SETTINGS,
  DEFAULT_DLA_SETTINGS,
  type AttachmentNeighborhood,
  type DisplaySettings,
  type DlaSettings,
} from '../types';
import { createGlbBlob, createObjBlob, type ExportInstanceData } from './modelExport';
import { createDisplayColor, setDisplayColor } from './displayColor';
import { disableWebGlFallback } from './nativeWebGpu';
import { createRequiredDeviceLimits } from './webGpuLimits';

const DEG_TO_RAD = Math.PI / 180;
const MODEL_ROTATION_DEGREES_PER_PIXEL = 0.35;
const KEY_LIGHT_DISTANCE_SCALE = 2.69284236449147;
const INITIAL_CAMERA_EXTENT = 24;
const INITIAL_CAMERA_DISTANCE_SCALE = 1.5;
const MAX_PIXEL_RATIO = 2;

export type RotationPhase = 'begin' | 'change' | 'end';

export interface DlaRendererOptions {
  onModelRotationChange?: (rotation: number, phase: RotationPhase) => void;
  onError?: (error: Error) => void;
}

export interface DlaRenderTargets {
  instanceMatrix: GPUBuffer;
  instanceBirth: GPUBuffer;
  indirectArgs: GPUBuffer;
  capacity: number;
  vertexCount: number;
}

export interface DlaRenderState {
  displayedCount: number;
  totalCount: number;
  seedCount: number;
  newestVisibleBirth: number;
  maxRadiusSq: number;
}

export interface CpuInstanceData {
  matrices: Float32Array;
  birthRanks: Float32Array;
  count: number;
}

export type InstanceDataProvider = (count: number) => Promise<CpuInstanceData>;

export interface SphereGeometryData {
  geometry: BufferGeometry;
  rawPositions: Float32Array;
  basePositions: Float32Array;
}

interface WebGpuBackendAccess {
  isWebGPUBackend?: boolean;
  device?: GPUDevice;
  createStorageAttribute: (attribute: unknown) => void;
  createIndirectStorageAttribute: (attribute: unknown) => void;
  get: (attribute: unknown) => { buffer?: GPUBuffer };
}

export class DlaRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly options: DlaRendererOptions;
  private readonly scene: Scene;
  private readonly camera: PerspectiveCamera;
  private readonly renderer: WebGPURenderer;
  private readonly controls: OrbitControls;
  private readonly material: MeshStandardNodeMaterial;
  private readonly skyLight: HemisphereLight;
  private readonly keyLight: DirectionalLight;
  private readonly rimLight: DirectionalLight;
  private readonly bounceLight: DirectionalLight;
  private readonly renderPipeline: RenderPipeline;
  private readonly bloomPass: ReturnType<typeof bloom>;
  private readonly indirect: IndirectStorageBufferAttribute;
  private readonly requiredDeviceLimits = createRequiredDeviceLimits();
  private readonly innerColorUniform = uniform(createDisplayColor(DEFAULT_DISPLAY_SETTINGS.innerColor));
  private readonly outerColorUniform = uniform(createDisplayColor(DEFAULT_DISPLAY_SETTINGS.outerColor));
  private readonly gradientCountUniform = uniform(1);
  private readonly seedCountUniform = uniform(1);
  private readonly brightnessUniform = uniform(DEFAULT_DISPLAY_SETTINGS.brightness);
  private readonly contrastUniform = uniform(DEFAULT_DISPLAY_SETTINGS.contrast);
  private geometry: BufferGeometry;
  private rawPositions: Float32Array;
  private basePositions: Float32Array;
  private birthAttribute: StorageInstancedBufferAttribute;
  private mesh: InstancedMesh;
  private targets: DlaRenderTargets | null = null;
  private instanceCapacity = 1;
  private sphereDetail = 0;
  private currentNeighborhood: AttachmentNeighborhood = 26;
  private currentSphereGap = 0;
  private currentRotation = 0;
  private displayedCount = 0;
  private seedCount = 1;
  private lastTotalCount = 0;
  private lastShadowParticleCount = 0;
  private lastShadowRefreshTime = 0;
  private lastMaxRadiusSq = 0;
  private currentExtent = INITIAL_CAMERA_EXTENT;
  private currentDla: DlaSettings = { ...DEFAULT_DLA_SETTINGS };
  private currentDisplay: DisplaySettings = { ...DEFAULT_DISPLAY_SETTINGS };
  private modelRotationPointerId: number | null = null;
  private modelRotationStartX = 0;
  private modelRotationStartDegrees = 0;
  private instanceDataProvider: InstanceDataProvider | null = null;
  private initialized = false;
  private disposed = false;
  private renderFailed = false;
  private bloomInitialized = false;

  constructor(canvas: HTMLCanvasElement, options: DlaRendererOptions = {}) {
    this.canvas = canvas;
    this.options = options;
    this.scene = new Scene();
    this.scene.background = new Color(0x000000);

    this.camera = new PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.01, 5000);
    this.camera.position.set(
      INITIAL_CAMERA_EXTENT * 0.92 * INITIAL_CAMERA_DISTANCE_SCALE,
      INITIAL_CAMERA_EXTENT * 0.72 * INITIAL_CAMERA_DISTANCE_SCALE,
      INITIAL_CAMERA_EXTENT * 1.18 * INITIAL_CAMERA_DISTANCE_SCALE,
    );

    this.renderer = new WebGPURenderer({
      canvas,
      antialias: true,
      alpha: false,
      forceWebGL: false,
      powerPreference: 'high-performance',
      requiredLimits: this.requiredDeviceLimits,
    });
    disableWebGlFallback(this.renderer);
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.toneMapping = ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = DEFAULT_DISPLAY_SETTINGS.exposure;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = PCFSoftShadowMap;
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.setPixelRatio(this.getPixelRatio());
    this.renderer.setSize(window.innerWidth, window.innerHeight);

    const sphere = createSphereGeometry(this.sphereDetail, this.currentDla.attachmentNeighborhood);
    this.geometry = sphere.geometry;
    this.rawPositions = sphere.rawPositions;
    this.basePositions = sphere.basePositions;
    this.indirect = new IndirectStorageBufferAttribute(
      new Uint32Array([this.getSphereVertexCount(), 0, 0, 0]),
      4,
    );
    this.geometry.setIndirect(this.indirect);

    this.material = new MeshStandardNodeMaterial({
      color: 0xffffff,
      roughness: DEFAULT_DISPLAY_SETTINGS.roughness,
      metalness: 0,
      flatShading: false,
    });
    this.material.dithering = true;
    const birthRank = attribute<'vec4'>('instanceBirth', 'vec4').x;
    const attachedCount = max(1, this.gradientCountUniform.sub(this.seedCountUniform));
    const gradientPosition = clamp(
      birthRank.sub(this.seedCountUniform).add(1).div(attachedCount),
      0,
      1,
    );
    const gradientColor = mix(this.innerColorUniform, this.outerColorUniform, gradientPosition);
    this.material.colorNode = clamp(
      gradientColor.sub(0.5).mul(this.contrastUniform).add(0.5),
      0,
      4,
    ).mul(this.brightnessUniform);

    this.birthAttribute = new StorageInstancedBufferAttribute(1, 4);
    this.geometry.setAttribute('instanceBirth', this.birthAttribute);
    this.mesh = this.createMesh(1, this.birthAttribute);
    this.scene.add(this.mesh);

    this.skyLight = new HemisphereLight(0xb9d4ff, 0x030403, DEFAULT_DISPLAY_SETTINGS.ambientFill);
    this.scene.add(this.skyLight);

    this.keyLight = new DirectionalLight(0xfff0d3, DEFAULT_DISPLAY_SETTINGS.keyBrightness);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.set(2048, 2048);
    this.keyLight.shadow.bias = -0.00035;
    this.keyLight.shadow.normalBias = 0.045;
    this.keyLight.shadow.radius = DEFAULT_DISPLAY_SETTINGS.shadowSoftness;
    this.keyLight.shadow.intensity = DEFAULT_DISPLAY_SETTINGS.shadowStrength;
    this.keyLight.shadow.autoUpdate = false;
    this.scene.add(this.keyLight.target);
    this.scene.add(this.keyLight);

    this.rimLight = new DirectionalLight(0x70a8ff, DEFAULT_DISPLAY_SETTINGS.rimBrightness);
    this.scene.add(this.rimLight.target);
    this.scene.add(this.rimLight);

    this.bounceLight = new DirectionalLight(0x8dd06e, DEFAULT_DISPLAY_SETTINGS.bounceBrightness);
    this.scene.add(this.bounceLight.target);
    this.scene.add(this.bounceLight);

    const scenePass = pass(this.scene, this.camera);
    const sceneColor = scenePass.getTextureNode('output');
    this.bloomPass = bloom(
      sceneColor,
      DEFAULT_DISPLAY_SETTINGS.bloomStrength,
      DEFAULT_DISPLAY_SETTINGS.bloomRadius,
      DEFAULT_DISPLAY_SETTINGS.bloomThreshold,
    );
    this.renderPipeline = new RenderPipeline(this.renderer);
    this.renderPipeline.outputNode = add(sceneColor, this.bloomPass);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.enablePan = true;
    this.controls.enableZoom = true;
    this.controls.minDistance = 12;
    this.controls.maxDistance = 900;
    this.controls.mouseButtons = {
      MIDDLE: MOUSE.PAN,
      RIGHT: MOUSE.ROTATE,
    };
    this.controls.target.set(0, INITIAL_CAMERA_EXTENT * 0.12, 0);
    this.controls.update();

    this.canvas.addEventListener('contextmenu', this.handleContextMenu);
    this.canvas.addEventListener('pointerdown', this.handleModelRotationPointerDown);
    this.canvas.ownerDocument.addEventListener('pointermove', this.handleModelRotationPointerMove);
    this.canvas.ownerDocument.addEventListener('pointerup', this.handleModelRotationPointerEnd);
    this.canvas.ownerDocument.addEventListener('pointercancel', this.handleModelRotationPointerEnd);
    window.addEventListener('contextmenu', this.handleContextMenu);
    window.addEventListener('resize', this.handleWindowResize);
  }

  async init(): Promise<void> {
    this.assertNotDisposed();
    if (this.initialized) {
      return;
    }
    try {
      await this.renderer.init();
    } catch (error) {
      const reason = normalizeError(error);
      throw new Error(
        `Native WebGPU is required to run 260716_DLAFractals. ${reason.message}`,
        { cause: reason },
      );
    }
    const backend = this.getWebGpuBackend();
    if (backend.isWebGPUBackend !== true || !backend.device) {
      throw new Error('Native WebGPU is required to run 260716_DLAFractals.');
    }
    this.initialized = true;
    this.prepareInstances(1, this.sphereDetail);
  }

  getWebGpuDevice(): GPUDevice {
    const device = this.getWebGpuBackend().device;
    if (!device || !this.initialized) {
      throw new Error('The native WebGPU renderer has not been initialized.');
    }
    return device;
  }

  getMaxSupportedCapacity(): number {
    const device = this.getWebGpuDevice();
    const maxStorageBinding = Number(device.limits.maxStorageBufferBindingSize);
    const maxBuffer = Number(device.limits.maxBufferSize);
    const matrixLimit = Math.floor(Math.min(maxStorageBinding, maxBuffer) / (16 * Float32Array.BYTES_PER_ELEMENT));
    const birthLimit = Math.floor(Math.min(maxStorageBinding, maxBuffer) / (4 * Float32Array.BYTES_PER_ELEMENT));
    return Math.max(1, Math.min(matrixLimit, birthLimit));
  }

  prepareInstances(capacity: number, sphereDetail: number): DlaRenderTargets {
    this.assertReady();
    const safeCapacity = Math.max(1, Math.floor(capacity));
    const maxCapacity = this.getMaxSupportedCapacity();
    if (safeCapacity > maxCapacity) {
      throw new Error(`This WebGPU device supports at most ${maxCapacity.toLocaleString()} DLA instances.`);
    }

    this.setSphereDetail(sphereDetail);
    if (this.instanceCapacity !== safeCapacity || this.mesh.instanceMatrix.count < safeCapacity) {
      this.replaceMesh(safeCapacity);
    }

    const backend = this.getWebGpuBackend();
    const matrixBuffer = this.ensureStorageBuffer(
      backend,
      this.mesh.instanceMatrix as StorageInstancedBufferAttribute,
    );
    const birthBuffer = this.ensureStorageBuffer(backend, this.birthAttribute);
    const indirectBuffer = this.ensureIndirectBuffer(backend, this.indirect);
    const vertexCount = this.getSphereVertexCount();
    const targets: DlaRenderTargets = {
      instanceMatrix: matrixBuffer,
      instanceBirth: birthBuffer,
      indirectArgs: indirectBuffer,
      capacity: safeCapacity,
      vertexCount,
    };
    this.targets = targets;
    this.getWebGpuDevice().queue.writeBuffer(
      indirectBuffer,
      0,
      new Uint32Array([vertexCount, 0, 0, 0]),
    );
    return targets;
  }

  getRenderTargets(): DlaRenderTargets | null {
    return this.targets;
  }

  update(dla: DlaSettings, display: DisplaySettings, state: DlaRenderState): void {
    this.updateDlaSettings(dla);
    this.updateDisplay(display);
    this.updateState(state);
  }

  updateDlaSettings(settings: DlaSettings): void {
    this.assertNotDisposed();
    const sphereScaleChanged = settings.sphereScale !== this.currentDla.sphereScale;
    const renderSettingsChanged =
      settings.sphereDetail !== this.currentDla.sphereDetail
      || settings.attachmentNeighborhood !== this.currentDla.attachmentNeighborhood
      || settings.sphereGap !== this.currentDla.sphereGap
      || settings.sphereScale !== this.currentDla.sphereScale
      || settings.rotation !== this.currentDla.rotation;
    this.currentDla = { ...settings };
    if (!renderSettingsChanged) {
      return;
    }
    this.setSphereDetail(settings.sphereDetail);
    this.setSphereNeighborhood(settings.attachmentNeighborhood);
    this.setSphereGap(settings.sphereGap);
    this.mesh.scale.setScalar(Math.max(0.01, settings.sphereScale));
    this.setModelRotation(settings.rotation);
    if (sphereScaleChanged) {
      const extent = extentFromRadius(this.lastMaxRadiusSq, settings.sphereScale);
      this.currentExtent = extent;
      this.updateLightRig(extent, this.currentDisplay);
    }
    this.keyLight.shadow.needsUpdate = true;
  }

  updateDisplay(settings: DisplaySettings): void {
    this.assertNotDisposed();
    if (displaySettingsEqual(this.currentDisplay, settings)) {
      return;
    }
    this.currentDisplay = { ...settings };
    setDisplayColor(this.innerColorUniform.value, settings.innerColor);
    setDisplayColor(this.outerColorUniform.value, settings.outerColor);
    this.brightnessUniform.value = Math.max(0, settings.brightness);
    this.contrastUniform.value = Math.max(0, settings.contrast);
    this.renderer.toneMappingExposure = Math.max(0, settings.exposure);
    this.material.roughness = clampNumber(settings.roughness, 0, 1);
    this.skyLight.intensity = Math.max(0, settings.ambientFill);
    this.keyLight.intensity = Math.max(0, settings.keyBrightness);
    this.keyLight.shadow.intensity = Math.max(0, settings.shadowStrength);
    this.keyLight.shadow.radius = Math.max(0, settings.shadowSoftness);
    this.rimLight.intensity = Math.max(0, settings.rimBrightness);
    this.bounceLight.intensity = Math.max(0, settings.bounceBrightness);
    this.bloomPass.strength.value = Math.max(0, settings.bloomStrength);
    this.bloomPass.radius.value = Math.max(0, settings.bloomRadius);
    this.bloomPass.threshold.value = Math.max(0, settings.bloomThreshold);
    this.updateLightRig(this.getCurrentExtent(), settings);
    this.keyLight.shadow.needsUpdate = true;
  }

  updateState(state: DlaRenderState): void {
    this.assertNotDisposed();
    const maxCount = this.targets?.capacity ?? this.instanceCapacity;
    this.displayedCount = clampInteger(state.displayedCount, 0, maxCount);
    const totalCount = Math.max(0, Math.floor(state.totalCount));
    this.seedCount = clampInteger(state.seedCount, 0, totalCount);
    const newestVisibleBirth = clampInteger(
      state.newestVisibleBirth,
      0,
      Math.max(0, totalCount - 1),
    );
    this.gradientCountUniform.value = Math.max(
      1,
      this.seedCount,
      newestVisibleBirth + 1,
    );
    this.seedCountUniform.value = this.seedCount;

    if (this.targets) {
      this.getWebGpuDevice().queue.writeBuffer(
        this.targets.indirectArgs,
        Uint32Array.BYTES_PER_ELEMENT,
        new Uint32Array([this.displayedCount]),
      );
    }

    const wasReset = state.totalCount < this.lastTotalCount;
    const aggregateChanged = state.totalCount !== this.lastTotalCount;
    this.lastTotalCount = totalCount;
    this.lastMaxRadiusSq = Math.max(0, state.maxRadiusSq);

    const extent = extentFromRadius(state.maxRadiusSq, this.currentDla.sphereScale);
    this.currentExtent = extent;
    this.updateLightRig(extent, this.currentDisplay);
    if (aggregateChanged) {
      this.requestAdaptiveShadowRefresh(totalCount, wasReset);
    }
  }

  setModelRotation(degrees: number): void {
    this.currentRotation = normalizeRotationDegrees(degrees);
    this.mesh.rotation.y = this.currentRotation * DEG_TO_RAD;
    this.keyLight.shadow.needsUpdate = true;
  }

  setInstanceDataProvider(provider: InstanceDataProvider | null): void {
    this.instanceDataProvider = provider;
  }

  async readInstanceData(count = this.displayedCount): Promise<ExportInstanceData> {
    this.assertReady();
    const safeCount = clampInteger(count, 0, this.displayedCount);
    if (safeCount <= 0) {
      throw new Error('No visible DLA particles are available to export.');
    }

    const instanceData = this.instanceDataProvider
      ? await this.instanceDataProvider(safeCount)
      : await this.readGpuInstanceData(safeCount);
    const exportCount = Math.min(safeCount, Math.max(0, instanceData.count));
    const positions = this.geometry.getAttribute('position').array;
    const normals = this.geometry.getAttribute('normal').array;

    return {
      matrices: instanceData.matrices.slice(0, exportCount * 16),
      birthRanks: instanceData.birthRanks.slice(0, exportCount),
      count: exportCount,
      seedCount: this.seedCount,
      gradientCount: displayedGradientCount(
        instanceData.birthRanks,
        exportCount,
        this.seedCount,
      ),
      spherePositions: new Float32Array(positions),
      sphereNormals: new Float32Array(normals),
      sphereScale: this.mesh.scale.x,
      rotationDegrees: this.currentRotation,
      innerColor: this.currentDisplay.innerColor,
      outerColor: this.currentDisplay.outerColor,
      brightness: this.currentDisplay.brightness,
      contrast: this.currentDisplay.contrast,
      materialRoughness: this.material.roughness,
    };
  }

  async exportGlb(): Promise<void> {
    const blob = await createGlbBlob(await this.readInstanceData());
    downloadBlob(blob, `260716_DLAFractals-${Date.now()}.glb`);
  }

  async exportObj(): Promise<void> {
    const blob = await createObjBlob(await this.readInstanceData());
    downloadBlob(blob, `260716_DLAFractals-${Date.now()}.obj`);
  }

  exportScreenshot(): void {
    this.render();
    const link = document.createElement('a');
    link.download = `260716_DLAFractals-${Date.now()}.png`;
    link.href = this.canvas.toDataURL('image/png');
    link.click();
  }

  resize(width = window.innerWidth, height = window.innerHeight): void {
    const safeWidth = Math.max(1, Math.floor(width));
    const safeHeight = Math.max(1, Math.floor(height));
    this.camera.aspect = safeWidth / safeHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(this.getPixelRatio());
    this.renderer.setSize(safeWidth, safeHeight);
    if (this.initialized && this.bloomInitialized) {
      this.bloomPass.setSize(safeWidth, safeHeight);
    }
  }

  render(): void {
    this.assertNotDisposed();
    if (!this.initialized || this.renderFailed) {
      return;
    }
    try {
      this.controls.update();
      if (this.bloomPass.strength.value > 0.0001) {
        this.renderPipeline.render();
        this.bloomInitialized = true;
      } else {
        this.renderer.render(this.scene, this.camera);
      }
    } catch (error) {
      this.renderFailed = true;
      this.options.onError?.(normalizeError(error));
    }
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    this.renderer.setAnimationLoop(null);
    this.controls.dispose();
    this.canvas.removeEventListener('contextmenu', this.handleContextMenu);
    this.canvas.removeEventListener('pointerdown', this.handleModelRotationPointerDown);
    this.canvas.ownerDocument.removeEventListener('pointermove', this.handleModelRotationPointerMove);
    this.canvas.ownerDocument.removeEventListener('pointerup', this.handleModelRotationPointerEnd);
    this.canvas.ownerDocument.removeEventListener('pointercancel', this.handleModelRotationPointerEnd);
    window.removeEventListener('contextmenu', this.handleContextMenu);
    window.removeEventListener('resize', this.handleWindowResize);
    this.scene.remove(this.mesh);
    this.mesh.dispose();
    this.geometry.dispose();
    this.material.dispose();
    this.keyLight.shadow.dispose();
    this.renderPipeline.dispose();
    this.renderer.dispose();
    this.targets = null;
  }

  private readonly handleWindowResize = (): void => {
    this.resize();
  };

  private readonly handleContextMenu = (event: Event): void => {
    event.preventDefault();
  };

  private readonly handleModelRotationPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || this.modelRotationPointerId !== null) {
      return;
    }
    this.modelRotationPointerId = event.pointerId;
    this.modelRotationStartX = event.clientX;
    this.modelRotationStartDegrees = this.currentRotation;
    this.canvas.setPointerCapture?.(event.pointerId);
    this.options.onModelRotationChange?.(this.currentRotation, 'begin');
    event.preventDefault();
  };

  private readonly handleModelRotationPointerMove = (event: PointerEvent): void => {
    if (event.pointerId !== this.modelRotationPointerId) {
      return;
    }
    const nextRotation = normalizeRotationDegrees(
      this.modelRotationStartDegrees
        + (event.clientX - this.modelRotationStartX) * MODEL_ROTATION_DEGREES_PER_PIXEL,
    );
    this.setModelRotation(nextRotation);
    this.options.onModelRotationChange?.(nextRotation, 'change');
    event.preventDefault();
  };

  private readonly handleModelRotationPointerEnd = (event: PointerEvent): void => {
    if (event.pointerId !== this.modelRotationPointerId) {
      return;
    }
    this.modelRotationPointerId = null;
    if (this.canvas.hasPointerCapture?.(event.pointerId)) {
      this.canvas.releasePointerCapture(event.pointerId);
    }
    this.options.onModelRotationChange?.(this.currentRotation, 'end');
    event.preventDefault();
  };

  private setSphereDetail(detail: number): void {
    const safeDetail = clampInteger(detail, 0, 2);
    if (safeDetail === this.sphereDetail) {
      return;
    }

    const previousGeometry = this.geometry;
    const sphere = createSphereGeometry(safeDetail, this.currentDla.attachmentNeighborhood);
    this.geometry = sphere.geometry;
    this.rawPositions = sphere.rawPositions;
    this.basePositions = sphere.basePositions;
    this.sphereDetail = safeDetail;
    this.currentNeighborhood = this.currentDla.attachmentNeighborhood;
    this.geometry.setAttribute('instanceBirth', this.birthAttribute);
    this.geometry.setIndirect(this.indirect);
    this.mesh.geometry = this.geometry;
    previousGeometry.dispose();
    this.currentSphereGap = Number.NaN;
    this.setSphereGap(this.currentDla.sphereGap);

    if (this.targets) {
      const vertexCount = this.getSphereVertexCount();
      this.targets.vertexCount = vertexCount;
      this.getWebGpuDevice().queue.writeBuffer(
        this.targets.indirectArgs,
        0,
        new Uint32Array([vertexCount]),
      );
    }
    this.keyLight.shadow.needsUpdate = true;
  }

  private setSphereNeighborhood(neighborhood: AttachmentNeighborhood): void {
    if (neighborhood === this.currentNeighborhood) {
      return;
    }
    this.basePositions = calibrateSphereForNeighborhood(this.rawPositions, neighborhood);
    this.currentNeighborhood = neighborhood;
    this.currentSphereGap = Number.NaN;
    this.setSphereGap(this.currentDla.sphereGap);
  }

  private setSphereGap(gap: number): void {
    const safeGap = Math.max(0, gap);
    if (Math.abs(safeGap - this.currentSphereGap) <= 0.0001) {
      return;
    }
    applySphereGapToGeometry(this.geometry, this.basePositions, safeGap);
    this.currentSphereGap = safeGap;
    this.keyLight.shadow.needsUpdate = true;
  }

  private replaceMesh(capacity: number): void {
    this.scene.remove(this.mesh);
    this.mesh.dispose();
    this.birthAttribute = new StorageInstancedBufferAttribute(capacity, 4);
    this.geometry.setAttribute('instanceBirth', this.birthAttribute);
    this.mesh = this.createMesh(capacity, this.birthAttribute);
    this.mesh.rotation.y = this.currentRotation * DEG_TO_RAD;
    this.mesh.scale.setScalar(Math.max(0.01, this.currentDla.sphereScale));
    this.scene.add(this.mesh);
    this.targets = null;
  }

  private createMesh(capacity: number, birthAttribute: StorageInstancedBufferAttribute): InstancedMesh {
    const safeCapacity = Math.max(1, Math.floor(capacity));
    this.geometry.setAttribute('instanceBirth', birthAttribute);
    this.geometry.setIndirect(this.indirect);
    const mesh = new InstancedMesh(this.geometry, this.material, safeCapacity);
    mesh.instanceMatrix = new StorageInstancedBufferAttribute(safeCapacity, 16);
    mesh.count = safeCapacity;
    mesh.frustumCulled = false;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    this.instanceCapacity = safeCapacity;
    return mesh;
  }

  private updateLightRig(extent: number, settings: DisplaySettings): void {
    const targetY = extent * 0.12;
    const shadowHalfSize = extent * 1.08;
    const azimuth = settings.lightAzimuth * DEG_TO_RAD;
    const elevation = settings.lightElevation * DEG_TO_RAD;
    const lightDistance = extent * KEY_LIGHT_DISTANCE_SCALE;
    const horizontalDistance = Math.cos(elevation) * lightDistance;

    this.keyLight.target.position.set(0, targetY, 0);
    this.keyLight.position.set(
      Math.cos(azimuth) * horizontalDistance,
      targetY + Math.sin(elevation) * lightDistance,
      Math.sin(azimuth) * horizontalDistance,
    );
    this.rimLight.target.position.set(0, targetY, 0);
    this.rimLight.position.set(extent * 1.4, extent * 0.95, -extent * 1.4);
    this.bounceLight.target.position.set(0, targetY * 0.4, 0);
    this.bounceLight.position.set(extent * 0.8, -extent * 0.15, extent * 1.05);

    const shadowCamera = this.keyLight.shadow.camera;
    shadowCamera.left = -shadowHalfSize;
    shadowCamera.right = shadowHalfSize;
    shadowCamera.top = shadowHalfSize;
    shadowCamera.bottom = -shadowHalfSize;
    shadowCamera.near = 1;
    shadowCamera.far = extent * 4.4;
    shadowCamera.updateProjectionMatrix();
  }

  private requestAdaptiveShadowRefresh(totalCount: number, force: boolean): void {
    const now = typeof performance === 'undefined' ? Date.now() : performance.now();
    const particleDelta = Math.abs(totalCount - this.lastShadowParticleCount);
    const countThreshold = Math.max(1, Math.floor(Math.max(1, totalCount) * 0.015));
    if (!force && particleDelta < countThreshold && now - this.lastShadowRefreshTime < 350) {
      return;
    }
    this.keyLight.shadow.needsUpdate = true;
    this.lastShadowParticleCount = totalCount;
    this.lastShadowRefreshTime = now;
  }

  private getCurrentExtent(): number {
    return this.currentExtent;
  }

  private getSphereVertexCount(): number {
    return this.geometry.getAttribute('position').count;
  }

  private getWebGpuBackend(): WebGpuBackendAccess {
    return this.renderer.backend as unknown as WebGpuBackendAccess;
  }

  private ensureStorageBuffer(
    backend: WebGpuBackendAccess,
    storageAttribute: StorageInstancedBufferAttribute,
  ): GPUBuffer {
    backend.createStorageAttribute(storageAttribute);
    const buffer = backend.get(storageAttribute).buffer;
    if (!buffer) {
      throw new Error('Unable to allocate a native WebGPU storage attribute buffer.');
    }
    return buffer;
  }

  private ensureIndirectBuffer(
    backend: WebGpuBackendAccess,
    indirectAttribute: IndirectStorageBufferAttribute,
  ): GPUBuffer {
    backend.createIndirectStorageAttribute(indirectAttribute);
    const buffer = backend.get(indirectAttribute).buffer;
    if (!buffer) {
      throw new Error('Unable to allocate a native WebGPU indirect draw buffer.');
    }
    return buffer;
  }

  private async readGpuInstanceData(count: number): Promise<CpuInstanceData> {
    if (!this.targets) {
      throw new Error('DLA render targets have not been prepared.');
    }
    const matrixByteLength = count * 16 * Float32Array.BYTES_PER_ELEMENT;
    const birthByteLength = count * 4 * Float32Array.BYTES_PER_ELEMENT;
    const [matrixData, paddedBirthData] = await Promise.all([
      this.readGpuBuffer(this.targets.instanceMatrix, matrixByteLength),
      this.readGpuBuffer(this.targets.instanceBirth, birthByteLength),
    ]);
    const paddedBirthRanks = new Float32Array(paddedBirthData);
    const birthRanks = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      birthRanks[i] = paddedBirthRanks[i * 4] ?? i;
    }
    return {
      matrices: new Float32Array(matrixData),
      birthRanks,
      count,
    };
  }

  private async readGpuBuffer(source: GPUBuffer, byteLength: number): Promise<ArrayBuffer> {
    const device = this.getWebGpuDevice();
    const bufferSize = alignedSize(byteLength);
    const readBuffer = device.createBuffer({
      size: bufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readBuffer, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    try {
      await readBuffer.mapAsync(GPUMapMode.READ, 0, bufferSize);
      return readBuffer.getMappedRange(0, byteLength).slice(0);
    } finally {
      if (readBuffer.mapState === 'mapped') {
        readBuffer.unmap();
      }
      readBuffer.destroy();
    }
  }

  private getPixelRatio(): number {
    return Math.min(window.devicePixelRatio * 1.15, MAX_PIXEL_RATIO);
  }

  private assertReady(): void {
    this.assertNotDisposed();
    if (!this.initialized) {
      throw new Error('Call DlaRenderer.init() before allocating or reading GPU buffers.');
    }
  }

  private assertNotDisposed(): void {
    if (this.disposed) {
      throw new Error('DlaRenderer has already been disposed.');
    }
  }
}

export function createSphereGeometry(
  detail: number,
  neighborhood: AttachmentNeighborhood = 26,
): SphereGeometryData {
  const subdivisions = [0, 1, 3][clampInteger(detail, 0, 2)] ?? 0;
  const source = new IcosahedronGeometry(0.5, subdivisions);
  const geometry = source.index ? source.toNonIndexed() : source;
  if (geometry !== source) {
    source.dispose();
  }

  const positions = geometry.getAttribute('position');
  const rawPositions = new Float32Array(positions.array);
  const basePositions = calibrateSphereForNeighborhood(rawPositions, neighborhood);
  (positions.array as Float32Array).set(basePositions);
  positions.needsUpdate = true;
  const normals = new Float32Array(positions.count * 3);
  const normal = new Float32Array(3);
  for (let i = 0; i < positions.count; i++) {
    normal[0] = positions.getX(i);
    normal[1] = positions.getY(i);
    normal[2] = positions.getZ(i);
    const length = Math.hypot(normal[0], normal[1], normal[2]) || 1;
    normals[i * 3] = normal[0] / length;
    normals[i * 3 + 1] = normal[1] / length;
    normals[i * 3 + 2] = normal[2] / length;
  }
  geometry.setAttribute('normal', new BufferAttribute(normals, 3));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return {
    geometry,
    rawPositions,
    basePositions,
  };
}

export function applySphereGapToGeometry(
  geometry: BufferGeometry,
  zeroGapPositions: Float32Array,
  gap: number,
): void {
  const positions = geometry.getAttribute('position');
  const array = positions.array as Float32Array;
  if (zeroGapPositions.length !== array.length) {
    throw new Error('Sphere gap geometry does not match its calibrated zero-gap positions.');
  }
  const radiusScale = Math.max(0.01, 1 - Math.max(0, gap));
  for (let index = 0; index < array.length; index += 1) {
    array[index] = zeroGapPositions[index] * radiusScale;
  }
  positions.needsUpdate = true;
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
}

function calibrateSphereForNeighborhood(
  rawPositions: Float32Array,
  neighborhood: AttachmentNeighborhood,
): Float32Array {
  let calibrationScale = 1;
  for (const [offsetX, offsetY, offsetZ] of latticeOffsets(neighborhood)) {
    const distance = Math.hypot(offsetX, offsetY, offsetZ);
    const directionX = offsetX / distance;
    const directionY = offsetY / distance;
    const directionZ = offsetZ / distance;
    let support = 0;
    for (let index = 0; index < rawPositions.length; index += 3) {
      support = Math.max(
        support,
        rawPositions[index] * directionX
          + rawPositions[index + 1] * directionY
          + rawPositions[index + 2] * directionZ,
      );
    }
    if (support > 0) {
      calibrationScale = Math.max(calibrationScale, distance / (2 * support));
    }
  }
  const calibrated = new Float32Array(rawPositions.length);
  for (let index = 0; index < rawPositions.length; index += 1) {
    calibrated[index] = rawPositions[index] * calibrationScale;
  }
  return calibrated;
}

function latticeOffsets(neighborhood: AttachmentNeighborhood): Array<[number, number, number]> {
  const offsets: Array<[number, number, number]> = [];
  for (let z = -1; z <= 1; z += 1) {
    for (let y = -1; y <= 1; y += 1) {
      for (let x = -1; x <= 1; x += 1) {
        const nonzeroAxes = Number(x !== 0) + Number(y !== 0) + Number(z !== 0);
        if (nonzeroAxes === 0) {
          continue;
        }
        if (neighborhood === 6 && nonzeroAxes !== 1) {
          continue;
        }
        if (neighborhood === 18 && nonzeroAxes === 3) {
          continue;
        }
        offsets.push([x, y, z]);
      }
    }
  }
  return offsets;
}

function extentFromRadius(maxRadiusSq: number, sphereScale: number): number {
  const radius = Math.sqrt(Math.max(0, maxRadiusSq));
  return Math.max(24, (radius * 2 + 2) * Math.max(0.01, sphereScale));
}

function normalizeRotationDegrees(value: number): number {
  const wrapped = ((((value + 360) % 720) + 720) % 720) - 360;
  return Math.round(wrapped === -360 ? 360 : wrapped);
}

function clampInteger(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Math.floor(Number.isFinite(value) ? value : min)));
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
}

function displaySettingsEqual(a: DisplaySettings, b: DisplaySettings): boolean {
  return (
    a.innerColor === b.innerColor
    && a.outerColor === b.outerColor
    && a.lightAzimuth === b.lightAzimuth
    && a.lightElevation === b.lightElevation
    && a.keyBrightness === b.keyBrightness
    && a.ambientFill === b.ambientFill
    && a.rimBrightness === b.rimBrightness
    && a.bounceBrightness === b.bounceBrightness
    && a.shadowStrength === b.shadowStrength
    && a.shadowSoftness === b.shadowSoftness
    && a.exposure === b.exposure
    && a.brightness === b.brightness
    && a.contrast === b.contrast
    && a.roughness === b.roughness
    && a.bloomStrength === b.bloomStrength
    && a.bloomRadius === b.bloomRadius
    && a.bloomThreshold === b.bloomThreshold
  );
}

function alignedSize(size: number): number {
  return Math.max(8, Math.ceil(size / 8) * 8);
}

function normalizeError(error: unknown): Error {
  return error instanceof Error ? error : new Error(String(error));
}

export function displayedGradientCount(
  birthRanks: Float32Array,
  count: number,
  seedCount: number,
): number {
  const safeCount = Math.min(Math.max(0, Math.floor(count)), birthRanks.length);
  let newestVisibleBirth = Math.max(0, Math.floor(seedCount) - 1);
  for (let index = 0; index < safeCount; index += 1) {
    const birth = birthRanks[index];
    if (Number.isFinite(birth)) {
      newestVisibleBirth = Math.max(newestVisibleBirth, Math.floor(birth));
    }
  }
  return Math.max(1, Math.floor(seedCount), newestVisibleBirth + 1);
}

function downloadBlob(blob: Blob, filename: string): void {
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.download = filename;
  link.href = url;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
