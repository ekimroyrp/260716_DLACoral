export type SeedShape = 'point' | 'sphere' | 'ring';
export type AttachmentNeighborhood = 6 | 18 | 26;

export interface SimulationSettings {
  running: boolean;
  rate: number;
  timeline: number;
  latestTimeline: number;
}

export interface DlaSettings {
  seed: number;
  seedShape: SeedShape;
  seedRadius: number;
  targetParticles: number;
  attachmentNeighborhood: AttachmentNeighborhood;
  stickNeighbors: number;
  stickChance: number;
  launchPadding: number;
  killPadding: number;
  growthBatch: number;
  walkerPool: number;
  seedRotation: number;
  particleSize: number;
  particleGap: number;
  particleScale: number;
  particleResolution: number;
  hideEnclosed: boolean;
}

export interface DisplaySettings {
  innerColor: string;
  outerColor: string;
  gradientContrast: number;
  gradientBias: number;
  gradientBlur: number;
  lightAzimuth: number;
  lightElevation: number;
  keyBrightness: number;
  ambientFill: number;
  rimBrightness: number;
  bounceBrightness: number;
  shadowStrength: number;
  shadowSoftness: number;
  exposure: number;
  brightness: number;
  contrast: number;
  roughness: number;
  bloomStrength: number;
  bloomRadius: number;
  bloomThreshold: number;
}

export interface Int3 {
  x: number;
  y: number;
  z: number;
}

export interface DlaSnapshot {
  positions: Int32Array;
  enclosed: Uint8Array;
  seedCount: number;
  currentCount: number;
  latestCount: number;
  maxRadiusSq: number;
  rngState: number;
  branchSerial: number;
}

export interface AppSnapshot {
  simulation: SimulationSettings;
  dla: DlaSettings;
  display: DisplaySettings;
  aggregate?: DlaSnapshot;
}

export const DEFAULT_SIMULATION_SETTINGS: SimulationSettings = {
  running: false,
  rate: 1,
  timeline: 0,
  latestTimeline: 0,
};

export const DEFAULT_DLA_SETTINGS: DlaSettings = {
  seed: 260716,
  seedShape: 'point',
  seedRadius: 8,
  targetParticles: 1_000_000,
  attachmentNeighborhood: 26,
  stickNeighbors: 1,
  stickChance: 1,
  launchPadding: 3,
  killPadding: 3,
  growthBatch: 256,
  walkerPool: 65_536,
  seedRotation: 0,
  particleSize: 1,
  particleGap: 0,
  particleScale: 1,
  particleResolution: 2,
  hideEnclosed: true,
};

export const DEFAULT_DISPLAY_SETTINGS: DisplaySettings = {
  innerColor: '#ac2a4a',
  outerColor: '#ffffff',
  gradientContrast: 1.37,
  gradientBias: -0.74,
  gradientBlur: 0.45,
  lightAzimuth: -3.08,
  lightElevation: 55.79,
  keyBrightness: 3.37,
  ambientFill: 0.8,
  rimBrightness: 0.49,
  bounceBrightness: 0.45,
  shadowStrength: 1.13,
  shadowSoftness: 2.09,
  exposure: 0.68,
  brightness: 1.15,
  contrast: 2.55,
  roughness: 0,
  bloomStrength: 0.13,
  bloomRadius: 0.24,
  bloomThreshold: 0.19,
};

export const MAX_HISTORY_ACTIONS = 120;
export const MAX_HISTORY_BYTES = 128 * 1024 * 1024;
