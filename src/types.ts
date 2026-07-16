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
  rotation: number;
  sphereScale: number;
  sphereGap: number;
  sphereDetail: number;
  hideEnclosed: boolean;
}

export interface DisplaySettings {
  innerColor: string;
  outerColor: string;
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
  rotation: 0,
  sphereScale: 1,
  sphereGap: 0,
  sphereDetail: 0,
  hideEnclosed: true,
};

export const DEFAULT_DISPLAY_SETTINGS: DisplaySettings = {
  innerColor: '#6b2f24',
  outerColor: '#f4e6d2',
  lightAzimuth: 25.65,
  lightElevation: 68.7,
  keyBrightness: 2.41,
  ambientFill: 0.3,
  rimBrightness: 0.49,
  bounceBrightness: 0.07,
  shadowStrength: 1.08,
  shadowSoftness: 2.6,
  exposure: 0.7,
  brightness: 1,
  contrast: 2.25,
  roughness: 0.92,
  bloomStrength: 0.08,
  bloomRadius: 0.26,
  bloomThreshold: 0,
};

export const MAX_HISTORY_ACTIONS = 120;
export const MAX_HISTORY_BYTES = 128 * 1024 * 1024;
