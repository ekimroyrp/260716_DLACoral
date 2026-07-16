export {
  SparseCellHash,
  HASH_COORD_MAX,
  HASH_COORD_MIN,
  buildNeighborMetadata,
  cellKey,
  colorAtBirth,
  countSeedPositions,
  countOccupiedNeighbors,
  evaluateCandidate,
  effectiveStickThreshold,
  generateSeedPositions,
  getNeighborOffsets,
  hash32,
  hashCell,
  isFullyEnclosed,
  isOutsideKillRadius,
  maxSeedRadiusForCapacity,
  nextPowerOfTwo,
  packCellKey,
  selectGrowthBatch,
  uniformSphereLaunch,
  unpackCellKey,
} from './cpu';
export type { CandidateOptions, CandidateResult, CellLookup, NeighborMetadata, RgbColor } from './cpu';

export {
  GpuDlaSimulator,
  MAX_COMPACT_SEED_RADIUS,
  initialSparseHashCapacity,
  projectedSparseHashCapacity,
} from './gpuDla';
export type {
  DlaStatus,
  DlaStepResult,
  GpuDlaSnapshot,
  GpuDlaInstanceTargets,
  GpuDlaLimits,
} from './gpuDla';
