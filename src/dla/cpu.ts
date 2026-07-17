import type { AttachmentNeighborhood, Int3, SeedShape } from '../types';

const UINT32_RANGE = 0x1_0000_0000;
export const HASH_COORD_MIN = -512;
export const HASH_COORD_MAX = 511;

export interface CellLookup {
  has(position: Int3): boolean;
}

export interface CandidateOptions {
  neighborhood: AttachmentNeighborhood;
  stickNeighbors: number;
  stickChance: number;
  /** A deterministic value in [0, 1). */
  roll: number;
}

export interface CandidateResult {
  accepted: boolean;
  neighborCount: number;
}

export interface RgbColor {
  r: number;
  g: number;
  b: number;
}

export interface NeighborMetadata {
  neighborCounts: Uint8Array;
  /** Birth rank at which the particle first became fully enclosed, or UINT32_MAX. */
  enclosedAt: Uint32Array;
}

/** Preferred stick threshold, reduced only when the epoch cannot attain it. */
export function effectiveStickThreshold(requested: number, maximumCandidateNeighbors: number): number {
  const preferred = Math.max(1, Math.floor(Number.isFinite(requested) ? requested : 1));
  const attainable = Math.max(1, Math.floor(Number.isFinite(maximumCandidateNeighbors) ? maximumCandidateNeighbors : 1));
  return Math.min(preferred, attainable);
}

export function hash32(value: number): number {
  let next = value >>> 0;
  next = (next ^ (next >>> 16)) >>> 0;
  next = Math.imul(next, 0x7feb_352d) >>> 0;
  next = (next ^ (next >>> 15)) >>> 0;
  next = Math.imul(next, 0x846c_a68b) >>> 0;
  return (next ^ (next >>> 16)) >>> 0;
}

export function hashCell(position: Int3): number {
  const x = hash32(position.x >>> 0);
  const y = hash32((position.y >>> 0) ^ 0x9e37_79b9);
  const z = hash32((position.z >>> 0) ^ 0x85eb_ca6b);
  return hash32(x ^ rotateLeft(y, 11) ^ rotateLeft(z, 22));
}

/** Exact 30-bit key used by the compact GPU hash; zero remains the empty sentinel. */
export function packCellKey(position: Int3): number | undefined {
  if (
    position.x < HASH_COORD_MIN ||
    position.x > HASH_COORD_MAX ||
    position.y < HASH_COORD_MIN ||
    position.y > HASH_COORD_MAX ||
    position.z < HASH_COORD_MIN ||
    position.z > HASH_COORD_MAX
  ) {
    return undefined;
  }
  const x = position.x - HASH_COORD_MIN;
  const y = position.y - HASH_COORD_MIN;
  const z = position.z - HASH_COORD_MIN;
  return ((x | (y << 10) | (z << 20)) + 1) >>> 0;
}

export function unpackCellKey(key: number): Int3 | undefined {
  const normalized = key >>> 0;
  if (normalized === 0 || normalized > 0x4000_0000) {
    return undefined;
  }
  const packed = normalized - 1;
  return {
    x: (packed & 1023) + HASH_COORD_MIN,
    y: ((packed >>> 10) & 1023) + HASH_COORD_MIN,
    z: ((packed >>> 20) & 1023) + HASH_COORD_MIN,
  };
}

export function nextPowerOfTwo(value: number): number {
  if (!Number.isFinite(value) || value <= 1) {
    return 1;
  }
  return 2 ** Math.ceil(Math.log2(value));
}

export function getNeighborOffsets(neighborhood: AttachmentNeighborhood): readonly Int3[] {
  return NEIGHBOR_OFFSETS[neighborhood];
}

export function countOccupiedNeighbors(
  position: Int3,
  occupied: CellLookup,
  neighborhood: AttachmentNeighborhood = 26,
): number {
  let count = 0;
  for (const offset of getNeighborOffsets(neighborhood)) {
    if (
      occupied.has({
        x: position.x + offset.x,
        y: position.y + offset.y,
        z: position.z + offset.z,
      })
    ) {
      count += 1;
    }
  }
  return count;
}

export function evaluateCandidate(
  position: Int3,
  occupied: CellLookup,
  options: CandidateOptions,
): CandidateResult {
  if (occupied.has(position)) {
    return { accepted: false, neighborCount: 0 };
  }
  const neighborCount = countOccupiedNeighbors(position, occupied, options.neighborhood);
  const required = Math.max(1, Math.min(options.neighborhood, Math.floor(options.stickNeighbors)));
  const chance = Math.max(0, Math.min(1, options.stickChance));
  return {
    accepted: neighborCount >= required && options.roll < chance,
    neighborCount,
  };
}

export function seedLatticeRadius(radius: number, particleSize = 1): number {
  const safeRadius = Number.isFinite(radius) ? radius : 1;
  const safeParticleSize = Math.max(0.001, Number.isFinite(particleSize) ? particleSize : 1);
  return Math.max(1, Math.round(safeRadius / safeParticleSize));
}

export function generateSeedPositions(shape: SeedShape, radius = 1, particleSize = 1): Int3[] {
  if (shape === 'point') {
    return [{ x: 0, y: 0, z: 0 }];
  }

  const r = seedLatticeRadius(radius, particleSize);
  const innerSq = (r - 0.5) ** 2;
  const outerSq = (r + 0.5) ** 2;
  const positions: Int3[] = [];

  if (shape === 'ring') {
    for (let z = -r; z <= r; z += 1) {
      appendShellX(positions, innerSq, outerSq, z * z, 0, z);
    }
  } else {
    for (let z = -r; z <= r; z += 1) {
      for (let y = -r; y <= r; y += 1) {
        appendShellX(positions, innerSq, outerSq, y * y + z * z, y, z);
      }
    }
  }

  // Very small radii can miss the half-cell shell on some integer lattices.
  return positions.length > 0 ? positions : [{ x: r, y: 0, z: 0 }];
}

/** Exact seed size without allocating per-cell objects. */
export function countSeedPositions(shape: SeedShape, radius = 1, particleSize = 1): number {
  if (shape === 'point') {
    return 1;
  }

  const r = seedLatticeRadius(radius, particleSize);
  const innerSq = (r - 0.5) ** 2;
  const outerSq = (r + 0.5) ** 2;
  let count = 0;
  if (shape === 'ring') {
    for (let z = -r; z <= r; z += 1) {
      count += shellXCount(innerSq, outerSq, z * z);
    }
    return Math.max(1, count);
  }

  for (let z = -r; z <= r; z += 1) {
    for (let y = -r; y <= r; y += 1) {
      count += shellXCount(innerSq, outerSq, y * y + z * z);
    }
  }
  return Math.max(1, count);
}

/** Largest exact structural seed radius that fits both particle and lattice capacity. */
export function maxSeedRadiusForCapacity(
  shape: SeedShape,
  particleCapacity: number,
  latticeMaximum: number,
  particleSize = 1,
): number {
  const safeParticleSize = Math.max(0.001, Number.isFinite(particleSize) ? particleSize : 1);
  const maximum = Math.max(1, Math.floor(latticeMaximum * safeParticleSize));
  if (shape === 'point') {
    return maximum;
  }
  const capacity = Math.max(1, Math.floor(particleCapacity));
  let low = 1;
  let high = maximum;
  let result = 1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (countSeedPositions(shape, middle, safeParticleSize) <= capacity) {
      result = middle;
      low = middle + 1;
    } else {
      high = middle - 1;
    }
  }
  return result;
}

function appendShellX(
  positions: Int3[],
  innerSq: number,
  outerSq: number,
  yzDistanceSq: number,
  y: number,
  z: number,
): void {
  const outerRemainder = outerSq - yzDistanceSq;
  if (outerRemainder < 0) {
    return;
  }
  const maxX = Math.floor(Math.sqrt(outerRemainder));
  const innerRemainder = innerSq - yzDistanceSq;
  const minAbsX = innerRemainder > 0 ? Math.ceil(Math.sqrt(innerRemainder)) : 0;
  if (minAbsX === 0) {
    for (let x = -maxX; x <= maxX; x += 1) {
      positions.push({ x, y, z });
    }
    return;
  }
  for (let x = -maxX; x <= -minAbsX; x += 1) {
    positions.push({ x, y, z });
  }
  for (let x = minAbsX; x <= maxX; x += 1) {
    positions.push({ x, y, z });
  }
}

function shellXCount(innerSq: number, outerSq: number, yzDistanceSq: number): number {
  const outerRemainder = outerSq - yzDistanceSq;
  if (outerRemainder < 0) {
    return 0;
  }
  const maxX = Math.floor(Math.sqrt(outerRemainder));
  const innerRemainder = innerSq - yzDistanceSq;
  const minAbsX = innerRemainder > 0 ? Math.ceil(Math.sqrt(innerRemainder)) : 0;
  if (minAbsX === 0) {
    return maxX * 2 + 1;
  }
  return minAbsX <= maxX ? (maxX - minAbsX + 1) * 2 : 0;
}

export function uniformSphereLaunch(seed: number, walkerIndex: number, radius: number): Int3 {
  let state = hash32((seed >>> 0) ^ Math.imul((walkerIndex + 1) >>> 0, 0x9e37_79b9));
  state = xorshift32(state);
  const u = state / UINT32_RANGE;
  state = xorshift32(state);
  const v = state / UINT32_RANGE;
  const theta = Math.PI * 2 * u;
  const phi = Math.acos(1 - 2 * v);
  const sinPhi = Math.sin(phi);
  const safeRadius = Math.max(1, radius);
  return {
    x: Math.round(safeRadius * sinPhi * Math.cos(theta)),
    y: Math.round(safeRadius * Math.cos(phi)),
    z: Math.round(safeRadius * sinPhi * Math.sin(theta)),
  };
}

export function isOutsideKillRadius(position: Int3, launchRadius: number, killPadding: number): boolean {
  const killRadius = Math.max(1, launchRadius) + Math.max(1, killPadding);
  return position.x ** 2 + position.y ** 2 + position.z ** 2 > killRadius ** 2;
}

export function colorAtBirth(innerHex: string, outerHex: string, birthRank: number, newestRank: number): RgbColor {
  const inner = parseHexColor(innerHex);
  const outer = parseHexColor(outerHex);
  const t = newestRank <= 0 ? 0 : Math.max(0, Math.min(1, birthRank / newestRank));
  return {
    r: inner.r + (outer.r - inner.r) * t,
    g: inner.g + (outer.g - inner.g) * t,
    b: inner.b + (outer.b - inner.b) * t,
  };
}

export function isFullyEnclosed(position: Int3, occupied: CellLookup): boolean {
  return countOccupiedNeighbors(position, occupied, 26) === 26;
}

export function buildNeighborMetadata(positions: readonly Int3[]): NeighborMetadata {
  const occupied = new SparseCellHash(Math.max(8, positions.length * 2));
  positions.forEach((position, birth) => occupied.set(position, birth));
  const neighborCounts = new Uint8Array(positions.length);
  const enclosedAt = new Uint32Array(positions.length);
  enclosedAt.fill(0xffff_ffff);
  positions.forEach((position, birth) => {
    let count = 0;
    let latestNeighborBirth = 0;
    for (const offset of getNeighborOffsets(26)) {
      const neighborBirth = occupied.get({
        x: position.x + offset.x,
        y: position.y + offset.y,
        z: position.z + offset.z,
      });
      if (neighborBirth !== undefined) {
        count += 1;
        latestNeighborBirth = Math.max(latestNeighborBirth, neighborBirth);
      }
    }
    neighborCounts[birth] = count;
    if (count === 26) {
      enclosedAt[birth] = latestNeighborBirth;
    }
  });
  return { neighborCounts, enclosedAt };
}

export function selectGrowthBatch(
  candidates: readonly Int3[],
  occupied: CellLookup,
  growthBatch: number,
): Int3[] {
  const limit = Math.max(1, Math.floor(growthBatch));
  const selected: Int3[] = [];
  const seen = new Set<string>();
  for (const candidate of candidates) {
    const key = cellKey(candidate);
    if (!occupied.has(candidate) && !seen.has(key)) {
      selected.push({ ...candidate });
      seen.add(key);
      if (selected.length === limit) {
        break;
      }
    }
  }
  return selected;
}

export function cellKey(position: Int3): string {
  return `${position.x},${position.y},${position.z}`;
}

/**
 * A deterministic CPU counterpart to the GPU open-addressed table. It is used
 * by tests, import validation, and explicit snapshot tooling, never by the live
 * simulation loop.
 */
export class SparseCellHash implements CellLookup {
  private keys: Array<Int3 | undefined>;
  private values: Uint32Array;
  private mask: number;
  private entryCount = 0;

  constructor(initialCapacity = 16) {
    const capacity = Math.max(8, nextPowerOfTwo(initialCapacity));
    this.keys = new Array<Int3 | undefined>(capacity);
    this.values = new Uint32Array(capacity);
    this.mask = capacity - 1;
  }

  get size(): number {
    return this.entryCount;
  }

  get capacity(): number {
    return this.keys.length;
  }

  has(position: Int3): boolean {
    return this.findSlot(position).found;
  }

  get(position: Int3): number | undefined {
    const slot = this.findSlot(position);
    return slot.found ? this.values[slot.index] : undefined;
  }

  set(position: Int3, birth: number): boolean {
    if ((this.entryCount + 1) * 2 >= this.keys.length) {
      this.resize(this.keys.length * 2);
    }
    const slot = this.findSlot(position);
    if (slot.found) {
      this.values[slot.index] = birth >>> 0;
      return false;
    }
    this.keys[slot.index] = { ...position };
    this.values[slot.index] = birth >>> 0;
    this.entryCount += 1;
    return true;
  }

  private findSlot(position: Int3): { index: number; found: boolean } {
    let index = hashCell(position) & this.mask;
    for (let probe = 0; probe < this.keys.length; probe += 1) {
      const key = this.keys[index];
      if (!key) {
        return { index, found: false };
      }
      if (key.x === position.x && key.y === position.y && key.z === position.z) {
        return { index, found: true };
      }
      index = (index + 1) & this.mask;
    }
    throw new Error('SparseCellHash is full.');
  }

  private resize(capacity: number): void {
    const previousKeys = this.keys;
    const previousValues = this.values;
    this.keys = new Array<Int3 | undefined>(capacity);
    this.values = new Uint32Array(capacity);
    this.mask = capacity - 1;
    this.entryCount = 0;
    previousKeys.forEach((key, index) => {
      if (key) {
        this.set(key, previousValues[index] ?? 0);
      }
    });
  }
}

function parseHexColor(hex: string): RgbColor {
  const normalized = hex.trim().replace(/^#/, '');
  const expanded = normalized.length === 3 ? normalized.replace(/(.)/g, '$1$1') : normalized;
  if (!/^[0-9a-fA-F]{6}$/.test(expanded)) {
    throw new Error(`Invalid RGB color: ${hex}`);
  }
  const packed = Number.parseInt(expanded, 16);
  return {
    r: ((packed >>> 16) & 255) / 255,
    g: ((packed >>> 8) & 255) / 255,
    b: (packed & 255) / 255,
  };
}

function xorshift32(value: number): number {
  let next = value >>> 0 || 0xa341_316c;
  next ^= next << 13;
  next ^= next >>> 17;
  next ^= next << 5;
  return next >>> 0;
}

function rotateLeft(value: number, shift: number): number {
  return ((value << shift) | (value >>> (32 - shift))) >>> 0;
}

function createNeighborOffsets(): Record<AttachmentNeighborhood, readonly Int3[]> {
  const faces: Int3[] = [];
  const edges: Int3[] = [];
  const corners: Int3[] = [];
  for (let z = -1; z <= 1; z += 1) {
    for (let y = -1; y <= 1; y += 1) {
      for (let x = -1; x <= 1; x += 1) {
        const distance = Math.abs(x) + Math.abs(y) + Math.abs(z);
        if (distance === 1) {
          faces.push({ x, y, z });
        } else if (distance === 2) {
          edges.push({ x, y, z });
        } else if (distance === 3) {
          corners.push({ x, y, z });
        }
      }
    }
  }
  return {
    6: Object.freeze(faces),
    18: Object.freeze([...faces, ...edges]),
    26: Object.freeze([...faces, ...edges, ...corners]),
  };
}

const NEIGHBOR_OFFSETS = createNeighborOffsets();
