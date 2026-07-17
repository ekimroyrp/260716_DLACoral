import { describe, expect, it } from 'vitest';
import {
  SparseCellHash,
  MAX_COMPACT_SEED_RADIUS,
  buildNeighborMetadata,
  colorAtBirth,
  countSeedPositions,
  effectiveStickThreshold,
  countOccupiedNeighbors,
  evaluateCandidate,
  generateSeedPositions,
  getNeighborOffsets,
  hashCell,
  initialSparseHashCapacity,
  isOutsideKillRadius,
  maxSeedRadiusForCapacity,
  packCellKey,
  projectedSparseHashCapacity,
  selectGrowthBatch,
  seedLatticeRadius,
  uniformSphereLaunch,
  unpackCellKey,
} from '../src/dla';
import type { Int3 } from '../src/types';

describe('DLA seed generation', () => {
  it('creates the classic one-cell point seed', () => {
    expect(generateSeedPositions('point', 64)).toEqual([{ x: 0, y: 0, z: 0 }]);
  });

  it('creates a unique one-cell-thick spherical shell', () => {
    const radius = 5;
    const seed = generateSeedPositions('sphere', radius);
    const keys = new Set(seed.map(({ x, y, z }) => `${x},${y},${z}`));
    expect(keys.size).toBe(seed.length);
    expect(seed.length).toBeGreaterThan(50);
    for (const { x, y, z } of seed) {
      const distance = Math.sqrt(x * x + y * y + z * z);
      expect(Math.abs(distance - radius)).toBeLessThanOrEqual(0.5);
    }
  });

  it('creates a planar circular ring', () => {
    const seed = generateSeedPositions('ring', 7);
    expect(seed.length).toBeGreaterThan(20);
    expect(seed.every(({ y }) => y === 0)).toBe(true);
  });

  it('packs a fixed-radius seed with more smaller particles and fewer larger particles', () => {
    const smallCount = countSeedPositions('ring', 8, 0.5);
    const defaultCount = countSeedPositions('ring', 8, 1);
    const largeCount = countSeedPositions('ring', 8, 2);
    expect(smallCount).toBeGreaterThan(defaultCount);
    expect(defaultCount).toBeGreaterThan(largeCount);
    expect(seedLatticeRadius(8, 0.5)).toBe(16);
    expect(seedLatticeRadius(8, 2)).toBe(4);

    for (const particleSize of [0.5, 1, 2]) {
      const seed = generateSeedPositions('ring', 8, particleSize);
      const worldRadii = seed.map(({ x, z }) => Math.hypot(x, z) * particleSize);
      expect(Math.max(...worldRadii.map((radius) => Math.abs(radius - 8))))
        .toBeLessThanOrEqual(particleSize * 0.51);
    }
  });

  it('keeps larger shell generation deterministic and unique', () => {
    const first = generateSeedPositions('sphere', 32);
    const second = generateSeedPositions('sphere', 32);
    expect(second).toEqual(first);
    expect(new Set(first.map(({ x, y, z }) => `${x},${y},${z}`)).size).toBe(first.length);
  });

  it('counts structural seeds before allocation and finds the capacity-safe radius', () => {
    for (const shape of ['point', 'sphere', 'ring'] as const) {
      for (const radius of [1, 5, 16]) {
        expect(countSeedPositions(shape, radius)).toBe(generateSeedPositions(shape, radius).length);
      }
    }
    const maximum = maxSeedRadiusForCapacity('sphere', 25_000, MAX_COMPACT_SEED_RADIUS);
    expect(countSeedPositions('sphere', maximum)).toBeLessThanOrEqual(25_000);
    expect(countSeedPositions('sphere', maximum + 1)).toBeGreaterThan(25_000);
    expect(maxSeedRadiusForCapacity('ring', 25_000, MAX_COMPACT_SEED_RADIUS)).toBe(MAX_COMPACT_SEED_RADIUS);
  });
});

describe('DLA deterministic walker helpers', () => {
  it('launches the same walker at the same uniformly sampled sphere position', () => {
    expect(uniformSphereLaunch(260716, 42, 30)).toEqual(uniformSphereLaunch(260716, 42, 30));
    expect(uniformSphereLaunch(260716, 42, 30)).not.toEqual(uniformSphereLaunch(260716, 43, 30));
    const launch = uniformSphereLaunch(260716, 42, 30);
    const distance = Math.sqrt(launch.x ** 2 + launch.y ** 2 + launch.z ** 2);
    expect(distance).toBeGreaterThanOrEqual(29);
    expect(distance).toBeLessThanOrEqual(31);
  });

  it('recycles only beyond launch radius plus kill padding', () => {
    expect(isOutsideKillRadius({ x: 13, y: 0, z: 0 }, 10, 3)).toBe(false);
    expect(isOutsideKillRadius({ x: 14, y: 0, z: 0 }, 10, 3)).toBe(true);
  });
});

describe('DLA sparse occupancy and neighbor metadata', () => {
  it('starts the GPU hash from the active aggregate and grows with bounded headroom', () => {
    const maximum = 8_388_608;
    expect(initialSparseHashCapacity(1, maximum)).toBe(64);
    expect(initialSparseHashCapacity(1_000_000, maximum)).toBe(8_388_608);
    expect(projectedSparseHashCapacity(64, 27, 256, maximum)).toBe(16_384);
    expect(projectedSparseHashCapacity(4_096, 2_800, 40, maximum)).toBe(8_192);
    expect(projectedSparseHashCapacity(8_192, 100, 0, maximum)).toBe(8_192);
    expect(projectedSparseHashCapacity(maximum, maximum - 1, 4_096, maximum)).toBe(maximum);
    expect(() => initialSparseHashCapacity(1_100_000, maximum)).toThrow(/above the device limit/);
  });

  it('reserves one compact-lattice cell around the largest structural seed', () => {
    expect(MAX_COMPACT_SEED_RADIUS).toBe(510);
    expect(packCellKey({ x: MAX_COMPACT_SEED_RADIUS + 1, y: 0, z: 0 })).toBeDefined();
  });

  it('packs compact hash coordinates exactly without key collisions', () => {
    const samples = [
      { x: -512, y: -512, z: -512 },
      { x: 511, y: 511, z: 511 },
      { x: 0, y: 0, z: 0 },
      { x: -17, y: 42, z: 301 },
    ];
    const keys = samples.map((position) => packCellKey(position));
    expect(new Set(keys).size).toBe(samples.length);
    samples.forEach((position, index) => expect(unpackCellKey(keys[index] ?? 0)).toEqual(position));
    expect(packCellKey({ x: 512, y: 0, z: 0 })).toBeUndefined();
    expect(packCellKey({ x: 0, y: -513, z: 0 })).toBeUndefined();
  });

  it('uses the exact 6, 18, and 26 neighborhoods', () => {
    expect(getNeighborOffsets(6)).toHaveLength(6);
    expect(getNeighborOffsets(18)).toHaveLength(18);
    expect(getNeighborOffsets(26)).toHaveLength(26);
  });

  it('resolves open-addressing collisions without losing birth ranks', () => {
    const groups = new Map<number, Int3[]>();
    for (let x = -100; x <= 100; x += 1) {
      const position = { x, y: x % 3, z: 0 };
      const bucket = hashCell(position) & 7;
      const group = groups.get(bucket) ?? [];
      group.push(position);
      groups.set(bucket, group);
    }
    const collision = [...groups.values()].find((group) => group.length >= 3);
    expect(collision).toBeDefined();
    const hash = new SparseCellHash(8);
    collision?.slice(0, 3).forEach((position, birth) => hash.set(position, birth + 10));
    collision?.slice(0, 3).forEach((position, birth) => expect(hash.get(position)).toBe(birth + 10));
  });

  it('computes cached counts and the birth rank that encloses a particle', () => {
    const center = { x: 0, y: 0, z: 0 };
    const positions = [center, ...getNeighborOffsets(26)];
    const metadata = buildNeighborMetadata(positions);
    expect(metadata.neighborCounts[0]).toBe(26);
    expect(metadata.enclosedAt[0]).toBe(26);
  });

  it('counts neighbors against sparse occupancy', () => {
    const occupied = new SparseCellHash();
    getNeighborOffsets(26).forEach((position, birth) => occupied.set(position, birth));
    expect(countOccupiedNeighbors({ x: 0, y: 0, z: 0 }, occupied, 6)).toBe(6);
    expect(countOccupiedNeighbors({ x: 0, y: 0, z: 0 }, occupied, 18)).toBe(18);
    expect(countOccupiedNeighbors({ x: 0, y: 0, z: 0 }, occupied, 26)).toBe(26);
  });
});

describe('DLA candidate and batch rules', () => {
  const occupied = new SparseCellHash();
  occupied.set({ x: 0, y: 0, z: 0 }, 0);

  it('applies neighbor thresholds and deterministic stick chance', () => {
    expect(
      evaluateCandidate(
        { x: 1, y: 0, z: 0 },
        occupied,
        { neighborhood: 6, stickNeighbors: 1, stickChance: 0.5, roll: 0.49 },
      ).accepted,
    ).toBe(true);
    expect(
      evaluateCandidate(
        { x: 1, y: 0, z: 0 },
        occupied,
        { neighborhood: 6, stickNeighbors: 2, stickChance: 1, roll: 0 },
      ).accepted,
    ).toBe(false);
    expect(
      evaluateCandidate(
        { x: 1, y: 0, z: 0 },
        occupied,
        { neighborhood: 6, stickNeighbors: 1, stickChance: 0.5, roll: 0.5 },
      ).accepted,
    ).toBe(false);
  });

  it('uses the preferred stick threshold when attainable and bootstraps to the densest frontier', () => {
    expect(effectiveStickThreshold(6, 1)).toBe(1);
    expect(effectiveStickThreshold(6, 4)).toBe(4);
    expect(effectiveStickThreshold(2, 4)).toBe(2);
    expect(effectiveStickThreshold(18, 0)).toBe(1);
    expect(
      evaluateCandidate(
        { x: 1, y: 0, z: 0 },
        occupied,
        {
          neighborhood: 6,
          stickNeighbors: effectiveStickThreshold(6, 1),
          stickChance: 1,
          roll: 0,
        },
      ).accepted,
    ).toBe(true);
  });

  it('makes Growth Batch 1 strict and larger batches bounded and de-duplicated', () => {
    const candidates = [
      { x: 1, y: 0, z: 0 },
      { x: 1, y: 0, z: 0 },
      { x: 0, y: 0, z: 0 },
      { x: 0, y: 1, z: 0 },
      { x: 0, y: 0, z: 1 },
    ];
    expect(selectGrowthBatch(candidates, occupied, 1)).toEqual([{ x: 1, y: 0, z: 0 }]);
    expect(selectGrowthBatch(candidates, occupied, 2)).toEqual([
      { x: 1, y: 0, z: 0 },
      { x: 0, y: 1, z: 0 },
    ]);
  });
});

describe('DLA timeline reconstruction and branching', () => {
  it('recomputes cached enclosure metadata from the displayed birth prefix', () => {
    const positions = [{ x: 0, y: 0, z: 0 }, ...getNeighborOffsets(26)];
    const partial = buildNeighborMetadata(positions.slice(0, 20));
    const complete = buildNeighborMetadata(positions);
    expect(partial.neighborCounts[0]).toBe(19);
    expect(partial.enclosedAt[0]).toBe(0xffff_ffff);
    expect(complete.neighborCounts[0]).toBe(26);
    expect(complete.enclosedAt[0]).toBe(26);
  });

  it('makes trimmed future cells available to a deterministic branch', () => {
    const seedOnly = new SparseCellHash();
    seedOnly.set({ x: 0, y: 0, z: 0 }, 0);
    const withFuture = new SparseCellHash();
    withFuture.set({ x: 0, y: 0, z: 0 }, 0);
    withFuture.set({ x: 1, y: 0, z: 0 }, 1);
    const options = { neighborhood: 6 as const, stickNeighbors: 1, stickChance: 1, roll: 0 };
    expect(evaluateCandidate({ x: 1, y: 0, z: 0 }, withFuture, options).accepted).toBe(false);
    expect(evaluateCandidate({ x: 1, y: 0, z: 0 }, seedOnly, options).accepted).toBe(true);
  });
});

describe('DLA birth color interpolation', () => {
  it('keeps the oldest inner, newest outer, and midpoint linear', () => {
    expect(colorAtBirth('#000000', '#ffffff', 0, 10)).toEqual({ r: 0, g: 0, b: 0 });
    expect(colorAtBirth('#000000', '#ffffff', 10, 10)).toEqual({ r: 1, g: 1, b: 1 });
    expect(colorAtBirth('#000000', '#ffffff', 5, 10)).toEqual({ r: 0.5, g: 0.5, b: 0.5 });
  });
});
