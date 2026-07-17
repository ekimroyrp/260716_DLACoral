import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import type { AttachmentNeighborhood } from '../src/types';
import {
  applyParticleDimensionsToGeometry,
  createSphereGeometry,
} from '../src/render/dlaRenderer';

describe('renderer icosphere geometry', () => {
  it.each([
    [0, 60],
    [1, 240],
    [2, 960],
  ])('creates resolution %i with %i non-indexed vertices', (resolution, expectedVertices) => {
    const { geometry } = createSphereGeometry(resolution);
    try {
      expect(geometry.index).toBeNull();
      expect(geometry.getAttribute('position').count).toBe(expectedVertices);
      expect(geometry.getAttribute('normal').count).toBe(expectedVertices);
    } finally {
      geometry.dispose();
    }
  });

  it('uses unit radial normals for smooth sphere shading', () => {
    const { geometry } = createSphereGeometry(2);
    try {
      const positions = geometry.getAttribute('position');
      const normals = geometry.getAttribute('normal');
      for (let index = 0; index < positions.count; index += 37) {
        const px = positions.getX(index);
        const py = positions.getY(index);
        const pz = positions.getZ(index);
        const nx = normals.getX(index);
        const ny = normals.getY(index);
        const nz = normals.getZ(index);
        expect(Math.hypot(nx, ny, nz)).toBeCloseTo(1, 6);
        expect(px * nx + py * ny + pz * nz).toBeGreaterThan(0);
      }
    } finally {
      geometry.dispose();
    }
  });

  it.each([0, 1, 2])(
    'uses one fixed resolution %i lattice-contact size with particle size, scale, and proportional gaps',
    (resolution) => {
      const { geometry, basePositions } = createSphereGeometry(resolution);
      try {
        const zeroGapDiameters = axisDiameters(geometry);
        expectAllowedNeighborsContact(geometry, 'full26');
        applyParticleDimensionsToGeometry(geometry, basePositions, 0.5, 0.25, 0.8);
        axisDiameters(geometry).forEach((diameter, axis) => {
          expect(diameter).toBeCloseTo(zeroGapDiameters[axis] * 0.5 * 0.75 * 0.8, 6);
        });
        applyParticleDimensionsToGeometry(geometry, basePositions, 1, 0, 1);
        axisDiameters(geometry).forEach((diameter, axis) => {
          expect(diameter).toBeCloseTo(zeroGapDiameters[axis], 6);
        });
      } finally {
        geometry.dispose();
      }
    },
  );

  it('keeps attachment neighborhood out of sphere geometry updates', () => {
    const rendererSource = readFileSync('src/render/dlaRenderer.ts', 'utf8');
    expect(rendererSource).not.toContain('settings.attachmentNeighborhood');
    expect(rendererSource).not.toContain('setSphereNeighborhood');
    expect(rendererSource).not.toContain('currentNeighborhood');
  });
});

function axisDiameters(
  geometry: ReturnType<typeof createSphereGeometry>['geometry'],
): number[] {
  const positions = geometry.getAttribute('position');
  const extents = [0, 0, 0];
  for (let index = 0; index < positions.count; index += 1) {
    extents[0] = Math.max(extents[0], Math.abs(positions.getX(index)));
    extents[1] = Math.max(extents[1], Math.abs(positions.getY(index)));
    extents[2] = Math.max(extents[2], Math.abs(positions.getZ(index)));
  }
  return extents.map((radius) => radius * 2);
}

function expectAllowedNeighborsContact(
  geometry: ReturnType<typeof createSphereGeometry>['geometry'],
  neighborhood: AttachmentNeighborhood,
): void {
  const positions = geometry.getAttribute('position');
  let minimumMargin = Number.POSITIVE_INFINITY;
  for (const [x, y, z] of offsets(neighborhood)) {
    const distance = Math.hypot(x, y, z);
    let support = 0;
    for (let index = 0; index < positions.count; index += 1) {
      support = Math.max(
        support,
        (positions.getX(index) * x + positions.getY(index) * y + positions.getZ(index) * z)
          / distance,
      );
    }
    const margin = 2 * support - distance;
    expect(margin).toBeGreaterThanOrEqual(-0.000001);
    minimumMargin = Math.min(minimumMargin, margin);
  }
  expect(minimumMargin).toBeCloseTo(0, 6);
}

function offsets(neighborhood: AttachmentNeighborhood): Array<[number, number, number]> {
  const result: Array<[number, number, number]> = [];
  for (let z = -1; z <= 1; z += 1) {
    for (let y = -1; y <= 1; y += 1) {
      for (let x = -1; x <= 1; x += 1) {
        const axes = Number(x !== 0) + Number(y !== 0) + Number(z !== 0);
        if (
          axes === 0
          || (neighborhood === 'faces6' && axes !== 1)
          || (neighborhood === 'facesEdges18' && axes === 3)
        ) {
          continue;
        }
        result.push([x, y, z]);
      }
    }
  }
  return result;
}
