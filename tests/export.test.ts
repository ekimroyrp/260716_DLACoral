import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import {
  createAgeGradientColors,
  createDisplayColor,
  createGlbBlob,
  createObjBlob,
  shapeAgeGradient,
  type ExportInstanceData,
} from '../src/render';
import { displayedGradientCount } from '../src/render/dlaRenderer';

function exportData(overrides: Partial<ExportInstanceData> = {}): ExportInstanceData {
  return {
    matrices: new Float32Array(5 * 16),
    birthRanks: new Float32Array([0, 1, 2, 3, 4]),
    count: 5,
    seedCount: 2,
    gradientCount: 5,
    spherePositions: new Float32Array(9),
    sphereNormals: new Float32Array(9),
    seedRotationDegrees: 0,
    innerColor: '#000000',
    outerColor: '#ffffff',
    gradientContrast: 1,
    gradientBias: 0,
    gradientBlur: 0,
    brightness: 1,
    contrast: 1,
    materialRoughness: 0.92,
    ...overrides,
  };
}

describe('export age colors', () => {
  it('matches the reference renderer raw hex-channel convention', () => {
    const color = createDisplayColor('#4fceee');
    expect(color.r).toBeCloseTo(0x4f / 0xff, 6);
    expect(color.g).toBeCloseTo(0xce / 0xff, 6);
    expect(color.b).toBeCloseTo(0xee / 0xff, 6);
  });

  it('keeps every seed at the inner color and makes the newest attachment exactly outer', () => {
    const colors = createAgeGradientColors(exportData());
    expect([...colors.slice(0, 6)]).toEqual([0, 0, 0, 0, 0, 0]);
    expect([...colors.slice(12, 15)]).toEqual([1, 1, 1]);
  });

  it('keeps seed albedo exactly at Gradient Start regardless of image grading', () => {
    const colors = createAgeGradientColors(exportData({
      innerColor: '#ac2a4a',
      brightness: 10,
      contrast: 10,
    }));
    expect(colors[0]).toBeCloseTo(0xac / 0xff, 6);
    expect(colors[1]).toBeCloseTo(0x2a / 0xff, 6);
    expect(colors[2]).toBeCloseTo(0x4a / 0xff, 6);
    expect([...colors.slice(0, 3)]).toEqual([...colors.slice(3, 6)]);
  });

  it('interpolates attached birth ranks and clamps display grading', () => {
    const colors = createAgeGradientColors(exportData());
    expect(colors[6]).toBeCloseTo(1 / 3, 5);
    expect(colors[9]).toBeCloseTo(2 / 3, 5);

    const graded = createAgeGradientColors(exportData({ brightness: 10, contrast: 10 }));
    expect([...graded].every((value) => value >= 0 && value <= 1)).toBe(true);
  });

  it('normalizes the export gradient to the newest displayed birth rank', () => {
    const visibleBirths = new Float32Array([0, 1, 4]);
    expect(displayedGradientCount(visibleBirths, 3, 1)).toBe(5);
    const colors = createAgeGradientColors(exportData({
      birthRanks: visibleBirths,
      count: 3,
      seedCount: 1,
      gradientCount: displayedGradientCount(visibleBirths, 3, 1),
    }));
    expect([...colors.slice(6, 9)]).toEqual([1, 1, 1]);
  });

  it('matches the reference gradient curve while preserving exact endpoints', () => {
    expect(shapeAgeGradient(0, 1.37, -0.74, 0.45)).toBe(0);
    expect(shapeAgeGradient(0.5, 1.37, -0.74, 0.45)).toBeCloseTo(0.225, 6);
    expect(shapeAgeGradient(1, 1.37, -0.74, 0.45)).toBe(1);

    const colors = createAgeGradientColors(exportData({
      gradientContrast: 1.37,
      gradientBias: -0.74,
      gradientBlur: 0.45,
    }));
    expect([...colors.slice(0, 6)]).toEqual([0, 0, 0, 0, 0, 0]);
    expect([...colors.slice(12, 15)]).toEqual([1, 1, 1]);
  });
});

describe('model exports', () => {
  beforeAll(() => {
    vi.stubGlobal('FileReader', TestFileReader);
  });

  afterAll(() => {
    vi.unstubAllGlobals();
  });

  it('writes GLB transforms and age colors with EXT_mesh_gpu_instancing', async () => {
    const blob = await createGlbBlob(exportData({
      matrices: translatedIdentityMatrices([0, 0, 0], [2, 3, 4]),
      birthRanks: new Float32Array([0, 1]),
      count: 2,
      seedCount: 1,
      gradientCount: 2,
      // Particle Size and Particle Scale are already baked into the geometry.
      spherePositions: new Float32Array([0, 0, 0, 0.8, 0, 0, 0, 0.8, 0]),
      sphereNormals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
      seedRotationDegrees: 30,
    }));
    const json = parseGlbJson(await blob.arrayBuffer());
    expect(json.extensionsUsed).toContain('EXT_mesh_gpu_instancing');
    expect(json.extensionsRequired).toContain('EXT_mesh_gpu_instancing');

    const node = json.nodes.find(
      (candidate: { name?: string }) => candidate.name === '260716_DLAFractals',
    );
    const attributes = node.extensions.EXT_mesh_gpu_instancing.attributes;
    expect(Object.keys(attributes)).toEqual(expect.arrayContaining([
      'TRANSLATION',
      'ROTATION',
      'SCALE',
      '_COLOR_0',
    ]));
    expect(json.accessors[attributes.TRANSLATION].count).toBe(2);
    expect(json.accessors[attributes._COLOR_0].count).toBe(2);
    expect(node.matrix).toHaveLength(16);
    expect(Math.hypot(node.matrix[0], node.matrix[1], node.matrix[2])).toBeCloseTo(1, 6);
  });

  it('expands colored OBJ triangles with baked local scale and one global transform', async () => {
    const blob = await createObjBlob(exportData({
      matrices: translatedIdentityMatrices([0, 0, 0], [2, 0, 0]),
      birthRanks: new Float32Array([0, 1]),
      count: 2,
      seedCount: 1,
      gradientCount: 2,
      // The exported geometry has already received Particle Size and Particle Scale.
      spherePositions: new Float32Array([0, 0, 0, 0.8, 0, 0, 0, 0.8, 0]),
      sphereNormals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
    }));
    const lines = (await blob.text()).trim().split('\n');
    expect(lines.filter((line) => line.startsWith('v '))).toHaveLength(6);
    expect(lines.filter((line) => line.startsWith('vn '))).toHaveLength(6);
    expect(lines.filter((line) => line.startsWith('f '))).toHaveLength(2);
    expect(lines.find((line) => line.startsWith('v 0 0 0 '))).toBe('v 0 0 0 0 0 0');
    expect(lines.find((line) => line.startsWith('v 0.8 0 0 '))).toBe('v 0.8 0 0 0 0 0');
    expect(lines.find((line) => line.startsWith('v 2 0 0 '))).toBe('v 2 0 0 1 1 1');
  });
});

function translatedIdentityMatrices(...translations: Array<[number, number, number]>): Float32Array {
  const matrices = new Float32Array(translations.length * 16);
  translations.forEach(([x, y, z], index) => {
    const offset = index * 16;
    matrices[offset] = 1;
    matrices[offset + 5] = 1;
    matrices[offset + 10] = 1;
    matrices[offset + 15] = 1;
    matrices[offset + 12] = x;
    matrices[offset + 13] = y;
    matrices[offset + 14] = z;
  });
  return matrices;
}

function parseGlbJson(buffer: ArrayBuffer): Record<string, any> {
  const view = new DataView(buffer);
  expect(view.getUint32(0, true)).toBe(0x46546c67);
  const jsonLength = view.getUint32(12, true);
  const json = new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLength)).trim();
  return JSON.parse(json) as Record<string, any>;
}

class TestFileReader {
  result: string | ArrayBuffer | null = null;
  onloadend: (() => void) | null = null;

  readAsArrayBuffer(blob: Blob): void {
    void blob.arrayBuffer().then((buffer) => {
      this.result = buffer;
      this.onloadend?.();
    });
  }

  readAsDataURL(blob: Blob): void {
    void blob.arrayBuffer().then((buffer) => {
      this.result = `data:${blob.type};base64,${Buffer.from(buffer).toString('base64')}`;
      this.onloadend?.();
    });
  }
}
