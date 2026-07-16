import { describe, expect, it } from 'vitest';
import { createAgeGradientColors, type ExportInstanceData } from '../src/render';

function exportData(overrides: Partial<ExportInstanceData> = {}): ExportInstanceData {
  return {
    matrices: new Float32Array(5 * 16),
    birthRanks: new Float32Array([0, 1, 2, 3, 4]),
    count: 5,
    seedCount: 2,
    gradientCount: 5,
    spherePositions: new Float32Array(9),
    sphereNormals: new Float32Array(9),
    sphereScale: 1,
    rotationDegrees: 0,
    innerColor: '#000000',
    outerColor: '#ffffff',
    brightness: 1,
    contrast: 1,
    materialRoughness: 0.92,
    ...overrides,
  };
}

describe('export age colors', () => {
  it('keeps every seed at the inner color and makes the newest attachment exactly outer', () => {
    const colors = createAgeGradientColors(exportData());
    expect([...colors.slice(0, 6)]).toEqual([0, 0, 0, 0, 0, 0]);
    expect([...colors.slice(12, 15)]).toEqual([1, 1, 1]);
  });

  it('interpolates attached birth ranks and clamps display grading', () => {
    const colors = createAgeGradientColors(exportData());
    expect(colors[6]).toBeCloseTo(1 / 3, 5);
    expect(colors[9]).toBeCloseTo(2 / 3, 5);

    const graded = createAgeGradientColors(exportData({ brightness: 10, contrast: 10 }));
    expect([...graded].every((value) => value >= 0 && value <= 1)).toBe(true);
  });
});
