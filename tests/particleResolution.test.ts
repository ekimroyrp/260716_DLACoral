import { describe, expect, it } from 'vitest';
import {
  automaticParticleResolution,
  PARTICLE_RESOLUTION_1_VISIBLE_LIMIT,
  PARTICLE_RESOLUTION_2_VISIBLE_LIMIT,
} from '../src/render/particleResolution';

describe('automatic particle resolution', () => {
  it.each([
    [PARTICLE_RESOLUTION_2_VISIBLE_LIMIT - 1, 2, 2],
    [PARTICLE_RESOLUTION_2_VISIBLE_LIMIT, 2, 1],
    [PARTICLE_RESOLUTION_1_VISIBLE_LIMIT - 1, 2, 1],
    [PARTICLE_RESOLUTION_1_VISIBLE_LIMIT, 2, 0],
    [PARTICLE_RESOLUTION_1_VISIBLE_LIMIT, 1, 0],
  ])(
    'at %i visible particles, resolution %i becomes %i',
    (visibleCount, currentResolution, expectedResolution) => {
      expect(automaticParticleResolution(visibleCount, currentResolution))
        .toBe(expectedResolution);
    },
  );

  it('never automatically increases a lower resolution', () => {
    expect(automaticParticleResolution(1, 0)).toBe(0);
    expect(automaticParticleResolution(1, 1)).toBe(1);
    expect(automaticParticleResolution(50_000, 0)).toBe(0);
  });
});
