export const PARTICLE_RESOLUTION_2_VISIBLE_LIMIT = 25_000;
export const PARTICLE_RESOLUTION_1_VISIBLE_LIMIT = 100_000;

export function automaticParticleResolution(
  visibleCount: number,
  currentResolution: number,
): number {
  const safeVisibleCount = Number.isFinite(visibleCount)
    ? Math.max(0, Math.floor(visibleCount))
    : 0;
  const safeResolution = Number.isFinite(currentResolution)
    ? Math.min(2, Math.max(0, Math.round(currentResolution)))
    : 0;

  if (safeVisibleCount >= PARTICLE_RESOLUTION_1_VISIBLE_LIMIT) {
    return 0;
  }
  if (safeVisibleCount >= PARTICLE_RESOLUTION_2_VISIBLE_LIMIT) {
    return Math.min(safeResolution, 1);
  }
  return safeResolution;
}
