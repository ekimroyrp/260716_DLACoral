interface ThreeRendererFallbackAccess {
  _getFallback: unknown;
}

/**
 * Three r185 installs a private WebGL2 fallback even when forceWebGL is false.
 * Clear it before init so a native WebGPU failure rejects immediately instead
 * of silently constructing a second backend.
 */
export function disableWebGlFallback(renderer: unknown): void {
  if (!renderer || typeof renderer !== 'object' || !('_getFallback' in renderer)) {
    throw new Error('This Three.js renderer does not expose the expected fallback hook.');
  }
  (renderer as ThreeRendererFallbackAccess)._getFallback = null;
}
