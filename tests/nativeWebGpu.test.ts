// @vitest-environment jsdom

import { describe, expect, it } from 'vitest';
import { WebGPURenderer } from 'three/webgpu';
import { disableWebGlFallback } from '../src/render/nativeWebGpu';

describe('native WebGPU renderer guard', () => {
  it('clears the WebGL2 fallback installed by Three r185', () => {
    const renderer = new WebGPURenderer({
      canvas: document.createElement('canvas'),
      forceWebGL: false,
    });
    const internals = renderer as unknown as { _getFallback: unknown };
    expect(internals._getFallback).toBeTypeOf('function');
    disableWebGlFallback(renderer);
    expect(internals._getFallback).toBeNull();
  });
});
