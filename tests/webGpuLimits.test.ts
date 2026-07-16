import { describe, expect, it } from 'vitest';
import {
  REQUIRED_STORAGE_BUFFER_LIMIT,
  createRequiredDeviceLimits,
} from '../src/render/webGpuLimits';

describe('WebGPU storage limit selection', () => {
  it('requests only the 128 MiB needed by the compact one-million-particle hash', () => {
    const compactHashBytes = 8_388_608 * 16;
    expect(REQUIRED_STORAGE_BUFFER_LIMIT).toBe(compactHashBytes);
    expect(createRequiredDeviceLimits()).toEqual({
      maxStorageBufferBindingSize: compactHashBytes,
      maxBufferSize: compactHashBytes,
    });
  });
});
