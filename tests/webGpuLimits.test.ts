import { describe, expect, it } from 'vitest';
import {
  PREFERRED_STORAGE_BUFFER_LIMIT,
  selectPreferredRequiredLimits,
} from '../src/render/webGpuLimits';

describe('WebGPU storage limit selection', () => {
  it('requests the 256 MiB limit needed for one million particles when supported', () => {
    expect(selectPreferredRequiredLimits({
      maxStorageBufferBindingSize: 2 * 1024 * 1024 * 1024,
      maxBufferSize: 2 * 1024 * 1024 * 1024,
    })).toEqual({
      maxStorageBufferBindingSize: PREFERRED_STORAGE_BUFFER_LIMIT,
      maxBufferSize: PREFERRED_STORAGE_BUFFER_LIMIT,
    });
  });

  it('uses lower advertised limits so device creation can still succeed', () => {
    const storageLimit = 128 * 1024 * 1024;
    const bufferLimit = 192 * 1024 * 1024;
    expect(selectPreferredRequiredLimits({
      maxStorageBufferBindingSize: storageLimit,
      maxBufferSize: bufferLimit,
    })).toEqual({
      maxStorageBufferBindingSize: storageLimit,
      maxBufferSize: bufferLimit,
    });
  });

  it('never requests a storage binding larger than the maximum buffer size', () => {
    const bufferLimit = 96 * 1024 * 1024;
    expect(selectPreferredRequiredLimits({
      maxStorageBufferBindingSize: PREFERRED_STORAGE_BUFFER_LIMIT,
      maxBufferSize: bufferLimit,
    })).toEqual({
      maxStorageBufferBindingSize: bufferLimit,
      maxBufferSize: bufferLimit,
    });
  });
});
