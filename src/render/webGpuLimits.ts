/**
 * A one-million-particle sparse hash needs 8,388,608 32-byte entries, or
 * exactly 256 MiB, in one storage binding. Requesting more provides no benefit
 * for the application's configured target.
 */
export const PREFERRED_STORAGE_BUFFER_LIMIT = 256 * 1024 * 1024;

export interface AdapterStorageLimits {
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
}

/**
 * Requests the useful adapter-supported limit without making lower-limit
 * devices fail device creation. Runtime capacity checks still clamp to the
 * limits granted by the resulting GPUDevice.
 */
export function selectPreferredRequiredLimits(
  supported: AdapterStorageLimits,
): Record<string, number> {
  const maxStorageBinding = normalizeLimit(supported.maxStorageBufferBindingSize);
  const maxBuffer = normalizeLimit(supported.maxBufferSize);
  if (maxStorageBinding === 0 || maxBuffer === 0) {
    return {};
  }

  return {
    maxStorageBufferBindingSize: Math.min(
      maxStorageBinding,
      maxBuffer,
      PREFERRED_STORAGE_BUFFER_LIMIT,
    ),
    maxBufferSize: Math.min(maxBuffer, PREFERRED_STORAGE_BUFFER_LIMIT),
  };
}

function normalizeLimit(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
}
