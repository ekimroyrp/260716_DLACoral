/**
 * A one-million-particle sparse hash needs 8,388,608 compact 16-byte entries,
 * or exactly 128 MiB, in one storage binding. Request only that baseline so
 * Three can create the device without a second adapter probe at startup.
 */
export const REQUIRED_STORAGE_BUFFER_LIMIT = 128 * 1024 * 1024;

export function createRequiredDeviceLimits(): Record<string, number> {
  return {
    maxStorageBufferBindingSize: REQUIRED_STORAGE_BUFFER_LIMIT,
    maxBufferSize: REQUIRED_STORAGE_BUFFER_LIMIT,
  };
}
