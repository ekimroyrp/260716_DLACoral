import { Color } from 'three';

const SIX_DIGIT_HEX_COLOR = /^#?([\da-f]{6})$/i;

/**
 * Stores UI hex channels directly in the material working space, matching the
 * instance-color convention used by the AutomataChunks reference renderer.
 */
export function setDisplayColor(target: Color, value: string): Color {
  const match = SIX_DIGIT_HEX_COLOR.exec(value.trim());
  if (!match) {
    throw new Error(`Invalid display color: ${value}`);
  }

  const packed = Number.parseInt(match[1], 16);
  return target.setRGB(
    ((packed >> 16) & 0xff) / 0xff,
    ((packed >> 8) & 0xff) / 0xff,
    (packed & 0xff) / 0xff,
  );
}

export function createDisplayColor(value: string): Color {
  return setDisplayColor(new Color(), value);
}
