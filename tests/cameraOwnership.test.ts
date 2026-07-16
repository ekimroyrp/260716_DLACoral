import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const rendererSource = readFileSync('src/render/dlaRenderer.ts', 'utf8');
const mainSource = readFileSync('src/main.ts', 'utf8');

describe('camera ownership', () => {
  it('assigns camera pose and orbit target only during renderer construction', () => {
    const constructorStart = rendererSource.indexOf('  constructor(');
    const constructorEnd = rendererSource.indexOf('\n  async init(', constructorStart);
    expect(constructorStart).toBeGreaterThanOrEqual(0);
    expect(constructorEnd).toBeGreaterThan(constructorStart);

    const constructorSource = rendererSource.slice(constructorStart, constructorEnd);
    const outsideConstructor = rendererSource.slice(0, constructorStart)
      + rendererSource.slice(constructorEnd);

    expect(constructorSource.match(/this\.camera\.position\.(?:set|copy)\s*\(/g)).toHaveLength(1);
    expect(constructorSource.match(/this\.controls\.target\.(?:set|copy)\s*\(/g)).toHaveLength(1);
    expect(outsideConstructor).not.toMatch(/this\.camera\.position\.(?:set|copy)\s*\(/);
    expect(outsideConstructor).not.toMatch(/this\.controls\.target\.(?:set|copy)\s*\(/);
    expect(rendererSource).not.toContain('frameCamera');
    expect(mainSource).not.toContain('frameCamera');
  });
});
