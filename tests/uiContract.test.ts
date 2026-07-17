// @vitest-environment jsdom

import { readFileSync } from 'node:fs';
import { beforeEach, describe, expect, it } from 'vitest';

const page = readFileSync('index.html', 'utf8');
const styles = readFileSync('src/style.css', 'utf8');
const mainSource = readFileSync('src/main.ts', 'utf8');

beforeEach(() => {
  const parsed = new DOMParser().parseFromString(page, 'text/html');
  document.head.innerHTML = parsed.head.innerHTML;
  document.body.innerHTML = parsed.body.innerHTML;
});

describe('first-version UI contract', () => {
  it('uses the repository name for the browser title and README header', () => {
    expect(document.title).toBe('260716_DLACoral');
    expect(readFileSync('README.md', 'utf8')).toMatch(/^# 260716_DLACoral\r?\n/);
  });

  it('keeps the required section and export order', () => {
    expect(textList('.panel-section-label')).toEqual([
      'Simulation',
      'Aggregation',
      'Display',
      'Export',
    ]);
    expect(textList('#export-glb, #export-obj, #screenshot')).toEqual(['GLB', 'OBJ', 'Screenshot']);
  });

  it('keeps the floating panel at the specified 320px and 0.75 scale contract', () => {
    expect(styles).toMatch(/--menu-scale:\s*0\.75;/);
    expect(styles).toMatch(/#ui-panel\s*\{[\s\S]*?width:\s*min\(320px,/);
    expect(styles).toMatch(/backdrop-filter:\s*blur\(18px\)\s+saturate\(145%\);/);
  });

  it('does not place rebuild or progress messages over the scene', () => {
    expect(document.getElementById('status-overlay')).toBeNull();
    expect(styles).not.toContain('#status-overlay');
    expect(mainSource).not.toMatch(/Rebuilding seed|Resetting aggregate|Preparing GLB|Preparing OBJ|Updating sphere geometry|Growing GPU buffers/);
    expect(document.getElementById('fatal-error-note')).not.toBeNull();
  });

  it('keeps every numeric control at its specified initial bounds, step, and default', () => {
    const contracts: Array<[string, number, number, number, number]> = [
      ['simulation-timeline', 0, 0, 1, 0],
      ['simulation-rate', 0.1, 3, 0.01, 1],
      ['seed', 1, 999_999, 1, 260_716],
      ['seed-radius', 1, 64, 1, 8],
      ['seed-rotation', -360, 360, 1, 0],
      ['particle-size', 0.1, 4, 0.01, 1],
      ['particle-gap', 0, 0.38, 0.01, 0],
      ['particle-scale', 0.1, 3, 0.01, 1],
      ['particle-resolution', 0, 2, 1, 2],
      ['target-particles', 1_000, 1_000_000, 1_000, 1_000_000],
      ['stick-neighbors', 1, 26, 1, 1],
      ['contact-hits', 1, 1_000, 1, 1],
      ['bootstrap-particles', 0, 10_000, 1, 50],
      ['stick-chance', 0.01, 1, 0.01, 1],
      ['launch-padding', 1, 32, 1, 3],
      ['kill-padding', 1, 64, 1, 3],
      ['growth-batch', 1, 4_096, 1, 256],
      ['walker-pool', 1_024, 131_072, 1_024, 65_536],
      ['gradient-contrast', 0.2, 3, 0.01, 1.37],
      ['gradient-bias', -1, 1, 0.01, -0.74],
      ['gradient-blur', 0, 1, 0.01, 0.45],
      ['light-azimuth', -180, 180, 0.01, -3.08],
      ['light-elevation', -20, 85, 0.01, 55.79],
      ['key-brightness', 0, 12, 0.01, 3.37],
      ['ambient-fill', 0, 2, 0.01, 0.8],
      ['rim-brightness', 0, 5, 0.01, 0.49],
      ['bounce-brightness', 0, 2, 0.01, 0.45],
      ['shadow-strength', 0, 1.5, 0.01, 1.13],
      ['shadow-softness', 0, 5, 0.01, 2.09],
      ['exposure', 0.1, 3, 0.01, 0.68],
      ['brightness', 0.1, 3, 0.01, 1.15],
      ['contrast', 0.1, 3, 0.01, 2.55],
      ['roughness', 0, 1, 0.01, 0],
      ['bloom-strength', 0, 2, 0.01, 0.13],
      ['bloom-radius', 0, 1, 0.01, 0.24],
      ['bloom-threshold', 0, 2, 0.01, 0.19],
    ];

    for (const [id, min, max, step, value] of contracts) {
      const slider = requiredInput(id);
      const field = requiredInput(`${id}-value`);
      expect(slider.type, id).toBe('range');
      expect([Number(slider.min), Number(slider.max), Number(slider.step), slider.valueAsNumber], id)
        .toEqual([min, max, step, value]);
      expect([Number(field.min), Number(field.max), Number(field.step), field.valueAsNumber], `${id}-value`)
        .toEqual([min, max, step, value]);
    }
  });

  it('keeps seed, neighborhood, colors, and toggles at the specified defaults', () => {
    expect(selectOptions('seed-shape')).toEqual([
      ['point', 'Point'],
      ['sphere', 'Sphere'],
      ['ring', 'Ring'],
    ]);
    expect(requiredSelect('seed-shape').value).toBe('point');
    expect(selectOptions('attachment-neighborhood')).toEqual([
      ['faces6', 'Faces 6'],
      ['facesEdges18', 'Faces + Edges 18'],
      ['full26', 'Full 26'],
      ['weightedFull26', 'Weighted Full 26'],
      ['radius2', 'Radius 2'],
      ['radius3', 'Radius 3'],
      ['surfaceHemisphere', 'Surface Hemisphere'],
      ['randomized', 'Randomized Neighborhood'],
    ]);
    expect(requiredSelect('attachment-neighborhood').value).toBe('full26');
    expect(requiredInput('inner-color').value).toBe('#ac2a4a');
    expect(requiredInput('outer-color').value).toBe('#ffffff');
    const swatches = document.querySelector('.gradient-swatches');
    expect(swatches).not.toBeNull();
    expect(swatches?.classList.contains('control-grid-2')).toBe(true);
    expect(textList('.gradient-swatches .control-row > span:first-child')).toEqual([
      'Gradient Start',
      'Gradient End',
    ]);
    const display = Array.from(document.querySelectorAll<HTMLElement>('.panel-section'))
      .find((section) => section.querySelector('.panel-section-label')?.textContent === 'Display');
    expect(Array.from(display!.querySelectorAll<HTMLElement>('.control-row > span:first-child'), (element) => element.textContent).slice(0, 5)).toEqual([
      'Gradient Start',
      'Gradient End',
      'Gradient Contrast',
      'Gradient Bias',
      'Gradient Blur',
    ]);
    expect(requiredInput('hide-enclosed').checked).toBe(true);
    const sections = Array.from(document.querySelectorAll<HTMLElement>('.panel-section'));
    const aggregation = sections.find((section) => section.querySelector('.panel-section-label')?.textContent === 'Aggregation');
    expect(aggregation).toBeDefined();
    expect(Array.from(aggregation!.querySelectorAll<HTMLElement>('.control-row > span:first-child'), (element) => element.textContent)).toEqual([
      'Seed',
      'Seed Shape',
      'Seed Radius',
      'Seed Rotation',
      'Particle Size',
      'Particle Gap',
      'Particle Scale',
      'Particle Resolution',
      'Target Particles',
      'Attachment Neighborhood',
      'Stick Neighbors',
      'Contact Hits',
      'Bootstrap Particles',
      'Stick Chance',
      'Launch Padding',
      'Kill Padding',
      'Growth Batch',
      'Walker Pool',
    ]);
    expect(aggregation!.querySelector('#particle-count-note')).toBeNull();
    expect(document.getElementById('particle-size')).not.toBeNull();
    expect(document.getElementById('particle-scale')).not.toBeNull();
    expect(document.getElementById('particle-scale-value')).not.toBeNull();
    expect(document.getElementById('rotation')).toBeNull();
    expect(document.getElementById('sphere-gap')).toBeNull();
    expect(document.getElementById('sphere-scale')).toBeNull();
    expect(document.getElementById('sphere-detail')).toBeNull();
    expect(document.getElementById('global-scale')).toBeNull();
    expect(document.getElementById('local-scale')).toBeNull();
    expect(document.getElementById('stick-neighbors-help')?.textContent).toBe(
      'Required neighborhood score after the bootstrap period.',
    );
    expect(requiredInput('stick-neighbors').getAttribute('aria-describedby')).toBe('stick-neighbors-help');
    expect(requiredInput('stick-neighbors-value').getAttribute('aria-describedby')).toBe('stick-neighbors-help');
    expect(document.querySelector('.control-hint')?.textContent).toBe(
      'Particle Count = 1',
    );
    expect(mainSource).toContain('ui.setParticleCount(next.visibleCount);');
    expect(mainSource).not.toContain('ui.setParticleCount(next.currentCount');
  });
});

function textList(selector: string): string[] {
  return Array.from(document.querySelectorAll<HTMLElement>(selector), (element) => element.textContent?.trim() ?? '');
}

function requiredInput(id: string): HTMLInputElement {
  const input = document.getElementById(id);
  if (!(input instanceof HTMLInputElement)) {
    throw new Error(`Missing input #${id}`);
  }
  return input;
}

function requiredSelect(id: string): HTMLSelectElement {
  const select = document.getElementById(id);
  if (!(select instanceof HTMLSelectElement)) {
    throw new Error(`Missing select #${id}`);
  }
  return select;
}

function selectOptions(id: string): Array<[string, string]> {
  return Array.from(requiredSelect(id).options, (option) => [option.value, option.textContent ?? '']);
}
