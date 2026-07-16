// @vitest-environment jsdom

import { readFileSync } from 'node:fs';
import { beforeEach, describe, expect, it } from 'vitest';

const page = readFileSync('index.html', 'utf8');
const styles = readFileSync('src/style.css', 'utf8');

beforeEach(() => {
  const parsed = new DOMParser().parseFromString(page, 'text/html');
  document.head.innerHTML = parsed.head.innerHTML;
  document.body.innerHTML = parsed.body.innerHTML;
});

describe('first-version UI contract', () => {
  it('uses the repository name for the browser title and README header', () => {
    expect(document.title).toBe('260716_DLAFractals');
    expect(readFileSync('README.md', 'utf8')).toMatch(/^# 260716_DLAFractals\r?\n/);
  });

  it('keeps the required section and export order', () => {
    expect(textList('.panel-section-label')).toEqual([
      'Simulation',
      'Diffusion-Limited Aggregation',
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

  it('keeps every numeric control at its specified initial bounds, step, and default', () => {
    const contracts: Array<[string, number, number, number, number]> = [
      ['simulation-timeline', 0, 0, 1, 0],
      ['simulation-rate', 0.1, 3, 0.01, 1],
      ['seed', 1, 999_999, 1, 260_716],
      ['seed-radius', 1, 64, 1, 8],
      ['target-particles', 1_000, 1_000_000, 1_000, 1_000_000],
      ['stick-neighbors', 1, 26, 1, 1],
      ['stick-chance', 0.01, 1, 0.01, 1],
      ['launch-padding', 1, 32, 1, 3],
      ['kill-padding', 1, 64, 1, 3],
      ['growth-batch', 1, 4_096, 1, 256],
      ['walker-pool', 1_024, 131_072, 1_024, 65_536],
      ['rotation', -360, 360, 1, 0],
      ['sphere-scale', 0.42, 1.15, 0.01, 1],
      ['sphere-gap', 0, 0.38, 0.01, 0],
      ['sphere-detail', 0, 2, 1, 0],
      ['light-azimuth', -180, 180, 0.01, 25.65],
      ['light-elevation', -20, 85, 0.01, 68.7],
      ['key-brightness', 0, 12, 0.01, 2.41],
      ['ambient-fill', 0, 2, 0.01, 0.3],
      ['rim-brightness', 0, 5, 0.01, 0.49],
      ['bounce-brightness', 0, 2, 0.01, 0.07],
      ['shadow-strength', 0, 1.5, 0.01, 1.08],
      ['shadow-softness', 0, 5, 0.01, 2.6],
      ['exposure', 0.1, 3, 0.01, 0.7],
      ['brightness', 0.1, 3, 0.01, 1],
      ['contrast', 0.1, 3, 0.01, 2.25],
      ['roughness', 0, 1, 0.01, 0.92],
      ['bloom-strength', 0, 2, 0.01, 0.08],
      ['bloom-radius', 0, 1, 0.01, 0.26],
      ['bloom-threshold', 0, 2, 0.01, 0],
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
      ['6', 'Faces 6'],
      ['18', 'Faces + Edges 18'],
      ['26', 'Full 26'],
    ]);
    expect(requiredSelect('attachment-neighborhood').value).toBe('26');
    expect(requiredInput('inner-color').value).toBe('#6b2f24');
    expect(requiredInput('outer-color').value).toBe('#f4e6d2');
    expect(requiredInput('hide-enclosed').checked).toBe(true);
    expect(document.getElementById('stick-neighbors-help')?.textContent).toBe(
      'Preferred occupied neighbors; growth uses the densest available candidates while bootstrapping.',
    );
    expect(requiredInput('stick-neighbors').getAttribute('aria-describedby')).toBe('stick-neighbors-help');
    expect(requiredInput('stick-neighbors-value').getAttribute('aria-describedby')).toBe('stick-neighbors-help');
    expect(document.querySelector('.control-hint')?.textContent).toBe(
      'Wheel = Zoom, MMB = Pan, RMB = Orbit, LMB = Rotate Model',
    );
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
