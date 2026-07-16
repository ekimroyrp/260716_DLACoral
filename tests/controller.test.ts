// @vitest-environment jsdom

import { readFileSync } from 'node:fs';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createUiController, type UiController } from '../src/ui/controller';

const page = readFileSync('index.html', 'utf8');
let controller: UiController | null = null;

beforeEach(() => {
  const parsed = new DOMParser().parseFromString(page, 'text/html');
  document.head.innerHTML = parsed.head.innerHTML;
  document.body.innerHTML = parsed.body.innerHTML;
  document.documentElement.className = '';
});

afterEach(() => {
  controller?.dispose();
  controller = null;
  vi.restoreAllMocks();
});

describe('UI numeric controllers', () => {
  it('extends slider bounds for valid typed values and restores invalid text', () => {
    controller = createUiController();
    const slider = input('seed');
    const field = input('seed-value');

    field.value = '1200000';
    field.dispatchEvent(new Event('change', { bubbles: true }));
    expect(controller.dla.seed).toBe(1_200_000);
    expect(Number(slider.max)).toBe(1_200_000);
    expect(Number(field.max)).toBe(1_200_000);

    field.value = '';
    field.dispatchEvent(new Event('change', { bubbles: true }));
    expect(field.value).toBe('1200000');
    expect(controller.dla.seed).toBe(1_200_000);
  });

  it('coalesces a slider gesture into one transaction', () => {
    const starts = vi.fn();
    const commits = vi.fn();
    controller = createUiController({
      onTransactionStart: starts,
      onTransactionCommit: commits,
    });
    const slider = input('sphere-scale');

    slider.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    slider.value = '0.8';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
    slider.value = '0.7';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
    slider.dispatchEvent(new Event('change', { bubbles: true }));

    expect(starts).toHaveBeenCalledTimes(1);
    expect(starts).toHaveBeenCalledWith('Sphere Scale');
    expect(commits).toHaveBeenCalledTimes(1);
    expect(controller.dla.sphereScale).toBe(0.7);
  });
});

describe('UI DLA and timeline invariants', () => {
  it('marks only seed structure edits as aggregate resets', () => {
    const changes = vi.fn();
    controller = createUiController({ onDlaChange: changes });

    const seedRadius = input('seed-radius-value');
    seedRadius.value = '12';
    seedRadius.dispatchEvent(new Event('change', { bubbles: true }));

    const target = input('target-particles-value');
    target.value = '2000';
    target.dispatchEvent(new Event('change', { bubbles: true }));

    expect(changes.mock.calls[0]?.[1]).toMatchObject({
      source: 'seedRadius',
      phase: 'commit',
      requiresReset: true,
    });
    expect(changes.mock.calls[1]?.[1]).toMatchObject({
      source: 'targetParticles',
      phase: 'commit',
      requiresReset: false,
    });
  });

  it('clamps the timeline to its latest birth and disables it while running', () => {
    const commits = vi.fn();
    controller = createUiController({ onTimelineCommit: commits });
    const slider = input('simulation-timeline');
    const field = input('simulation-timeline-value');

    controller.setTimeline(7, 10);
    expect(slider.max).toBe('10');
    controller.setRunning(true);
    expect(slider.disabled).toBe(true);
    expect(field.disabled).toBe(true);

    controller.setRunning(false);
    field.value = '99';
    field.dispatchEvent(new Event('change', { bubbles: true }));
    expect(controller.simulation.timeline).toBe(10);
    expect(commits).toHaveBeenLastCalledWith(10);
  });
});

function input(id: string): HTMLInputElement {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLInputElement)) {
    throw new Error(`Missing input #${id}`);
  }
  return element;
}
