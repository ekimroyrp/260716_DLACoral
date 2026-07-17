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

  it('commits typed values on blur even when no change event is emitted', () => {
    const changes = vi.fn();
    controller = createUiController({ onDlaChange: changes });
    const field = input('particle-scale-value');

    field.focus();
    field.value = '0.73';
    field.dispatchEvent(new Event('input', { bubbles: true }));
    field.blur();

    expect(controller.dla.particleScale).toBe(0.73);
    expect(changes).toHaveBeenLastCalledWith(
      expect.objectContaining({ particleScale: 0.73 }),
      expect.objectContaining({ source: 'particleScale', phase: 'commit' }),
    );

    field.focus();
    field.value = '';
    field.dispatchEvent(new Event('input', { bubbles: true }));
    field.blur();
    expect(field.value).toBe('0.73');
    expect(controller.dla.particleScale).toBe(0.73);
  });

  it('coalesces a slider gesture into one transaction', () => {
    const starts = vi.fn();
    const commits = vi.fn();
    controller = createUiController({
      onTransactionStart: starts,
      onTransactionCommit: commits,
    });
    const slider = input('particle-scale');

    slider.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    slider.value = '0.8';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
    slider.value = '0.7';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
    slider.dispatchEvent(new Event('change', { bubbles: true }));

    expect(starts).toHaveBeenCalledTimes(1);
    expect(starts).toHaveBeenCalledWith('Particle Scale');
    expect(commits).toHaveBeenCalledTimes(1);
    expect(controller.dla.particleScale).toBe(0.7);
  });

  it('updates gradient grading live and extends a typed gradient bound', () => {
    const changes = vi.fn();
    controller = createUiController({ onDisplayChange: changes });
    const contrast = input('gradient-contrast');

    contrast.value = '2.1';
    contrast.dispatchEvent(new Event('input', { bubbles: true }));
    expect(controller.display.gradientContrast).toBe(2.1);
    expect(changes).toHaveBeenLastCalledWith(
      expect.objectContaining({ gradientContrast: 2.1 }),
      expect.objectContaining({ source: 'gradientContrast', phase: 'input' }),
    );

    const biasField = input('gradient-bias-value');
    biasField.value = '-1.5';
    biasField.dispatchEvent(new Event('change', { bubbles: true }));
    expect(controller.display.gradientBias).toBe(-1.5);
    expect(Number(input('gradient-bias').min)).toBe(-1.5);
  });
});

describe('UI DLA and timeline invariants', () => {
  it('shows only the formatted visible particle count in the top note', () => {
    controller = createUiController();
    controller.setParticleCount(12_345);
    expect(document.getElementById('particle-count-note')?.textContent).toBe('Particle Count = 12,345');
  });

  it('marks seed geometry edits as aggregate resets', () => {
    const changes = vi.fn();
    controller = createUiController({ onDlaChange: changes });

    const seedRadius = input('seed-radius-value');
    seedRadius.value = '12';
    seedRadius.dispatchEvent(new Event('change', { bubbles: true }));

    const target = input('target-particles-value');
    target.value = '2000';
    target.dispatchEvent(new Event('change', { bubbles: true }));

    const particleSize = input('particle-size-value');
    particleSize.value = '0.5';
    particleSize.dispatchEvent(new Event('change', { bubbles: true }));

    const particleScale = input('particle-scale-value');
    particleScale.value = '1.4';
    particleScale.dispatchEvent(new Event('change', { bubbles: true }));

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
    expect(changes.mock.calls[2]?.[1]).toMatchObject({
      source: 'particleSize',
      phase: 'commit',
      requiresReset: true,
    });
    expect(changes.mock.calls[3]?.[1]).toMatchObject({
      source: 'particleScale',
      phase: 'commit',
      requiresReset: false,
    });
  });

  it('coalesces a Particle Scale slider gesture into one transaction', () => {
    const starts = vi.fn();
    const commits = vi.fn();
    controller = createUiController({
      onTransactionStart: starts,
      onTransactionCommit: commits,
    });
    const slider = input('particle-scale');

    slider.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    slider.value = '1.2';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
    slider.value = '1.4';
    slider.dispatchEvent(new Event('input', { bubbles: true }));
    slider.dispatchEvent(new Event('change', { bubbles: true }));

    expect(starts).toHaveBeenCalledTimes(1);
    expect(starts).toHaveBeenCalledWith('Particle Scale');
    expect(commits).toHaveBeenCalledTimes(1);
    expect(controller.dla.particleScale).toBe(1.4);
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

  it('preserves the run state when Reset is requested', () => {
    const resets = vi.fn();
    controller = createUiController({ onReset: resets });
    controller.setRunning(true);

    requiredButton('reset-sim').click();

    expect(resets).toHaveBeenCalledTimes(1);
    expect(controller.simulation.running).toBe(true);
  });
});

describe('UI global shortcuts and browser-menu blocking', () => {
  it('handles undo and redo outside fields while leaving field editing alone', () => {
    const undo = vi.fn();
    const redo = vi.fn();
    controller = createUiController({ onUndo: undo, onRedo: redo });

    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'z', ctrlKey: true }));
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'z', ctrlKey: true, shiftKey: true }));
    window.dispatchEvent(new KeyboardEvent('keydown', { key: 'y', metaKey: true }));
    expect(undo).toHaveBeenCalledTimes(1);
    expect(redo).toHaveBeenCalledTimes(2);

    const field = input('seed-value');
    field.dispatchEvent(new KeyboardEvent('keydown', { key: 'z', ctrlKey: true, bubbles: true }));
    expect(undo).toHaveBeenCalledTimes(1);
  });

  it('prevents the native context menu globally', () => {
    controller = createUiController();
    const event = new MouseEvent('contextmenu', { bubbles: true, cancelable: true });
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
  });
});

function input(id: string): HTMLInputElement {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLInputElement)) {
    throw new Error(`Missing input #${id}`);
  }
  return element;
}

function requiredButton(id: string): HTMLButtonElement {
  const element = document.getElementById(id);
  if (!(element instanceof HTMLButtonElement)) {
    throw new Error(`Missing button #${id}`);
  }
  return element;
}
