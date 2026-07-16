import {
  DEFAULT_DISPLAY_SETTINGS,
  DEFAULT_DLA_SETTINGS,
  DEFAULT_SIMULATION_SETTINGS,
  type AppSnapshot,
  type AttachmentNeighborhood,
  type DisplaySettings,
  type DlaSettings,
  type SeedShape,
  type SimulationSettings,
} from '../types';

export type UiChangePhase = 'input' | 'commit';

export interface UiChangeMeta {
  source: string;
  phase: UiChangePhase;
}

export interface DlaUiChangeMeta extends UiChangeMeta {
  requiresReset: boolean;
}

export interface UiControllerCallbacks {
  onSimulationChange?: (settings: SimulationSettings, meta: UiChangeMeta) => void;
  onDlaChange?: (settings: DlaSettings, meta: DlaUiChangeMeta) => void;
  onDisplayChange?: (settings: DisplaySettings, meta: UiChangeMeta) => void;
  onStartPause?: (running: boolean) => void;
  onReset?: () => void;
  onTimelineInput?: (value: number) => void;
  onTimelineCommit?: (value: number) => void;
  onUndo?: () => void;
  onRedo?: () => void;
  onExportGlb?: () => void;
  onExportObj?: () => void;
  onScreenshot?: () => void;
  onTransactionStart?: (label: string) => void;
  onTransactionCommit?: (label: string) => void;
}

export interface UiSettingsSnapshot {
  simulation: SimulationSettings;
  dla: DlaSettings;
  display: DisplaySettings;
}

export interface UiController {
  readonly simulation: SimulationSettings;
  readonly dla: DlaSettings;
  readonly display: DisplaySettings;
  readonly settings: UiSettingsSnapshot;
  sync(snapshot: Partial<Pick<AppSnapshot, 'simulation' | 'dla' | 'display'>>): void;
  setRunning(running: boolean): void;
  setTimeline(value: number, latest: number): void;
  setParticleCount(count: number, target?: number): void;
  setRotation(rotation: number): void;
  setBusy(message: string): void;
  setReady(): void;
  setError(message: string): void;
  beginTransaction(label: string): void;
  commitTransaction(label: string): void;
  dispose(): void;
}

interface NumericBindingOptions {
  id: string;
  label: string;
  decimals: number;
  integer?: boolean;
  extendBounds?: boolean;
  emitOnInput?: boolean;
  sanitize?: (value: number) => number;
  onValue: (value: number, phase: UiChangePhase) => void;
}

interface NumericBinding {
  readonly slider: HTMLInputElement;
  readonly valueInput: HTMLInputElement;
  set(value: number): number;
  setDisabled(disabled: boolean): void;
  setMaximum(maximum: number): void;
}

interface SelectBinding {
  readonly select: HTMLSelectElement;
  set(value: string): void;
  setDisabled(disabled: boolean): void;
}

interface Transaction {
  begin(): void;
  commit(): void;
}

export function createUiController(callbacks: UiControllerCallbacks = {}): UiController {
  let simulation: SimulationSettings = { ...DEFAULT_SIMULATION_SETTINGS };
  let dla: DlaSettings = { ...DEFAULT_DLA_SETTINGS };
  let display: DisplaySettings = { ...DEFAULT_DISPLAY_SETTINGS };
  let currentParticleCount = 1;
  let busy = false;
  let disposed = false;

  const cleanups: Array<() => void> = [];
  const activeTransactions = new Set<Transaction>();

  const panel = requiredElement('ui-panel', HTMLElement);
  const handleTop = requiredElement('ui-handle', HTMLElement);
  const handleBottom = requiredElement('ui-handle-bottom', HTMLElement);
  const collapseToggle = requiredElement('collapse-toggle', HTMLButtonElement);
  const startButton = requiredElement('start-sim', HTMLButtonElement);
  const resetButton = requiredElement('reset-sim', HTMLButtonElement);
  const particleCountNote = requiredElement('particle-count-note', HTMLElement);
  const status = requiredElement('status-overlay', HTMLElement);
  const exportGlb = requiredElement('export-glb', HTMLButtonElement);
  const exportObj = requiredElement('export-obj', HTMLButtonElement);
  const screenshot = requiredElement('screenshot', HTMLButtonElement);
  const hideEnclosed = requiredElement('hide-enclosed', HTMLInputElement);
  const innerColor = requiredElement('inner-color', HTMLInputElement);
  const outerColor = requiredElement('outer-color', HTMLInputElement);

  const listen = (
    target: EventTarget,
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: boolean | AddEventListenerOptions,
  ): void => {
    target.addEventListener(type, listener, options);
    cleanups.push(() => target.removeEventListener(type, listener, options));
  };

  const makeTransaction = (label: string): Transaction => {
    let active = false;
    const transaction: Transaction = {
      begin(): void {
        if (active || disposed) {
          return;
        }
        active = true;
        activeTransactions.add(transaction);
        callbacks.onTransactionStart?.(label);
      },
      commit(): void {
        if (!active || disposed) {
          return;
        }
        active = false;
        activeTransactions.delete(transaction);
        callbacks.onTransactionCommit?.(label);
      },
    };
    return transaction;
  };

  const runTransaction = (label: string, action: () => void): void => {
    if (disposed) {
      return;
    }
    callbacks.onTransactionStart?.(label);
    try {
      action();
    } finally {
      callbacks.onTransactionCommit?.(label);
    }
  };

  const emitSimulationChange = (source: string, phase: UiChangePhase): void => {
    callbacks.onSimulationChange?.({ ...simulation }, { source, phase });
  };

  const emitDlaChange = (source: string, phase: UiChangePhase, requiresReset = false): void => {
    callbacks.onDlaChange?.({ ...dla }, { source, phase, requiresReset });
  };

  const emitDisplayChange = (source: string, phase: UiChangePhase): void => {
    callbacks.onDisplayChange?.({ ...display }, { source, phase });
  };

  const bindNumericControl = ({
    id,
    label,
    decimals,
    integer = false,
    extendBounds = true,
    emitOnInput = false,
    sanitize = (value) => value,
    onValue,
  }: NumericBindingOptions): NumericBinding => {
    const slider = requiredElement(id, HTMLInputElement);
    const valueInput = requiredElement(`${id}-value`, HTMLInputElement);
    const transaction = makeTransaction(label);
    let valueInputDirty = false;

    const set = (rawValue: number): number => {
      const normalized = normalizeNumericValue(sanitize(rawValue), decimals, integer);
      const nextValue = extendBounds ? normalized : clampToInputBounds(slider, normalized);
      if (extendBounds) {
        extendInputBounds(slider, nextValue);
        extendInputBounds(valueInput, nextValue);
      }
      slider.value = formatNumericValue(nextValue, decimals);
      const sliderValue = Number.parseFloat(slider.value);
      const appliedValue = Number.isFinite(sliderValue) ? sliderValue : nextValue;
      setValueInput(valueInput, appliedValue, decimals, true);
      updateRangeProgress(slider);
      return appliedValue;
    };

    const readSlider = (): number => {
      const parsed = Number.parseFloat(slider.value);
      return set(Number.isFinite(parsed) ? parsed : 0);
    };

    listen(slider, 'input', () => {
      transaction.begin();
      const value = readSlider();
      if (emitOnInput) {
        onValue(value, 'input');
      }
    });
    listen(slider, 'change', () => {
      transaction.begin();
      const value = readSlider();
      onValue(value, 'commit');
      transaction.commit();
    });
    listen(slider, 'pointercancel', () => transaction.commit());

    const commitValueInput = (): void => {
      transaction.begin();
      const parsed = Number.parseFloat(valueInput.value);
      valueInputDirty = false;
      if (!Number.isFinite(parsed)) {
        setValueInput(valueInput, Number.parseFloat(slider.value), decimals, true);
        transaction.commit();
        return;
      }
      const value = set(parsed);
      onValue(value, 'commit');
      transaction.commit();
    };

    listen(valueInput, 'focus', () => {
      valueInputDirty = false;
      valueInput.select();
    });
    listen(valueInput, 'input', () => {
      valueInputDirty = true;
      transaction.begin();
    });
    listen(valueInput, 'keydown', (event) => {
      const keyboardEvent = event as KeyboardEvent;
      if (keyboardEvent.key === 'Enter') {
        keyboardEvent.preventDefault();
        valueInput.blur();
      } else if (keyboardEvent.key === 'Escape') {
        keyboardEvent.preventDefault();
        valueInputDirty = false;
        setValueInput(valueInput, Number.parseFloat(slider.value), decimals, true);
        transaction.commit();
        valueInput.blur();
      }
    });
    listen(valueInput, 'change', commitValueInput);
    listen(valueInput, 'blur', () => {
      if (valueInputDirty) {
        commitValueInput();
      } else {
        transaction.commit();
      }
    });

    set(Number.parseFloat(slider.value));

    return {
      slider,
      valueInput,
      set,
      setDisabled(disabled: boolean): void {
        slider.disabled = disabled;
        valueInput.disabled = disabled;
      },
      setMaximum(maximum: number): void {
        const formatted = formatBoundValue(maximum);
        slider.max = formatted;
        valueInput.max = formatted;
        set(Number.parseFloat(slider.value));
      },
    };
  };

  const simulationRateControl = bindNumericControl({
    id: 'simulation-rate',
    label: 'Simulation Rate',
    decimals: 2,
    emitOnInput: true,
    sanitize: (value) => Math.max(0.01, value),
    onValue(value, phase) {
      simulation.rate = value;
      emitSimulationChange('simulationRate', phase);
    },
  });

  const timelineControl = bindNumericControl({
    id: 'simulation-timeline',
    label: 'Simulation Timeline',
    decimals: 0,
    integer: true,
    extendBounds: false,
    emitOnInput: true,
    sanitize: (value) => clamp(value, 0, simulation.latestTimeline),
    onValue(value, phase) {
      simulation.timeline = value;
      if (phase === 'input') {
        callbacks.onTimelineInput?.(value);
      } else {
        callbacks.onTimelineCommit?.(value);
      }
    },
  });

  const seedControl = bindNumericControl({
    id: 'seed',
    label: 'Seed',
    decimals: 0,
    integer: true,
    sanitize: (value) => Math.max(1, value),
    onValue(value, phase) {
      dla.seed = value;
      emitDlaChange('seed', phase, true);
    },
  });

  const seedRadiusControl = bindNumericControl({
    id: 'seed-radius',
    label: 'Seed Radius',
    decimals: 0,
    integer: true,
    sanitize: (value) => Math.max(1, value),
    onValue(value, phase) {
      dla.seedRadius = value;
      emitDlaChange('seedRadius', phase, true);
    },
  });

  const targetParticlesControl = bindNumericControl({
    id: 'target-particles',
    label: 'Target Particles',
    decimals: 0,
    integer: true,
    sanitize: (value) => Math.max(1, value),
    onValue(value, phase) {
      dla.targetParticles = value;
      updateParticleCountNote();
      emitDlaChange('targetParticles', phase);
    },
  });

  const stickNeighborsControl = bindNumericControl({
    id: 'stick-neighbors',
    label: 'Stick Neighbors',
    decimals: 0,
    integer: true,
    extendBounds: false,
    sanitize: (value) => clamp(value, 1, dla.attachmentNeighborhood),
    onValue(value, phase) {
      dla.stickNeighbors = value;
      emitDlaChange('stickNeighbors', phase);
    },
  });

  const stickChanceControl = bindNumericControl({
    id: 'stick-chance',
    label: 'Stick Chance',
    decimals: 2,
    extendBounds: false,
    sanitize: (value) => clamp(value, 0.01, 1),
    onValue(value, phase) {
      dla.stickChance = value;
      emitDlaChange('stickChance', phase);
    },
  });

  const launchPaddingControl = bindNumericControl({
    id: 'launch-padding',
    label: 'Launch Padding',
    decimals: 0,
    integer: true,
    sanitize: (value) => Math.max(1, value),
    onValue(value, phase) {
      dla.launchPadding = value;
      emitDlaChange('launchPadding', phase);
    },
  });

  const killPaddingControl = bindNumericControl({
    id: 'kill-padding',
    label: 'Kill Padding',
    decimals: 0,
    integer: true,
    sanitize: (value) => Math.max(1, value),
    onValue(value, phase) {
      dla.killPadding = value;
      emitDlaChange('killPadding', phase);
    },
  });

  const growthBatchControl = bindNumericControl({
    id: 'growth-batch',
    label: 'Growth Batch',
    decimals: 0,
    integer: true,
    sanitize: (value) => Math.max(1, value),
    onValue(value, phase) {
      dla.growthBatch = value;
      emitDlaChange('growthBatch', phase);
    },
  });

  const walkerPoolControl = bindNumericControl({
    id: 'walker-pool',
    label: 'Walker Pool',
    decimals: 0,
    integer: true,
    sanitize: (value) => Math.max(1, value),
    onValue(value, phase) {
      dla.walkerPool = value;
      emitDlaChange('walkerPool', phase);
    },
  });

  const rotationControl = bindNumericControl({
    id: 'rotation',
    label: 'Rotation',
    decimals: 0,
    integer: true,
    emitOnInput: true,
    onValue(value, phase) {
      dla.rotation = value;
      emitDlaChange('rotation', phase);
    },
  });

  const sphereScaleControl = bindNumericControl({
    id: 'sphere-scale',
    label: 'Sphere Scale',
    decimals: 2,
    emitOnInput: true,
    sanitize: (value) => Math.max(0.01, value),
    onValue(value, phase) {
      dla.sphereScale = value;
      emitDlaChange('sphereScale', phase);
    },
  });

  const sphereGapControl = bindNumericControl({
    id: 'sphere-gap',
    label: 'Sphere Gap',
    decimals: 2,
    emitOnInput: true,
    onValue(value, phase) {
      dla.sphereGap = value;
      emitDlaChange('sphereGap', phase);
    },
  });

  const sphereDetailControl = bindNumericControl({
    id: 'sphere-detail',
    label: 'Sphere Detail',
    decimals: 0,
    integer: true,
    extendBounds: false,
    sanitize: (value) => clamp(value, 0, 2),
    onValue(value, phase) {
      dla.sphereDetail = value;
      emitDlaChange('sphereDetail', phase);
    },
  });

  const displayControls = {
    lightAzimuth: bindDisplayNumber('light-azimuth', 'Light Azimuth', 2, (value) => {
      display.lightAzimuth = value;
    }),
    lightElevation: bindDisplayNumber('light-elevation', 'Light Elevation', 2, (value) => {
      display.lightElevation = value;
    }),
    keyBrightness: bindDisplayNumber('key-brightness', 'Key Brightness', 2, (value) => {
      display.keyBrightness = value;
    }),
    ambientFill: bindDisplayNumber('ambient-fill', 'Ambient Fill', 2, (value) => {
      display.ambientFill = value;
    }),
    rimBrightness: bindDisplayNumber('rim-brightness', 'Rim Brightness', 2, (value) => {
      display.rimBrightness = value;
    }),
    bounceBrightness: bindDisplayNumber('bounce-brightness', 'Bounce Brightness', 2, (value) => {
      display.bounceBrightness = value;
    }),
    shadowStrength: bindDisplayNumber('shadow-strength', 'Shadow Strength', 2, (value) => {
      display.shadowStrength = value;
    }),
    shadowSoftness: bindDisplayNumber('shadow-softness', 'Shadow Softness', 2, (value) => {
      display.shadowSoftness = value;
    }),
    exposure: bindDisplayNumber('exposure', 'Exposure', 2, (value) => {
      display.exposure = value;
    }),
    brightness: bindDisplayNumber('brightness', 'Brightness', 2, (value) => {
      display.brightness = value;
    }),
    contrast: bindDisplayNumber('contrast', 'Contrast', 2, (value) => {
      display.contrast = value;
    }),
    roughness: bindDisplayNumber('roughness', 'Roughness', 2, (value) => {
      display.roughness = value;
    }),
    bloomStrength: bindDisplayNumber('bloom-strength', 'Bloom Strength', 2, (value) => {
      display.bloomStrength = value;
    }),
    bloomRadius: bindDisplayNumber('bloom-radius', 'Bloom Radius', 2, (value) => {
      display.bloomRadius = value;
    }),
    bloomThreshold: bindDisplayNumber('bloom-threshold', 'Bloom Threshold', 2, (value) => {
      display.bloomThreshold = value;
    }),
  };

  function bindDisplayNumber(
    id: string,
    label: string,
    decimals: number,
    assign: (value: number) => void,
  ): NumericBinding {
    return bindNumericControl({
      id,
      label,
      decimals,
      emitOnInput: true,
      onValue(value, phase) {
        assign(value);
        emitDisplayChange(toCamelCase(id), phase);
      },
    });
  }

  const closeSelectMenus = (): void => {
    document.querySelectorAll<HTMLElement>('.select-control.is-open').forEach((shell) => {
      shell.classList.remove('is-open');
      const menu = shell.querySelector<HTMLUListElement>('.select-menu');
      const trigger = shell.querySelector<HTMLButtonElement>('.select-trigger');
      if (menu) {
        menu.hidden = true;
      }
      trigger?.setAttribute('aria-expanded', 'false');
    });
  };

  const bindSelect = (
    id: string,
    label: string,
    onApply: (value: string) => void,
  ): SelectBinding => {
    const select = requiredElement(id, HTMLSelectElement);
    const trigger = requiredElement(`${id}-trigger`, HTMLButtonElement);
    const menu = requiredElement(`${id}-menu`, HTMLUListElement);
    const shell = trigger.closest<HTMLElement>('.select-control');
    const transaction = makeTransaction(label);

    if (!shell) {
      throw new Error(`Missing select shell for #${id}`);
    }

    const set = (value: string): void => {
      select.value = value;
      const option = Array.from(select.options).find((candidate) => candidate.value === value);
      trigger.textContent = option?.textContent ?? value;
      menu.querySelectorAll<HTMLButtonElement>('.select-option').forEach((button) => {
        button.classList.toggle('is-selected', button.dataset.value === value);
        button.setAttribute('aria-selected', button.dataset.value === value ? 'true' : 'false');
      });
    };

    const apply = (value: string): void => {
      transaction.begin();
      set(value);
      onApply(value);
      closeSelectMenus();
      transaction.commit();
    };

    menu.setAttribute('role', 'listbox');
    menu.replaceChildren();
    trigger.setAttribute('aria-haspopup', 'listbox');
    trigger.setAttribute('aria-expanded', 'false');

    Array.from(select.options).forEach((option) => {
      const item = document.createElement('li');
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'select-option';
      button.dataset.value = option.value;
      button.textContent = option.textContent;
      button.setAttribute('role', 'option');
      listen(button, 'click', (event) => {
        event.stopPropagation();
        apply(option.value);
      });
      item.append(button);
      menu.append(item);
    });

    listen(trigger, 'click', (event) => {
      event.stopPropagation();
      const opening = menu.hidden;
      closeSelectMenus();
      if (opening) {
        menu.hidden = false;
        shell.classList.add('is-open');
        trigger.setAttribute('aria-expanded', 'true');
      }
    });
    listen(select, 'change', () => apply(select.value));
    set(select.value);

    return {
      select,
      set,
      setDisabled(disabled: boolean): void {
        select.disabled = disabled;
        trigger.disabled = disabled;
        if (disabled) {
          closeSelectMenus();
        }
      },
    };
  };

  const seedShapeSelect = bindSelect('seed-shape', 'Seed Shape', (value) => {
    dla.seedShape = value as SeedShape;
    updateSeedRadiusAvailability();
    emitDlaChange('seedShape', 'commit', true);
  });

  const attachmentNeighborhoodSelect = bindSelect(
    'attachment-neighborhood',
    'Attachment Neighborhood',
    (value) => {
      dla.attachmentNeighborhood = parseAttachmentNeighborhood(value);
      stickNeighborsControl.setMaximum(dla.attachmentNeighborhood);
      dla.stickNeighbors = stickNeighborsControl.set(dla.stickNeighbors);
      emitDlaChange('attachmentNeighborhood', 'commit');
    },
  );

  const bindToggle = (
    input: HTMLInputElement,
    label: string,
    apply: (checked: boolean) => void,
  ): void => {
    listen(input, 'change', () => {
      runTransaction(label, () => apply(input.checked));
    });
  };

  bindToggle(hideEnclosed, 'Hide Enclosed', (checked) => {
    dla.hideEnclosed = checked;
    emitDlaChange('hideEnclosed', 'commit');
  });

  const bindColor = (
    input: HTMLInputElement,
    label: string,
    source: string,
    assign: (value: string) => void,
  ): void => {
    const transaction = makeTransaction(label);
    listen(input, 'input', () => {
      transaction.begin();
      assign(input.value);
      emitDisplayChange(source, 'input');
    });
    listen(input, 'change', () => {
      transaction.begin();
      assign(input.value);
      emitDisplayChange(source, 'commit');
      transaction.commit();
    });
    listen(input, 'blur', () => transaction.commit());
  };

  bindColor(innerColor, 'Inner Color', 'innerColor', (value) => {
    display.innerColor = value;
  });
  bindColor(outerColor, 'Outer Color', 'outerColor', (value) => {
    display.outerColor = value;
  });

  listen(startButton, 'click', () => {
    const nextRunning = !simulation.running;
    runTransaction(nextRunning ? 'Start Simulation' : 'Pause Simulation', () => {
      setRunning(nextRunning);
      callbacks.onStartPause?.(nextRunning);
    });
  });
  listen(resetButton, 'click', () => {
    runTransaction('Reset Simulation', () => callbacks.onReset?.());
  });
  listen(exportGlb, 'click', () => callbacks.onExportGlb?.());
  listen(exportObj, 'click', () => callbacks.onExportObj?.());
  listen(screenshot, 'click', () => callbacks.onScreenshot?.());

  listen(window, 'click', closeSelectMenus);
  listen(window, 'keydown', (event) => {
    const keyboardEvent = event as KeyboardEvent;
    if (keyboardEvent.key === 'Escape') {
      closeSelectMenus();
    }
    if (!(keyboardEvent.ctrlKey || keyboardEvent.metaKey) || keyboardEvent.altKey) {
      return;
    }
    if (isTextEditingTarget(keyboardEvent.target)) {
      return;
    }
    const key = keyboardEvent.key.toLowerCase();
    if (key === 'z') {
      const callback = keyboardEvent.shiftKey ? callbacks.onRedo : callbacks.onUndo;
      if (callback) {
        keyboardEvent.preventDefault();
        callback();
      }
    } else if (key === 'y' && callbacks.onRedo) {
      keyboardEvent.preventDefault();
      callbacks.onRedo();
    }
  });
  listen(
    window,
    'contextmenu',
    (event) => event.preventDefault(),
    { capture: true },
  );

  bindPanel();
  bindSections();
  syncControlsFromState();
  document.documentElement.classList.add('ui-ready');

  function bindPanel(): void {
    let dragging = false;
    let activeHandle: HTMLElement | null = null;
    let pointerId = -1;
    const offset = { x: 0, y: 0 };

    const clampPanel = (): void => {
      const rect = panel.getBoundingClientRect();
      const padding = window.innerWidth <= 700 ? 12 : 18;
      const maxLeft = Math.max(padding, window.innerWidth - rect.width - padding);
      const maxTop = Math.max(padding, window.innerHeight - rect.height - padding);
      panel.style.left = `${Math.min(maxLeft, Math.max(padding, rect.left))}px`;
      panel.style.top = `${Math.min(maxTop, Math.max(padding, rect.top))}px`;
      panel.style.right = 'auto';
      panel.style.bottom = 'auto';
    };

    const begin = (event: Event): void => {
      const pointerEvent = event as PointerEvent;
      if (pointerEvent.button !== 0) {
        return;
      }
      if (pointerEvent.target instanceof Element && pointerEvent.target.closest('.collapse-button')) {
        return;
      }
      const handle = pointerEvent.currentTarget as HTMLElement;
      const rect = panel.getBoundingClientRect();
      pointerEvent.preventDefault();
      dragging = true;
      activeHandle = handle;
      pointerId = pointerEvent.pointerId;
      offset.x = pointerEvent.clientX - rect.left;
      offset.y = pointerEvent.clientY - rect.top;
      panel.style.left = `${rect.left}px`;
      panel.style.top = `${rect.top}px`;
      panel.style.right = 'auto';
      panel.style.bottom = 'auto';
      handle.setPointerCapture?.(pointerId);
    };

    const move = (event: Event): void => {
      if (!dragging) {
        return;
      }
      const pointerEvent = event as PointerEvent;
      panel.style.left = `${pointerEvent.clientX - offset.x}px`;
      panel.style.top = `${pointerEvent.clientY - offset.y}px`;
      clampPanel();
    };

    const end = (): void => {
      if (activeHandle?.hasPointerCapture?.(pointerId)) {
        activeHandle.releasePointerCapture(pointerId);
      }
      dragging = false;
      activeHandle = null;
      pointerId = -1;
    };

    listen(handleTop, 'pointerdown', begin);
    listen(handleBottom, 'pointerdown', begin);
    listen(window, 'pointermove', move);
    listen(window, 'pointerup', end);
    listen(window, 'pointercancel', end);
    listen(window, 'resize', clampPanel);
    listen(collapseToggle, 'click', () => {
      const collapsed = panel.classList.toggle('is-collapsed');
      collapseToggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
      collapseToggle.setAttribute('aria-label', collapsed ? 'Expand controls' : 'Collapse controls');
      requestAnimationFrame(clampPanel);
    });
  }

  function bindSections(): void {
    document.querySelectorAll<HTMLElement>('.panel-section-header').forEach((header) => {
      const section = header.closest<HTMLElement>('.panel-section');
      if (!section) {
        return;
      }
      header.setAttribute('role', 'button');
      header.setAttribute('tabindex', '0');
      header.setAttribute('aria-expanded', 'true');
      const toggle = (): void => {
        closeSelectMenus();
        const collapsed = section.classList.toggle('is-collapsed');
        header.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
      };
      listen(header, 'click', toggle);
      listen(header, 'keydown', (event) => {
        const keyboardEvent = event as KeyboardEvent;
        if (keyboardEvent.key === 'Enter' || keyboardEvent.key === ' ') {
          keyboardEvent.preventDefault();
          toggle();
        }
      });
    });
  }

  function syncControlsFromState(): void {
    simulationRateControl.set(simulation.rate);
    timelineControl.setMaximum(Math.max(0, Math.round(simulation.latestTimeline)));
    timelineControl.set(simulation.timeline);

    seedControl.set(dla.seed);
    seedShapeSelect.set(dla.seedShape);
    seedRadiusControl.set(dla.seedRadius);
    targetParticlesControl.set(dla.targetParticles);
    attachmentNeighborhoodSelect.set(String(dla.attachmentNeighborhood));
    stickNeighborsControl.setMaximum(dla.attachmentNeighborhood);
    dla.stickNeighbors = stickNeighborsControl.set(dla.stickNeighbors);
    stickChanceControl.set(dla.stickChance);
    launchPaddingControl.set(dla.launchPadding);
    killPaddingControl.set(dla.killPadding);
    growthBatchControl.set(dla.growthBatch);
    walkerPoolControl.set(dla.walkerPool);
    rotationControl.set(dla.rotation);
    sphereScaleControl.set(dla.sphereScale);
    sphereGapControl.set(dla.sphereGap);
    sphereDetailControl.set(dla.sphereDetail);
    hideEnclosed.checked = dla.hideEnclosed;

    innerColor.value = display.innerColor;
    outerColor.value = display.outerColor;
    displayControls.lightAzimuth.set(display.lightAzimuth);
    displayControls.lightElevation.set(display.lightElevation);
    displayControls.keyBrightness.set(display.keyBrightness);
    displayControls.ambientFill.set(display.ambientFill);
    displayControls.rimBrightness.set(display.rimBrightness);
    displayControls.bounceBrightness.set(display.bounceBrightness);
    displayControls.shadowStrength.set(display.shadowStrength);
    displayControls.shadowSoftness.set(display.shadowSoftness);
    displayControls.exposure.set(display.exposure);
    displayControls.brightness.set(display.brightness);
    displayControls.contrast.set(display.contrast);
    displayControls.roughness.set(display.roughness);
    displayControls.bloomStrength.set(display.bloomStrength);
    displayControls.bloomRadius.set(display.bloomRadius);
    displayControls.bloomThreshold.set(display.bloomThreshold);

    updateSeedRadiusAvailability();
    updateRunningControls();
    updateParticleCountNote();
  }

  function updateRunningControls(): void {
    startButton.textContent = simulation.running ? 'Pause' : 'Start';
    startButton.classList.toggle('is-active', simulation.running);
    startButton.setAttribute('aria-pressed', simulation.running ? 'true' : 'false');
    const timelineDisabled = busy || simulation.running;
    timelineControl.setDisabled(timelineDisabled);
  }

  function updateSeedRadiusAvailability(): void {
    seedRadiusControl.setDisabled(dla.seedShape === 'point');
  }

  function updateParticleCountNote(): void {
    particleCountNote.textContent = `Particles = ${formatCount(currentParticleCount)} / ${formatCount(dla.targetParticles)}`;
  }

  function setRunning(running: boolean): void {
    simulation.running = running;
    updateRunningControls();
  }

  function setTimeline(value: number, latest: number): void {
    simulation.latestTimeline = Math.max(0, Math.round(latest));
    simulation.timeline = clamp(Math.round(value), 0, simulation.latestTimeline);
    timelineControl.setMaximum(simulation.latestTimeline);
    timelineControl.set(simulation.timeline);
  }

  function setBusy(message: string): void {
    busy = true;
    status.textContent = message;
    status.className = message ? 'is-visible' : '';
    [startButton, resetButton, exportGlb, exportObj, screenshot].forEach((button) => {
      button.disabled = true;
    });
    updateRunningControls();
  }

  function setReady(): void {
    busy = false;
    status.textContent = '';
    status.className = '';
    [startButton, resetButton, exportGlb, exportObj, screenshot].forEach((button) => {
      button.disabled = false;
    });
    updateRunningControls();
  }

  return {
    get simulation(): SimulationSettings {
      return { ...simulation };
    },
    get dla(): DlaSettings {
      return { ...dla };
    },
    get display(): DisplaySettings {
      return { ...display };
    },
    get settings(): UiSettingsSnapshot {
      return {
        simulation: { ...simulation },
        dla: { ...dla },
        display: { ...display },
      };
    },
    sync(snapshot): void {
      if (snapshot.simulation) {
        simulation = { ...simulation, ...snapshot.simulation };
      }
      if (snapshot.dla) {
        dla = { ...dla, ...snapshot.dla };
      }
      if (snapshot.display) {
        display = { ...display, ...snapshot.display };
      }
      syncControlsFromState();
    },
    setRunning,
    setTimeline,
    setParticleCount(count: number, target?: number): void {
      currentParticleCount = Math.max(0, Math.round(count));
      if (target !== undefined) {
        dla.targetParticles = targetParticlesControl.set(Math.max(1, Math.round(target)));
      }
      updateParticleCountNote();
    },
    setRotation(rotation: number): void {
      dla.rotation = rotationControl.set(rotation);
    },
    setBusy,
    setReady,
    setError(message: string): void {
      busy = false;
      status.textContent = message;
      status.className = 'is-visible is-error';
      [startButton, resetButton, exportGlb, exportObj, screenshot].forEach((button) => {
        button.disabled = false;
      });
      updateRunningControls();
    },
    beginTransaction(label: string): void {
      callbacks.onTransactionStart?.(label);
    },
    commitTransaction(label: string): void {
      callbacks.onTransactionCommit?.(label);
    },
    dispose(): void {
      if (disposed) {
        return;
      }
      activeTransactions.forEach((transaction) => transaction.commit());
      disposed = true;
      closeSelectMenus();
      cleanups.splice(0).reverse().forEach((cleanup) => cleanup());
      document.documentElement.classList.remove('ui-ready');
    },
  };
}

function requiredElement<T extends typeof HTMLElement>(id: string, constructor: T): InstanceType<T> {
  const element = document.getElementById(id);
  if (!(element instanceof constructor)) {
    throw new Error(`Missing #${id}`);
  }
  return element as InstanceType<T>;
}

function parseAttachmentNeighborhood(value: string): AttachmentNeighborhood {
  const parsed = Number.parseInt(value, 10);
  if (parsed === 6 || parsed === 18 || parsed === 26) {
    return parsed;
  }
  return 26;
}

function normalizeNumericValue(value: number, decimals: number, integer: boolean): number {
  if (integer) {
    return Math.round(value);
  }
  return Number(value.toFixed(decimals));
}

function clampToInputBounds(input: HTMLInputElement, value: number): number {
  const min = Number.parseFloat(input.min);
  const max = Number.parseFloat(input.max);
  let clamped = value;
  if (Number.isFinite(min)) {
    clamped = Math.max(min, clamped);
  }
  if (Number.isFinite(max)) {
    clamped = Math.min(max, clamped);
  }
  return clamped;
}

function extendInputBounds(input: HTMLInputElement, value: number): void {
  const min = Number.parseFloat(input.min);
  const max = Number.parseFloat(input.max);
  if (Number.isFinite(min) && value < min) {
    input.min = formatBoundValue(value);
  }
  if (Number.isFinite(max) && value > max) {
    input.max = formatBoundValue(value);
  }
}

function setValueInput(input: HTMLInputElement, value: number, decimals: number, force = false): void {
  if (!force && document.activeElement === input) {
    return;
  }
  input.value = formatNumericValue(value, decimals);
}

function formatNumericValue(value: number, decimals: number): string {
  return decimals === 0 ? `${Math.round(value)}` : value.toFixed(decimals);
}

function formatBoundValue(value: number): string {
  return `${value}`;
}

function updateRangeProgress(input: HTMLInputElement): void {
  const min = Number.parseFloat(input.min);
  const max = Number.parseFloat(input.max);
  const value = Number.parseFloat(input.value);
  const progress = max === min ? 0 : ((value - min) / (max - min)) * 100;
  input.style.setProperty('--range-progress', `${clamp(progress, 0, 100)}%`);
}

function formatCount(value: number): string {
  return Math.round(value).toLocaleString('en-US');
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function toCamelCase(value: string): string {
  return value.replace(/-([a-z])/g, (_, character: string) => character.toUpperCase());
}

function isTextEditingTarget(target: EventTarget | null): boolean {
  if (target instanceof HTMLTextAreaElement) {
    return true;
  }
  if (target instanceof HTMLInputElement) {
    return ['text', 'number', 'search', 'email', 'url', 'password'].includes(target.type);
  }
  return target instanceof HTMLElement && target.isContentEditable;
}
