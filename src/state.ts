import type { AppSnapshot, DisplaySettings, DlaSettings, DlaSnapshot, SimulationSettings } from './types';
import {
  DEFAULT_DISPLAY_SETTINGS,
  DEFAULT_DLA_SETTINGS,
  DEFAULT_SIMULATION_SETTINGS,
} from './types';

export interface MutableAppState {
  simulation: SimulationSettings;
  dla: DlaSettings;
  display: DisplaySettings;
}

export function createInitialAppState(): MutableAppState {
  return {
    simulation: { ...DEFAULT_SIMULATION_SETTINGS },
    dla: { ...DEFAULT_DLA_SETTINGS },
    display: { ...DEFAULT_DISPLAY_SETTINGS },
  };
}

export function cloneDlaSnapshot(snapshot: DlaSnapshot): DlaSnapshot {
  const cloned = {
    ...snapshot,
    positions: snapshot.positions.slice(),
    enclosed: snapshot.enclosed.slice(),
  };
  const gpuSnapshot = snapshot as DlaSnapshot & { walkerState?: Int32Array };
  if (gpuSnapshot.walkerState instanceof Int32Array) {
    (cloned as DlaSnapshot & { walkerState: Int32Array }).walkerState = gpuSnapshot.walkerState.slice();
  }
  return cloned;
}

export function createAppSnapshot(state: MutableAppState, aggregate?: DlaSnapshot): AppSnapshot {
  return {
    simulation: { ...state.simulation },
    dla: { ...state.dla },
    display: { ...state.display },
    aggregate: aggregate ? cloneDlaSnapshot(aggregate) : undefined,
  };
}

export function applySettingsSnapshot(state: MutableAppState, snapshot: AppSnapshot): void {
  Object.assign(state.simulation, snapshot.simulation);
  Object.assign(state.dla, snapshot.dla);
  Object.assign(state.display, snapshot.display);
}
