# 260716_DLACoral

260716_DLACoral is an interactive Three.js application for growing and exploring three-dimensional diffusion-limited aggregation (DLA) coral forms. It uses native WebGPU compute, indirect sphere instancing, birth-order color gradients, a branchable simulation timeline, and a compact floating control panel designed for large aggregates.

## Features

- Native WebGPU simulation and rendering with no WebGL fallback
- Point, spherical-shell, and ring seed shapes
- Eight fixed attachment-neighborhood presets, strict neighbor threshold, persistent contact hits, bootstrap growth, sticking chance, walker pool, launch/kill padding, and growth batch
- Independently sized seed particles, allowing fixed-radius shells and rings to use more small particles or fewer large ones
- Shared resolution-adjustable icosphere mesh with indirect instanced rendering
- Zero-gap particle contact independent of attachment-neighborhood rules
- Birth-order Gradient Start/End colors with adjustable contrast, bias, and blur
- Studio lighting, soft shadows, ACES tone mapping, and bloom
- Start/Pause, Reset, Simulation Rate, and branchable Simulation Timeline
- Ctrl/Cmd+Z and Ctrl/Cmd+Y action history
- GLB, OBJ, and PNG screenshot export
- Draggable, collapsible floating controls

## Requirements

- A browser and GPU with native WebGPU support
- Node.js `20.19+` or `22.12+` and npm

Unsupported browsers display an in-app WebGPU error instead of silently switching renderers.

## Getting Started

```powershell
git clone https://github.com/ekimroyrp/260716_DLACoral.git
cd 260716_DLACoral
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

Build and test with:

```powershell
npm run build
npm test
```

## Controls

### Navigation

- Mouse wheel: zoom
- Middle mouse drag: pan
- Right mouse drag: orbit camera
- Left mouse drag on the canvas: rotate the aggregate
- Growth, timeline changes, resets, and settings never reframe the camera; only mouse navigation changes its pose and target
- Right-click browser menus are disabled inside the app

### Simulation

| Control | Default | Initial range / behavior |
|---|---:|---|
| Start / Pause | Paused | Toggles live growth |
| Reset | — | Returns to the selected seed while preserving run state |
| Simulation Timeline | 0 | Attached-particle birth prefix; available while paused |
| Simulation Rate | 1.00 | `0.10–3.00`, step `0.01`; scales compute epochs |

Starting from an earlier timeline point removes the later future and grows a new deterministic branch.

### Aggregation

| Control | Default | Initial range / options |
|---|---:|---|
| Seed | 260716 | `1–999999`, step `1` |
| Seed Shape | Point | Point, Sphere, Ring |
| Seed Radius | 8 | `1–64`, step `1` |
| Seed Rotation | 0 | `-360–360`, step `1` |
| Particle Size | 1.00 | `0.10–4.00`, step `0.01` |
| Particle Gap | 0.00 | `0–0.38`, step `0.01` |
| Particle Scale | 1.00 | `0.10–3.00`, step `0.01` |
| Particle Resolution | 2 | `0–2`, step `1` (60 / 240 / 960 vertices) |
| Target Particles | 1,000,000 | `1,000–1,000,000`, step `1,000` |
| Attachment Neighborhood | Full 26 | Faces 6, Faces + Edges 18, Full 26, Weighted Full 26, Radius 2, Radius 3, Surface Hemisphere, Randomized Neighborhood |
| Stick Neighbors | 1 | `1` through the selected neighborhood score maximum |
| Contact Hits | 1 | `1–1,000`, step `1` |
| Bootstrap Particles | 50 | `0–10,000`, step `1` |
| Stick Chance | 1.00 | `0.01–1.00`, step `0.01` |
| Launch Padding | 3 | `1–32`, step `1` |
| Kill Padding | 3 | `1–64`, step `1` |
| Growth Batch | 256 | `1–4096`, step `1` |
| Walker Pool | 65,536 | `1,024–131,072`, step `1,024` |
| Hide Enclosed | On | Omits fully surrounded particles from drawing |

`Contact Hits` is the number of aggregate contacts an individual walker must accumulate before it can stick. The count persists across compute updates and resets when that walker is relaunched. `Bootstrap Particles` allows one-neighbor growth for that many attached particles, then enforces `Stick Neighbors` strictly. The defaults (`Contact Hits = 1`, `Bootstrap Particles = 50`, and `Stick Neighbors = 1`) preserve the original default growth result.

`Weighted Full 26` scores face, edge, and corner contacts as `3`, `2`, and `1`. Radius 2 and Radius 3 count occupied cells inside fixed spherical lattice radii. Surface Hemisphere selects the inward member of 13 opposite direction pairs. Randomized Neighborhood selects a deterministic 13-direction mask from the existing Seed. These presets use the existing Stick Neighbors threshold and add no additional settings.

`Particle Size` sets both the simulation lattice spacing and the base sphere diameter. Seed Radius remains a world-space radius, so decreasing Particle Size packs more particles into Sphere and Ring seeds while increasing it uses fewer. Changing Particle Size rebuilds the seed. `Particle Scale` resizes each rendered particle around its center without changing lattice spacing, while `Particle Gap` adds proportional separation. Attachment Neighborhood affects aggregation only and never changes particle size.

`Growth Batch = 1` commits one attachment per compute epoch. Larger values accept a bounded deterministic batch evaluated against the same pre-commit aggregate. The GPU device limit can reduce exceptionally large typed targets, walker pools, or seed shells.

### Display

All seed particles use the exact Gradient Start albedo. Attached particles interpolate by birth rank, and the newest displayed attachment reaches Gradient End exactly. Gradient Contrast and Gradient Bias shape the age ramp using the DifferentialGrowth curve; Gradient Blur softens that grading toward the underlying linear ramp. Brightness and Contrast grade attached age colors without replacing the seed color.

| Control | Default | Initial range / step |
|---|---:|---|
| Gradient Start | `#ac2a4a` | Color |
| Gradient End | `#ffffff` | Color |
| Gradient Contrast | 1.37 | `0.2–3` / `0.01` |
| Gradient Bias | -0.74 | `-1–1` / `0.01` |
| Gradient Blur | 0.45 | `0–1` / `0.01` |
| Light Azimuth | -3.08 | `-180–180` / `0.01` |
| Light Elevation | 55.79 | `-20–85` / `0.01` |
| Key Brightness | 3.37 | `0–12` / `0.01` |
| Ambient Fill | 0.80 | `0–2` / `0.01` |
| Rim Brightness | 0.49 | `0–5` / `0.01` |
| Bounce Brightness | 0.45 | `0–2` / `0.01` |
| Shadow Strength | 1.13 | `0–1.5` / `0.01` |
| Shadow Softness | 2.09 | `0–5` / `0.01` |
| Exposure | 0.68 | `0.1–3` / `0.01` |
| Brightness | 1.15 | `0.1–3` / `0.01` |
| Contrast | 2.55 | `0.1–3` / `0.01` |
| Roughness | 0.00 | `0–1` / `0.01` |
| Bloom Strength | 0.13 | `0–2` / `0.01` |
| Bloom Radius | 0.24 | `0–1` / `0.01` |
| Bloom Threshold | 0.19 | `0–2` / `0.01` |

Every continuous slider has a selectable numeric input. Enter or blur commits a value, invalid text reverts, and valid values beyond an initial range extend the matching slider bound unless a simulation or WebGPU invariant requires clamping.

Parameter names and row backgrounds are non-interactive. Numeric fields, sliders, selects, color inputs, and toggles respond only when clicked directly.

### History

- Ctrl/Cmd+Z: undo
- Ctrl/Cmd+Y or Shift+Ctrl/Cmd+Z: redo
- History covers deliberate control changes, Start/Pause, Reset, committed timeline seeks and branches, and complete seed-rotation gestures.
- Automatic simulation ticks and camera navigation are excluded. History retains up to 120 actions and budgets losslessly compressed aggregate snapshots to 128 MiB while preserving exact walker continuation state.

### Export

- **GLB** exports the displayed aggregate with `EXT_mesh_gpu_instancing`, current transforms, and age colors.
- **OBJ** exports every displayed sphere as expanded colored geometry.
- **Screenshot** saves the current canvas as a PNG.

Export filenames begin with `260716_DLACoral`.

## Architecture

The live path uses raw WGSL compute buffers for walkers, occupied/frontier hash cells, cached neighbor counts, birth-ordered particles, instance matrices, birth ranks, counters, and indirect draw arguments. The sparse hash starts at the active seed size and grows on the GPU with projected 70% headroom instead of allocating its one-million-particle maximum at startup. Normal growth reads back only a persistent 40-byte status buffer; sphere rendering remains one indirect instanced draw per render pass with no per-particle JavaScript render loop. Uniform spherical launches, nearby kill-radius recycling, squared-distance checks, neighbor caching, enclosed-particle hiding, and adaptive update work follow the practical guidance in [Softology's 3D DLA notes](https://softologyblog.wordpress.com/2017/05/22/pushing-3d-diffusion-limited-aggregation-even-further/).
