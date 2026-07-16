# 260716_DLAFractals

260716_DLAFractals is an interactive Three.js application for growing and exploring three-dimensional diffusion-limited aggregation (DLA) fractals. It uses native WebGPU compute, indirect sphere instancing, birth-order color gradients, a branchable simulation timeline, and a compact floating control panel designed for large aggregates.

## Features

- Native WebGPU simulation and rendering with no WebGL fallback
- Point, spherical-shell, and ring seed shapes
- Adjustable sticking neighborhood, neighbor threshold, sticking chance, walker pool, launch/kill padding, and growth batch
- Shared detail-adjustable icosphere mesh with indirect instanced rendering
- Inner-to-outer color gradient based on particle birth order
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
git clone https://github.com/ekimroyrp/260716_DLAFractals.git
cd 260716_DLAFractals
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
- Right-click browser menus are disabled inside the app

### Simulation

| Control | Default | Initial range / behavior |
|---|---:|---|
| Start / Pause | Paused | Toggles live growth |
| Reset | — | Returns to the selected seed while preserving run state |
| Simulation Timeline | 0 | Attached-particle birth prefix; available while paused |
| Simulation Rate | 1.00 | `0.10–3.00`, step `0.01`; scales compute epochs |

Starting from an earlier timeline point removes the later future and grows a new deterministic branch.

### Diffusion-Limited Aggregation

| Control | Default | Initial range / options |
|---|---:|---|
| Seed | 260716 | `1–999999`, step `1` |
| Seed Shape | Point | Point, Sphere, Ring |
| Seed Radius | 8 | `1–64`, step `1` |
| Target Particles | 1,000,000 | `1,000–1,000,000`, step `1,000` |
| Attachment Neighborhood | Full | Faces 6, Faces + Edges 18, Full 26 |
| Stick Neighbors | 1 | `1` through the selected neighborhood size |
| Stick Chance | 1.00 | `0.01–1.00`, step `0.01` |
| Launch Padding | 3 | `1–32`, step `1` |
| Kill Padding | 3 | `1–64`, step `1` |
| Growth Batch | 256 | `1–4096`, step `1` |
| Walker Pool | 65,536 | `1,024–131,072`, step `1,024` |
| Rotation | 0 | `-360–360`, step `1` |
| Sphere Scale | 1.00 | `0.42–1.15`, step `0.01` |
| Sphere Gap | 0.00 | `0–0.38`, step `0.01` |
| Sphere Detail | 0 | `0–2`, step `1` (60 / 240 / 960 vertices) |
| Hide Enclosed | On | Omits fully surrounded particles from drawing |

`Growth Batch = 1` commits one attachment per compute epoch. Larger values accept a bounded deterministic batch evaluated against the same pre-commit aggregate. The GPU device limit can reduce exceptionally large typed targets, walker pools, or seed shells.

### Display

All seed particles use Inner Color. Attached particles interpolate by birth rank, and the newest displayed attachment reaches Outer Color exactly.

| Control | Default | Initial range / step |
|---|---:|---|
| Inner Color | `#6b2f24` | Color |
| Outer Color | `#f4e6d2` | Color |
| Light Azimuth | 25.65 | `-180–180` / `0.01` |
| Light Elevation | 68.70 | `-20–85` / `0.01` |
| Key Brightness | 2.41 | `0–12` / `0.01` |
| Ambient Fill | 0.30 | `0–2` / `0.01` |
| Rim Brightness | 0.49 | `0–5` / `0.01` |
| Bounce Brightness | 0.07 | `0–2` / `0.01` |
| Shadow Strength | 1.08 | `0–1.5` / `0.01` |
| Shadow Softness | 2.60 | `0–5` / `0.01` |
| Exposure | 0.70 | `0.1–3` / `0.01` |
| Brightness | 1.00 | `0.1–3` / `0.01` |
| Contrast | 2.25 | `0.1–3` / `0.01` |
| Roughness | 0.92 | `0–1` / `0.01` |
| Bloom Strength | 0.08 | `0–2` / `0.01` |
| Bloom Radius | 0.26 | `0–1` / `0.01` |
| Bloom Threshold | 0.00 | `0–2` / `0.01` |

Every continuous slider has a selectable numeric input. Enter or blur commits a value, invalid text reverts, and valid values beyond an initial range extend the matching slider bound unless a simulation or WebGPU invariant requires clamping.

### History

- Ctrl/Cmd+Z: undo
- Ctrl/Cmd+Y or Shift+Ctrl/Cmd+Z: redo
- History covers deliberate control changes, Start/Pause, Reset, committed timeline seeks and branches, and complete model-rotation gestures.
- Automatic simulation ticks and camera navigation are excluded. History retains up to 120 actions and budgets compact aggregate snapshots to 128 MiB.

### Export

- **GLB** exports the displayed aggregate with `EXT_mesh_gpu_instancing`, current transforms, and age colors.
- **OBJ** exports every displayed sphere as expanded colored geometry.
- **Screenshot** saves the current canvas as a PNG.

Export filenames begin with `260716_DLAFractals`.

## Architecture

The live path uses raw WGSL compute buffers for walkers, occupied/frontier hash cells, cached neighbor counts, birth-ordered particles, instance matrices, birth ranks, counters, and indirect draw arguments. Normal growth reads back only a persistent 32-byte status buffer; sphere rendering remains one indirect instanced draw per render pass with no per-particle JavaScript render loop. Uniform spherical launches, nearby kill-radius recycling, squared-distance checks, neighbor caching, enclosed-particle hiding, and adaptive update work follow the practical guidance in [Softology's 3D DLA notes](https://softologyblog.wordpress.com/2017/05/22/pushing-3d-diffusion-limited-aggregation-even-further/).
