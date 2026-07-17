# 260716_DLACoral

260716_DLACoral is a Vite + TypeScript + Three.js application for growing and exploring three-dimensional diffusion-limited aggregation coral forms. Native WebGPU compute evolves large particle aggregates entirely on the GPU, while indirect icosphere instancing, birth-order color gradients, studio lighting, timeline branching, and model export turn the simulation into an interactive coral-design tool.

## Features

- Native WebGPU simulation and rendering with raw WGSL compute and no WebGL fallback.
- GPU walker pools, sparse occupied/frontier hash buffers, cached neighbor counts, adaptive capacity growth, and indirect instanced drawing for large aggregates.
- Point, spherical-shell, and ring seeds with independently controlled world-space radius, rotation, and particle size.
- Eight attachment-neighborhood modes: Faces 6, Faces + Edges 18, Full 26, Weighted Full 26, Radius 2, Radius 3, Surface Hemisphere, and deterministic Randomized Neighborhood.
- Detailed aggregation controls for adaptive or strict neighbor thresholds, persistent contact hits, bootstrap growth, sticking chance, walker count, launch/kill padding, and growth batch size.
- Resolution-adjustable shared icosphere geometry with automatic live detail reduction from resolution `2` to `1` at 25,000 visible particles and from `1` to `0` at 100,000.
- Particle Size, Gap, and Scale controls that separate simulation spacing from rendered particle appearance.
- Birth-order Gradient Start/End colors with contrast, bias, blur, image grading, ACES tone mapping, studio lighting, soft shadows, and bloom.
- Start/Pause, Reset, Simulation Rate, and a branchable Simulation Timeline that can resume deterministic growth from an earlier prefix.
- Undo/redo history for control changes, resets, timeline branches, and seed-rotation gestures, including compressed aggregate snapshots and exact walker continuation state.
- Compatible GLB and OBJ exports containing every displayed sphere with its current transform and age color, plus PNG screenshot export.
- Draggable and collapsible control panel, direct numeric entry, device-aware limits, aggregate rotation, orbit, pan, and zoom navigation.

## Getting Started

1. Install Node.js `20.19+` or `22.12+` and use a browser/GPU with native WebGPU support.
2. Clone the repository with `git clone https://github.com/ekimroyrp/260716_DLACoral.git`, then run `cd 260716_DLACoral`.
3. Run `npm install` to install the locked project dependencies.
4. Run `npm run dev` and open `http://127.0.0.1:5173`.
5. Run `npm test` to execute the Vitest suite.
6. Run `npm run build` to type-check the project and emit the production bundle.

Unsupported browsers show an in-app WebGPU error instead of silently switching to another renderer.

## Controls

- **Start / Pause:** Toggles live GPU growth without resetting the current aggregate.
- **Reset:** Returns to the selected seed configuration while preserving the current run-state setting.
- **Simulation Timeline:** Scrubs the attached-particle birth prefix while paused. Starting from an earlier point discards the later future and grows a new deterministic branch.
- **Simulation Rate:** Scales the amount of compute work performed during live growth.
- **Seed controls:** Seed selects the deterministic random sequence; Seed Shape chooses Point, Sphere, or Ring; Seed Radius sets shell/ring size; Seed Rotation rotates the aggregate without moving the camera.
- **Particle controls:** Particle Size changes simulation lattice spacing and base sphere diameter; Particle Gap adds proportional separation; Particle Scale resizes only the rendered spheres.
- **Particle Resolution:** Selects 60, 240, or 960 vertices per sphere with values `0`, `1`, or `2`. The application automatically lowers this value as visible particle counts cross 25,000 and 100,000, never raises it automatically, and keeps growth running during the switch.
- **Target Particles:** Sets the intended aggregate size and expands within the active WebGPU device limits.
- **Adaptive:** Enabled by default. Uses the highest neighbor score attained by eligible candidates during each growth epoch, capped by Stick Neighbors. Disable it to enforce Stick Neighbors strictly after the bootstrap period.
- **Attachment Neighborhood / Stick Neighbors:** Selects which nearby cells contribute to attachment and the preferred adaptive cap or strict occupied-neighbor requirement.
- **Contact Hits / Bootstrap Particles / Stick Chance:** Controls repeated walker contact requirements, the initial one-neighbor growth period, and the probability of accepting an eligible attachment.
- **Launch Padding / Kill Padding:** Controls where walkers spawn outside the aggregate and how far they may travel before recycling.
- **Growth Batch / Walker Pool:** Controls attachments accepted per compute epoch and the number of walkers evaluated in parallel.
- **Hide Enclosed:** Omits fully surrounded particles from the visible draw and exported displayed prefix.
- **Gradient controls:** Gradient Start/End define birth-order color endpoints; Gradient Contrast, Bias, and Blur shape the age ramp.
- **Lighting and image controls:** Light Azimuth/Elevation, Key Brightness, Ambient Fill, Rim Brightness, Bounce Brightness, Shadow Strength/Softness, Exposure, Brightness, Contrast, Roughness, and Bloom tune the final render.
- **Viewport navigation:** Mouse wheel zooms, middle-drag pans, right-drag orbits the camera, and left-drag rotates the aggregate. Simulation changes never reframe the camera.
- **History shortcuts:** `Ctrl/Cmd+Z` undoes; `Ctrl/Cmd+Y` or `Shift+Ctrl/Cmd+Z` redoes.
- **Export:** GLB and OBJ download the complete displayed sphere geometry; Screenshot downloads the current canvas as a PNG. Export filenames begin with `260716_DLACoral`.

## Deployment

- **Local production preview:** Run `npm install`, then `npm run build` followed by `npm run preview` and open `http://127.0.0.1:4173` to inspect the compiled bundle.
- **Publish to GitHub Pages:** From a clean `main`, run `npm run build -- --base=./`. In a separate temporary clone or worktree, create or check out the orphan `gh-pages` branch, remove the development source, copy everything inside `dist/` to the branch root, include `.nojekyll`, `.gitignore`, and any required `env/` assets, commit with a descriptive message, and run `git push origin gh-pages`. Configure GitHub Pages to deploy from `gh-pages` at `/ (root)`.
- **Live demo:** [https://ekimroyrp.github.io/260716_DLACoral/](https://ekimroyrp.github.io/260716_DLACoral/)
