# 260716_DLACoral

This branch contains the flat, prebuilt static bundle for deploying 260716_DLACoral with GitHub Pages. The application uses native WebGPU compute and rendering to grow and explore three-dimensional diffusion-limited aggregation coral forms.

Open the published application at [ekimroyrp.github.io/260716_DLACoral](https://ekimroyrp.github.io/260716_DLACoral/).

## Bundle structure

- `index.html` is the GitHub Pages entry point.
- `assets/` contains the hashed JavaScript and CSS production bundles.
- `env/` is reserved for deployable environment assets.
- `.nojekyll` disables Jekyll processing.

All runtime asset references are relative so the application works from the repository subpath. Development source and build tooling remain on the `main` branch.
