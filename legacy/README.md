# legacy/

Archived code, kept for historical reference only. **Nothing here is imported by
any active file of the project**, and none of it is maintained.

| File | What it was |
|---|---|
| `nerf_module.py` | The pre-refactor custom NeRF implementation (model + training + query). Replaced by the `nerf/` package, which implements vanilla bmild/nerf in PyTorch with coarse+fine hierarchical sampling and optional depth hints for the indirect irradiance pass. |
| `tiny_nerf_data.npz` | The toy dataset `nerf_module.py` was developed against. |
| `pbr_solver_coscone.py.bak` | The PBR solver as it was before the 2026-07-16 model revision, when `L_j` was a cosine-weighted integral over the cone (`"coscone"`) instead of a pure solid-angle mean. Kept because the thesis discusses the difference. |
| `inspect_pbr_results.py`, `inspect_pbr_results2.py` | Diagnostic viewers for the PBR fit. Deprecated: they read the old flat output layout and the `"rings"` file patterns, both of which predate the `sources/{source}/` layout and the `"cones"` bake format. Use `scripts/inspect_final_maps.py` and `scripts/inspect_spec_cone.py` instead. |
