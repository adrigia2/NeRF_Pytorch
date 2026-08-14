# NeRF_Pytorch — hybrid rendering pipeline

Python side of a thesis project that reconstructs PBR material maps by combining
**GPU ray tracing** (NVIDIA OptiX) with **neural rendering** (a NeRF trained in PyTorch).

The repository has two halves:

- **[`OptixProjectCMake/`](../OptixProjectCMake/README.md)** (sibling folder, C++/CUDA) — the
  OptiX ray tracing passes: depth, positions, normals, masks, inverse UV mapping, visibility,
  baked colour textures, skybox irradiance, indirect irradiance and specular cones. It is
  exposed to Python as the module `OptixProgrammablePasses`.
- **`NeRF_Pytorch/`** (this folder) — the pipeline that calls those passes, trains the
  NeRF, bakes everything into texture space and fits the PBR model.

---

## 1. What the pipeline does

`images_generator.py` runs four steps, each independently toggle-able:

| Step | What it produces | Needs OptiX | Needs a GPU |
|---|---|:---:|:---:|
| **1** | Per-frame depth, world position, normals and mask; copies the source images; writes `transforms_extended.json` (the minimum a NeRF needs). | yes | yes |
| **2** | Trains the NeRF (`nerf/train.py`) and saves the checkpoint. Optional **Step 2b** re-renders every training frame with the trained model. | no | yes |
| **3** | The texture-space bake: inverse UV mapping → visibility → colour texture → skybox irradiance → indirect irradiance → specular cones. | yes | yes |
| **4** | The reconstruction: PBR fit (`pbr_solver.py`) plus the Lambertian albedo. Reads **only** the Step 3 cache on disk. | no | no |

The last row is the useful one for a quick look: **Step 4 and every script under
`figures/` read from disk only**, so they run on a machine with no CUDA toolchain and
no OptiX SDK, against a run that already exists.

### The physical model

The solver fits, per texel, the model

```
C_j = (a·x/π)·E + (1 − x)·L_j
```

where `C_j` is the colour that camera *j* sees, `E` is the incident irradiance
(skybox + indirect), `a` the albedo, `x` the diffuse weight and `L_j` the mean
environment radiance over a cone around the reflected ray of camera *j*. The cones are
baked at a set of apertures (`spec_cone_apertures_deg`); the fit picks the aperture that
minimises the residual, and writes `metallic = 1 − x`, `roughness = aperture/180` and
`albedo_pbr = a`.

---

## 2. Installation

### 2.1 Python environment

```bash
conda env create -f environment.yml
conda activate tesi-nerf
```

This creates an environment named **`tesi-nerf`** (Python 3.10) with pinned versions of
`torch` (CUDA 13.0 build), `numpy`, `scipy`, `openexr`, `pillow`, `opencv-python`,
`matplotlib`, `tqdm` and `tensorboard`. The torch wheel is roughly 3 GB, so expect the
solve+download to take several minutes.

Check it:

```bash
python -c "import torch, cv2, scipy, OpenEXR, Imath, matplotlib; print(torch.__version__, torch.cuda.is_available())"
```

### 2.2 The native OptiX module (optional, needed by Steps 1 and 3)

`OptixProgrammablePasses` is **not** in `environment.yml` on purpose: it is a local
pybind11/CMake extension that has to be compiled, and listing it would make
`conda env create` fail on any machine without the toolchain.

Prerequisites:

- Visual Studio 2022 (MSVC toolset, x64)
- NVIDIA CUDA Toolkit
- NVIDIA OptiX SDK 9.0.0
- the pybind11 submodule, which the build needs and a plain clone leaves empty:

```bash
git -C ../OptixProjectCMake submodule update --init --recursive
```

Then, from inside the activated environment:

```bash
pip install -e ../OptixProjectCMake
python -c "import OptixProgrammablePasses; print('ok')"
```

If the SDK is not at its default path
(`C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0`), point the build at it:

```bash
OPTIX_INSTALL_DIR="D:/SDKs/OptiX SDK 9.0.0" pip install -e ../OptixProjectCMake
```

See [`OptixProjectCMake/README.md`](../OptixProjectCMake/README.md) for the passes
themselves, the architecture, and how to add one.

To verify the OptiX path end to end:

```bash
python scripts/test_spec_cone_smoke.py
python scripts/test_hemivis_shared.py
```

Both print a final `✓` line. **Ignore their exit code**: the process exits non-zero at
interpreter shutdown because of a known OptiX cleanup issue, not because of a failure.

---

## 3. Loading a scene

Scenes live outside this folder, under
`OptixProjectCMake/Scenes/<SceneName>/`. The layout is a convention, not something the
code enforces; the pipeline only ever reads three or four explicit paths.

```
OptixProjectCMake/Scenes/TableAndOtherInterior/
├── Nerf<Variant>/                     ← a capture: cameras + images
│   ├── transforms.json
│   └── images/
│       ├── render_Camera_Shell10_0.exr
│       └── ...
├── Models<Variant>/
│   └── Baked.obj                      ← the geometry the OptiX passes trace
├── BlenderBaked<Variant>/             ← reference bakes out of Blender
│   ├── BakedMaterial_base_color.exr
│   ├── BakedMaterial_metallic.exr
│   ├── BakedMaterial_normal.exr
│   └── BakedMaterial_roughness.exr
└── Blender/assets/hdri/*.exr          ← equirectangular HDR environment maps
```

Different variants of the same scene (different lighting, different mesh smoothing) are
separate `Nerf*`/`Models*`/`BlenderBaked*` folder triples.

### `transforms.json`

Standard NeRF/instant-ngp format. The fields the pipeline reads:

| Field | Meaning |
|---|---|
| `camera_angle_x` | Horizontal FOV, in radians. Used when `fl_x` is absent. |
| `fl_x`, `fl_y` | Focal lengths in pixels. Non-square pixels are supported; when `fl_y` is missing, `fl_x` is used for both axes. |
| `cx`, `cy` | Principal point, in pixels. |
| `w`, `h` | Image size. |
| `frames[].file_path` | Path to the image, resolved **relative to the json file**. |
| `frames[].transform_matrix` | 4×4 camera-to-world matrix. |

**World convention: Z-up, Y-forward (Blender-native).** OBJ vertices are loaded as-is,
with no axis swap, and the envmap lookup uses +Z as zenith. Normal maps baked in Blender
in object/world space are already in this frame and need no remapping.
`skybox_yaw_degrees` (default `0.0`) rotates the equirectangular lookup; at 0° Blender's
−Y forward direction sits at the centre of the envmap.

**A note on the mesh normals.** The OBJ loader reads positions, indices and UVs only, so
the IUM pass computes *face* normals by cross product (CCW winding assumed). For smooth
shading, supply a baked normal map through `external_normal_path`.

---

## 4. Configuring a run

`images_generator.py` **has no command-line interface**: the run is described by the
`if __name__ == "__main__":` block at the bottom of the file (from line ~3877), which you
edit. There are three levels of configuration.

### 4.1 `SceneConfig` — one per scene

```python
SceneConfig(
    name                 = "TableAndOtherInteriorWithSpecularNight",  # output subfolder name
    transforms_path      = f"{REPO}/Scenes/TableAndOtherInterior/NerfOpenEXRSmoothNight/transforms.json",
    model_path           = f"{REPO}/Scenes/TableAndOtherInterior/ModelsSmooth/Baked.obj",
    external_normal_path = f"{REPO}/Scenes/TableAndOtherInterior/BlenderBakedSmoothNight/BakedMaterial_normal.exr",
    skybox_path          = None,   # only used when skybox_source == "file"
)
```

Add or comment out entries in the `SCENES` list to choose what gets processed. Every
scene lands in `<output_root>/<scene.name>/`.

### 4.2 `RenderConfig` — what to bake and how

Passed as `PipelineConfig(render=RenderConfig(...))`. The knobs that matter in practice:

| Field | Default | Notes |
|---|---|---|
| `render_depth` / `_position` / `_normal` / `_mask` | `True` | Step 1 layers. |
| `render_ium`, `render_visibility` | `True` | Step 3 geometry passes. Everything below depends on them. |
| `ium_texture_size` | `[512, 512]` | Texture-space resolution, `[w, h]`. Production runs use `[4096, 4096]`. |
| `external_normal_path` | `None` | Baked normal map replacing the OptiX face normals. |
| `external_normal_resolution_mode` | `None` | `"resample"` (rescale the map), `"adapt"` (rescale the IUM instead), `None` (ask at runtime). |
| `apply_scale` | `False` | **Must stay `False`.** The `scale` in `transforms.json` would also have to be applied to the camera translations; applying it on one side only puts the NeRF query points off the mesh surface. |
| `render_color_texture` | `False` | Bakes the observed colour into texture space. |
| `color_texture_image_sources` | `["gt"]` | Which images to bake from: `"gt"` (ground truth) and/or `"nerf"` (the Step 2b predictions). **Every entry gets a full, independent set of outputs** under `sources/<name>/`. |
| `color_texture_grazing_max_deg` | `75.0` | Cameras seeing a texel at a more grazing angle than this are discarded (background bleed at silhouettes). `90.0` disables it. |
| `render_irradiance` | `False` | Per-texel skybox irradiance, by deterministic Fibonacci quadrature. |
| `skybox_source` | `"file"` | `"file"` reads `skybox_path`; `"nerf"` bakes the trained NeRF's background sphere and saves it as `skybox_nerf_baked.exr`. |
| `irradiance_sample_side` | `16` | N → N×N directions per hemisphere. |
| `precompute_indirect` | `False` | Indirect irradiance queried from the NeRF. Needs the checkpoint. |
| `precompute_spec_cone` | `False` | The specular-cone bake. Needs the checkpoint, the IUM and the visibility. |
| `spec_cone_scheme` | `"per_camera"` | `"shared"` traces one Fibonacci set per texel and lets every camera bin the same rays: cost `S + m` rays per texel instead of `m·ΣN_i`. Recommended. |
| `spec_cone_shared_samples` | `16384` | S, the shared set size (`"shared"` only). Angular resolution goes as `1/√S`. |
| `spec_cone_apertures_deg` | 11 values | Total cone apertures in degrees, increasing, first element **must be 0** (the mirror ray). With `"shared"`, refining this grid costs zero extra rays. |
| `spec_cone_tile_size` | `8192` | **With `spec_cone_scheme="shared"` this must be a multiple of the IUM width** (each tile is a block of whole scanlines, streamed into the per-camera EXRs). Validated at startup. |
| `spec_cone_chunk_texels` | `256` | Torch sub-block size (`"shared"` only). Peak VRAM only. |
| `spec_cone_nerf_chunk` | `None` | Rays per NeRF batch during the bake. This, not `chunk_texels`, is what caps GPU occupancy. |
| `render_pbr_maps` | `False` | Runs `pbr_solver.py` at the end of Step 4. |
| `pbr_spec_threshold` | `0.2` | Minimum metallic for the fitted roughness to be trusted; `0.0` writes it everywhere. |
| `pbr_tile_texels` | `1 << 20` | Solver band size. Caps peak RAM (~2.5 GiB at 1 M texels, 14 candidates) without changing the result. |
| `render_albedo` | `False` | Classic Lambertian albedo `π·C / (E_sky + E_ind)`. |
| `roi_rect`, `roi_mask_path`, `roi_tag` | unset | Texture-space test ROI — see §6.2. |

### 4.3 `PipelineConfig` — steps and NeRF hyper-parameters

| Field | Default | Notes |
|---|---|---|
| `run_step1` … `run_step4` | `True` | The four toggles. |
| `resume_skip_step2_if_ckpt` | `False` | Skip training when the checkpoint already exists. Careful: an *interrupted* checkpoint is reused too. Delete the `.pt` to force a retrain. |
| `nerf_num_iters` | `10000` | Training budget. |
| `nerf_batch_size` | `4096` | Rays per iteration. |
| `nerf_lr`, `nerf_lr_decay`, `nerf_lr_decay_steps` | `5e-4`, `0.2`, `0` | `lr_decay_steps = 0` means "use `num_iters`". Setting it to the *total planned* length keeps the decay continuous across resumes. |
| `nerf_rgb_activation` | `"exp"` | `"exp"` (HDR) or `"softplus"`. **Checkpoints are not compatible between the two.** |
| `nerf_loss_type` | `"rel_mse_raw"` | `"l1"`, `"mse"`, `"rel_mse"`, `"rel_mse_raw"`, `"log_l1"`. |
| `nerf_depth_window`, `_end`, `_samples` | `0.5`, `0.5`, `32` | Depth-guided sampling window around the OptiX `t_hit`. |
| `nerf_bg_radius_mult` | `6.0` | Background-sphere radius = this × the **maximum distance from the world origin** of the foreground hit points. Must be > 1; a warning is printed if the shell does not enclose every camera. |
| `nerf_raw_noise_std` | `0.0` | Pre-ReLU noise on the density. **Training only** — it is never applied at inference (see §7). |
| `enable_nerf_render_train_images` | `False` | Step 2b: re-render every training frame with the trained model. Required if you want `color_texture_image_sources = ["nerf"]`. |
| `nerf_interactive_loop` | `True` | Ask at the end of each round whether to keep training. Set `False` for unattended runs. |

### 4.4 Two things that bite

**`run_pipeline_multi` injects the scene paths.** In the `template` `RenderConfig`,
`transforms_path`, `model_path`, `external_normal_path` and `output_dir` are left empty
on purpose — they are overwritten from each `SceneConfig`. Filling them in the template
has no effect.

**Overrides in the loop win.** The current `__main__` sets `nerf_num_iters = 50000` in
the template and then `cfg.nerf_num_iters = 75000` inside the sweep loop. The loop is
what actually runs. The same holds for `nerf_lr_decay_steps`, `nerf_rgb_activation`,
`nerf_loss_type`, `nerf_lr_decay` and `render.skybox_source`.

---

## 5. Running

```bash
conda activate tesi-nerf
cd NeRF_Pytorch
python images_generator.py
```

For a long background run on Windows, set the console encoding first, otherwise the
progress bars and the `✓` markers will raise:

```bash
PYTHONIOENCODING=utf-8 python images_generator.py
```

**The exit code is unreliable** (OptiX exits non-zero at interpreter shutdown). Judge the
outcome from the final `run_pipeline_multi summary:` block, and from `console.log` inside
each scene folder.

### Output layout

```
<output_root>/<scene name>/
├── transforms_extended.json     Step 1: cameras + per-frame layer paths
├── images/                      copies of the source images
├── depth/  position/  normal/  mask/          Step 1 per-frame layers
├── model/nerf_model_cache.pt    Step 2: the checkpoint
├── nerf_train/                  Step 2: training_metrics.csv, epoch_metrics.csv, previews
├── nerf_render_images/iter_*/   Step 2b: per-frame pred/gt EXRs + metrics
├── ium/                         Step 3: inverse UV mapping (positions, normals, mask)
├── visibility/                  Step 3: per-texel, per-camera visibility (occlusion)
├── camera_mask/                 Step 3: per-camera masks, occlusion + frustum + grazing
├── irradiance/                  Step 3: skybox irradiance, and irradiance_indirect.exr
├── spec_cone/                   Step 3: cam_000.exr … + spec_cone_meta.json
├── skybox_nerf_baked.exr        Step 3: envmap baked from the NeRF (skybox_source="nerf")
├── sources/<gt|nerf>/
│   ├── color_texture/  camera_texture/  pixel_change/
│   ├── albedo/                  Step 4: Lambertian albedo
│   ├── metallic/  roughness/  albedo_pbr/     Step 4: the PBR maps
│   └── pbr/                     Step 4: diagnostics (x, α, residual, n_views, previews)
├── roi/<tag>/                   ROI sandboxes, mirroring the layout above
├── run_manifest.json            the full resolved configuration of the run
└── console.log
```

The visibility kernel only answers "is this texel occluded from this camera?"; the
frustum test and the grazing-angle filter are applied later, when the colour texture is
baked. `camera_mask/` therefore holds the *authoritative* per-camera masks, and it
overwrites `visibility.exr` — read `camera_mask/`, not `visibility/`, if you want to know
which cameras a texel was actually fitted from.

`metallic.exr` and `roughness.exr` are single-channel (`Z`). Because Blender's baker
expects the value replicated over R/G/B, `metallic_rgb.exr` and `roughness_rgb.exr` are
written next to them (disable with `pbr_write_blender_rgb=False`). The single-channel
files stay the input of every internal reader.

**`roughness` is a cone-width index, not a GGX α.** It is the winning cone's aperture
divided by 180, quantised to the `spec_cone_apertures_deg` grid (mirror → 0). Exporting
it as a Disney/Principled roughness would need an aperture→α calibration that this
pipeline does not do. Where the fit is quasi-diffuse the lobe is ill-conditioned; the
`pbr/r_valid.png` diagnostic marks where it is trustworthy.

---

## 6. Iterating cheaply

Steps 1–3 are the expensive ones. Two ways to avoid paying for them twice.

### 6.1 Re-fit only

```python
run_step1 = False
run_step2 = False
run_step3 = False
run_step4 = True
```

Step 4 touches neither OptiX nor the NeRF checkpoint — it reads the Step 3 cache from
disk. This is how you iterate on the solver. Equivalently, standalone:

```bash
python pbr_solver.py "D:/tesi_output/<sweep>/<tag>/<scene>" --source gt --spec-threshold 0
```

> **Pass `--spec-threshold` explicitly.** The CLI defaults it to `0.2`, whereas the
> pipeline's `__main__` sets `pbr_spec_threshold = 0.0`. Running the solver standalone
> without the flag therefore rewrites `roughness.exr` with a *different* gate than the
> one your run was produced with — every other map is threshold-independent and comes
> out bit-identical, so the discrepancy is easy to miss. The same applies to
> `--cv-gate`, whose neutral value is `0`.

### 6.2 Restrict to a region of the texture

A ROI restricts Steps 3 and 4 to a rectangle (and/or a mask) in IUM texel space, turning
an hours-long bake into minutes. It is **not an approximation**: the ROI is applied once,
as an extra factor on the IUM mask, and every downstream kernel already returns early on
masked texels, so the texels inside get bit-identical values to a full run. Everything
lands in a sandbox `<output_dir>/roi/<tag>/` — the full-resolution caches are never
touched.

```bash
python scripts/roi_rerun.py <run_dir> --rect 3623 2712 473 473 --tag my_test
python scripts/compare_roi_run.py <run_dir> --tag my_test
```

`roi_rerun.py` rebuilds the configuration from that run's `run_manifest.json` and asserts
it matches on every key except the steps, the output dir and the ROI, so the comparison
stays valid. `compare_roi_run.py` then checks the sandbox map by map against the full run
and verifies everything outside the ROI is exactly zero.

---

## 7. Gotchas worth knowing

**Camera rays are metric.** `rays_d` is a unit vector, and every `t_hit` is a Euclidean
world-space distance from an OptiX depth map — not a camera-axis Z depth. Directions and
distances must share one parametrisation; feeding a metric distance to a non-unit
direction places the sample at `depth·|rays_d|`, an error that is exactly zero at the
principal point and grows towards the corners. Checkpoints written before this was fixed
are **rejected on load** by `nerf/checkpoint.py`, on both the resume and the inference
path.

**`raw_noise_std` is training-only.** It used to leak into every inference path, which
made the indirect irradiance, the cones, the PBR maps and `skybox_nerf_baked.exr`
irreproducible (two runs on identical inputs agreed on ~52 % of values). Since
2026-08-12 the noise is opt-in and only the training forward pass passes it. Bakes made
before that date carry a systematic bias, not just noise, and are **not comparable** with
later ones.

**Old spec-cone bakes are unreadable.** Only the `"cones"` format is accepted; the older
`"rings"` / `"rings_shared"` formats (per-ring means) have to be re-baked. The solver
raises rather than guessing.

**The NeRF background sphere is anchored at the world origin.** `bg_radius_mult` is
relative to `max |p|` measured from the origin, not to the AABB side — the same numeric
value gives a smaller sphere than it did under the old convention.

**Result objects from C++ expose zero-copy views.** The `*_np` NumPy arrays returned by
the OptiX generators are views into C++ memory; the Python caller must keep the result
object alive for as long as it uses them.

---

## 8. Additional scripts

Everything in `figures/` and `scripts/` is **outside the pipeline**: one-off tools, tests
and figure generators. They are not imported by `images_generator.py`. Each folder has a
tiny `_paths.py` that puts the repository root on `sys.path`, so they can be run directly
from anywhere:

```bash
python figures/<name>.py ...
python scripts/<name>.py ...
```

### `figures/` — thesis figures

| Script | What it draws | Example |
|---|---|---|
| `compare_runs.py` | Full comparison of the runs of a sweep: metric matrix, per-band error budget, value spectrum, markdown report. | `python figures/compare_runs.py <sweep_root>` |
| `compare_training_curves.py` | Training curves of two sweeps differing by one hyper-parameter. | `python figures/compare_training_curves.py <ref_root> <new_root>` |
| `compare_exr.py` | Pixel-by-pixel comparison of two sets of EXR renders. | `python figures/compare_exr.py --original A --computed B` |
| `make_pipeline_diagram.py` | Block diagram of the pipeline. | `python figures/make_pipeline_diagram.py` |
| `make_geometry_diagrams.py` | Geometric diagrams of the Implementation chapter. | `python figures/make_geometry_diagrams.py` |
| `make_cone_diagram.py` | Figures and worked example for the cone equation. | `python figures/make_cone_diagram.py` |
| `make_pbr_model_diagram.py` | Figures for the PBR model equation. | `python figures/make_pbr_model_diagram.py` |
| `make_pbr_diagram.py` | Illustrative diagram of the multi-view cone approach. | `python figures/make_pbr_diagram.py` |
| `make_pbr_fit_figures.py` | Worked example of the per-texel PBR fit. | `python figures/make_pbr_fit_figures.py` |
| `make_nerf_sampling_figure.py` | The two NeRF sampling strategies. | `python figures/make_nerf_sampling_figure.py` |
| `make_depth_figure.py` | The geometric layers of a run, as PNGs. | `python figures/make_depth_figure.py <run_dir>` |
| `make_ium_normal_figure.py` | Recovering the geometric normal in the IUM pass. | `python figures/make_ium_normal_figure.py <run_dir>` |
| `make_uv_unwrap_figure.py` | The UV unwrap panels. | `python figures/make_uv_unwrap_figure.py <run_dir>` |
| `make_visibility_figure.py` | Texture-space visibility / irradiance / pixel-change maps. | `python figures/make_visibility_figure.py <run_dir>` |
| `make_atlas_pngs.py` | Converts the channels of a Blender bake into PNGs. | `python figures/make_atlas_pngs.py <bake_dir>` |
| `make_scenes_figure.py` | Scene overview figures. | `python figures/make_scenes_figure.py` |
| `make_lighting_figure.py` | The two lighting conditions of the Results chapter. | `python figures/make_lighting_figure.py` |
| `make_skybox_figure.py` | The skyboxes baked from a sweep's NeRFs. | `python figures/make_skybox_figure.py <sweep_root>` |
| `make_results_figures.py` | The Results-chapter panels built from the runs. `mode` is one of `maps`, `views`, `highfreq`, `curves`, `grids`, `mapdiff`, `spectrum`, `all`. | `python figures/make_results_figures.py maps --out <dir>` |

### `scripts/` — reruns, inspection, tests

| Script | What it does | Example |
|---|---|---|
| `roi_rerun.py` | Re-runs Steps 3+4 of an existing run restricted to a ROI, rebuilding the config from `run_manifest.json`. | `python scripts/roi_rerun.py <run_dir> --rect X0 Y0 W H --tag NAME` |
| `compare_roi_run.py` | Compares a ROI sandbox against the full run, map by map and band by band. | `python scripts/compare_roi_run.py <run_dir> --tag NAME` |
| `rerun_irradiance.py` | Regenerates the irradiance and Step 4 over an existing run tree (work on a copy). | `python scripts/rerun_irradiance.py <root> --verify` |
| `retrain_from_manifest.py` | Retrains the scenes of a sweep into a new folder, changing a few parameters. | `python scripts/retrain_from_manifest.py <src_root> <dst_root> --iters N` |
| `bake_skyboxes.py` | Bakes the NeRF skybox for every run of a sweep. | `python scripts/bake_skyboxes.py <root> --gt <hdr.exr>` |
| `rerender_run.py` | Re-renders a finished run in Blender using its reconstructed textures. | `python scripts/rerender_run.py <run_dir>` |
| `blender_renderer.py` | Standalone Blender rendering helper: renders a model from every camera of a `transforms.json`. Runs **inside Blender's Python**, not the conda env. | `blender --background --python scripts/blender_renderer.py -- --model M.obj --transforms t.json --output-dir out/` |
| `nerf_viewer.py` | Interactive novel-view viewer for a trained NeRF (OpenCV window, orbit camera, live checkpoint reload). | `python scripts/nerf_viewer.py --ckpt <model.pt> --transforms t.json --obj m.obj` |
| `inspect_final_maps.py` | Summary figure of the final metallic/roughness maps. | `python scripts/inspect_final_maps.py <run_dir> [source]` |
| `inspect_spec_cone.py` | Browses the baked cones: contact sheets per camera, optional full-res unpack. | `python scripts/inspect_spec_cone.py <output_dir> --cams 0 1 2` |
| `regen_heatmaps.py` | Regenerates the PNG heatmaps from the EXRs already on disk. | `python scripts/regen_heatmaps.py <run_dir> [gt_skybox.exr]` |
| `pbr_toy_solver.py` | Numerical toy that checks the identifiability of the multi-view PBR system. | `python scripts/pbr_toy_solver.py` |
| `test_spec_cone_smoke.py` | Envmap-only smoke test of the SpecCone pass (needs OptiX, no NeRF). | `python scripts/test_spec_cone_smoke.py` |
| `test_hemivis_shared.py` | Envmap-only test of the shared-ray scheme, including kernel↔torch direction parity (needs OptiX, no NeRF). | `python scripts/test_hemivis_shared.py` |

Two more tools live at the top level because the pipeline imports them:
`exr_to_blender_rgb.py` (converts single-channel EXRs to the R/G/B convention Blender
expects — also usable standalone on a file or a directory) and `monitoring.py` (GPU/CPU
sampling during long runs).

### Other folders

| Folder | Content |
|---|---|
| `nerf/` | The NeRF implementation: `config`, `dataset`, `encoding`, `model`, `rays`, `render`, `train`, `metrics`, `csv_logger`, `checkpoint`. |
| `blender/` | `bake_unified_material.py` — run **inside Blender** (Text Editor → Run Script, or install as an add-on). Joins the scene meshes, bakes every material into one PBR texture set and assembles a new "Baked" scene. |
| `legacy/` | Archived code, kept for reference only. Nothing here is imported by an active file. See `legacy/README.md`. |
| `assets/` | A small self-contained sample scene (`SwordShield`). |

---

## 9. Where to look first

If you have a run on disk and want to see what this produces without building anything:

```bash
conda activate tesi-nerf
python scripts/inspect_spec_cone.py <run_dir>             # the baked specular cones
python figures/make_results_figures.py maps --out out/    # the reconstructed PBR maps
python pbr_solver.py <run_dir> --source gt                 # re-run the fit itself
```
