"""
diagnose_rays.py
----------------
Diagnostic script: loads NerfDataset, computes 3D query points for the test
frame, checks whether they land on the mesh surface (KD-tree distance to OBJ
vertices), and saves a matplotlib 3D scatter for visual inspection — no
external tools required.

Usage:
    python diagnose_rays.py <transforms_extended.json> [<scene.obj>]

Example:
    python diagnose_rays.py output/sworshield_render_nerf_2/transforms_extended.json \\
        ../OptixProjectCMake/Scenes/SwordShield/Models/sworshield.obj
"""
from __future__ import annotations
import sys
import json
import struct
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nerf_module import NerfDataset, get_ray_bundle


def write_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Save (N,3) float32 points (+ optional (N,3) uint8 colors) as ASCII PLY."""
    N = len(points)
    has_color = colors is not None
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            row = f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f}"
            if has_color:
                row += f" {colors[i,0]} {colors[i,1]} {colors[i,2]}"
            f.write(row + "\n")
    print(f"  PLY → {path}  ({N} points)")


def write_edges_ply(path: str, origins: np.ndarray, ends: np.ndarray,
                    colors: np.ndarray | None = None) -> None:
    """Save N ray segments as an edge PLY (2N vertices + N edges).

    origins, ends: (N,3) float32.  colors: (N,3) uint8 (applied to both endpoints).
    MeshLab: Render → Show → Edges to display lines.
    """
    N = len(origins)
    has_color = colors is not None
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {2 * N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {N}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")
        for i in range(N):
            c = f" {colors[i,0]} {colors[i,1]} {colors[i,2]}" if has_color else ""
            f.write(f"{origins[i,0]:.6f} {origins[i,1]:.6f} {origins[i,2]:.6f}{c}\n")
            f.write(f"{ends[i,0]:.6f} {ends[i,1]:.6f} {ends[i,2]:.6f}{c}\n")
        for i in range(N):
            f.write(f"{2*i} {2*i+1}\n")
    print(f"  Edge PLY → {path}  ({N} segments)")


def load_obj_vertices(obj_path: str) -> np.ndarray:
    """Parse only 'v x y z' lines from an OBJ; returns (N,3) float32."""
    verts = []
    with open(obj_path, "r", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float32)


def kd_distance_stats(query_pts: np.ndarray, ref_pts: np.ndarray,
                      sample: int = 20_000) -> dict:
    """Mean / p50 / p95 distance from query_pts to nearest vertex in ref_pts."""
    from scipy.spatial import cKDTree
    if len(query_pts) > sample:
        idx = np.random.choice(len(query_pts), sample, replace=False)
        q = query_pts[idx]
    else:
        q = query_pts
    tree = cKDTree(ref_pts)
    dists, _ = tree.query(q, k=1)
    return {"mean": float(dists.mean()), "p50": float(np.percentile(dists, 50)),
            "p95": float(np.percentile(dists, 95)), "max": float(dists.max())}


def save_scatter3d(fg_q: np.ndarray, fg_q_flip: np.ndarray,
                   cam_pos: np.ndarray, mesh_verts: np.ndarray | None,
                   out_path: str, title: str = "") -> None:
    """Save a matplotlib 3D scatter: normal (blue) + flipped (red) query pts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    sub = 3000  # max points per cloud for speed
    rng = np.random.default_rng(0)

    def _pick(pts):
        if len(pts) > sub:
            return pts[rng.choice(len(pts), sub, replace=False)]
        return pts

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection='3d')

    if mesh_verts is not None and len(mesh_verts) > 0:
        mv = _pick(mesh_verts)
        ax.scatter(mv[:, 0], mv[:, 1], mv[:, 2], s=1, c='lightgrey',
                   alpha=0.3, label='mesh verts')

    fq = _pick(fg_q)
    ax.scatter(fq[:, 0], fq[:, 1], fq[:, 2], s=4, c='royalblue',
               alpha=0.7, label='query pts (normal)')

    fqf = _pick(fg_q_flip)
    ax.scatter(fqf[:, 0], fqf[:, 1], fqf[:, 2], s=4, c='tomato',
               alpha=0.7, label='query pts (flipped dirs)')

    ax.scatter(*cam_pos[0], s=80, c='gold', marker='*', label='camera')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(loc='upper left', fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  3D scatter → {out_path}")


def main(json_path: str, obj_path: str | None = None) -> None:
    json_path = Path(json_path).resolve()
    print(f"Loading dataset: {json_path}")

    device = torch.device("cpu")
    dataset = NerfDataset(str(json_path), device=device)

    H, W, focal = dataset.H, dataset.W, dataset.focal
    print(f"\n=== Scene info ===")
    print(f"  H={H}  W={W}  H==W? {H==W}")
    print(f"  fl_x (focal) = {focal:.4f}")

    # Read fl_y from JSON if present
    with open(json_path) as fh:
        raw = json.load(fh)
    fl_y = raw.get("fl_y", None)
    scale = raw.get("scale", 1.0)
    print(f"  fl_y = {fl_y}  (None → same as fl_x)")
    print(f"  scene scale = {scale}")
    if fl_y is not None and abs(fl_y - focal) > 0.1:
        print(f"  *** MISMATCH fl_x={focal:.2f}  fl_y={fl_y:.2f} — "
              f"NeRF uses fl_x for BOTH axes; vertical rays will be off by "
              f"{abs(fl_y/focal - 1)*100:.1f}%")

    # Test frame data
    _, test_pose, test_depth, _ = dataset.get_test_frame()
    test_idx = dataset.test_idx
    print(f"\n=== Test frame idx={test_idx} ===")

    c2w = test_pose.float()  # (4,4)
    origins, dirs = get_ray_bundle(H, W, torch.tensor(focal), c2w,
                                   focal_y=torch.tensor(dataset.focal_y))
    # origins, dirs: (H, W, 3)

    # ── Camera basis sanity print ────────────────────────────────────────────
    c2w_np = c2w.numpy()
    print(f"\n=== Camera c2w matrix ===")
    print(c2w_np)
    print(f"  col0 (right)     = {c2w_np[:3, 0]}")
    print(f"  col1 (up)        = {c2w_np[:3, 1]}")
    print(f"  col2 (-forward)  = {c2w_np[:3, 2]}")
    print(f"  col3 (position)  = {c2w_np[:3, 3]}")
    expected_center_dir = -c2w_np[:3, 2]  # should match dirs at center pixel
    ch, cw = H // 2, W // 2
    actual_center_dir = dirs[ch, cw].numpy()
    print(f"\n  Expected center-pixel dir (-col2) = {expected_center_dir}")
    print(f"  Actual   center-pixel dir          = {actual_center_dir}")
    dot = float(np.dot(expected_center_dir / (np.linalg.norm(expected_center_dir) + 1e-8),
                       actual_center_dir  / (np.linalg.norm(actual_center_dir)  + 1e-8)))
    print(f"  Cosine similarity = {dot:.4f}  "
          f"(+1 = same dir, -1 = OPPOSITE ← bug here, 0 = perpendicular)")

    depth_flat = test_depth.float().reshape(-1)     # (H*W,)
    origins_flat = origins.reshape(-1, 3).numpy()
    dirs_flat    = dirs.reshape(-1, 3).numpy()
    depth_np     = depth_flat.numpy()

    # Foreground mask: depth > 0 and < 1e10
    fg_mask = (depth_np > 0) & (depth_np < 1e10)
    bg_mask = ~fg_mask
    n_fg = fg_mask.sum()
    n_bg = bg_mask.sum()
    n_sentinel = (depth_np > 1e10).sum()

    print(f"  Foreground pixels (depth>0 & <1e10): {n_fg}/{H*W} "
          f"({100*n_fg/(H*W):.1f}%)")
    print(f"  Background zeros (depth==0): {(depth_np == 0).sum()}")
    print(f"  Miss sentinel (depth>1e10): {n_sentinel} "
          f"← these should be 0 after Fix A")
    if n_fg > 0:
        fg_d = depth_np[fg_mask]
        print(f"  FG depth  min={fg_d.min():.4f}  "
              f"p25={np.percentile(fg_d,25):.4f}  "
              f"p50={np.percentile(fg_d,50):.4f}  "
              f"p75={np.percentile(fg_d,75):.4f}  "
              f"max={fg_d.max():.4f}")

    # Compute query points: origins + dirs * (depth / ||dirs||)
    ray_norms = np.linalg.norm(dirs_flat, axis=-1, keepdims=True).clip(1e-8)
    depth_t = depth_np / ray_norms[:, 0]   # parametric t
    query_pts = origins_flat + dirs_flat * depth_t[:, None]

    fg_q       = query_pts[fg_mask]
    fg_depths  = depth_np[fg_mask]
    # Flipped version: shoot dirs in the opposite direction
    fg_q_flipped = origins_flat[fg_mask] - dirs_flat[fg_mask] * depth_t[fg_mask, np.newaxis]

    print(f"\n=== FG query points 3D bounds ===")
    if n_fg > 0:
        print(f"  X: [{fg_q[:,0].min():.3f}, {fg_q[:,0].max():.3f}]")
        print(f"  Y: [{fg_q[:,1].min():.3f}, {fg_q[:,1].max():.3f}]")
        print(f"  Z: [{fg_q[:,2].min():.3f}, {fg_q[:,2].max():.3f}]")
        print("  (Compare these bounds against the scene mesh in MeshLab/Blender)")
    else:
        print("  No foreground pixels found! Check depth EXR and mask.")

    # Color query points by depth (blue=near, red=far), capped at 99th percentile
    out_dir = json_path.parent / "debug"
    out_dir.mkdir(exist_ok=True)

    # ── Depth-value sanity check ─────────────────────────────────────────────
    print(f"\n=== Depth raw values (before fg filter) ===")
    print(f"  All pixels: min={depth_np.min():.4f}  max={depth_np.max():.4f}")
    zero_count = (depth_np == 0.0).sum()
    large_count = (depth_np > 1e10).sum()
    print(f"  Zero  values: {zero_count}  (background / no-hit → OK to have many)")
    print(f"  >1e10 values: {large_count}  (miss sentinel NOT yet clamped — expected 0)")

    if n_fg > 0:
        cap = float(np.percentile(fg_depths, 99))
        t = np.clip(fg_depths / max(cap, 1e-8), 0, 1)  # 0=near, 1=far
        colors = np.stack([
            (t * 255).astype(np.uint8),
            np.zeros(n_fg, dtype=np.uint8),
            ((1 - t) * 255).astype(np.uint8),
        ], axis=-1)
        # Downsample for large images
        if n_fg > 200_000:
            idx = np.random.choice(n_fg, 200_000, replace=False)
            write_ply(str(out_dir / "fg_query_points.ply"), fg_q[idx], colors[idx])
        else:
            write_ply(str(out_dir / "fg_query_points.ply"), fg_q, colors)
        print("  Color: blue=near, red=far")

        # Also export a flipped version (negate dirs) — if THIS PLY lands on
        # the mesh, the fix is to negate dirs in get_ray_bundle.
        if n_fg > 200_000:
            write_ply(str(out_dir / "fg_query_points_flipped.ply"),
                      fg_q_flipped[idx], colors[idx])
        else:
            write_ply(str(out_dir / "fg_query_points_flipped.ply"),
                      fg_q_flipped, colors)
        print("  → If 'flipped' PLY lands on mesh but normal does not:")
        print("    root cause = directions have wrong sign in get_ray_bundle.")

    # Also export camera position as single red point
    cam_pos = c2w[:3, 3].numpy().reshape(1, 3)
    write_ply(str(out_dir / "camera_pos.ply"), cam_pos,
              np.array([[255, 0, 0]], dtype=np.uint8))

    # ── Ray segments for MeshLab (2000 subsampled) ───────────────────────────
    SUBSAMPLE_RAYS = 2000
    if n_fg > 0:
        n_pick = min(SUBSAMPLE_RAYS, n_fg)
        rng_sub = np.random.default_rng(0)
        sub_idx = rng_sub.choice(n_fg, n_pick, replace=False)

        cam_origin  = c2w[:3, 3].numpy()
        origins_sub = np.tile(cam_origin, (n_pick, 1))
        ends_normal  = fg_q[sub_idx]
        ends_flipped = fg_q_flipped[sub_idx]

        sub_depths = fg_depths[sub_idx]
        cap_sub = float(np.percentile(fg_depths, 99))
        t_sub = np.clip(sub_depths / max(cap_sub, 1e-8), 0, 1)
        cols_normal = np.stack([
            (t_sub * 255).astype(np.uint8),
            np.zeros(n_pick, dtype=np.uint8),
            ((1 - t_sub) * 255).astype(np.uint8),
        ], axis=-1)
        cols_flipped = np.tile(np.array([255, 80, 80], dtype=np.uint8),
                               (n_pick, 1))

        write_edges_ply(str(out_dir / "rays_normal.ply"),
                        origins_sub, ends_normal, cols_normal)
        write_edges_ply(str(out_dir / "rays_flipped.ply"),
                        origins_sub, ends_flipped, cols_flipped)
        print("\n  MeshLab: File → Import Mesh → seleziona tutti i PLY in debug/")
        print("  Render → Show → Edges per visualizzare i raggi come linee.")

    # ── OBJ-based surface check ──────────────────────────────────────────────
    mesh_verts = None
    if obj_path is not None:
        obj_p = Path(obj_path)
        if obj_p.exists():
            print(f"\n=== Mesh surface check (OBJ: {obj_p.name}) ===")
            mesh_verts = load_obj_vertices(str(obj_p))
            print(f"  OBJ vertices loaded: {len(mesh_verts)}")
            grey = np.full((len(mesh_verts), 3), 180, dtype=np.uint8)
            write_ply(str(out_dir / "mesh_verts.ply"), mesh_verts, grey)
            print(f"  Mesh AABB  X:[{mesh_verts[:,0].min():.3f}, {mesh_verts[:,0].max():.3f}]"
                  f"  Y:[{mesh_verts[:,1].min():.3f}, {mesh_verts[:,1].max():.3f}]"
                  f"  Z:[{mesh_verts[:,2].min():.3f}, {mesh_verts[:,2].max():.3f}]")
            if n_fg > 0:
                stats_n = kd_distance_stats(fg_q, mesh_verts)
                stats_f = kd_distance_stats(fg_q_flipped, mesh_verts)
                print(f"\n  Distance  query→mesh (NORMAL dirs):   "
                      f"mean={stats_n['mean']:.4f}  p50={stats_n['p50']:.4f}  "
                      f"p95={stats_n['p95']:.4f}  max={stats_n['max']:.4f}")
                print(f"  Distance  query→mesh (FLIPPED dirs):  "
                      f"mean={stats_f['mean']:.4f}  p50={stats_f['p50']:.4f}  "
                      f"p95={stats_f['p95']:.4f}  max={stats_f['max']:.4f}")
                which = "NORMAL" if stats_n['mean'] < stats_f['mean'] else "FLIPPED"
                print(f"\n  *** {which} dirs place query points closer to the mesh surface ***")
                if which == "FLIPPED":
                    print("      → fix: negate ray directions in get_ray_bundle / training loop")
                else:
                    print("      → ray directions look correct; if color is still wrong,")
                    print("        check depth_window, color-space, or network capacity.")
        else:
            print(f"\n  OBJ not found at {obj_p} — skipping surface distance check.")

    # ── 3D scatter (no MeshLab needed) ──────────────────────────────────────
    if n_fg > 0:
        scatter_title = (f"Blue=normal dirs, Red=flipped dirs, Gold★=camera\n"
                         f"cosine={dot:.3f}  |  "
                         f"FG depth p50={np.percentile(fg_depths,50):.3f}")
        save_scatter3d(fg_q, fg_q_flipped, cam_pos, mesh_verts,
                       str(out_dir / "scatter3d.png"), title=scatter_title)
        print(f"\n  Open {out_dir / 'scatter3d.png'} to visually inspect:")
        print("  Blue points should cluster ON/NEAR the mesh (grey cloud).")
        print("  If Red points are closer → directions are flipped.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("ERROR: provide path to transforms_extended.json")
        sys.exit(1)
    main(sys.argv[1], obj_path=sys.argv[2] if len(sys.argv) >= 3 else None)
