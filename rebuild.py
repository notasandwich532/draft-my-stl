#!/home/nicholas/.pyenv/versions/3.10.14/envs/mesh-env/bin/python
"""
rebuild.py — Poisson surface reconstruction using open3d.

Settings are read from settings.toml [reconstruct] section.

Usage:
    ./rebuild.py                         # uses settings.toml
    ./rebuild.py input.stl output.stl    # override paths
    ./rebuild.py input.stl output.stl settings.toml
"""

import sys
import os

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("[ERROR] Install tomli: pip install tomli")
        sys.exit(1)

import warnings
warnings.filterwarnings("ignore")  # suppress numpy longdouble warnings from open3d venv

import open3d as o3d
import numpy as np
import trimesh

# ── Args ──────────────────────────────────────────────────────────────────────
args = sys.argv[1:]
if len(args) == 0:
    settings_path = "settings.toml"
    input_path    = None
    output_path   = None
elif len(args) == 1:
    settings_path = args[0]
    input_path    = None
    output_path   = None
else:
    input_path    = args[0]
    output_path   = args[1]
    settings_path = args[2] if len(args) > 2 else "settings.toml"

# ── Settings ──────────────────────────────────────────────────────────────────
with open(settings_path, "rb") as f:
    cfg = tomllib.load(f)

rc = cfg.get("reconstruct", {})
POINT_COUNT      = int(rc.get("point_count",       150_000))
POISSON_DEPTH    = int(rc.get("poisson_depth",     9))
DENSITY_TRIM     = float(rc.get("density_trim",    0.0))
FA_ANGLE         = float(rc.get("fa_laplacian_angle", 45.0))
FA_ITERS         = int(rc.get("fa_laplacian_iters",   30))
BOUNDARY_ITERS   = int(rc.get("boundary_lap_iters",   5))
PIN_BOTTOM_THRESH = float(rc.get("pin_bottom_z_thresh", 0.5))

if input_path is None:
    input_path  = cfg["mesh"].get("reconstruct_input",  cfg["mesh"]["output_file"])
    output_path = cfg["mesh"].get("reconstruct_output", "rebuilt.stl")

print(f"[INFO] Input:  {input_path}")
print(f"[INFO] Output: {output_path}")
print(f"[INFO] depth={POISSON_DEPTH}  points={POINT_COUNT:,}  density_trim={DENSITY_TRIM}"
      f"  fa_angle={FA_ANGLE}  fa_iters={FA_ITERS}  boundary_lap={BOUNDARY_ITERS}")

# ── Load ──────────────────────────────────────────────────────────────────────
print(f"\n[INFO] Loading mesh...")
mesh = o3d.io.read_triangle_mesh(input_path)

if not mesh.has_triangles():
    print("[ERROR] Invalid mesh or no triangles found.")
    sys.exit(1)

print(f"[INFO] Triangles={len(mesh.triangles):,}  Vertices={len(mesh.vertices):,}")

# ── Normals ───────────────────────────────────────────────────────────────────
print("[INFO] Computing vertex normals...")
mesh.compute_vertex_normals()

# ── Sample point cloud ────────────────────────────────────────────────────────
print(f"[INFO] Sampling {POINT_COUNT:,} points (Poisson disk)...")
pcd = mesh.sample_points_poisson_disk(POINT_COUNT)

print("[INFO] Estimating point cloud normals...")
pcd.estimate_normals()

# ── Poisson reconstruction ────────────────────────────────────────────────────
print(f"[INFO] Running Poisson reconstruction (depth={POISSON_DEPTH})...")
mesh_rec, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=POISSON_DEPTH
)
densities = np.asarray(densities)
print(f"[INFO] Reconstructed: Triangles={len(mesh_rec.triangles):,}  Vertices={len(mesh_rec.vertices):,}")

# ── Density trim ──────────────────────────────────────────────────────────────
if DENSITY_TRIM > 0:
    print(f"[INFO] Trimming lowest {DENSITY_TRIM*100:.1f}% density vertices...")
    threshold = np.quantile(densities, DENSITY_TRIM)
    mesh_rec.remove_vertices_by_mask(densities < threshold)
    print(f"[INFO] After trim: Triangles={len(mesh_rec.triangles):,}")

# ── Smoothing ─────────────────────────────────────────────────────────────────
# Convert to trimesh for feature-adaptive + boundary-aware Laplacian smoothing
tm = trimesh.Trimesh(
    vertices=np.asarray(mesh_rec.vertices),
    faces=np.asarray(mesh_rec.triangles),
    process=False,
)

# Pin vertices near the bottom of the mesh so smoothing doesn't round the base edge.
# Computed once here and reused in both passes.
z_min   = tm.vertices[:, 2].min()
pinned  = tm.vertices[:, 2] < z_min + PIN_BOTTOM_THRESH
n_pinned = int(pinned.sum())
if n_pinned:
    print(f"[INFO] Pinning {n_pinned:,} bottom vertices (z < {z_min + PIN_BOTTOM_THRESH:.3f})")

if FA_ITERS > 0:
    print(f"[INFO] Feature-adaptive Laplacian ({FA_ITERS} iters, {FA_ANGLE}deg threshold)...")
    sharp_cos = np.cos(np.radians(FA_ANGLE))
    adj_faces = tm.face_adjacency
    adj_edges = tm.face_adjacency_edges
    for _ in range(FA_ITERS):
        fnormals = tm.face_normals
        n0  = fnormals[adj_faces[:, 0]]
        n1  = fnormals[adj_faces[:, 1]]
        dot = (n0 * n1).sum(axis=1).clip(-1.0, 1.0)
        w   = ((dot - sharp_cos) / (1.0 - sharp_cos + 1e-8)).clip(0.0, 1.0)
        a, b  = adj_edges[:, 0], adj_edges[:, 1]
        verts = tm.vertices
        w_sum = np.ones(len(verts))
        w_pos = verts.copy()
        np.add.at(w_sum, a, w)
        np.add.at(w_sum, b, w)
        np.add.at(w_pos, a, w[:, None] * verts[b])
        np.add.at(w_pos, b, w[:, None] * verts[a])
        new_v = w_pos / w_sum[:, None]
        new_v[pinned] = verts[pinned]
        tm.vertices = new_v
    print(f"[INFO] After FA-Laplacian: Triangles={len(tm.faces):,}")

if BOUNDARY_ITERS > 0:
    print(f"[INFO] Boundary-aware Laplacian polish ({BOUNDARY_ITERS} iters)...")
    all_edges = set(map(tuple, np.sort(tm.edges_unique, axis=1)))
    int_edges = set(map(tuple, np.sort(tm.face_adjacency_edges, axis=1)))
    bnd_verts = set()
    for e in all_edges - int_edges:
        bnd_verts.add(e[0]); bnd_verts.add(e[1])
    boundary = np.zeros(len(tm.vertices), dtype=bool)
    if bnd_verts:
        boundary[list(bnd_verts)] = True
    boundary |= pinned  # also pin bottom vertices in the polish pass
    a, b = tm.edges_unique[:, 0], tm.edges_unique[:, 1]
    for _ in range(BOUNDARY_ITERS):
        verts  = tm.vertices
        n_sum  = np.zeros_like(verts)
        n_cnt  = np.zeros(len(verts))
        np.add.at(n_sum, a, verts[b])
        np.add.at(n_cnt, a, 1)
        np.add.at(n_sum, b, verts[a])
        np.add.at(n_cnt, b, 1)
        ok    = n_cnt > 0
        delta = np.zeros_like(verts)
        delta[ok] = n_sum[ok] / n_cnt[ok, None] - verts[ok]
        new_v = verts + 0.5 * delta
        new_v[boundary] = verts[boundary]
        tm.vertices = new_v

# ── Export ────────────────────────────────────────────────────────────────────
print(f"\n[INFO] Saving: {output_path}")
tm.export(output_path)
print(f"[DONE] Triangles={len(tm.faces):,}  Vertices={len(tm.vertices):,}")
