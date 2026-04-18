#!/usr/bin/env python3
# NOTE: rebuild.py requires open3d, which only supports Python <=3.11.
# If your system Python is newer, create a venv with an older version:
#   python3.10 -m venv venv-reconstruct
#   source venv-reconstruct/bin/activate
#   pip install open3d tomli
# Then update this shebang to point at that venv's python, e.g.:
#   #!/path/to/venv-reconstruct/bin/python
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

import open3d as o3d
import numpy as np

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
POINT_COUNT   = int(rc.get("point_count",         150_000))
POISSON_DEPTH = int(rc.get("poisson_depth",        9))
SMOOTHING     = int(rc.get("smoothing_iterations", 10))
DENSITY_TRIM  = float(rc.get("density_trim",       0.0))  # 0 = disabled

if input_path is None:
    input_path  = cfg["mesh"].get("reconstruct_input",  cfg["mesh"]["output_file"])
    output_path = cfg["mesh"].get("reconstruct_output", "rebuilt.stl")

print(f"[INFO] Input:  {input_path}")
print(f"[INFO] Output: {output_path}")
print(f"[INFO] depth={POISSON_DEPTH}  points={POINT_COUNT:,}  smooth={SMOOTHING}  density_trim={DENSITY_TRIM}")

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
if SMOOTHING > 0:
    print(f"[INFO] Taubin smoothing ({SMOOTHING} iterations)...")
    mesh_rec = mesh_rec.filter_smooth_taubin(number_of_iterations=SMOOTHING)

# ── Export ────────────────────────────────────────────────────────────────────
mesh_rec.compute_vertex_normals()
print(f"\n[INFO] Saving: {output_path}")
o3d.io.write_triangle_mesh(output_path, mesh_rec)
print(f"[DONE] Triangles={len(mesh_rec.triangles):,}  Vertices={len(mesh_rec.vertices):,}")
