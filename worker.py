#!/bin/python3
import json
import sys
import os
import time
import tomllib
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

# -----------------------------
# SETTINGS
# -----------------------------
INPUT_FILE  = sys.argv[1]
OUTPUT_FILE = sys.argv[2]
SETTINGS    = sys.argv[3] if len(sys.argv) > 3 else "settings.toml"

with open(SETTINGS, "rb") as f:
    cfg = tomllib.load(f)

VOXEL_SIZE      = float(cfg["voxel"]["voxel_size"])
DRAFT_ANGLE_DEG = float(cfg["voxel"]["draft_angle_deg"])

# OVERLAP and PAD can be passed from controller (auto-computed from mesh height)
# or fall back to settings.toml values
if len(sys.argv) > 5:
    OVERLAP = float(sys.argv[4])
    PAD     = int(sys.argv[5])
else:
    OVERLAP = float(cfg["tiling"].get("overlap", 5.0))
    PAD     = int(cfg["voxel"]["pad"])

# -----------------------------
# TIMER
# -----------------------------
_t = time.perf_counter()
def tick(label):
    global _t
    now = time.perf_counter()
    print(f"  [{now - _t:6.2f}s] {label}", flush=True)
    _t = now

print(f"\n── Processing {INPUT_FILE} ──", flush=True)
print(f"  PAD={PAD}  OVERLAP={OVERLAP:.2f}mm", flush=True)

# -----------------------------
# LOAD MESH
# -----------------------------
tick("start")
mesh = trimesh.load(INPUT_FILE)
tick("load mesh")

print(f"  Bounds: {mesh.bounds}")
print(f"  Size:   {mesh.extents}")
print(f"  Watertight input: {mesh.is_watertight}")

# -----------------------------
# LOAD Z OFFSET
# -----------------------------
tile_name   = os.path.basename(INPUT_FILE)
tile_id_str = tile_name.split("_")[1].split(".")[0]
offset_file = os.path.join(os.path.dirname(INPUT_FILE), "offsets.json")

z_offset = 0.0

if os.path.exists(offset_file):
    with open(offset_file, "r") as f:
        offsets = json.load(f)
    z_offset = float(offsets.get(tile_id_str, 0.0))
    print(f"  Z offset from JSON [{tile_id_str}]: {z_offset:.6f}")
else:
    print(f"  WARNING: offsets.json not found — using 0.0")

# -----------------------------
# APPLY OFFSET BEFORE VOXEL
# -----------------------------
if z_offset != 0.0:
    mesh.vertices[:, 2] += z_offset

# -----------------------------
# VOXELIZE
# -----------------------------
print("\n── Voxelizing ──")
tick("start")

vox    = mesh.voxelized(pitch=VOXEL_SIZE, method='ray')
volume = vox.matrix.astype(bool)
origin = np.array(vox.transform[:3, 3])

nx, ny, nz = volume.shape
print(f"  Grid: ({nx}, {ny}, {nz})")
tick("voxelized")

# -----------------------------
# EXTEND GRID DOWNWARD
# -----------------------------
extra_layers = int(abs(z_offset) / VOXEL_SIZE) + 10

if extra_layers > 0:
    volume = np.pad(volume, ((0,0),(0,0),(extra_layers,0)), mode='constant')
    origin[2] -= extra_layers * VOXEL_SIZE

# -----------------------------
# PAD
# -----------------------------
volume  = np.pad(volume, PAD, mode='constant')
origin -= PAD * VOXEL_SIZE

# -----------------------------
# CLEAR XY BOUNDARY BEFORE UNDERCUT/DRAFT
# -----------------------------
BORDER = PAD - 1
if BORDER > 0:
    volume[:BORDER,  :, :] = False
    volume[-BORDER:, :, :] = False
    volume[:,  :BORDER, :] = False
    volume[:, -BORDER:, :] = False

# -----------------------------
# UNDERCUT REMOVAL
# -----------------------------
rev    = volume[:, :, ::-1]
filled = np.maximum.accumulate(rev, axis=2)[:, :, ::-1]

# -----------------------------
# DRAFT
# -----------------------------
angle_rad        = np.deg2rad(DRAFT_ANGLE_DEG)
growth_per_layer = np.tan(angle_rad)

def dilate(a):
    """Dilate in XY without wrapping — np.roll wraps edges causing boundary artifacts."""
    out = a.copy()
    out[1:,  :] |= a[:-1, :]   # shift +X
    out[:-1, :] |= a[1:,  :]   # shift -X
    out[:, 1:]  |= a[:, :-1]   # shift +Y
    out[:, :-1] |= a[:, 1:]    # shift -Y
    out[1:,  1:]  |= a[:-1, :-1]  # shift +X+Y
    out[1:,  :-1] |= a[:-1, 1:]   # shift +X-Y
    out[:-1, 1:]  |= a[1:,  :-1]  # shift -X+Y
    out[:-1, :-1] |= a[1:,  1:]   # shift -X-Y
    return out

draft = filled.copy()
# Use full padded grid height, not original nz
full_nz = draft.shape[2]

# Start accumulation from the topmost solid Z slice rather than the very top
# of the padded grid (which is empty). This ensures the draft begins correctly
# at the actual model surface rather than wasting iterations on empty PAD space.
top_solid_z = full_nz - 1
while top_solid_z > 0 and not draft[:, :, top_solid_z].any():
    top_solid_z -= 1

accum        = draft[:, :, top_solid_z].copy()
growth_accum = 0.0

for z in reversed(range(top_solid_z)):
    growth_accum += growth_per_layer
    while growth_accum >= 1.0:
        accum = dilate(accum)
        growth_accum -= 1.0

    accum = accum | draft[:, :, z]
    draft[:, :, z] = accum

# -----------------------------
# SMOOTH
# -----------------------------
# size=2 uniform filter eats 1 voxel from every surface — use a lighter
# smooth that cleans noise without shrinking the top/sides of the model
smoothed = ndimage.uniform_filter(draft.astype(np.float32), size=1.5) > 0.5

# -----------------------------
# MARCHING CUBES
# -----------------------------
smoothed = np.pad(smoothed, 1)

try:
    verts, faces, _, _ = measure.marching_cubes(
        smoothed.astype(np.float32),
        level=0.5,
        allow_degenerate=False
    )
except RuntimeError:
    sys.exit(0)

verts -= 1
verts = verts * VOXEL_SIZE + origin

# -----------------------------
# UNDO OFFSET
# -----------------------------
if z_offset != 0.0:
    verts[:, 2] -= z_offset

# -----------------------------
# BUILD MESH
# -----------------------------
out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

# -----------------------------
# PLANE CUT
# Cut below the mesh's actual Z minimum so the plane never intersects
# real geometry — this preserves the floor face rather than discarding
# faces that lie exactly on Z=0.
# -----------------------------
try:
    z_cut = out.bounds[0][2] - VOXEL_SIZE * 0.5
    out = out.slice_plane(
        plane_origin=[0, 0, z_cut],
        plane_normal=[0, 0, 1]
    )
except:
    pass

# -----------------------------
# CLEANUP
# -----------------------------
out.remove_unreferenced_vertices()
out.merge_vertices()
out.update_faces(out.nondegenerate_faces())
out.fix_normals()

# -----------------------------
# DEBUG
# -----------------------------
print(f"\n── Debug ──")
print(f"  Faces: {len(out.faces)}")
print(f"  Watertight: {out.is_watertight}")

# -----------------------------
# EXPORT
# -----------------------------
tick("start")
out.export(OUTPUT_FILE)
tick("export")

print("Done\n")
