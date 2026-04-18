#!/usr/bin/env python3

import os
import sys
import tomllib
import subprocess
import numpy as np
import trimesh
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# -----------------------------
# SETTINGS
# -----------------------------
SETTINGS = sys.argv[1] if len(sys.argv) > 1 else "settings.toml"

if not os.path.exists(SETTINGS):
    print(f"ERROR: settings file not found: {SETTINGS}")
    sys.exit(1)

with open(SETTINGS, "rb") as f:
    cfg = tomllib.load(f)

INPUT_FILE       = cfg["mesh"]["input_file"]
TILE_SIZE        = float(cfg["tiling"]["tile_size"])
MAX_WORKERS      = int(cfg["tiling"]["max_workers"])
TILES_DIR        = cfg["paths"]["tiles_dir"]
PROCESSED_DIR    = cfg["paths"]["processed_dir"]
REDUCED_DIR      = cfg["paths"].get("reduced_dir", "reduced")
WORKER           = cfg["paths"]["worker_script"]
MERGE_SCRIPT     = cfg["paths"].get("merge_script", "./merge_coplanar.py")
VOXEL_SIZE       = float(cfg["voxel"]["voxel_size"])
DRAFT_ANGLE_DEG  = float(cfg["voxel"]["draft_angle_deg"])

WORKER_PATH = os.path.abspath(WORKER)

TILE_DEBUG = False  # set True to print per-plane clip info for each tile

os.makedirs(TILES_DIR,     exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REDUCED_DIR,   exist_ok=True)

# -----------------------------
# LOAD
# -----------------------------
print(f"Loading {INPUT_FILE}...")
mesh = trimesh.load(INPUT_FILE)
min_corner, max_corner = mesh.bounds
print(f"  Bounds: {mesh.bounds}")
print(f"  Size:   {mesh.extents}")

# -----------------------------
# AUTO OVERLAP + PAD
# -----------------------------
# The draft grows geometry outward by tan(angle) per unit of height.
# Overlap must cover the maximum possible horizontal draft growth so that
# adjacent tiles see each other's drafted geometry and boolean correctly.
# PAD must cover the same growth in voxel units plus marching cubes margin.
import math
mesh_height  = float(max_corner[2] - min_corner[2])
draft_rad    = math.radians(DRAFT_ANGLE_DEG)
draft_growth = mesh_height * math.tan(draft_rad)   # max XY growth from draft
OVERLAP      = math.ceil(draft_growth * 1.25)       # 25% safety margin, round up
PAD          = math.ceil(OVERLAP / VOXEL_SIZE) + 4  # voxels + marching cubes buffer

print(f"Draft angle : {DRAFT_ANGLE_DEG}°")
print(f"  Mesh height : {mesh_height:.2f} mm")
print(f"  Max draft growth : {draft_growth:.2f} mm")
print(f"  Auto overlap     : {OVERLAP:.2f} mm")
print(f"  Auto pad         : {PAD} voxels")

# -----------------------------
# PLANE CLIP (NEW — triangle splitting)
# -----------------------------
def clip_to_box(mesh, xmin, xmax, ymin, ymax, debug_label=""):
    m = mesh.copy()

    center = mesh.bounds.mean(axis=0)

    def dbg(stage, m):
        if not TILE_DEBUG: return
        if m is None: print(f"    [{debug_label}] {stage}: NONE")
        else:         print(f"    [{debug_label}] {stage}: faces={len(m.faces)}")
    if TILE_DEBUG:
        print(f"  TILE {debug_label} x[{xmin:.1f},{xmax:.1f}] y[{ymin:.1f},{ymax:.1f}]")

    try:
        m = trimesh.intersections.slice_mesh_plane(m, plane_normal=[1,0,0], plane_origin=[xmin, center[1], center[2]])
        dbg("xmin", m)
        if m is None or len(m.faces) == 0: return None

        m = trimesh.intersections.slice_mesh_plane(m, plane_normal=[-1,0,0], plane_origin=[xmax, center[1], center[2]])
        dbg("xmax", m)
        if m is None or len(m.faces) == 0: return None

        m = trimesh.intersections.slice_mesh_plane(m, plane_normal=[0,1,0], plane_origin=[center[0], ymin, center[2]])
        dbg("ymin", m)
        if m is None or len(m.faces) == 0: return None

        m = trimesh.intersections.slice_mesh_plane(m, plane_normal=[0,-1,0], plane_origin=[center[0], ymax, center[2]])
        dbg("ymax", m)
        if m is None or len(m.faces) == 0: return None

    except Exception as e:
        if TILE_DEBUG: print(f"    [{debug_label}] EXCEPTION: {e}")
        return None

    m.remove_unreferenced_vertices()
    m.merge_vertices()
    m.fix_normals()

    return m

# -----------------------------
# GENERATE TILES
# -----------------------------
print("\n------------------------------------------------------------\n  TILING\n------------------------------------------------------------")

xs_range = np.arange(min_corner[0], max_corner[0], TILE_SIZE)
ys_range = np.arange(min_corner[1], max_corner[1], TILE_SIZE)

tile_meta = []
tile_id   = 0
offsets   = {}

with tqdm(total=len(xs_range) * len(ys_range), desc="Cutting tiles") as pbar:
    for x0 in xs_range:
        for y0 in ys_range:
            pbar.update(1)

            x1 = min(x0 + TILE_SIZE, max_corner[0])
            y1 = min(y0 + TILE_SIZE, max_corner[1])

            tile = clip_to_box(
                mesh,
                x0 - OVERLAP, x1 + OVERLAP,
                y0 - OVERLAP, y1 + OVERLAP,
                debug_label=f"{tile_id}"
            )

            if tile is None:
                continue

            # -----------------------------
            # Z OFFSET DEBUG
            # -----------------------------
            original_z_min = mesh.bounds[0][2]
            tile_z_min     = tile.bounds[0][2]
            z_offset       = original_z_min - tile_z_min

            if TILE_DEBUG: print(f"  tile_{tile_id}  Z_offset={z_offset:.6f}")
            offsets[tile_id] = z_offset

            tile_file = os.path.join(TILES_DIR,     f"tile_{tile_id}.stl")
            out_file  = os.path.join(PROCESSED_DIR, f"tile_{tile_id}.stl")

            tile.export(tile_file)

            tile_meta.append({
                "tile_file": tile_file,
                "out_file":  out_file,
            })

            tile_id += 1

print(f"  Generated {len(tile_meta)} tiles  (overlap={OVERLAP:.1f}mm  pad={PAD}vx)")

# -----------------------------
# SAVE OFFSETS
# -----------------------------
offset_file = os.path.join(TILES_DIR, "offsets.json")

with open(offset_file, "w") as f:
    json.dump(offsets, f, indent=2)

print(f"Saved offsets → {offset_file}")

# -----------------------------
# PARALLEL WORKERS
# -----------------------------
def run_worker(tile):
    if os.path.exists(tile["out_file"]):
        return tile["out_file"], "skipped", None

    result = subprocess.run(
        ["python3", WORKER_PATH, tile["tile_file"], tile["out_file"], SETTINGS,
         str(OVERLAP), str(PAD)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return tile["out_file"], "failed", result.stderr.strip()

    return tile["out_file"], "ok", None

print(f"\n------------------------------------------------------------\n  DRAFTING  ({len(tile_meta)} tiles, {MAX_WORKERS} workers)\n------------------------------------------------------------")
failed = []

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(run_worker, t): t for t in tile_meta}
    with tqdm(total=len(tile_meta), desc="Workers") as pbar:
        for future in as_completed(futures):
            out_file, status, err = future.result()
            pbar.update(1)
            if status == "failed":
                failed.append(out_file)
                pbar.write(f"  FAILED: {out_file}\n    {err}")

if failed:
    print(f"\nWARNING: {len(failed)} tiles failed — output may have gaps")

# -----------------------------
# MERGE COPLANAR (triangle reduction)
# -----------------------------
print(f"\n------------------------------------------------------------\n  REDUCING  -> {REDUCED_DIR}/\n------------------------------------------------------------")

MERGE_PATH = os.path.abspath(MERGE_SCRIPT)

def run_merge(tile):
    """Run merge_coplanar.py on a single processed tile."""
    in_file  = tile["out_file"]
    out_file = os.path.join(REDUCED_DIR, os.path.basename(in_file))

    if not os.path.exists(in_file):
        return out_file, "missing", None

    if os.path.exists(out_file):
        return out_file, "skipped", None

    result = subprocess.run(
        ["python3", MERGE_PATH, in_file, out_file, "--quiet"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return out_file, "failed", result.stderr.strip()

    return out_file, "ok", None

merge_failed = []

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(run_merge, t): t for t in tile_meta}
    with tqdm(total=len(tile_meta), desc="Reducing") as pbar:
        for future in as_completed(futures):
            out_file, status, err = future.result()
            pbar.update(1)
            if status == "failed":
                merge_failed.append(out_file)
                pbar.write(f"  MERGE FAILED: {out_file}\n    {err}")
            elif status == "missing":
                pbar.write(f"  SKIPPED (worker failed): {out_file}")

if merge_failed:
    print(f"\nWARNING: {len(merge_failed)} tiles failed reduction — originals will be used")
    # Copy originals as fallback so combiner always has something to work with
    import shutil
    for t in tile_meta:
        reduced = os.path.join(REDUCED_DIR, os.path.basename(t["out_file"]))
        if not os.path.exists(reduced) and os.path.exists(t["out_file"]):
            shutil.copyfile(t["out_file"], reduced)  # copyfile = bytes only, no metadata (WSL safe)
            print(f"  Copied original: {os.path.basename(t['out_file'])}")

COMBINER_PATH = os.path.abspath(cfg["paths"].get("combiner_script", "./combiner.py"))
print(f"\n------------------------------------------------------------\n  COMBINING\n------------------------------------------------------------")
result = subprocess.run(["python3", COMBINER_PATH, SETTINGS], text=True)
if result.returncode != 0:
    print(f"WARNING: combiner exited with code {result.returncode}")

REBUILD_PATH = os.path.abspath(cfg["paths"].get("rebuild_script", "./rebuild.py"))
print(f"\n------------------------------------------------------------\n  RECONSTRUCTION\n------------------------------------------------------------")
result = subprocess.run([REBUILD_PATH, SETTINGS], text=True)
if result.returncode != 0:
    print(f"WARNING: rebuild exited with code {result.returncode}")

print(f"\n============================================================\n  PIPELINE COMPLETE\n============================================================\n")
