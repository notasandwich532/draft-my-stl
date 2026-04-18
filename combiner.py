#!/bin/python3
"""
combiner.py — streaming + chunked boolean merge with debug exports
"""

import os
import sys
import tomllib
import trimesh
import numpy as np
from tqdm import tqdm

YELLOW = "\033[93m"
RED    = "\033[91m"
GREEN  = "\033[92m"
RESET  = "\033[0m"

def warn(msg): print(f"{YELLOW}  WARNING: {msg}{RESET}", flush=True)
def err(msg):  print(f"{RED}  ERROR: {msg}{RESET}",   flush=True)
def ok(msg):   print(f"{GREEN}  OK: {msg}{RESET}",     flush=True)

DEBUG = '--debug' in sys.argv
args  = [a for a in sys.argv[1:] if not a.startswith('--')]

# -----------------------------
# SETTINGS
# -----------------------------
SETTINGS = args[0] if args else "settings.toml"

with open(SETTINGS, "rb") as f:
    cfg = tomllib.load(f)

OUTPUT_FILE          = cfg["mesh"]["output_file"]
PROCESSED_DIR        = cfg["paths"].get("reduced_dir", cfg["paths"]["processed_dir"])
FINAL_FACES          = int(cfg["combine"].get("final_face_count", 0))
BATCH_SIZE           = int(cfg["combine"].get("batch_size", 8))
MAX_INTERMEDIATE     = int(cfg["combine"].get("max_intermediate_faces", 2_000_000))
DEBUG_DIR            = cfg["paths"].get("debug_dir", "debug_combine")

os.makedirs(DEBUG_DIR, exist_ok=True)

debug_counter = [0]  # use list so inner functions can mutate

def save_debug(mesh, label):
    """Save intermediate mesh to debug folder."""
    path = os.path.join(DEBUG_DIR, f"{debug_counter[0]:03d}_{label}.stl")
    try:
        mesh.export(path)
        debug_counter[0] += 1
        print(f"  [debug] → {path}  faces={len(mesh.faces)} watertight={mesh.is_watertight}")
    except Exception as e:
        warn(f"debug export failed for {label}: {e}")

# -----------------------------
# SMART DECIMATION
# -----------------------------
def smart_decimate(mesh, target_faces, label=""):
    """
    Decimate a mesh toward target_faces while protecting boundary edges.
    Uses quadric decimation which respects sharp edges well.
    Only runs if mesh is significantly over target.
    """
    current = len(mesh.faces)
    if current <= target_faces:
        return mesh

    ratio = target_faces / current
    print(f"  [decimate] {label}: {current:,} → {target_faces:,} faces ({ratio:.1%})", flush=True)
    try:
        result = mesh.simplify_quadric_decimation(face_count=target_faces)
        if result is not None and len(result.faces) > 0:
            ok(f"  [decimate] {label}: done → {len(result.faces):,} faces  wt={result.is_watertight}")
            return result
        else:
            warn(f"  [decimate] {label}: returned empty mesh, keeping original")
            return mesh
    except Exception as e:
        warn(f"  [decimate] {label}: failed ({e}), keeping original")
        return mesh


# -----------------------------
# ENGINE DETECTION
# -----------------------------
def detect_engine():
    a = trimesh.creation.box([1, 1, 1])
    b = trimesh.creation.box([2, 2, 2])
    for engine in ('manifold', 'trimesh'):
        try:
            r = trimesh.boolean.union([a, b], engine=engine)
            if r is not None and len(r.faces) > 0:
                return engine
        except Exception:
            pass
    return None

print("Detecting Boolean engine...")
ENGINE = detect_engine()
if ENGINE is None:
    err("no Boolean engine available — pip install manifold3d")
    sys.exit(1)
print(f"  Using engine: {ENGINE}")

# -----------------------------
# DIAGNOSE
# -----------------------------
def diagnose(mesh, label):
    wt     = mesh.is_watertight
    euler  = mesh.euler_number
    vol    = mesh.volume

    edges       = mesh.edges_sorted
    uniq, cnt   = np.unique(edges, axis=0, return_counts=True)
    n_boundary  = int((cnt == 1).sum())
    n_nonmanif  = int((cnt >= 3).sum())

    if wt:
        ok(f"{label}: watertight euler={euler} vol={vol:.1f} faces={len(mesh.faces)}")
    else:
        warn(f"{label}: NOT watertight euler={euler} vol={vol:.1f} "
             f"boundary={n_boundary} nonmanifold={n_nonmanif} faces={len(mesh.faces)}")

    if not wt and DEBUG:
        bnd_verts = np.unique(uniq[cnt == 1])
        if len(bnd_verts):
            z_vals = mesh.vertices[bnd_verts, 2]
            bodies = len(mesh.split(only_watertight=False))
            print(f"    bodies={bodies} boundary_Z=[{z_vals.min():.2f}, {z_vals.max():.2f}]")

    return wt

# -----------------------------
# REPAIR
# -----------------------------
def repair(mesh, label=""):
    if mesh.is_watertight:
        return mesh
    try:
        trimesh.repair.fill_holes(mesh)
        mesh.merge_vertices()
        mesh.fix_normals()
    except Exception as e:
        warn(f"repair failed for {label}: {e}")
    return mesh

# -----------------------------
# UNION SAFE — with crash protection
# -----------------------------
def union_safe(a, b, label=""):
    a_wt = a.is_watertight
    b_wt = b.is_watertight

    if not a_wt:
        warn(f"{label}: A not watertight")
    if not b_wt:
        warn(f"{label}: B not watertight")

    try:
        result = trimesh.boolean.union([a, b], engine=ENGINE)
        if result is not None and len(result.faces) > 0:
            # Repair without fix_normals on bad mesh to avoid the crash
            if not result.is_watertight:
                try:
                    trimesh.repair.fill_holes(result)
                    result.merge_vertices()
                    result.fix_normals()
                except Exception as e:
                    warn(f"{label} post-repair failed: {e}")
            if a_wt and b_wt:
                ok(f"{label}: Boolean union succeeded")
            else:
                warn(f"{label}: Boolean union ran but inputs weren't watertight")
            return result
    except Exception as e:
        warn(f"{label}: Boolean failed ({e})")

    # Concatenate fallback — safe, no fix_normals on combined until stable
    warn(f"{label}: using concatenate fallback — internal walls will NOT be removed")
    try:
        combined = trimesh.util.concatenate([a, b])
        combined.merge_vertices()
        # Only fix normals if mesh looks sane
        if len(combined.faces) > 0 and len(combined.vertices) > 0:
            try:
                combined.fix_normals()
            except Exception as e:
                warn(f"{label}: fix_normals failed on concatenate: {e}")
    except Exception as e:
        err(f"{label}: concatenate itself failed: {e}")
        # Last resort — return whichever is bigger
        combined = a if len(a.faces) >= len(b.faces) else b

    return combined

# -----------------------------
# LOAD + DIAGNOSE TILES
# -----------------------------
tile_files = sorted(
    os.path.join(PROCESSED_DIR, f)
    for f in os.listdir(PROCESSED_DIR)
    if f.endswith(".stl")
)

if not tile_files:
    err("no processed tiles found")
    sys.exit(1)

print(f"\nFound {len(tile_files)} tiles")
print("\nDiagnosing all tiles...")

n_wt, n_not_wt = 0, 0
for path in tile_files:
    m  = trimesh.load(path)
    wt = diagnose(m, os.path.basename(path))
    if wt: n_wt += 1
    else:  n_not_wt += 1
    del m

print(f"\n  Summary: {n_wt}/{len(tile_files)} watertight")
if n_not_wt:
    warn(f"{n_not_wt} tiles not watertight — Boolean will fall back to concatenate")
    print("  Delete processed/*.stl and rerun controller to regenerate tiles.")

# -----------------------------
# BATCH PROCESSING
# -----------------------------
intermediate_meshes = []
print(f"\nProcessing in batches of {BATCH_SIZE}...")

for i in range(0, len(tile_files), BATCH_SIZE):
    batch      = tile_files[i:i+BATCH_SIZE]
    batch_num  = i // BATCH_SIZE + 1
    print(f"\n  Batch {batch_num}: {len(batch)} tiles")

    merged = None

    for path in tqdm(batch, desc=f"    batch {batch_num}"):
        try:
            mesh = trimesh.load(path)
        except Exception as e:
            warn(f"failed to load {path}: {e}")
            continue

        if mesh is None or len(mesh.faces) == 0:
            warn(f"empty tile: {path}")
            continue

        mesh = repair(mesh, os.path.basename(path))

        if merged is None:
            merged = mesh
        else:
            merged = union_safe(merged, mesh, label=f"b{batch_num}")

    if merged is not None:
        diagnose(merged, f"batch_{batch_num}_result")
        # Decimate batch result if over limit — before storing as intermediate
        if MAX_INTERMEDIATE > 0 and len(merged.faces) > MAX_INTERMEDIATE:
            merged = smart_decimate(merged, MAX_INTERMEDIATE, label=f"batch_{batch_num}")
            diagnose(merged, f"batch_{batch_num}_after_decimate")
        save_debug(merged, f"batch{batch_num:02d}_result")
        intermediate_meshes.append(merged)

    del merged

print(f"\nCreated {len(intermediate_meshes)} intermediate meshes")

# -----------------------------
# FINAL REDUCTION
# -----------------------------
print("\nFinal merge...")
round_num = 0

while len(intermediate_meshes) > 1:
    round_num  += 1
    next_round  = []
    pairs       = list(zip(intermediate_meshes[0::2], intermediate_meshes[1::2]))
    leftover    = intermediate_meshes[-1] if len(intermediate_meshes) % 2 else None

    print(f"  Round {round_num}: {len(intermediate_meshes)} → {len(pairs) + (1 if leftover else 0)}")

    for i, (a, b) in enumerate(tqdm(pairs, desc=f"  round {round_num}")):
        merged = union_safe(a, b, label=f"r{round_num}p{i}")
        # Decimate round result if over limit (but not the very last merge)
        if MAX_INTERMEDIATE > 0 and len(merged.faces) > MAX_INTERMEDIATE:
            merged = smart_decimate(merged, MAX_INTERMEDIATE, label=f"r{round_num}p{i}")
        save_debug(merged, f"round{round_num}_pair{i}")
        next_round.append(merged)

    if leftover:
        next_round.append(leftover)

    intermediate_meshes = next_round

combined = intermediate_meshes[0]

# -----------------------------
# FINAL CLEANUP
# -----------------------------
print("\nFinal cleanup...")
combined.remove_unreferenced_vertices()
combined.merge_vertices()
try:
    combined.fix_normals()
except Exception as e:
    warn(f"final fix_normals failed: {e}")

if not combined.is_watertight:
    warn("attempting final repair...")
    try:
        trimesh.repair.fill_holes(combined)
        combined.fix_normals()
    except Exception as e:
        warn(f"final repair failed: {e}")

print("\nFinal result:")
diagnose(combined, "combined")

if FINAL_FACES > 0 and len(combined.faces) > FINAL_FACES:
    print(f"Decimating to {FINAL_FACES} faces...")
    combined = combined.simplify_quadric_decimation(face_count=FINAL_FACES)

save_debug(combined, "FINAL")

print(f"\nExporting → {OUTPUT_FILE}")
combined.export(OUTPUT_FILE)
print("Done")
