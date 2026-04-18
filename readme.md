# Draft My STL

**This code was written by Claude (Anthropic's AI assistant).** It is provided as-is. If something doesn't work, consult an AI — it will be able to help you debug it faster than any issue tracker.

A pipeline for preparing organic 3D mesh files for sand casting molds. It fills in undercuts (overhanging geometry that would trap a mold), applies draft angles to all surfaces, and produces a clean reconstructed mesh ready for mold making.

> **Tested on:** Linux only · 32GB RAM · Intel i7-10750H · Resolution ≥ 0.1mm

---

## What it does

Sand casting requires that a pattern can be pulled cleanly from the mold — this means no undercuts and all vertical surfaces must have a slight outward taper (the "draft angle"). This pipeline automates that process:

1. **Tiles** the mesh into overlapping sections so large models can be processed in parallel
2. **Voxelizes** each tile at a chosen resolution
3. **Fills undercuts** — any geometry that overhangs is filled in downward
4. **Applies draft angle** — surfaces are grown outward as they go downward, at the specified angle
5. **Merges** all tiles back together using boolean union
6. **Reconstructs** the surface using Poisson reconstruction to produce a clean, smooth mesh

---

## Installation

### Main environment (Python 3.12+)

```bash
pip install -r requirements.txt
```

### Reconstruction environment

`rebuild.py` requires `open3d`, which only supports Python ≤3.11. You need a separate environment for this step:

```bash
python3.10 -m venv venv-reconstruct
source venv-reconstruct/bin/activate
pip install -r requirements-reconstruct.txt
```

Then update the shebang at the top of `rebuild.py` to point at this environment:

```python
#!/path/to/venv-reconstruct/bin/python
```

---

## Usage

1. Copy `settings.toml` into your working directory
2. Set `input_file` to the path of your STL
3. Run:

```bash
./controller.py
```

That's it. The full pipeline runs automatically — tiling, drafting, reducing, combining, and reconstructing.

### Running stages manually

```bash
./worker.py tiles/tile_0.stl processed/tile_0.stl settings.toml
./merge_coplanar.py processed/tile_0.stl reduced/tile_0.stl
./combiner.py
./rebuild.py
```

---

## Settings

```toml
[mesh]
input_file         = "../my_model.stl"   # your input STL
output_file        = "output.stl"        # combined mesh before reconstruction
reconstruct_input  = "output.stl"
reconstruct_output = "rebuilt.stl"       # final output

[voxel]
voxel_size      = 0.1   # mm per voxel
draft_angle_deg = 3     # draft angle in degrees

[tiling]
tile_size   = 15.0   # XY tile size in mm
max_workers = 6      # parallel worker processes

[combine]
batch_size             = 8
max_tile_faces         = 0   # per-tile decimation before boolean (0 = off)
max_intermediate_faces = 0   # intermediate decimation (0 = off)
final_face_count       = 0   # final decimation (0 = off)

[reconstruct]
point_count          = 1000000   # points sampled for Poisson
poisson_depth        = 10        # reconstruction detail level
smoothing_iterations = 3         # Taubin smoothing passes (0 = raw only)
```

---

## Choosing resolution

> ⚠️ **Start at 0.2mm or 0.3mm for testing.** At 0.1mm the pipeline can take up to an hour for a 200–300mm wide mesh on an i7-10750H. Only drop to 0.1mm once you are happy with the result at a coarser resolution.

This code has **not been tested below 0.1mm**. Going finer may work but is untested and will be significantly slower. Processing time scales roughly with the cube of resolution — halving `voxel_size` makes it ~8× slower.

### Choosing `poisson_depth`

Set depth so that `model_size / 2^depth ≈ voxel_size`:

| Model size | voxel_size | Recommended depth |
|---|---|---|
| ~50mm | 0.1mm | 9 |
| ~100mm | 0.1mm | 10 |
| ~200mm | 0.1mm | 11 |

---

## Output files

| File | Description |
|---|---|
| `output.stl` | Combined boolean mesh (input to reconstruction) |
| `rebuilt_raw.stl` | Poisson reconstruction, no smoothing |
| `rebuilt.stl` | Poisson reconstruction with Taubin smoothing |
| `debug_combine/` | Intermediate meshes saved during boolean combining |
| `tile_cache.json` | Hash cache so tile diagnosis is fast on re-runs |

---

## Notes

- Overlap between tiles and voxel padding are computed automatically from your mesh height and draft angle — you don't need to set these manually
- Two output meshes are always produced: `rebuilt_raw.stl` (no smoothing) and `rebuilt.stl` (smoothed) — compare both and use whichever suits your application
- Set `TILE_DEBUG = True` at the top of `controller.py` to print detailed per-tile clipping info when diagnosing tiling problems
- **If `tiles/`, `processed/`, or `reduced/` are not cleared between runs, their existing contents will be reused.** This is intentional — it lets you resume a failed run without re-processing everything. If you change settings or your input model, delete these directories before running again to avoid stale data
- Only tested on Linux — Windows and macOS are untested
- Only tested on a 32GB system — lower RAM may work for smaller models at coarser resolution but is untested
