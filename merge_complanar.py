#!/usr/bin/env python3
"""
merge_coplanar.py – Merge coplanar triangles on axis-aligned planes
                    while preserving watertightness at every step.

Approach (no Shapely, no new vertices):
  1. Cluster faces by axis-aligned plane + plane offset (connected components).
  2. Extract boundary loops of each component.
  3. Drop zero-area (degenerate) loops.
  4. Use point-in-polygon test to build correct outer/hole hierarchy:
       - Sort loops by |area| descending.
       - Outer = first loop (largest area).
       - A subsequent loop is a HOLE only if its centroid is INSIDE the outer loop.
       - Otherwise it's a separate disconnected polygon → skip (can't handle with
         a single earcut call without new vertices).
  5. Triangulate with earcut mapping indices back to original vertices.
  6. Check watertight + euler + body_count; rollback if any fail.

Usage:
    python merge_coplanar.py input.stl output.stl [--passes 3] [--quiet]

Dependencies:
    pip install trimesh numpy mapbox-earcut
"""

import sys
import argparse
import traceback
import time
from collections import defaultdict

import numpy as np
import trimesh

try:
    import mapbox_earcut as earcut
    HAS_EARCUT = True
    print("[INFO] mapbox-earcut found ✅")
except ImportError:
    HAS_EARCUT = False
    print("[WARN] mapbox-earcut not found.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

AXIS_NORMALS = np.array([
    [ 1,  0,  0], [-1,  0,  0],
    [ 0,  1,  0], [ 0, -1,  0],
    [ 0,  0,  1], [ 0,  0, -1],
], dtype=np.float64)

AXIS_LABELS = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

# (u, v) basis per axis
AXIS_BASIS = [
    (np.array([0.,0.,1.]), np.array([0.,1.,0.])),  # +X
    (np.array([0.,1.,0.]), np.array([0.,0.,1.])),  # -X
    (np.array([1.,0.,0.]), np.array([0.,0.,1.])),  # +Y
    (np.array([0.,0.,1.]), np.array([1.,0.,0.])),  # -Y
    (np.array([1.,0.,0.]), np.array([0.,1.,0.])),  # +Z
    (np.array([0.,1.,0.]), np.array([1.,0.,0.])),  # -Z
]

ALIGN_TOL = 0.005
MIN_AREA  = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def face_info(mesh, label=""):
    wt  = mesh.is_watertight
    tag = "✅ watertight" if wt else "❌ NOT watertight"
    print(f"  [{label}] faces={len(mesh.faces):,}  verts={len(mesh.vertices):,}  "
          f"euler={mesh.euler_number}  bodies={mesh.body_count}  {tag}")
    return wt


# ─────────────────────────────────────────────────────────────────────────────
# Axis classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_face_normals(face_normals):
    dots    = face_normals @ AXIS_NORMALS.T
    best    = np.argmax(dots, axis=1)
    best_val = dots[np.arange(len(dots)), best]
    return np.where(best_val > (1.0 - ALIGN_TOL), best, -1).astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Connected components
# ─────────────────────────────────────────────────────────────────────────────

def connected_components_faces(face_subset_indices, all_faces):
    n  = len(face_subset_indices)
    fi = face_subset_indices

    edge_to_local = defaultdict(list)
    for loc, gi in enumerate(fi):
        tri = all_faces[gi]
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k+1)%3])
            edge_to_local[(min(a,b), max(a,b))].append(loc)

    adj = defaultdict(set)
    for nbrs in edge_to_local.values():
        if len(nbrs) == 2:
            adj[nbrs[0]].add(nbrs[1])
            adj[nbrs[1]].add(nbrs[0])

    visited = np.zeros(n, dtype=bool)
    comps   = []
    for start in range(n):
        if visited[start]:
            continue
        comp  = []
        stack = [start]
        while stack:
            cur = stack.pop()
            if visited[cur]: continue
            visited[cur] = True
            comp.append(cur)
            for nb in adj[cur]:
                if not visited[nb]:
                    stack.append(nb)
        comps.append(fi[np.array(comp, dtype=np.int64)])
    return comps


def cluster_axis_aligned(mesh, dist_tol=1e-4):
    print(f"\n[cluster] Classifying {len(mesh.faces):,} face normals …")

    axis_ids = classify_face_normals(mesh.face_normals)
    aligned  = np.where(axis_ids >= 0)[0]
    print(f"[cluster] Axis-aligned : {len(aligned):,} / {len(mesh.faces):,}")
    if len(aligned) == 0:
        return []

    ax      = axis_ids[aligned]
    normals = AXIS_NORMALS[ax]
    v0      = mesh.vertices[mesh.faces[aligned, 0]]
    offsets = np.einsum('ij,ij->i', normals, v0)
    q_off   = np.round(offsets / dist_tol).astype(np.int64)

    plane_bucket = defaultdict(list)
    for i, gi in enumerate(aligned):
        plane_bucket[(int(ax[i]), int(q_off[i]))].append(gi)

    all_comps = []
    for (aid, _), fl in plane_bucket.items():
        fi_arr = np.array(fl, dtype=np.int64)
        if len(fi_arr) < 2:
            continue
        for comp in connected_components_faces(fi_arr, mesh.faces):
            if len(comp) >= 2:
                all_comps.append((aid, comp))

    all_comps.sort(key=lambda x: -len(x[1]))
    print(f"[cluster] Components ≥2 faces : {len(all_comps):,}")
    if all_comps:
        sizes = [len(c) for _, c in all_comps]
        print(f"[cluster] Largest={sizes[0]:,}  median={int(np.median(sizes)):,}")
    return all_comps


# ─────────────────────────────────────────────────────────────────────────────
# Boundary loop extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_boundary_loops(face_indices, all_faces):
    tri = all_faces[face_indices]

    edge_count    = defaultdict(int)
    edge_directed = {}
    for t in tri:
        for i in range(3):
            a, b = int(t[i]), int(t[(i+1)%3])
            key  = (min(a,b), max(a,b))
            edge_count[key] += 1
            if key not in edge_directed:
                edge_directed[key] = (a, b)

    boundary = [edge_directed[k] for k, cnt in edge_count.items() if cnt == 1]
    if not boundary:
        return None

    adj = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)

    visited = set()
    loops   = []
    for a0, b0 in boundary:
        if (a0, b0) in visited:
            continue
        loop = [a0]
        visited.add((a0, b0))
        cur = b0
        for _ in range(len(boundary) + 2):
            if cur == a0:
                break
            loop.append(cur)
            advanced = False
            for nxt in adj[cur]:
                if (cur, nxt) not in visited:
                    visited.add((cur, nxt))
                    cur = nxt
                    advanced = True
                    break
            if not advanced:
                cur = None
                break
        if cur == a0 and len(loop) >= 3:
            loops.append(np.array(loop, dtype=np.int64))

    return loops if loops else None


# ─────────────────────────────────────────────────────────────────────────────
# 2D geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def signed_area_2d(pts):
    x, y = pts[:,0], pts[:,1]
    return 0.5 * float(np.dot(x, np.roll(y,-1)) - np.dot(np.roll(x,-1), y))


def point_in_polygon_2d(pt, poly):
    """Ray-casting point-in-polygon test."""
    x, y   = pt
    px, py = poly[:,0], poly[:,1]
    n      = len(px)
    inside = False
    j      = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def centroid_2d(pts):
    return pts.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Merge one connected component  (no new vertices, no Shapely)
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_component(mesh, axis_id, face_indices, cid):
    """
    Deep diagnostic: print everything about a component's boundary loops,
    the edges, and what earcut produces. Call instead of merge_component
    for specific components to understand why they fail.
    """
    import sys
    u_vec, v_vec = AXIS_BASIS[axis_id]
    all_faces = mesh.faces

    tri = all_faces[face_indices]
    print(f"\n{'='*70}")
    print(f"  DIAGNOSE C{cid:05d}  axis={['−Z','+Z','−X','+X','−Y','+Y'][axis_id]}  n_input_faces={len(face_indices)}")

    # ── edge census ──────────────────────────────────────────────────────────
    edge_count    = defaultdict(int)
    edge_directed = {}
    for t in tri:
        for i in range(3):
            a, b = int(t[i]), int(t[(i+1)%3])
            key  = (min(a,b), max(a,b))
            edge_count[key] += 1
            if key not in edge_directed:
                edge_directed[key] = (a, b)

    interior_edges  = [k for k,cnt in edge_count.items() if cnt == 2]
    boundary_edges  = [edge_directed[k] for k,cnt in edge_count.items() if cnt == 1]
    n_verts_total   = len(set(v for f in tri for v in f))
    print(f"  Verts in component : {n_verts_total}")
    print(f"  Interior edges     : {len(interior_edges)}")
    print(f"  Boundary edges     : {len(boundary_edges)}")
    euler_local = n_verts_total - (len(interior_edges)+len(boundary_edges)) + len(face_indices)
    print(f"  Local euler (V-E+F): {euler_local}  (expect 1 for disk, 0 for cylinder, etc.)")

    # ── trace loops ─────────────────────────────────────────────────────────
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)

    visited = set()
    loops_vi = []
    for a0, b0 in boundary_edges:
        if (a0, b0) in visited:
            continue
        loop = [a0]
        visited.add((a0, b0))
        cur = b0
        for _ in range(len(boundary_edges) + 2):
            if cur == a0: break
            loop.append(cur)
            advanced = False
            for nxt in adj[cur]:
                if (cur, nxt) not in visited:
                    visited.add((cur, nxt))
                    cur = nxt
                    advanced = True
                    break
            if not advanced:
                cur = None
                break
        if cur == a0 and len(loop) >= 3:
            loops_vi.append(np.array(loop, dtype=np.int64))

    print(f"  Boundary loops     : {len(loops_vi)}")

    # Check for vertices shared between loops
    all_loop_sets = [set(lp.tolist()) for lp in loops_vi]
    for i in range(len(all_loop_sets)):
        for j in range(i+1, len(all_loop_sets)):
            shared = all_loop_sets[i] & all_loop_sets[j]
            if shared:
                print(f"  ⚠️  Loops {i} and {j} share vertices: {shared}")

    # Check for vertices with degree != 2 in boundary
    from collections import Counter
    bdeg = Counter()
    for a, b in boundary_edges:
        bdeg[a] += 1
        bdeg[b] += 1
    bad_verts = {v: d for v,d in bdeg.items() if d != 2}
    if bad_verts:
        print(f"  ⚠️  {len(bad_verts)} boundary verts with degree != 2: {dict(list(bad_verts.items())[:10])}")
    else:
        print(f"  ✅ All boundary verts have degree 2")

    # ── per-loop analysis ────────────────────────────────────────────────────
    loops_2d = []
    areas    = []
    for lp in loops_vi:
        pts = mesh.vertices[lp]
        p2d = np.column_stack([pts @ u_vec, pts @ v_vec])
        loops_2d.append(p2d)
        areas.append(signed_area_2d(p2d))

    # Sort by area descending
    order = sorted(range(len(loops_vi)), key=lambda i: -abs(areas[i]))
    loops_vi = [loops_vi[i] for i in order]
    loops_2d = [loops_2d[i] for i in order]
    areas    = [areas[i]    for i in order]

    print(f"\n  {'Loop':>4}  {'n_verts':>7}  {'area_2d':>12}  {'winding':>8}  {'class':>6}  centroid")
    print(f"  {'-'*4}  {'-'*7}  {'-'*12}  {'-'*8}  {'-'*6}  {'-'*20}")
    for idx, (lp, p2d, a) in enumerate(zip(loops_vi, loops_2d, areas)):
        winding = "CCW" if a > 0 else "CW"
        c = centroid_2d(p2d)
        # classify vs loop 0 (largest)
        if idx == 0:
            cls = "outer"
        else:
            cls = "hole" if point_in_polygon_2d(c, loops_2d[0]) else "outer?"
        # 3D extent
        pts3 = mesh.vertices[lp]
        span = pts3.max(axis=0) - pts3.min(axis=0)
        print(f"  {idx:>4}  {len(lp):>7}  {a:>12.4f}  {winding:>8}  {cls:>6}  c=({c[0]:.3f},{c[1]:.3f})  3Dspan=({span[0]:.3f},{span[1]:.3f},{span[2]:.3f})")

    # ── earcut per-ring experiment ───────────────────────────────────────────
    print(f"\n  --- Per-ring earcut ---")
    total_new = 0
    ring_results = []
    for idx, (lp, p2d, a) in enumerate(zip(loops_vi, loops_2d, areas)):
        if abs(a) < MIN_AREA:
            print(f"  Ring {idx}: SKIP degenerate")
            ring_results.append(None)
            continue
        flat      = p2d.astype(np.float64)
        ring_ends = np.array([len(lp)], dtype=np.uint32)
        raw = earcut.triangulate_float64(flat, ring_ends)
        if raw is None or len(raw) == 0:
            print(f"  Ring {idx}: earcut FAILED")
            ring_results.append(None)
            continue
        tris = raw.reshape(-1,3)
        ring_faces = lp[tris.astype(np.int64)]
        valid = ((ring_faces[:,0]!=ring_faces[:,1])&(ring_faces[:,1]!=ring_faces[:,2])&(ring_faces[:,0]!=ring_faces[:,2]))
        ring_faces = ring_faces[valid]
        expected = len(lp) - 2
        print(f"  Ring {idx} (n={len(lp)}, a={a:.3f}): earcut→{len(ring_faces)} tris (expected {expected})")
        # check for any edge in ring_faces that also appears in boundary_edges
        ring_edge_count = defaultdict(int)
        for f in ring_faces:
            for i in range(3):
                a2,b2 = int(f[i]), int(f[(i+1)%3])
                ring_edge_count[(min(a2,b2),max(a2,b2))] += 1
        new_edges = [e for e in ring_edge_count if e not in edge_count]
        shared_bdry = [e for e in ring_edge_count if edge_count.get(e,0)==1]
        print(f"    → {len(new_edges)} NEW interior edges (not in original mesh)")
        print(f"    → {len(shared_bdry)} edges shared with original boundary (will be doubled → interior)")
        ring_results.append(ring_faces)
        total_new += len(ring_faces)

    print(f"\n  Per-ring total new faces: {total_new}  vs original: {len(face_indices)}")

    # ── simulate apply and check edge manifold ───────────────────────────────
    print(f"\n  --- Manifold simulation (per-ring) ---")
    all_ring_faces_valid = [r for r in ring_results if r is not None]
    if all_ring_faces_valid:
        combined_new = np.vstack(all_ring_faces_valid)
        # Build full edge map: original mesh minus component + new faces
        keep_mask = np.ones(len(mesh.faces), dtype=bool)
        keep_mask[face_indices] = False
        remaining = mesh.faces[keep_mask]
        test_faces = np.vstack([remaining, combined_new])
        ec = defaultdict(int)
        for f in test_faces:
            for i in range(3):
                a2,b2 = int(f[i]), int(f[(i+1)%3])
                ec[(min(a2,b2),max(a2,b2))] += 1
        open_edges   = sum(1 for cnt in ec.values() if cnt == 1)
        multi_edges  = sum(1 for cnt in ec.values() if cnt >  2)
        print(f"  Open edges (cnt=1) : {open_edges}")
        print(f"  Non-manifold (cnt>2): {multi_edges}")
        if open_edges == 0 and multi_edges == 0:
            print(f"  ✅ Would be WATERTIGHT")
        else:
            print(f"  ❌ NOT watertight")
            # Show some open edges
            open_ex = [(e,ec[e]) for e in ec if ec[e]==1][:5]
            print(f"  Open edge examples: {open_ex}")
            # For each open edge, which loop does it belong to?
            for e, _ in open_ex:
                for idx, lp in enumerate(loops_vi):
                    lp_set = set(zip(lp.tolist(), np.roll(lp,-1).tolist()))
                    lp_set |= set(zip(np.roll(lp,-1).tolist(), lp.tolist()))
                    if (e[0],e[1]) in lp_set or (e[1],e[0]) in lp_set:
                        print(f"    Edge {e} belongs to loop {idx} (n={len(loops_vi[idx])}, area={areas[idx]:.3f})")
                        break

    print(f"{'='*70}\n")



def merge_component(mesh, axis_id, face_indices, cid, debug=True):
    """
    Returns new_faces (Nx3 global vertex indices) or None to skip.
    
    Handles multi-region components by grouping boundary loops into
    (outer, holes) clusters using containment testing, then triangulating
    each cluster independently with earcut.
    
    Key constraint: no new vertices are ever introduced.
    """
    u_vec, v_vec = AXIS_BASIS[axis_id]

    # ── boundary loops ────────────────────────────────────────────────────
    loops_vi = get_boundary_loops(face_indices, mesh.faces)
    if loops_vi is None:
        if debug: print("  → no boundary")
        return None

    # ── project to 2D ─────────────────────────────────────────────────────
    loops_2d = []
    for lp in loops_vi:
        pts = mesh.vertices[lp]
        loops_2d.append(np.column_stack([pts @ u_vec, pts @ v_vec]))

    areas = [signed_area_2d(p) for p in loops_2d]

    # ── drop degenerate loops ─────────────────────────────────────────────
    keep = [(lp, p2d, a) for lp, p2d, a in zip(loops_vi, loops_2d, areas)
            if abs(a) >= MIN_AREA]
    if not keep:
        if debug: print("  → all degenerate")
        return None

    # ── sort by |area| descending ─────────────────────────────────────────
    keep.sort(key=lambda x: -abs(x[2]))

    # ── group loops into (outer, holes) clusters ──────────────────────────
    # A loop is an "outer" if its centroid is NOT inside any larger loop.
    # A loop is a "hole" of the smallest enclosing outer.
    # This handles: simple polygon, polygon with holes, multiple disjoint polys.
    
    clusters = []   # list of (outer_lp, outer_2d, [hole_lp, ...], [hole_2d, ...])
    
    # First pass: identify outers (not inside any other loop)
    outer_indices = []
    for i, (lp_i, p2d_i, a_i) in enumerate(keep):
        c = centroid_2d(p2d_i)
        is_inside = False
        for j, (lp_j, p2d_j, a_j) in enumerate(keep):
            if j == i: continue
            if abs(a_j) <= abs(a_i): continue  # only check larger loops
            if point_in_polygon_2d(c, p2d_j):
                is_inside = True
                break
        if not is_inside:
            outer_indices.append(i)
    
    # Second pass: assign holes to their smallest enclosing outer
    for oi in outer_indices:
        lp_o, p2d_o, a_o = keep[oi]
        # ensure CCW
        if a_o < 0:
            lp_o = lp_o[::-1]; p2d_o = p2d_o[::-1]
        
        hole_lps, hole_2ds = [], []
        for i, (lp_i, p2d_i, a_i) in enumerate(keep):
            if i in outer_indices: continue
            # Is this loop inside our outer and not inside a smaller outer?
            c = centroid_2d(p2d_i)
            if not point_in_polygon_2d(c, p2d_o):
                continue
            # Check it's not inside a smaller outer that's also inside p2d_o
            in_smaller = False
            for oj in outer_indices:
                if oj == oi: continue
                lp_oj, p2d_oj, a_oj = keep[oj]
                if abs(a_oj) >= abs(a_o): continue
                if point_in_polygon_2d(centroid_2d(p2d_oj), p2d_o):
                    if point_in_polygon_2d(c, p2d_oj):
                        in_smaller = True
                        break
            if not in_smaller:
                lp_h, p2d_h = lp_i, p2d_i
                if a_i > 0:  # ensure CW for holes
                    lp_h = lp_h[::-1]; p2d_h = p2d_h[::-1]
                hole_lps.append(lp_h)
                hole_2ds.append(p2d_h)
        
        clusters.append((lp_o, p2d_o, hole_lps, hole_2ds))

    if not clusters:
        if debug: print("  → no clusters")
        return None

    n_outers = len(clusters)
    n_holes  = sum(len(c[2]) for c in clusters)
    if debug:
        print(f"  {n_outers} region(s), {n_holes} hole(s)", end="")

    # ── triangulate each ring INDEPENDENTLY (no hole bridging) ───────────────
    # Key insight proven by diagnostic:
    # - earcut WITH holes creates bridge edges between outer and hole rings.
    #   These edges are new (don't exist in surrounding mesh) → open seams → not watertight.
    # - earcut per ring independently works: each ring's boundary edges are shared
    #   with adjacent mesh faces, so they get counted twice → interior → manifold.
    # - Both outer rings AND hole rings are valid simply-connected polygons when
    #   triangulated independently. The holes' boundary edges are already "sewn"
    #   to the stud/recess geometry on the other side.
    def rotate_to_most_convex(lp, p2d):
        """Rotate loop so earcut starts at the most convex vertex.
        Reduces long spoke triangles on concave boundary loops."""
        n = len(p2d)
        if n <= 3:
            return lp, p2d
        prev_pts = np.roll(p2d,  1, axis=0)
        next_pts = np.roll(p2d, -1, axis=0)
        d1 = p2d - prev_pts
        d2 = next_pts - p2d
        cross = d1[:,0]*d2[:,1] - d1[:,1]*d2[:,0]
        best = int(np.argmax(cross))
        if best == 0:
            return lp, p2d
        return np.roll(lp, -best), np.roll(p2d, -best, axis=0)

    expected_normal = AXIS_NORMALS[axis_id]

    def earcut_ring(lp, p2d):
        """Triangulate a ring with earcut, fix winding to match expected normal."""
        n = len(lp)
        # Special case: rectangle (4 verts) → 2 triangles, optimal layout
        if n == 4:
            # Two possible diagonals — pick the shorter one for better aspect ratio
            v = mesh.vertices[lp]
            d02 = np.linalg.norm(v[2] - v[0])
            d13 = np.linalg.norm(v[3] - v[1])
            if d02 <= d13:
                tris = np.array([[lp[0],lp[1],lp[2]], [lp[0],lp[2],lp[3]]], dtype=np.int64)
            else:
                tris = np.array([[lp[0],lp[1],lp[3]], [lp[1],lp[2],lp[3]]], dtype=np.int64)
        else:
            lp2, p2d2 = rotate_to_most_convex(lp, p2d)
            flat      = p2d2.astype(np.float64)
            ring_ends = np.array([n], dtype=np.uint32)
            raw = earcut.triangulate_float64(flat, ring_ends)
            if raw is None or len(raw) == 0:
                return None
            tris = lp2[raw.reshape(-1, 3).astype(np.int64)]

        valid = ((tris[:,0] != tris[:,1]) & (tris[:,1] != tris[:,2]) & (tris[:,0] != tris[:,2]))
        tris = tris[valid]
        if len(tris) == 0:
            return None

        # Fix winding: check first valid triangle's normal vs expected
        v0 = mesh.vertices[tris[0,0]]
        v1 = mesh.vertices[tris[0,1]]
        v2 = mesh.vertices[tris[0,2]]
        tri_normal = np.cross(v1 - v0, v2 - v0)
        if np.dot(tri_normal, expected_normal) < 0:
            tris = tris[:, ::-1]  # flip all winding
        return tris

    all_new_faces = []

    for lp_o, p2d_o, hole_lps, hole_2ds in clusters:
        tris = earcut_ring(lp_o, p2d_o)
        if tris is not None:
            all_new_faces.append(tris)

        for lp_h, p2d_h in zip(hole_lps, hole_2ds):
            tris = earcut_ring(lp_h, p2d_h)
            if tris is not None:
                all_new_faces.append(tris)

    if not all_new_faces:
        if debug: print("  → earcut failed all rings")
        return None

    new_faces = np.vstack(all_new_faces)
    old_n, new_n = len(face_indices), len(new_faces)
    if debug:
        print(f"  {old_n}→{new_n} {chr(9660) if new_n < old_n else chr(9650)}", end="")

    if new_n >= old_n:
        if debug: print("  no gain")
        return None

    return new_faces


# ─────────────────────────────────────────────────────────────────────────────
# Apply replacement  (no new vertices, no fix_normals on candidate)
# ─────────────────────────────────────────────────────────────────────────────

def apply_replacement(mesh, face_indices_to_remove, new_faces, fix_norms=False):
    keep      = np.ones(len(mesh.faces), dtype=bool)
    keep[face_indices_to_remove] = False
    all_faces = np.vstack([mesh.faces[keep], new_faces])
    out = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                          faces=all_faces, process=False)
    # Do NOT remove_unreferenced_vertices here — it renumbers vertex indices
    # which invalidates all subsequent face_indices from the original clustering.
    # Cleanup happens once at end of pass.
    if fix_norms:
        out.fix_normals()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def apply_batch(mesh, pending):
    """Apply a list of (face_keys, new_faces) replacements in one shot.
    Resolves face_keys to current row indices at apply time."""
    lookup = {frozenset(f.tolist()): fi for fi, f in enumerate(mesh.faces)}
    keep = np.ones(len(mesh.faces), dtype=bool)
    new_face_list = []
    for face_keys, new_faces in pending:
        indices = [lookup[k] for k in face_keys if k in lookup]
        if len(indices) != len(face_keys):
            return None  # stale — some faces already replaced
        keep[indices] = False
        new_face_list.append(new_faces)
    all_faces = np.vstack([mesh.faces[keep]] + new_face_list)
    return trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=all_faces, process=False)


def validate(candidate):
    """Fast watertight check."""
    return candidate is not None and candidate.is_watertight


def apply_and_validate_batch(base_mesh, pending, debug=False):
    """
    Try to apply all pending replacements. If the combined result fails,
    bisect: try first half, then second half, rolling back the bad half.
    pending items: (face_keys, new_faces)
    Returns (accepted_mesh, n_accepted, n_rolled).
    """
    if not pending:
        return base_mesh, 0, 0

    if len(pending) == 1:
        candidate = apply_batch(base_mesh, pending)
        if validate(candidate):
            if debug: print(f"    [batch-1] ✅")
            return candidate, 1, 0
        else:
            if debug: print(f"    [batch-1] ❌ → rolled back")
            return base_mesh, 0, 1

    # Try whole batch first
    candidate = apply_batch(base_mesh, pending)
    if validate(candidate):
        if debug: print(f"    [batch-{len(pending)}] ✅ all accepted")
        return candidate, len(pending), 0

    # Batch failed — bisect
    if debug: print(f"    [batch-{len(pending)}] ❌ → bisecting")
    mid = len(pending) // 2
    left, right = pending[:mid], pending[mid:]

    # Apply left half against base_mesh, then right half against result
    mesh_after_left, n_left, r_left = apply_and_validate_batch(base_mesh, left, debug)
    # Right half re-resolves face keys against mesh_after_left automatically
    mesh_final, n_right, r_right   = apply_and_validate_batch(mesh_after_left, right, debug)
    return mesh_final, n_left + n_right, r_left + r_right


def merge_coplanar_pass(mesh, dist_tol=1e-4, debug=True, batch_size=32,
                        force_fail_test=False):
    t0 = time.perf_counter()

    print("\n" + "═"*62)
    print("  COPLANAR MERGE PASS")
    print("═"*62)
    face_info(mesh, "pass-input")

    components = cluster_axis_aligned(mesh, dist_tol=dist_tol)
    if not components:
        print("[merge] No eligible components.")
        return mesh

    # Pre-compute face content keys for stable resolution across merges
    component_face_keys = []
    for axis_id, face_indices in components:
        keys = [frozenset(mesh.faces[fi].tolist()) for fi in face_indices]
        component_face_keys.append(keys)

    current  = mesh.copy()
    saved    = 0
    accepted = 0
    rolled   = 0
    skipped  = 0
    t_lookup   = 0.0
    t_merge    = 0.0
    t_validate = 0.0

    # Face lookup: frozenset(verts) → row in current.faces
    current_face_lookup = {frozenset(f.tolist()): fi
                           for fi, f in enumerate(current.faces)}

    print(f"\n[merge] {len(components):,} components, batch_size={batch_size} …\n")

    # Collect pending replacements; flush when batch is full or at end
    pending = []  # list of (face_keys, new_faces)

    def flush_batch(force_fail=False):
        nonlocal current, current_face_lookup, accepted, rolled, saved, t_validate
        if not pending:
            return

        batch = [(fk, nf) for (fk, nf) in pending]

        if force_fail:
            # Corrupt last replacement to force a validation failure
            bad_fk, bad_nf = batch[-1]
            bogus = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)  # duplicate = non-manifold
            batch[-1] = (bad_fk, np.vstack([bad_nf, bogus]))
            print(f"  [TEST] Injecting bad faces into last replacement to force failure")

        _t = time.perf_counter()
        current, n_acc, n_roll = apply_and_validate_batch(current, batch, debug=debug)
        t_validate += time.perf_counter() - _t

        accepted += n_acc
        rolled   += n_roll

        # Rebuild lookup after batch
        current_face_lookup.clear()
        for fi, f in enumerate(current.faces):
            current_face_lookup[frozenset(f.tolist())] = fi

        pending.clear()

    for cid, (axis_id, _) in enumerate(components):
        face_keys = component_face_keys[cid]

        _t = time.perf_counter()
        face_indices = np.array(
            [current_face_lookup[k] for k in face_keys if k in current_face_lookup],
            dtype=np.int64)
        t_lookup += time.perf_counter() - _t

        if len(face_indices) < len(face_keys):
            if debug:
                print(f"[C{cid:05d}] {AXIS_LABELS[axis_id]} n={len(face_keys):4d} → stale")
            skipped += 1
            continue

        if len(face_indices) < 35:
            skipped += 1
            continue

        if debug:
            print(f"[C{cid:05d}] {AXIS_LABELS[axis_id]} n={len(face_indices):4d}  ",
                  end="", flush=True)

        _t = time.perf_counter()
        new_faces = merge_component(current, axis_id, face_indices, cid, debug=debug)
        t_merge += time.perf_counter() - _t

        if new_faces is None:
            skipped += 1
            if debug: print()
            continue

        delta = len(new_faces) - len(face_indices)
        pending.append((face_keys, new_faces))
        if debug: print(f"  queued (Δ={delta:+d})")

        if len(pending) >= batch_size:
            flush_batch(force_fail=force_fail_test and len(pending) >= batch_size)

    flush_batch()  # flush remaining

    # Recompute saved from face count difference
    saved = len(mesh.faces) - len(current.faces)

    # Only remove orphaned vertices
    current.remove_unreferenced_vertices()

    elapsed = time.perf_counter() - t0
    print(f"\n{'═'*62}")
    print(f"  PASS DONE  ({elapsed:.1f}s)")
    print(f"  in={len(mesh.faces):,}  out={len(current.faces):,}  "
          f"saved={saved:,} ({100*saved/max(len(mesh.faces),1):.1f}%)")
    print(f"  accepted={accepted}  rolled={rolled}  skipped={skipped}")
    print(f"[timing] lookup: {t_lookup:.2f}s  merge: {t_merge:.2f}s  "
          f"validate: {t_validate:.2f}s")
    face_info(current, "pass-output")
    print("═"*62)
    return current


def run(input_path, output_path, passes=1, dist_tol=1e-4, debug=True,
        batch_size=32, force_fail_test=False):
    print(f"\n{'#'*62}")
    print(f"  {input_path}")
    print(f"{'#'*62}\n")

    mesh = trimesh.load(input_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh)}")

    print(f"  Bounds : {mesh.bounds}")
    print(f"  Extents: {mesh.extents}")
    face_info(mesh, "loaded")

    current = mesh
    for p in range(1, passes + 1):
        print(f"\n{'━'*62}  PASS {p}/{passes}  {'━'*62}")
        before  = len(current.faces)
        current = merge_coplanar_pass(current, dist_tol=dist_tol, debug=debug,
                                      batch_size=batch_size,
                                      force_fail_test=force_fail_test)
        if len(current.faces) >= before:
            print(f"[run] Pass {p}: no reduction – stopping.")
            break

    # Ensure consistent winding before export — is_watertight can pass with
    # inconsistent normals but manifold3d boolean requires is_volume=True.
    if not current.is_volume:
        current.fix_normals()

    print(f"\n[export] → {output_path}")
    current.export(output_path)
    face_info(current, "final")
    print("[export] Done.\n")


def run_diagnose(input_path, cids, dist_tol=1e-4):
    """Load mesh, cluster, and run diagnose_component on specific component IDs."""
    mesh = trimesh.load(input_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh, got {type(mesh)}")
    face_info(mesh, "loaded")
    components = cluster_axis_aligned(mesh, dist_tol=dist_tol)
    print(f"[diagnose] {len(components)} components found. Diagnosing {cids} ...")
    for cid in cids:
        if cid >= len(components):
            print(f"[diagnose] CID {cid} out of range (max {len(components)-1})")
            continue
        axis_id, face_indices = components[cid]
        print(f"\n[C{cid:05d}] {AXIS_LABELS[axis_id]} n={len(face_indices)}")
        diagnose_component(mesh, axis_id, face_indices, cid)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output", nargs="?", default=None,
                    help="Output STL (omit when using --diagnose)")
    ap.add_argument("--passes",        type=int,   default=1)
    ap.add_argument("--dist-tol",      type=float, default=1e-4)
    ap.add_argument("--quiet",         action="store_true")
    ap.add_argument("--batch-size",    type=int,   default=32,
                    help="Number of merges to validate at once (default 32)")
    ap.add_argument("--force-fail-test", action="store_true",
                    help="Inject a bad face into one batch to test rollback")
    ap.add_argument("--diagnose", type=str, default=None,
                    help="Comma-separated component IDs to diagnose, e.g. 1,4,14")

    args = ap.parse_args()
    try:
        if args.diagnose is not None:
            cids = [int(x.strip()) for x in args.diagnose.split(",")]
            run_diagnose(args.input, cids, dist_tol=args.dist_tol)

        else:
            if args.output is None:
                ap.error("output is required unless --diagnose or --vis is used")
            run(args.input, args.output,
                passes=args.passes, dist_tol=args.dist_tol,
                debug=not args.quiet, batch_size=args.batch_size,
                force_fail_test=args.force_fail_test)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
