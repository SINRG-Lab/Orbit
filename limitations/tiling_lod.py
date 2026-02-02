#!/usr/bin/env python3
"""
tile_lod_demo.py

Load a .ply (mesh or point cloud), partition into 3D tiles, and apply
per-tile LOD (4 discrete quality levels) via voxel downsampling.

Goal: visibly demonstrate tile-based viewport-adaptive artifacts:
- hard seams at tile borders
- uneven quality across a single object spanning multiple tiles
- popping when the viewport center moves

Requires:
  pip install open3d numpy

Examples:
  python tile_lod_demo.py --ply bunny.ply --tile 0.03 --steps 12 --show_boxes --colorize_lod
  python tile_lod_demo.py --ply scene.ply --tile 0.15 --near 1.0 --mid 2.5 --far 5.0 --colorize_lod
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import open3d as o3d


@dataclass
class LOD4Config:
    # Distance thresholds that map each tile to one of 4 LODs
    d1: float  # <= d1 => LOD0 (best)
    d2: float  # <= d2 => LOD1
    d3: float  # <= d3 => LOD2
    # > d3 => LOD3 (worst)

    # Voxel sizes for downsampling (smaller voxel = higher quality)
    v0: float
    v1: float
    v2: float
    v3: float


def load_ply_as_point_cloud(path: str, mesh_sample_points: int = 300000) -> o3d.geometry.PointCloud:
    """
    Loads a PLY file.
    - If it contains a triangle mesh, sample points from it (Poisson disk sampling).
    - Otherwise load as point cloud.
    """
    mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
    if mesh is not None and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        pcd = mesh.sample_points_poisson_disk(number_of_points=mesh_sample_points)
        return pcd

    pcd = o3d.io.read_point_cloud(path)
    if pcd is None or len(pcd.points) == 0:
        raise ValueError(f"Could not load usable geometry from: {path}")
    return pcd


def compute_tile_ids(points: np.ndarray, min_bound: np.ndarray, tile_size: float) -> np.ndarray:
    rel = (points - min_bound) / tile_size
    return np.floor(rel).astype(np.int32)


def pack_tile_keys(ixyz: np.ndarray) -> List[Tuple[int, int, int]]:
    return [tuple(v) for v in ixyz]


def build_tiles(tile_keys: List[Tuple[int, int, int]]) -> Dict[Tuple[int, int, int], np.ndarray]:
    """
    tile_key -> indices of points in that tile
    """
    tiles: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, key in enumerate(tile_keys):
        tiles.setdefault(key, []).append(idx)
    return {k: np.array(v, dtype=np.int32) for k, v in tiles.items()}


def tile_center(tile_key: Tuple[int, int, int], min_bound: np.ndarray, tile_size: float) -> np.ndarray:
    i, j, k = tile_key
    return min_bound + (np.array([i, j, k], dtype=np.float64) + 0.5) * tile_size


def choose_lod_index(tile_ctr: np.ndarray, viewport_ctr: np.ndarray, cfg: LOD4Config) -> int:
    d = float(np.linalg.norm(tile_ctr - viewport_ctr))
    if d <= cfg.d1:
        return 0
    if d <= cfg.d2:
        return 1
    if d <= cfg.d3:
        return 2
    return 3


def lod_voxel(lod_idx: int, cfg: LOD4Config) -> float:
    return [cfg.v0, cfg.v1, cfg.v2, cfg.v3][lod_idx]


def make_tile_aabb(tile_key: Tuple[int, int, int], min_bound: np.ndarray, tile_size: float) -> o3d.geometry.AxisAlignedBoundingBox:
    i, j, k = tile_key
    bmin = min_bound + np.array([i, j, k], dtype=np.float64) * tile_size
    bmax = bmin + tile_size
    return o3d.geometry.AxisAlignedBoundingBox(bmin, bmax)


def downsample_tile(points: np.ndarray, colors: Optional[np.ndarray], voxel: float) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd.voxel_down_sample(voxel_size=max(voxel, 1e-9))


def merge_point_clouds(pcds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud()
    if not pcds:
        return out

    all_pts = []
    all_cols = []
    any_colors = any(p.has_colors() for p in pcds)

    for p in pcds:
        pts = np.asarray(p.points)
        if len(pts) == 0:
            continue
        all_pts.append(pts)

        if any_colors:
            if p.has_colors():
                all_cols.append(np.asarray(p.colors))
            else:
                all_cols.append(np.zeros((len(pts), 3), dtype=np.float64))

    if not all_pts:
        return out

    out.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    if any_colors and all_cols:
        out.colors = o3d.utility.Vector3dVector(np.vstack(all_cols))
    return out


# 4 distinct colors for 4 LOD levels (only used when --colorize_lod is set)
LOD_COLORS = [
    np.array([0.10, 0.85, 0.10]),  # LOD0 best
    np.array([0.95, 0.85, 0.10]),  # LOD1
    np.array([0.95, 0.45, 0.10]),  # LOD2
    np.array([0.80, 0.10, 0.10]),  # LOD3 worst
]


def overwrite_color_by_lod(pcd: o3d.geometry.PointCloud, lod_idx: int) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud(pcd)
    n = len(p.points)
    if n == 0:
        return p
    c = LOD_COLORS[lod_idx]
    p.colors = o3d.utility.Vector3dVector(np.tile(c, (n, 1)))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="Input .ply (mesh or point cloud).")
    ap.add_argument("--outdir", default="tile_lod_out", help="Output directory for generated frames.")
    ap.add_argument("--tile", type=float, default=0.05, help="Tile size (world units).")

    # Dist thresholds (you can override; defaults are later scaled if you pass --auto_scale)
    ap.add_argument("--near", type=float, default=1.0, help="Distance threshold d1 (LOD0).")
    ap.add_argument("--mid", type=float, default=2.5, help="Distance threshold d2 (LOD1).")
    ap.add_argument("--far", type=float, default=5.0, help="Distance threshold d3 (LOD2).")

    # 4 LOD voxel sizes (default values are intentionally far apart -> very visible artifacts)
    ap.add_argument("--v0", type=float, default=0.0015, help="Voxel size for LOD0 (best).")
    ap.add_argument("--v1", type=float, default=0.0075, help="Voxel size for LOD1.")
    ap.add_argument("--v2", type=float, default=0.0250, help="Voxel size for LOD2.")
    ap.add_argument("--v3", type=float, default=0.0700, help="Voxel size for LOD3 (worst).")

    ap.add_argument("--steps", type=int, default=10, help="Number of viewport positions (frames).")
    ap.add_argument("--path_radius", type=float, default=0.45, help="Viewport motion radius as fraction of bbox diagonal.")
    ap.add_argument("--mesh_sample_points", type=int, default=300000, help="If input is mesh, how many points to sample.")

    ap.add_argument("--show_boxes", action="store_true", help="Draw tile bounding boxes.")
    ap.add_argument("--colorize_lod", action="store_true", help="Overwrite colors to visualize LOD regions clearly.")
    ap.add_argument("--no_vis", action="store_true", help="Don't open an interactive window; just write frames.")
    ap.add_argument("--auto_scale", action="store_true",
                    help="Auto-scale distance thresholds to bbox size (recommended if you don't know scene scale).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pcd = load_ply_as_point_cloud(args.ply, mesh_sample_points=args.mesh_sample_points)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None

    minb = pts.min(axis=0)
    maxb = pts.max(axis=0)
    scene_center = 0.5 * (minb + maxb)
    diag = float(np.linalg.norm(maxb - minb))

    # If you don't know the scale of the PLY, this makes the bands "reasonable" automatically.
    if args.auto_scale:
        d1 = 0.20 * diag
        d2 = 0.45 * diag
        d3 = 0.80 * diag
    else:
        d1, d2, d3 = args.near, args.mid, args.far

    cfg = LOD4Config(d1=d1, d2=d2, d3=d3, v0=args.v0, v1=args.v1, v2=args.v2, v3=args.v3)

    # Tiling
    tile_ids = compute_tile_ids(pts, minb, args.tile)
    tile_keys = pack_tile_keys(tile_ids)
    tiles = build_tiles(tile_keys)
    tile_keys_sorted = sorted(tiles.keys())

    # Prebuild tile boxes (optional)
    tile_boxes: List[o3d.geometry.AxisAlignedBoundingBox] = []
    if args.show_boxes:
        for k in tile_keys_sorted:
            box = make_tile_aabb(k, minb, args.tile)
            box.color = (0.25, 0.25, 0.25)
            tile_boxes.append(box)

    # Viewport motion path (circle in XY plane)
    radius = args.path_radius * diag
    viewport_centers = []
    for t in range(args.steps):
        ang = 2.0 * math.pi * (t / max(args.steps, 1))
        vc = scene_center + np.array([radius * math.cos(ang), radius * math.sin(ang), 0.0])
        viewport_centers.append(vc)

    # Generate frames
    frames: List[o3d.geometry.PointCloud] = []
    for fi, vc in enumerate(viewport_centers):
        tile_pcds = []

        for k in tile_keys_sorted:
            idx = tiles[k]
            if len(idx) == 0:
                continue

            tc = tile_center(k, minb, args.tile)
            lod_idx = choose_lod_index(tc, vc, cfg)
            voxel = lod_voxel(lod_idx, cfg)

            tp = downsample_tile(pts[idx], cols[idx] if cols is not None else None, voxel)
            if args.colorize_lod:
                tp = overwrite_color_by_lod(tp, lod_idx)

            tile_pcds.append(tp)

        merged = merge_point_clouds(tile_pcds)

        out_path = os.path.join(args.outdir, f"frame_{fi:03d}.ply")
        o3d.io.write_point_cloud(out_path, merged, write_ascii=False, compressed=True)

        print(
            f"[wrote] {out_path}  points={len(merged.points)}  "
            f"d1={cfg.d1:.3g} d2={cfg.d2:.3g} d3={cfg.d3:.3g}  viewport_center={vc}"
        )

        frames.append(merged)

    if args.no_vis:
        return

    # Visualize the first frame (+ optional tile boxes and viewport marker)
    geoms: List[o3d.geometry.Geometry] = [frames[0]]
    if args.show_boxes:
        geoms += tile_boxes

    # Viewport center marker
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.02 * diag)
    sph.translate(viewport_centers[0])
    sph.paint_uniform_color([0.1, 0.4, 0.9])
    geoms.append(sph)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Tile-based 4-LOD artifact demo (frame 0)",
        point_show_normal=False,
    )


if __name__ == "__main__":
    main()

