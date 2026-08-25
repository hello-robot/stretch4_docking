import numba
import numpy as np
from scipy import ndimage
from dataclasses import dataclass, field

from .floor_ransac import fit_perpendicular_floor_plane, signed_height_above_plane


@dataclass
class FloorAnalysisConfig:
    map_radius_m: float = 2.0
    resolution_m: float = 0.05
    base_radius: float = 0.26
    voxel_leaf_size: float = 0.05
    floor_detect_z_min: float = -0.4
    floor_detect_z_max: float = 0.1
    floor_fit_threshold_m: float = 0.015
    floor_observed_threshold_m: float = 0.04
    floor_max_tilt_deg: float = 10.0
    floor_ransac_iterations: int = 30
    lidar_obstacle_min_height_m: float = 0.05
    thresh_cliff_m: float = 0.04
    min_cliff_cells: int = 4
    min_edge_floor_cells: int = 2
    overhead_z_min_m: float = 0.30
    base_collision_z_max_m: float = 0.35
    crop_z_min_m: float = -1.0
    crop_z_max_m: float = 1.51


@dataclass
class LabeledLayers:
    obstacle_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    cliff_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    occlusion_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


def _voxel_downsample_dense(points: np.ndarray, leaf_size: float, xy_origin: float | None = None) -> np.ndarray:
    if len(points) == 0 or leaf_size <= 0.0:
        return points

    if xy_origin is not None:
        kx = np.floor((points[:, 0] - xy_origin) / leaf_size).astype(np.int64)
        ky = np.floor((points[:, 1] - xy_origin) / leaf_size).astype(np.int64)
    else:
        kx = np.floor(points[:, 0] / leaf_size).astype(np.int64)
        ky = np.floor(points[:, 1] / leaf_size).astype(np.int64)
    kz = np.floor(points[:, 2] / leaf_size).astype(np.int64)

    kx -= kx.min()
    ky -= ky.min()
    kz -= kz.min()
    dim0 = kx.max() + 1
    dim1 = ky.max() + 1
    dim2 = kz.max() + 1

    n_bins = dim0 * dim1 * dim2
    flat = kx + ky * dim0 + kz * dim0 * dim1
    counts = np.bincount(flat, minlength=n_bins)
    sum_x = np.bincount(flat, weights=points[:, 0], minlength=n_bins)
    sum_y = np.bincount(flat, weights=points[:, 1], minlength=n_bins)
    sum_z = np.bincount(flat, weights=points[:, 2], minlength=n_bins)

    occupied = counts > 0
    occ_counts = counts[occupied].astype(np.float64)
    return np.column_stack([
        sum_x[occupied] / occ_counts,
        sum_y[occupied] / occ_counts,
        sum_z[occupied] / occ_counts,
    ])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    labeled, n = ndimage.label(mask)
    if n == 0:
        return []
    components = []
    for i in range(1, n + 1):
        idx = np.argwhere(labeled == i)
        components.append(idx)
    return components


def _drop_small_components(mask: np.ndarray, min_cells: int) -> np.ndarray:
    if min_cells <= 1 or not np.any(mask):
        return mask
    out = np.zeros_like(mask, dtype=bool)
    for comp in _connected_components(mask):
        if comp.shape[0] >= min_cells:
            out[comp[:, 0], comp[:, 1]] = True
    return out


@numba.njit(cache=True)
def _raycast_occlusions(has_tall, missing_floor, occlusion_mask, size):
    cr = size // 2
    cc = size // 2
    for r in range(size):
        for c in range(size):
            if has_tall[r, c]:
                dr = r - cr
                dc = c - cc
                if dr == 0 and dc == 0:
                    continue
                steps = max(abs(dr), abs(dc))
                for t in range(steps + 1, size + 1):
                    sr = cr + int(round(dr * t / steps))
                    sc = cc + int(round(dc * t / steps))
                    if not (0 <= sr < size and 0 <= sc < size):
                        break
                    if missing_floor[sr, sc]:
                        occlusion_mask[sr, sc] = True


class FloorAnalysis:

    def __init__(self, config: FloorAnalysisConfig | None = None):
        self.config = config if config is not None else FloorAnalysisConfig()
        self.resolution = config.resolution_m
        self.radius = config.map_radius_m
        self.size = int(np.ceil(2.0 * self.radius / self.resolution))
        self.origin = -self.radius + self.resolution / 2.0

    def _indices(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cols = np.floor((points[:, 0] - self.origin) / self.resolution).astype(np.int64)
        rows = np.floor((points[:, 1] - self.origin) / self.resolution).astype(np.int64)
        valid = (cols >= 0) & (cols < self.size) & (rows >= 0) & (rows < self.size)
        return rows[valid], cols[valid], points[valid]

    def process(self, points: np.ndarray) -> LabeledLayers:
        cfg = self.config

        merged = np.asarray(points[:, :3], dtype=np.float64)
        if len(merged) > 0:
            r2 = merged[:, 0] * merged[:, 0] + merged[:, 1] * merged[:, 1]
            cropped = merged[(r2 <= cfg.map_radius_m * cfg.map_radius_m)
                             & (r2 > cfg.base_radius * cfg.base_radius)
                             & (merged[:, 2] >= cfg.crop_z_min_m)
                             & (merged[:, 2] <= cfg.crop_z_max_m)]
        else:
            cropped = merged
        cropped = _voxel_downsample_dense(cropped, cfg.voxel_leaf_size, xy_origin=self.origin)
        if len(cropped) == 0:
            return LabeledLayers()

        n = self.size
        has_any = np.zeros((n, n), dtype=bool)
        has_tall = np.zeros((n, n), dtype=bool)
        has_low_obstacle = np.zeros((n, n), dtype=bool)

        rows, cols, pts = self._indices(cropped)
        if len(rows) == 0:
            return LabeledLayers()

        has_any[rows, cols] = True
        floor_band = (
            (pts[:, 2] >= cfg.floor_detect_z_min)
            & (pts[:, 2] <= cfg.floor_detect_z_max)
        )
        tall = pts[:, 2] >= cfg.overhead_z_min_m
        has_tall[rows[tall], cols[tall]] = True

        floor_pts = cropped[
            (cropped[:, 2] >= cfg.floor_detect_z_min)
            & (cropped[:, 2] <= cfg.floor_detect_z_max)
        ]
        plane, _ = fit_perpendicular_floor_plane(
            floor_pts,
            max_tilt_deg=cfg.floor_max_tilt_deg,
            iterations=cfg.floor_ransac_iterations,
            threshold=cfg.floor_fit_threshold_m,
        )

        observed_floor = np.zeros((n, n), dtype=bool)
        lower_floor = np.zeros((n, n), dtype=bool)
        if plane is not None:
            heights = signed_height_above_plane(pts, plane)
            floor_mask = np.abs(heights) <= cfg.floor_observed_threshold_m
            observed_floor[rows[floor_mask], cols[floor_mask]] = True

            low_band = (
                (heights >= cfg.lidar_obstacle_min_height_m)
                & (heights <= cfg.base_collision_z_max_m)
            )
            has_low_obstacle[rows[low_band], cols[low_band]] = True

            below = floor_band & (heights < -cfg.thresh_cliff_m)
            lower_floor[rows[below], cols[below]] = True

        else:
            floor_mask = (
                (pts[:, 2] >= cfg.floor_detect_z_min)
                & (pts[:, 2] <= cfg.floor_detect_z_max)
            )
            observed_floor[rows[floor_mask], cols[floor_mask]] = True

        ray_hit = has_any.copy()
        non_floor_occupied = has_low_obstacle | has_tall
        missing_floor = ray_hit & (~observed_floor) & (~non_floor_occupied)
        clear_floor = observed_floor & (~lower_floor)

        cliff_mask = np.zeros((n, n), dtype=bool)
        for comp in _connected_components(missing_floor):
            if comp.shape[0] < cfg.min_cliff_cells:
                continue
            floor_neighbors = 0
            for row, col in comp:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < n and 0 <= nc < n and observed_floor[nr, nc]:
                            floor_neighbors += 1
            if floor_neighbors >= cfg.min_edge_floor_cells:
                for row, col in comp:
                    cliff_mask[row, col] = True

        cliff_mask |= lower_floor & ray_hit & (~non_floor_occupied)
        cliff_mask = _drop_small_components(cliff_mask, cfg.min_cliff_cells)

        occlusion_mask = has_tall & (~observed_floor) & (~has_low_obstacle)
        _raycast_occlusions(has_tall, missing_floor, occlusion_mask, self.size)
        occlusion_mask &= ~clear_floor

        obstacle_mask = has_low_obstacle & (~cliff_mask) & (~occlusion_mask)

        return LabeledLayers(
            obstacle_xy=self._mask_to_xy(obstacle_mask),
            cliff_xy=self._mask_to_xy(cliff_mask),
            occlusion_xy=self._mask_to_xy(occlusion_mask),
        )

    def _mask_to_xy(self, mask: np.ndarray) -> np.ndarray:
        idx = np.argwhere(mask)
        if len(idx) == 0:
            return np.zeros((0, 2))
        rows = idx[:, 0].astype(np.float64)
        cols = idx[:, 1].astype(np.float64)
        x = self.origin + cols * self.resolution
        y = self.origin + rows * self.resolution
        return np.column_stack([x, y])
