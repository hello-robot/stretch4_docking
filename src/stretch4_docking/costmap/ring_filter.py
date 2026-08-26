import numba
import numpy as np
from dataclasses import dataclass


LIDAR_LEFT_ORIGIN = np.array([0.03874421, 0.12441374, 1.500672], dtype=np.float32)
LIDAR_RIGHT_ORIGIN = np.array([0.03874421, -0.12441374, 1.500672], dtype=np.float32)
LIDAR_LEFT_ROT = np.array([[ 0.51955507,  0.72351543,  0.45451948],
                           [ 0.48359135, -0.68755171,  0.54167522],
                           [ 0.70441603, -0.06162842, -0.70710678]], dtype=np.float32)
LIDAR_RIGHT_ROT = np.array([[-0.51955507,  0.72351543,  0.45451948],
                            [ 0.48359135,  0.68755171, -0.54167522],
                            [-0.70441603, -0.06162842, -0.70710678]], dtype=np.float32)

def fetch_origins():
    from stretch4_urdf import get_urdf_from_robot_params, get_transform
    urdf = get_urdf_from_robot_params()
    left_origin = get_transform(urdf, frame_to="lidar_left_link", frame_from="base_link")[:3,3]
    right_origin = get_transform(urdf, frame_to="lidar_right_link", frame_from="base_link")[:3,3]
    return left_origin, right_origin


@dataclass
class RingFilterConfig:
    num_rings: int = 128
    min_angle_deg: float = 30.0
    max_angle_deg: float = 150.0
    invert_filter: bool = False
    cylinder_radius_m: float = 0.33


_AZ_BIN_RAD = 1e-3
_AZ_NBINS = int(np.ceil(2.0 * np.pi / _AZ_BIN_RAD)) + 2  # +1 NaN bin, +1 rounding headroom


@numba.njit(cache=True)
def _counting_argsort_jit(keys: np.ndarray, n_keys: int) -> np.ndarray:
    counts = np.zeros(n_keys + 1, dtype=np.int64)
    for i in range(len(keys)):
        counts[keys[i] + 1] += 1
    for k in range(1, n_keys + 1):
        counts[k] += counts[k - 1]
    order = np.empty(len(keys), dtype=np.int64)
    for i in range(len(keys)):
        order[counts[keys[i]]] = i
        counts[keys[i]] += 1
    return order


@numba.njit(cache=True)
def _insertion_refine_jit(order: np.ndarray, ring_keys: np.ndarray, angles: np.ndarray) -> None:
    for i in range(1, len(order)):
        idx = order[i]
        ring = ring_keys[idx]
        angle = angles[idx]
        j = i - 1
        while j >= 0 and ring_keys[order[j]] == ring and angles[order[j]] > angle:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = idx


@numba.njit(cache=True)
def _filter_shadow_by_index_jit(xyz: np.ndarray, xyz_baselink: np.ndarray, rings: np.ndarray, sort_idx: np.ndarray, min_angle_tan: float, max_angle_tan: float, invert_filter: bool, cylinder_radius_m: float) -> np.ndarray:
    n = sort_idx.shape[0]
    nan_mask = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return nan_mask
    cyl_r2 = cylinder_radius_m * cylinder_radius_m

    prev = sort_idx[0]
    x0 = xyz[prev, 0]
    y0 = xyz[prev, 1]
    z0 = xyz[prev, 2]
    r0 = np.sqrt(x0 * x0 + y0 * y0 + z0 * z0)
    prev_ring = rings[prev]

    for k in range(1, n):
        cur = sort_idx[k]
        cur_ring = rings[cur]
        x1 = xyz[cur, 0]
        y1 = xyz[cur, 1]
        z1 = xyz[cur, 2]
        r1 = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)

        if cur_ring == prev_ring:
            in_cyl = True
            if cylinder_radius_m > 0.0:
                bx = xyz_baselink[prev, 0]
                by = xyz_baselink[prev, 1]
                in_cyl = bx * bx + by * by <= cyl_r2
            if in_cyl:
                dot = x0 * x1 + y0 * y1 + z0 * z1
                denom = r0 * r1
                if denom > 0.0:
                    cos_alpha = dot / denom
                    if cos_alpha < -1.0:
                        cos_alpha = -1.0
                    elif cos_alpha > 1.0:
                        cos_alpha = 1.0

                    sin_alpha = np.sqrt(-cos_alpha * cos_alpha + 1.0)
                    perpendicular_x = r0 - r1 * cos_alpha
                    perpendicular_y = r1 * sin_alpha

                    is_sh = False
                    if perpendicular_x == 0.0:
                        is_sh = True
                    else:
                        tan_theta = np.abs(perpendicular_y) / perpendicular_x
                        if tan_theta > 0.0:
                            if tan_theta < min_angle_tan:
                                is_sh = True
                        else:
                            if tan_theta > max_angle_tan:
                                is_sh = True

                    if is_sh == (not invert_filter):
                        nan_mask[prev] = True

        prev = cur
        prev_ring = cur_ring
        x0 = x1
        y0 = y1
        z0 = z1
        r0 = r1

    return nan_mask


def _organized_grid_cols(rings: np.ndarray, num_rings: int) -> int:
    n = rings.shape[0]
    if n == 0 or n % num_rings != 0:
        return 0
    ncols = n // num_rings
    expected = np.tile(np.arange(num_rings), ncols)
    if np.array_equal(rings.astype(np.int64), expected):
        return ncols
    return 0


def _warm_start_cloud(dtype: np.dtype) -> np.ndarray:
    n_rings, per_ring = 4, 16
    rings = np.repeat(np.arange(n_rings), per_ring)
    azimuth = np.tile(np.linspace(-np.pi, np.pi, per_ring, endpoint=False), n_rings)
    radius = 1.0 + 0.05 * (rings % 2)
    return np.column_stack([
        radius * np.cos(azimuth),
        radius * np.sin(azimuth),
        0.1 * rings - 0.2,
        np.full(rings.shape, 100.0),
        rings,
    ]).astype(dtype, copy=False)


class RingFilter:

    def __init__(self, config: RingFilterConfig | None = None):
        self.config = config if config is not None else RingFilterConfig()

    def warm_start(self):
        for dtype in (np.float32, np.float64):
            self.process(_warm_start_cloud(dtype), sensor_origin=LIDAR_LEFT_ORIGIN)

    def process(self, points: np.ndarray, sensor_origin: np.ndarray, compact: bool = True) -> np.ndarray:
        if len(points) == 0:
            return points.copy()
        if points.shape[1] < 5:
            raise NotImplementedError('need point clouds w/ring field')

        xyz_orig = points[:, :3]
        rings_orig = points[:, 4]
        xyz_sensor = xyz_orig - sensor_origin.astype(np.float32)
        angles = np.arctan2(xyz_sensor[:, 1], xyz_sensor[:, 0])
        ring_keys = np.clip(rings_orig.astype(np.int64), 0, self.config.num_rings - 1)

        num_rings = self.config.num_rings
        ncols = _organized_grid_cols(rings_orig, num_rings)
        if ncols > 0:
            base_idx = np.arange(len(points), dtype=np.int64).reshape(ncols, num_rings).T
            ang_grid = angles.reshape(ncols, num_rings).T
            order = np.argsort(ang_grid, axis=1, kind='stable')
            sort_idx = np.take_along_axis(base_idx, order, axis=1).reshape(-1)
        else:
            az_bins = ((angles + np.pi) * (1.0 / _AZ_BIN_RAD)).astype(np.int64)
            np.clip(az_bins, 0, _AZ_NBINS - 1, out=az_bins)
            az_bins[np.isnan(angles)] = _AZ_NBINS - 1
            keys = ring_keys * _AZ_NBINS + az_bins
            sort_idx = _counting_argsort_jit(keys, num_rings * _AZ_NBINS)
            _insertion_refine_jit(sort_idx, ring_keys, angles)

        xyz = np.ascontiguousarray(xyz_sensor, dtype=np.float32)
        xyz_baselink = np.ascontiguousarray(xyz_orig, dtype=np.float32)
        rings = np.ascontiguousarray(ring_keys, dtype=np.int32)
        min_angle_rad = np.deg2rad(self.config.min_angle_deg)
        max_angle_rad = np.deg2rad(self.config.max_angle_deg)
        min_angle_tan = np.tan(min_angle_rad)
        max_angle_tan = np.tan(max_angle_rad)
        nan_mask = _filter_shadow_by_index_jit(
            xyz,
            xyz_baselink,
            rings,
            sort_idx,
            min_angle_tan,
            max_angle_tan,
            self.config.invert_filter,
            self.config.cylinder_radius_m,
        )

        if compact:
            return points[~nan_mask]

        out_points = points.copy()
        out_points[nan_mask, 0] = np.nan
        out_points[nan_mask, 1] = np.nan
        out_points[nan_mask, 2] = np.nan
        return out_points
