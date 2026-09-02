import numpy as np
from dataclasses import dataclass, field

from .floor_analysis import FloorAnalysisConfig


def _as_xy(points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 2:
        raise ValueError('hazard points must have at least x and y columns')
    return np.ascontiguousarray(points[:, :2], dtype=np.float64)


@dataclass(frozen=True)
class ClearanceFilterConfig:
    footprint_m: float = 0.21
    obstacle_buffer_m: float = 0.06
    cliff_buffer_m: float = 0.20
    horizon_s: float = 0.25
    min_linear_speed_mps: float = 1e-4
    sensor_floor_m: float = FloorAnalysisConfig.base_radius

    @property
    def obstacle_stop_radius_m(self) -> float:
        return self.footprint_m + self.obstacle_buffer_m

    @property
    def cliff_stop_radius_m(self) -> float:
        return self.footprint_m + self.cliff_buffer_m


@dataclass(frozen=True)
class ClearanceFilterResult:
    vx: float
    vy: float
    wz: float
    blocked_by_obstacle: bool = False
    blocked_by_cliff: bool = False
    blocking_obstacle_count: int = 0
    blocking_cliff_count: int = 0
    blocking_obstacles: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    blocking_cliffs: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


def get_closing_points(points_xy: np.ndarray, step: np.ndarray, stop_radius_m: float) -> np.ndarray:
    points = _as_xy(points_xy)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    after = points - np.asarray(step, dtype=np.float64).reshape(2)
    d_next_sq = np.einsum('ij,ij->i', after, after)
    d_now_sq = np.einsum('ij,ij->i', points, points)
    radius = max(float(stop_radius_m), 0.0)
    closing = (
        np.isfinite(points).all(axis=1)
        & (d_next_sq < radius * radius)
        & (d_next_sq < d_now_sq)
    )
    return points[closing]


def filter_clearance_velocity(
    vx: float,
    vy: float,
    wz: float,
    obstacle_xy: np.ndarray,
    cliff_xy: np.ndarray,
    config: ClearanceFilterConfig | None = None,
) -> ClearanceFilterResult:
    cfg = config or ClearanceFilterConfig()
    linear = np.asarray([float(vx), float(vy)], dtype=np.float64)
    speed = float(np.linalg.norm(linear))
    if speed <= cfg.min_linear_speed_mps:
        return ClearanceFilterResult(float(vx), float(vy), float(wz))

    step = linear * cfg.horizon_s
    blocking_obstacles = get_closing_points(
        obstacle_xy, step, cfg.obstacle_stop_radius_m)
    blocking_cliffs = get_closing_points(
        cliff_xy, step, cfg.cliff_stop_radius_m)

    blocked_by_obstacle = len(blocking_obstacles) > 0
    blocked_by_cliff = len(blocking_cliffs) > 0

    out_vx, out_vy = float(vx), float(vy)
    if blocked_by_obstacle or blocked_by_cliff:
        out_vx, out_vy = 0.0, 0.0

    return ClearanceFilterResult(
        vx=out_vx,
        vy=out_vy,
        wz=float(wz),
        blocked_by_obstacle=blocked_by_obstacle,
        blocked_by_cliff=blocked_by_cliff,
        blocking_obstacle_count=len(blocking_obstacles),
        blocking_cliff_count=len(blocking_cliffs),
        blocking_obstacles=blocking_obstacles,
        blocking_cliffs=blocking_cliffs,
    )
