import numpy as np
from dataclasses import dataclass
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation

from .floor_analysis import FloorAnalysis, FloorAnalysisConfig, LabeledLayers, _drop_small_components


@dataclass(frozen=True)
class CostmapConfig:
    obstacle_cost: float = 500.0
    cliff_cost: float = 490.0
    occlusion_cost: float = 150.0
    lethal_threshold: float = 450.0
    min_hazard_cells: int = 3
    inflation_hard_m: float = 0.21
    inflation_soft_m: float = 0.20
    inflation_soft_peak: float = 250.0
    inflation_soft_tau: float = 0.07
    dock_free_half_m: float = 0.1524
    dock_free_depth_m: float = 0.3556
    dock_free_margin_m: float = 0.025
    dock_channel_half_m: float = 0.16
    dock_channel_behind_m: float = 0.10
    dock_channel_reach_m: float = 0.80


@dataclass
class CostmapResult:
    costmap: np.ndarray
    obstacle_xy: np.ndarray
    cliff_xy: np.ndarray
    occlusion_xy: np.ndarray
    origin: float = 0.0
    resolution: float = 0.05
    size: int = 0


class Costmap:
    def __init__(self, fa_config: FloorAnalysisConfig | None = None, config: CostmapConfig | None = None):
        self.fa_config = fa_config if fa_config is not None else FloorAnalysisConfig()
        self.config = config if config is not None else CostmapConfig()
        self.analysis = FloorAnalysis(self.fa_config)

        self.size = self.analysis.size
        self.resolution = self.analysis.resolution
        self.origin = self.analysis.origin
        self.cell_origin = self.origin + self.resolution / 2.0

    def process(self, points: np.ndarray, dock_pose: np.ndarray | None = None, inflate: bool = True) -> CostmapResult:
        return self.build_costmap(self.analysis.process(points), dock_pose, inflate)

    def build_costmap(self, hits: LabeledLayers, dock_pose: np.ndarray | None = None, inflate: bool = True) -> CostmapResult:
        cfg = self.config

        obstacle = self._layer_mask(hits.obstacle_xy)
        cliff = self._layer_mask(hits.cliff_xy)
        occlusion = self._layer_mask(hits.occlusion_xy)

        free, channel = self._dock_masks(dock_pose)
        if free is not None:
            obstacle &= ~free
            cliff &= ~free
            occlusion &= ~free

        costmap = np.zeros((self.size, self.size), dtype=np.float32)
        costmap[occlusion] = cfg.occlusion_cost
        costmap[cliff] = cfg.cliff_cost
        costmap[obstacle] = cfg.obstacle_cost

        if inflate:
            costmap = self._inflate_lethal(costmap, free, channel)

        return CostmapResult(
            costmap=costmap,
            obstacle_xy=self._mask_to_xy(obstacle),
            cliff_xy=self._mask_to_xy(cliff),
            occlusion_xy=self._mask_to_xy(occlusion),
            origin=self.origin,
            resolution=self.resolution,
            size=self.size,
        )

    def _layer_mask(self, xy: np.ndarray) -> np.ndarray:
        mask = np.zeros((self.size, self.size), dtype=bool)
        if len(xy) == 0:
            return mask
        ix = np.round((xy[:, 0] - self.origin) / self.resolution).astype(np.int64)
        iy = np.round((xy[:, 1] - self.origin) / self.resolution).astype(np.int64)
        keep = (ix >= 0) & (ix < self.size) & (iy >= 0) & (iy < self.size)
        mask[ix[keep], iy[keep]] = True
        return _drop_small_components(mask, self.config.min_hazard_cells)

    def _mask_to_xy(self, mask: np.ndarray) -> np.ndarray:
        idx = np.argwhere(mask)
        if len(idx) == 0:
            return np.zeros((0, 2))
        xy = self.cell_origin + idx.astype(np.float64) * self.resolution
        return np.ascontiguousarray(xy, dtype=np.float64)

    def _dock_masks(self, dock_pose) -> tuple[np.ndarray | None, np.ndarray | None]:
        if dock_pose is None:
            return None, None
        pose = np.asarray(dock_pose, dtype=np.float64).ravel()
        if pose.size >= 7:
            yaw = float(Rotation.from_quat(pose[3:7]).as_euler('xyz')[2])
        elif pose.size == 3:
            yaw = float(pose[2])
        else:
            raise ValueError(
                f"dock_pose must be (x, y, z, qx, qy, qz, qw) or (x, y, yaw), got {pose.size} elements"
            )

        axis = self.cell_origin + np.arange(self.size) * self.resolution
        gx, gy = np.meshgrid(axis, axis, indexing='ij')
        rx, ry = gx - pose[0], gy - pose[1]
        c, s = np.cos(yaw), np.sin(yaw)
        lateral = rx * c + ry * s
        out = -(rx * -s + ry * c)

        cfg = self.config
        margin = cfg.dock_free_margin_m
        free = ((np.abs(lateral) <= cfg.dock_free_half_m + margin)
                & (out >= -margin)
                & (out <= cfg.dock_free_depth_m + margin))
        channel = ((np.abs(lateral) <= cfg.dock_channel_half_m)
                   & (out >= -cfg.dock_channel_behind_m)
                   & (out <= cfg.dock_channel_reach_m))
        return free, channel

    def _inflate_lethal(self, costmap, free, channel) -> np.ndarray:
        cfg = self.config
        source = costmap >= cfg.lethal_threshold
        keep_out = None if free is None else (free | channel)
        if keep_out is not None:
            source &= ~keep_out
        if not source.any():
            return costmap

        dist = distance_transform_edt(~source, sampling=self.resolution)
        inflated = np.zeros_like(costmap)
        skirt = dist - cfg.inflation_hard_m
        soft = cfg.inflation_soft_peak * np.exp(-np.maximum(skirt, 0.0) / cfg.inflation_soft_tau)
        np.copyto(inflated, soft, where=skirt <= cfg.inflation_soft_m)
        np.copyto(inflated, cfg.obstacle_cost, where=dist <= cfg.inflation_hard_m)
        if keep_out is not None:
            inflated[keep_out] = 0.0

        out = np.maximum(costmap, inflated)
        if free is not None:
            out[free] = 0.0
        return out.astype(np.float32)
