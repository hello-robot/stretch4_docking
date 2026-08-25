import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from stretch4_docking.detectors.numba_detector import region_grow, GROW_RADIUS, EPSILON
from stretch4_docking.estimators.kabsch_estimator import KabschEstimator


APEX, RIGHT, LEFT = 0, 1, 2
MARKER_NAMES = {APEX: 'apex', RIGHT: 'right', LEFT: 'left'}


class DockAmbiguityError(RuntimeError):
    """Raised when seeding sees more than one candidate dock."""


def Rt_from_pose7(pose):
    """(x, y, z, qx, qy, qz, qw) -> (R, t)."""
    x, y, z, qx, qy, qz, qw = pose
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix(), np.array([x, y, z])


def synthesize_missing(p_a, p_b, p_c, q_a, q_b, R_prev):
    e1 = p_b - p_a
    n1 = np.linalg.norm(e1)
    if n1 < 1e-9:
        raise ValueError("degenerate model pair")
    e1 = e1 / n1
    d = p_c - p_a
    alpha = float(d @ e1)
    e2 = d - alpha * e1
    beta = float(np.linalg.norm(e2))
    if beta < 1e-9:
        raise ValueError("model markers are collinear; third point unrecoverable")
    e2 = e2 / beta
    f1 = q_b - q_a
    n2 = np.linalg.norm(f1)
    if n2 < 1e-9:
        raise ValueError("degenerate observed pair")
    f1 = f1 / n2
    f2 = R_prev @ e2
    f2 = f2 - (f2 @ f1) * f1
    n3 = np.linalg.norm(f2)
    if n3 < 1e-6:
        raise ValueError("prior inconsistent with observed pair; cannot recover "
                         "the unobservable axis")
    f2 = f2 / n3
    return q_a + alpha * f1 + beta * f2


class DockTracker(KabschEstimator):

    def __init__(self):
        self.R_prev = None
        self.t_prev = None

        super().__init__()

        self.pair_specs = [
            (RIGHT, LEFT, APEX, self.Lbase),
            (APEX, RIGHT, LEFT, self.Leq),
            (APEX, LEFT, RIGHT, self.Leq),
        ]
        self.assoc_radius = 0.5 * min(self.Lbase, self.Leq)

    def identify(self, points, allow_ambiguity=False):
        """
        `allow_ambiguity` prevents the tracker from raising DockAmbiguityError when more than 1
        dock candidate is seen. If true, it picks the one at random.
        """
        super().identify(points)
        detections = list(self.targets)
        self.targets = []
        if self.R_prev is None:
            self._seed(detections, allow_ambiguity)
        elif detections:
            self._accept(self._nearest(detections))
        else:
            self._recover()

    def is_tracking(self):
        return len(self.targets) > 0

    def get_pose(self):
        return self.targets[0]

    def drop_track(self):
        self.R_prev = None
        self.t_prev = None

    def _seed(self, detections, allow_ambiguity=False):
        if len(detections) > 1 and allow_ambiguity is False:
            raise DockAmbiguityError(
                "Robot is not sure which dock to dock with: {} candidate docks "
                "detected while seeding the tracker. Reposition so that exactly "
                "one dock is in view, or select a target explicitly."
                .format(len(detections))
            )
        if not detections:
            return
        self._accept(detections[0])

    def _nearest(self, detections):
        d = [np.linalg.norm(np.array(p[:3]) - self.t_prev) for p in detections]
        return detections[int(np.argmin(d))]

    def _accept(self, pose):
        self.R_prev, self.t_prev = Rt_from_pose7(pose)
        self.targets = [pose]

    def _recover(self):
        clusters = self._cluster()
        if not clusters:
            return

        pred = self.P_local @ self.R_prev.T + self.t_prev
        C = np.asarray(clusters)
        D = np.linalg.norm(C[:, None, :] - pred[None, :, :], axis=2)  # (n, 3)
        admissible = D <= self.assoc_radius
        if not admissible.any():
            return

        best, best_score = None, np.inf
        for i in range(len(clusters)):
            if not admissible[i].any():
                continue
            for j in range(len(clusters)):
                if i == j or not admissible[j].any():
                    continue
                ci, cj = clusters[i], clusters[j]
                sep = np.linalg.norm(ci - cj)
                for la, lb, lc, model_sep in self.pair_specs:
                    if abs(sep - model_sep) > EPSILON:
                        continue
                    if not (admissible[i, la] and admissible[j, lb]):
                        continue
                    score = D[i, la] + D[j, lb]
                    if score < best_score:
                        best, best_score = (ci, cj, la, lb, lc), score

        if best is None:
            return

        ci, cj, la, lb, lc = best
        try:
            q_c = synthesize_missing(self.P_local[la], self.P_local[lb],
                                     self.P_local[lc], ci, cj, self.R_prev)
        except ValueError:
            return

        observed = {la: ci, lb: cj, lc: q_c}
        self.calculate_pose(observed[APEX], observed[RIGHT], observed[LEFT])
        self.R_prev, self.t_prev = Rt_from_pose7(self.targets[-1])

    def _cluster(self):
        pts = self.filtered_points
        if len(pts) < 2:
            return []

        xyz = np.ascontiguousarray(pts[:, :3], dtype=np.float64)
        tree = KDTree(xyz)
        active = np.ones(len(xyz), dtype=bool)
        centroids = []
        for seed in range(len(xyz)):
            if not active[seed]:
                continue
            idx = region_grow(tree, xyz, seed, GROW_RADIUS, active)
            if not idx:
                active[seed] = False
                continue
            active[idx] = False
            centroids.append(xyz[idx].mean(axis=0))

        return centroids
