import numpy as np
from scipy.spatial.transform import Rotation

from stretch4_docking.detectors.numba_detector import IsocelesDetector


class KabschEstimator(IsocelesDetector):
    def calculate_pose(self, super_apex, super_right, super_left):
        P = self.P_local
        Q = np.array([super_apex, super_right, super_left])
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        p_centered = P - centroid_P
        q_centered = Q - centroid_Q
        rot, _ = Rotation.align_vectors(q_centered, p_centered)
        t = centroid_Q - rot.apply(centroid_P)

        # Output SE(3) pose as (x, y, z, qx, qy, qz, qw)
        qx, qy, qz, qw = rot.as_quat()
        pose = (t[0], t[1], t[2], qx, qy, qz, qw)
        self._add_detection(pose)
