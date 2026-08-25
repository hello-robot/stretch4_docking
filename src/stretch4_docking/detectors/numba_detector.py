import numpy as np
from stretch4_urdf import get_accessory, get_transform
from scipy.spatial import KDTree
from numba import njit


GROW_RADIUS = 0.04
EPSILON = 0.03


@njit(fastmath=True, cache=True)
def kabsch(P, Q):
    centroid_P = np.array([np.mean(P[:, 0]), np.mean(P[:, 1]), np.mean(P[:, 2])])
    centroid_Q = np.array([np.mean(Q[:, 0]), np.mean(Q[:, 1]), np.mean(Q[:, 2])])
    p = P - centroid_P
    q = Q - centroid_Q
    H = p.T @ q
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt_copy = Vt.copy()
        Vt_copy[2, :] *= -1
        R = Vt_copy.T @ U.T
    t = centroid_Q - R @ centroid_P
    return R, t


@njit(fastmath=True, cache=True)
def check_roll_pitch(apex_cand, right_cand, left_cand, P_local):
    Q = np.zeros((3, 3))
    Q[0] = apex_cand
    Q[1] = right_cand
    Q[2] = left_cand

    R, _ = kabsch(P_local, Q)

    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    roll = np.arctan2(R[2, 1], R[2, 2])

    pitch_deg = np.abs(pitch * 180.0 / np.pi)
    roll_deg = np.abs(roll * 180.0 / np.pi)

    return roll_deg <= 5.0 and pitch_deg <= 5.0


@njit(fastmath=True, cache=True)
def numba_find_triangle(pts, Leq, Lbase, P_local, epsilon=EPSILON):
    n = pts.shape[0]

    for i in range(n):
        apex = pts[i]
        candidates = np.zeros(n, dtype=np.int32)
        c_count = 0
        for j in range(n):
            if i == j: continue

            dx = apex[0] - pts[j, 0]
            dy = apex[1] - pts[j, 1]
            dz = apex[2] - pts[j, 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            if abs(dist - Leq) <= epsilon:
                candidates[c_count] = j
                c_count += 1

        for j in range(c_count):
            for k in range(j + 1, c_count):
                n1 = candidates[j]
                n2 = candidates[k]
                dx = pts[n1, 0] - pts[n2, 0]
                dy = pts[n1, 1] - pts[n2, 1]
                dz = pts[n1, 2] - pts[n2, 2]
                dist_base = np.sqrt(dx*dx + dy*dy + dz*dz)
                if abs(dist_base - Lbase) <= epsilon:
                    v_m_x = pts[n1, 0] - pts[i, 0]
                    v_m_y = pts[n1, 1] - pts[i, 1]
                    v_n_x = pts[n2, 0] - pts[i, 0]
                    v_n_y = pts[n2, 1] - pts[i, 1]
                    cross_product = v_m_x * v_n_y - v_m_y * v_n_x
                    if cross_product > 0:
                        right_cand = pts[n1]
                        left_cand = pts[n2]
                    else:
                        right_cand = pts[n2]
                        left_cand = pts[n1]
                    if check_roll_pitch(pts[i], right_cand, left_cand, P_local):
                        # FOUND! triangle is formed by indices (i, n1, n2)
                        return i, n1, n2

    return -1, -1, -1


def region_grow(tree, points, seed_idx, grow_radius, active_mask):
    cluster = []
    queue = [seed_idx]
    visited = {seed_idx}

    while queue:
        curr_idx = queue.pop(0)
        if not active_mask[curr_idx]:
            continue
        cluster.append(curr_idx)
        neighbors = tree.query_ball_point(points[curr_idx], grow_radius)
        for n in neighbors:
            if active_mask[n] and n not in visited:
                visited.add(n)
                queue.append(n)

    return cluster


class IsocelesDetector:

    def __init__(self):
        self.targets = []
        self.reset()

        urdf = get_accessory("docking_station")
        self.T_left = get_transform(urdf, "left_aruco_marker_link", "docking_station_link")
        self.T_right = get_transform(urdf, "right_aruco_marker_link", "docking_station_link")
        self.T_apex = get_transform(urdf, "apex_aruco_marker_link", "docking_station_link")
        self.left_local = self.T_left[:3, 3]
        self.right_local = self.T_right[:3, 3]
        self.apex_local = self.T_apex[:3, 3]
        self.P_local = np.array([self.apex_local, self.right_local, self.left_local])
        left = self.left_local
        right = self.right_local
        apex = self.apex_local
        self.Lbase = np.linalg.norm(left - right)
        self.Leq = np.linalg.norm(left - apex)
        self.filtered_points = np.empty((0, 4))

    def reset(self):
        self.targets = []
        self.filtered_points = np.empty((0, 4))

    def identify(self, points):
        self.reset()
        z = points[:, 2]
        valid_z = (z >= -0.05) & (z <= 0.2)
        points = points[valid_z]
        i = points[:, 3]
        valid_i = (i > 235)
        points = points[valid_i]
        self.filtered_points = points

        n_points = len(points)
        active_mask = np.ones(n_points, dtype=bool)
        global_tree = KDTree(points[:, :3])
        grow_radius = GROW_RADIUS
        raw_pts = np.ascontiguousarray(points[:, :3], dtype=np.float64)
        active_indices = np.where(active_mask)[0]
        while len(active_indices) >= 3:
            active_points = raw_pts[active_indices]
            i_idx, n1_idx, n2_idx = numba_find_triangle(active_points, self.Leq, self.Lbase, self.P_local, 0.03)
            if i_idx == -1:
                break

            seed_apex = active_indices[i_idx]
            seed_leg1 = active_indices[n1_idx]
            seed_leg2 = active_indices[n2_idx]
            c_apex = region_grow(global_tree, points[:, :3], seed_apex, grow_radius, active_mask)
            c_leg1 = region_grow(global_tree, points[:, :3], seed_leg1, grow_radius, active_mask)
            c_leg2 = region_grow(global_tree, points[:, :3], seed_leg2, grow_radius, active_mask)
            for idx in c_apex + c_leg1 + c_leg2:
                active_mask[idx] = False
            apex_pts = points[c_apex, :3]
            leg1_pts = points[c_leg1, :3]
            leg2_pts = points[c_leg2, :3]
            super_apex = np.mean(apex_pts, axis=0)
            super_leg1 = np.mean(leg1_pts, axis=0)
            super_leg2 = np.mean(leg2_pts, axis=0)
            v_m = super_leg1[:2] - super_apex[:2]
            v_n = super_leg2[:2] - super_apex[:2]
            cross_product = v_m[0] * v_n[1] - v_m[1] * v_n[0]
            if cross_product > 0:
                super_right = super_leg1
                super_left = super_leg2
                c_right, c_left = c_leg1, c_leg2
            else:
                super_right = super_leg2
                super_left = super_leg1
                c_right, c_left = c_leg2, c_leg1
            self.calculate_pose(super_apex, super_right, super_left)
            active_indices = np.where(active_mask)[0]

    def calculate_pose(self, super_apex, super_right, super_left):
        raise NotImplementedError("Subclasses must implement calculate_pose")

    def _add_detection(self, pose):
        self.targets.append(pose)

    def is_triangle_found(self):
        return len(self.targets) > 0

    def get_pose(self):
        return self.targets
