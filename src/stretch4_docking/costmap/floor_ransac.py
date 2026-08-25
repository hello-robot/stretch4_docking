import numpy as np
import numba

PlaneCoeffs = tuple[float, float, float, float]


@numba.njit(cache=True)
def _normal_tilt_deg(normal: np.ndarray) -> float:
    z = abs(float(normal[2]))
    z = min(max(z, 0.0), 1.0)
    return float(np.degrees(np.arccos(z)))


@numba.njit(cache=True)
def _numba_fit_floor_plane(points: np.ndarray, max_tilt_deg: float, iterations: int, threshold: float, seed: int = 42) -> tuple[np.ndarray, np.ndarray] | None:
    n = points.shape[0]
    if n < 3:
        return None

    np.random.seed(seed)

    best_inlier_count = -1
    best_plane = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    best_inliers = np.zeros(n, dtype=numba.bool_)

    max_tilt_rad = np.radians(max_tilt_deg)
    cos_max_tilt = np.cos(max_tilt_rad)

    for _ in range(iterations):
        idx1 = np.random.randint(0, n)
        idx2 = np.random.randint(0, n)
        idx3 = np.random.randint(0, n)
        if idx1 == idx2 or idx1 == idx3 or idx2 == idx3:
            continue

        p1 = points[idx1]
        p2 = points[idx2]
        p3 = points[idx3]

        v1 = p2 - p1
        v2 = p3 - p1
        nx = v1[1] * v2[2] - v1[2] * v2[1]
        ny = v1[2] * v2[0] - v1[0] * v2[2]
        nz = v1[0] * v2[1] - v1[1] * v2[0]

        norm = np.sqrt(nx*nx + ny*ny + nz*nz)
        if norm <= 0.0:
            continue

        nx /= norm
        ny /= norm
        nz /= norm
        if nz < 0.0:
            nx = -nx
            ny = -ny
            nz = -nz
        if nz < cos_max_tilt:
            continue

        d = -(nx * p1[0] + ny * p1[1] + nz * p1[2])
        inlier_count = 0
        current_inliers = np.zeros(n, dtype=numba.bool_)
        for j in range(n):
            dist = np.abs(nx * points[j, 0] + ny * points[j, 1] + nz * points[j, 2] + d)
            if dist <= threshold:
                current_inliers[j] = True
                inlier_count += 1

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_plane = np.array([nx, ny, nz, d], dtype=np.float64)
            best_inliers = current_inliers

    if best_inlier_count < 3:
        return None

    inlier_pts = points[best_inliers]
    num_inliers = len(inlier_pts)

    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    sum_xx = 0.0
    sum_yy = 0.0
    sum_xy = 0.0
    sum_xz = 0.0
    sum_yz = 0.0

    for j in range(num_inliers):
        x = inlier_pts[j, 0]
        y = inlier_pts[j, 1]
        z = inlier_pts[j, 2]
        sum_x += x
        sum_y += y
        sum_z += z
        sum_xx += x * x
        sum_yy += y * y
        sum_xy += x * y
        sum_xz += x * z
        sum_yz += y * z

    M = np.array([
        [sum_xx, sum_xy, sum_x],
        [sum_xy, sum_yy, sum_y],
        [sum_x,  sum_y,  float(num_inliers)]
    ], dtype=np.float64)
    Y = np.array([sum_xz, sum_yz, sum_z], dtype=np.float64)

    try:
        coeffs = np.linalg.solve(M, Y)
        normal = np.array([-coeffs[0], -coeffs[1], 1.0], dtype=np.float64)
        norm = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
        if norm > 0.0:
            nx = normal[0] / norm
            ny = normal[1] / norm
            nz = normal[2] / norm
            if nz < 0.0:
                nx = -nx
                ny = -ny
                nz = -nz
            d = -coeffs[2] / norm
            best_plane = np.array([nx, ny, nz, d], dtype=np.float64)
    except:
        pass

    return best_plane, best_inliers


def signed_height_above_plane(points: np.ndarray, coeffs: PlaneCoeffs) -> np.ndarray:
    a, b, c, d = coeffs
    norm = np.sqrt(a * a + b * b + c * c)
    if norm <= 0.0:
        return np.zeros(points.shape[0])
    return (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / norm


def fit_perpendicular_floor_plane(
    points: np.ndarray,
    *,
    max_tilt_deg: float = 10.0,
    iterations: int = 200,
    threshold: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[PlaneCoeffs | None, np.ndarray]:
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    n = points.shape[0]
    if n < 3:
        return None, np.zeros(n, dtype=bool)

    seed = 42
    if rng is not None:
        seed = int(rng.integers(0, 2**31 - 1))

    result = _numba_fit_floor_plane(
        points[:, :3],
        max_tilt_deg,
        max(iterations, 1),
        threshold,
        seed
    )

    if result is None:
        return None, np.zeros(n, dtype=bool)

    plane_arr, inliers = result
    coeffs = (float(plane_arr[0]), float(plane_arr[1]), float(plane_arr[2]), float(plane_arr[3]))

    if _normal_tilt_deg(plane_arr[:3]) > max_tilt_deg:
        return None, np.zeros(n, dtype=bool)

    return coeffs, inliers
