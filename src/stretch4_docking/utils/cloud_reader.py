import math
import numba
import numpy as np
from scipy.spatial.transform import Rotation


POINTFIELD_DTYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64
}
_POINTFIELD_FLOAT32 = 7


@numba.njit(cache=True)
def _fused_gather_xform_jit(xs, ys, zs, intens, rings, R, t, out, start):
    r00 = R[0, 0]; r01 = R[0, 1]; r02 = R[0, 2]
    r10 = R[1, 0]; r11 = R[1, 1]; r12 = R[1, 2]
    r20 = R[2, 0]; r21 = R[2, 1]; r22 = R[2, 2]
    t0 = t[0]; t1 = t[1]; t2 = t[2]
    count = start
    for i in range(xs.shape[0]):
        x = xs[i]
        y = ys[i]
        z = zs[i]
        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            continue
        out[count, 0] = r00 * x + r01 * y + r02 * z + t0
        out[count, 1] = r10 * x + r11 * y + r12 * z + t1
        out[count, 2] = r20 * x + r21 * y + r22 * z + t2
        out[count, 3] = intens[i]
        out[count, 4] = np.float32(rings[i])
        count += 1
    return count


def _strided_field_view(data, np_dtype, offset, point_step, n_points):
    return np.ndarray(
        shape=(n_points,), dtype=np_dtype, buffer=data,
        offset=offset, strides=(point_step,),
    )


def fused_read_transform_points(cloud, rotation, translation, out, start=0):
    n_points = cloud.width * cloud.height
    if n_points == 0 or len(cloud.data) == 0:
        return start
    fields = {f.name: f for f in cloud.fields}
    missing = [name for name in ('x', 'y', 'z', 'intensity', 'ring') if name not in fields]
    if missing:
        raise ValueError(f"cloud is missing required fields {missing}")
    for name in ('x', 'y', 'z', 'intensity'):
        if fields[name].datatype != _POINTFIELD_FLOAT32:
            raise ValueError(f"field '{name}' has datatype {fields[name].datatype}, expected float32")
    ring_dtype = POINTFIELD_DTYPES.get(fields['ring'].datatype)
    if ring_dtype is None or ring_dtype in (np.float64,):
        raise ValueError(f"unsupported ring datatype {fields['ring'].datatype}")
    if out.shape[0] - start < n_points:
        raise ValueError(
            f"output buffer too small: {out.shape[0] - start} rows free, need up to {n_points}"
        )
    step = cloud.point_step
    xs = _strided_field_view(cloud.data, np.float32, fields['x'].offset, step, n_points)
    ys = _strided_field_view(cloud.data, np.float32, fields['y'].offset, step, n_points)
    zs = _strided_field_view(cloud.data, np.float32, fields['z'].offset, step, n_points)
    intens = _strided_field_view(cloud.data, np.float32, fields['intensity'].offset, step, n_points)
    rings = _strided_field_view(cloud.data, ring_dtype, fields['ring'].offset, step, n_points)
    return _fused_gather_xform_jit(
        xs, ys, zs, intens, rings,
        np.ascontiguousarray(rotation, dtype=np.float32),
        np.ascontiguousarray(translation, dtype=np.float32),
        out, start,
    )


def transform_to_rt(transform):
    """Convert a TransformStamped into a (3x3 rotation, 3-vector translation) numpy pair."""
    q = transform.transform.rotation
    t = transform.transform.translation
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return R, np.array([t.x, t.y, t.z])
