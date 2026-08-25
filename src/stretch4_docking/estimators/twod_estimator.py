import math
from stretch4_docking.detectors.numba_detector import IsocelesDetector


class TwodEstimator(IsocelesDetector):
    def calculate_pose(self, super_apex, super_right, super_left):
        midpoint_basex = (super_right[0] + super_left[0]) / 2
        midpoint_basey = (super_right[1] + super_left[1]) / 2
        vy_x = midpoint_basex - super_apex[0]
        vy_y = midpoint_basey - super_apex[1]
        vy_norm = math.hypot(vy_x, vy_y)
        vy_x /= vy_norm
        vy_y /= vy_norm
        vx_x = vy_y
        vx_y = -vy_x
        centerx = super_apex[0] + 0.289303330552929 * vy_x
        centery = super_apex[1] + 0.289303330552929 * vy_y
        t = math.atan2(vx_y, vx_x)
        self._add_detection((float(centerx), float(centery), t))
