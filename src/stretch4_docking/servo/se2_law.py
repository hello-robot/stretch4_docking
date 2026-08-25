import math


class XYThetaServo:
    def __init__(self, kp=0.4, kp_th=1.0, max_vel=0.1, max_omega=0.5):
        self.kp = kp
        self.kp_th = kp_th

        self.max_vel = max_vel
        self.max_omega = max_omega

    def _axis_pid(self, err, kp):
        # P only
        p = kp * err

        return p

    def step(self, err_x, err_y, err_theta):
        err_theta = math.atan2(math.sin(err_theta), math.cos(err_theta))
        vx = self._axis_pid(err_x, self.kp)
        vy = self._axis_pid(err_y, self.kp)
        omega = self._axis_pid(err_theta, self.kp_th)

        speed = math.hypot(vx, vy)
        if speed > self.max_vel:
            scale = self.max_vel / speed
            vx *= scale
            vy *= scale
        if abs(omega) > self.max_omega:
            omega = math.copysign(self.max_omega, omega)

        return vx, vy, omega