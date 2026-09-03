import math
import time


class XYThetaServo:
    def __init__(self, kp=0.4, kp_th=1.0, max_vel=0.1, max_omega=0.5,
                 ki=0.25, ki_th=0.6,
                 max_integral_vel=0.05, max_integral_omega=0.25,
                 integral_band=0.10, integral_band_th=0.15,
                 max_dt=0.5):
        self.kp = kp
        self.kp_th = kp_th

        self.max_vel = max_vel
        self.max_omega = max_omega

        self.ki = ki
        self.ki_th = ki_th
        self.max_integral_vel = max_integral_vel
        self.max_integral_omega = max_integral_omega
        self.integral_band = integral_band
        self.integral_band_th = integral_band_th

        self.max_dt = max_dt

        self.reset()

    def reset(self):
        """Drop the wind-up. Call when starting or restarting a servo attempt."""
        self._i_x = 0.0
        self._i_y = 0.0
        self._i_th = 0.0
        self._prev_x = None
        self._prev_y = None
        self._prev_th = None
        self._last_step = None

    def _integrate(self, state, err, prev_err, dt, band, ki, limit):
        if prev_err is not None and err * prev_err < 0.0:
            return 0.0  # crossed the goal; whatever is wound up is now pushing the wrong way
        if abs(err) > band:
            return 0.0  # still travelling, not stalled
        state += err * dt
        # Clamp the state rather than the term, so the stored value cannot run away
        # behind a saturated output and take time to unwind later.
        bound = limit / ki
        return max(-bound, min(bound, state))

    def _axis_pid(self, err, kp, integral, ki):
        return kp * err + ki * integral

    def step(self, err_x, err_y, err_theta):
        err_theta = math.atan2(math.sin(err_theta), math.cos(err_theta))

        now = time.monotonic()
        dt = 0.0 if self._last_step is None else now - self._last_step
        if dt > self.max_dt:
            # A gap this long means the servo was not running -- the dock was lost and
            # renavigated to, or a new goal arrived. Whatever is wound up describes a
            # situation that no longer holds.
            self.reset()
            dt = 0.0
        self._last_step = now

        if dt > 0.0:
            self._i_x = self._integrate(self._i_x, err_x, self._prev_x, dt,
                                        self.integral_band, self.ki, self.max_integral_vel)
            self._i_y = self._integrate(self._i_y, err_y, self._prev_y, dt,
                                        self.integral_band, self.ki, self.max_integral_vel)
            self._i_th = self._integrate(self._i_th, err_theta, self._prev_th, dt,
                                         self.integral_band_th, self.ki_th,
                                         self.max_integral_omega)
        self._prev_x, self._prev_y, self._prev_th = err_x, err_y, err_theta

        vx = self._axis_pid(err_x, self.kp, self._i_x, self.ki)
        vy = self._axis_pid(err_y, self.kp, self._i_y, self.ki)
        omega = self._axis_pid(err_theta, self.kp_th, self._i_th, self.ki_th)

        speed = math.hypot(vx, vy)
        if speed > self.max_vel:
            scale = self.max_vel / speed
            vx *= scale
            vy *= scale
        if abs(omega) > self.max_omega:
            omega = math.copysign(self.max_omega, omega)

        return vx, vy, omega