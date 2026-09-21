import time
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from stretch4_body.robot.robot_client import RobotClient
from stretch4_pyhesai_wrapper import stream_lidar_both
from stretch4_pyhesai_wrapper.ptc_client import (
    LEFT_LIDAR_IP,
    RIGHT_LIDAR_IP,
    get_return_mode,
    set_return_mode,
    RETURN_MODE_NAMES,
)

from stretch4_docking.trackers import DockTracker
from stretch4_docking.costmap import Costmap, RingFilter, filter_clearance_velocity, LIDAR_LEFT_ORIGIN, LIDAR_RIGHT_ORIGIN, LIDAR_LEFT_ROT, LIDAR_RIGHT_ROT
from stretch4_docking.servo import XYThetaServo, Mppi
from stretch4_docking.utils import cloud_reader, ensure_stow



def autodock(robot):
    if robot.power_periph.status['adapter_voltage_present']:
        logger.info("Already charging.")
        return

    stiffness_rise_per_s = 1.0
    contact_settle_s = 0.3
    _stiffness = 0.0
    _contact_since = None

    def slew_stiffness(target, elapsed_s):
        """Track the plant's stiffness target, instantly down and slowly up.

        Softening is inaudible, so a decrease is applied as soon as it is asked
        for. Firming up is not: the steppers make a noise when the gains jump,
        which is exactly what happens on the frame after the robot backs away
        from the dock or the goal is lost. Increases are therefore slewed at
        stiffness_rise_per_s.
        """
        nonlocal _stiffness
        target = float(np.clip(target, 0.0, 1.0))
        if target <= _stiffness:
            _stiffness = target
        else:
            step = stiffness_rise_per_s * max(elapsed_s, 0.0)
            _stiffness = min(target, _stiffness + step)
        return _stiffness

    def contact_settled():
        """True once adapter_voltage_present has held for contact_settle_s.

        A single high reading is not a dock: the plate makes and breaks contact
        as it slides down the rails, and latching on the first sample stops the
        base mid-seat with the charger dropping in and out. Any low reading
        restarts the clock, so only an uninterrupted run counts.
        """
        nonlocal _contact_since
        robot.pull_status()
        if not robot.power_periph.status['adapter_voltage_present']:
            _contact_since = None
            return False
        now = time.perf_counter()
        if _contact_since is None:
            _contact_since = now
            return False
        return (now - _contact_since) >= contact_settle_s

    logger.info("Warm starting...")
    warm_start_start = time.perf_counter()

    tracker = DockTracker()
    tracker.warm_start()

    ring_filter = RingFilter()
    ring_filter.warm_start()

    costmapper = Costmap()
    costmapper.warm_start()

    cloud_reader.warm_start()

    logger.info(f"Warm start took {time.perf_counter() - warm_start_start:.1f}s")

    use_mppi = Mppi.is_online()
    servo_law = Mppi() if use_mppi else XYThetaServo()
    if use_mppi:
        servo_law.connect()
    logger.info("GPU docking" if use_mppi else "CPU docking")
    last_request_time = None

    measured_control_rate = 10.0
    last_loop_time = None
    deadline = time.time() + 25.0

    for pair in stream_lidar_both():
        if time.time() > deadline:
            logger.error("Timed out")
            return
        if pair is None:
            robot.base.enable_freewheel_mode()
            robot.push_command()
            continue
        left_frame, right_frame = pair

        loop_start = time.perf_counter()
        if last_loop_time is not None:
            dt = loop_start - last_loop_time
            if dt > 0:
                rate = 1.0 / dt
                measured_control_rate = 0.1 * rate + 0.9 * measured_control_rate
            if dt > 0.25:
                logger.warning('Loop latency too high, may see abnormal behavior')
        last_loop_time = loop_start
        t0 = time.perf_counter()

        # Put both clouds in base_link, stacked into one [x, y, z, intensity, ring]
        # buffer. A fresh buffer each cycle, since tracker.identify() holds on to
        # a reference to what it was given.
        n_left = len(left_frame.points)
        n_right = len(right_frame.points)
        points = np.empty((n_left + n_right, 5), dtype=np.float32)
        cloud_reader.transform_frame_points(
            left_frame, LIDAR_LEFT_ROT, LIDAR_LEFT_ORIGIN, points, 0)
        cloud_reader.transform_frame_points(
            right_frame, LIDAR_RIGHT_ROT, LIDAR_RIGHT_ORIGIN, points, n_left)
        # The ring filter wants xyzr, the dock tracker xyzi, so both read their
        # columns straight out of the shared buffer.
        left_points = points[:n_left]
        right_points = points[n_left:]
        t1 = time.perf_counter()

        tracker.identify(points[:, :4])
        if not tracker.is_tracking():
            logger.warning("No dock seen...")
            continue
        dock_pose = tracker.get_pose()
        t2 = time.perf_counter()

        filtered_left_points = ring_filter.process(left_points, sensor_origin=LIDAR_LEFT_ORIGIN)
        filtered_right_points = ring_filter.process(right_points, sensor_origin=LIDAR_RIGHT_ORIGIN)
        filtered_xyz = np.vstack([filtered_left_points[:, :3], filtered_right_points[:, :3]])
        t3 = time.perf_counter()

        costmap = costmapper.process(filtered_xyz, dock_pose=dock_pose)
        t4 = time.perf_counter()

        # Predock pose 55cm in front of the dock
        x, y, z, qx, qy, qz, qw = dock_pose
        rot, t = Rotation.from_quat([qx, qy, qz, qw]), np.array([x, y, z]) 
        p_local = np.array([0.0, -0.55, 0.0])
        p_robot = rot.apply(p_local) + t

        errx, erry, errt = p_robot[0], p_robot[1], rot.as_euler('xyz')[2]
        elapsed_s = 0.0
        now = time.perf_counter()
        if last_request_time is not None:
            elapsed_s = now - last_request_time
        vx, vy, wz, stiffness = servo_law.step(
            err_x=errx, err_y=erry, err_theta=errt,
            costmap=costmap.costmap,
            origin_x=float(costmap.origin),
            origin_y=float(costmap.origin),
            resolution=float(costmap.resolution),
            elapsed_s=elapsed_s,
        )
        last_request_time = now
        if not use_mppi:
            filtered = filter_clearance_velocity(vx, vy, wz, costmap.obstacle_xy, costmap.cliff_xy)
            vx, vy, wz = filtered.vx, filtered.vy, filtered.wz
        stiffness = slew_stiffness(stiffness, elapsed_s)
        robot.base.set_velocity(vx, vy, wz, stiffness=stiffness)
        robot.push_command()
        t5 = time.perf_counter()

        logger.debug(
            f"Rate: {measured_control_rate:.1f}Hz | "
            f"ps: {(t1 - t0) * 1000.0:.0f}ms | "
            f"ID: {(t2 - t1) * 1000.0:.0f}ms | "
            f"Ring: {(t3 - t2) * 1000.0:.0f}ms | "
            f"Map: {(t4 - t3) * 1000.0:.0f}ms | "
            f"u: {(t5 - t4) * 1000.0:.0f}ms | "
            f"Total: {(t5 - loop_start) * 1000.0:.0f}ms"
        )

        if not use_mppi:
            # 5mm / 1deg tolerance
            if abs(errx) < 0.005 and abs(erry) < 0.005 and abs(errt) < 0.0175:
                ret = robot.routines.routine_blind_dock()
                logger.info("Success!" if ret else "Failure")
                return
        else:
            if contact_settled():
                robot.base.enable_freewheel_mode()
                robot.push_command()
                logger.info("Success!")
                return

def main():
    robot = RobotClient()
    robot.startup()
    assert robot.is_homed()
    assert ensure_stow.ensure_stowed(robot)
    modes_before = None
    try:
        # Set LiDARs to single-return if needed
        lmode = get_return_mode(LEFT_LIDAR_IP)
        rmode = get_return_mode(RIGHT_LIDAR_IP)
        if RETURN_MODE_NAMES.get(lmode, 'unknown') != 'strongest' or \
           RETURN_MODE_NAMES.get(rmode, 'unknown') != 'strongest':
            logger.warning("Switching LiDARs temporarily to single-return")
            modes_before = (lmode, rmode)
            set_return_mode(LEFT_LIDAR_IP, 1) # 1 -> strongest
            set_return_mode(RIGHT_LIDAR_IP, 1)

        autodock(robot)
    finally:
        robot.base.enable_freewheel_mode()
        robot.push_command()
        robot.stop()
        if modes_before is not None:
            logger.warning("Switching LiDARs back to prior return-mode")
            lmode, rmode = modes_before
            set_return_mode(LEFT_LIDAR_IP, lmode)
            set_return_mode(RIGHT_LIDAR_IP, rmode)

if __name__ == "__main__":
    main()
