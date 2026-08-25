import time
from loguru import logger

from stretch4_body.robot.robot_client import RobotClient
from stretch4_pyhesai_wrapper import stream_lidar_left

from stretch4_docking.trackers import DockTracker
from stretch4_docking.costmap import Costmap, RingFilter, LIDAR_LEFT_ORIGIN, LIDAR_RIGHT_ORIGIN
from stretch4_docking.servo import XYThetaServo, Mppi


def autodock(robot):
    tracker = DockTracker()
    tracker.warm_start()

    ring_filter = RingFilter()
    ring_filter.warm_start()

    costmapper = Costmap()
    costmapper.warm_start()

    servo = XYThetaServo()

    control_rate = 10.0
    last_loop_time = None
    deadline = time.time() + 25.0

    for frame in stream_lidar_left():
        if time.time() > deadline:
            logger.error("Timed out")
            return
        if frame is None:
            continue

        loop_start = time.perf_counter()
        if last_loop_time is not None:
            dt = loop_start - last_loop_time
            if dt > 0:
                rate = 1.0 / dt
                control_rate = 0.1 * rate + 0.9 * control_rate
            if dt > 0.25:
                logger.warn('Loop latency too high, may see abnormal behavior')
        last_loop_time = loop_start
        t0 = time.perf_counter()

        # Put clouds in base_link
        print(f"Points shape: {frame.points.shape}, timestamp: {frame.timestamp}")
        points = np.vstack([left_frame.points, right_frame.points])
        t1 = time.perf_counter()

        tracker.identify(points)
        if tracker.is_tracking():
            logger.warn("No dock seen...")
            continue

        dock_pose = tracker.get_pose()
        t2 = time.perf_counter()

        left = ring_filter.process(left_frame.points, sensor_origin=LIDAR_LEFT_ORIGIN)
        right = ring_filter.process(right_frame.points, sensor_origin=LIDAR_RIGHT_ORIGIN)
        filtered_xyz = np.vstack([left[:, :3], right[:, :3]])
        t3 = time.perf_counter()

        costmap = costmapper.process(filtered_xyz, dock_pose=dock_pose)
        t4 = time.perf_counter()

        x, y, z, qx, qy, qz, qw = dock_pose
        rot, t = Rotation.from_quat([qx, qy, qz, qw]), np.array([x, y, z]) 

        # Warn about excessive dock tilt (roll or pitch)
        roll, pitch, _ = rot.as_euler('xyz', degrees=True)
        if abs(roll) > 5.0 or abs(pitch) > 5.0:
            logger.warn(f"Unsafe dock alignment! Roll: {roll:.1f}°, Pitch: {pitch:.1f}°")

        # Predock pose 55cm in front of the dock
        p_local = np.array([0.0, -0.55, 0.0])
        p_robot = rot.apply(p_local) + t

        errx, erry, errt = p_robot[0], p_robot[1], rot.as_euler('xyz')[2]
        vx, vy, wz = servo.step(errx, erry, errt)

        filtered = filter_clearance_velocity(vx, vy, wz, costmap.obstacle_xy, costmap.cliff_xy)
        robot.set_velocity(filtered.vx, filtered.vy, filtered.wz)
        robot.push_command()
        t5 = time.perf_counter()

        logger.debug(
            f"Rate: {self.cloud_cb_rate:.1f}Hz | "
            f"ps: {(t1 - t0) * 1000.0:.0f}ms | "
            f"ID: {(t2 - t1) * 1000.0:.0f}ms | "
            f"Ring: {(t3 - t2) * 1000.0:.0f}ms | "
            f"Map: {(t4 - t3) * 1000.0:.0f}ms | "
            f"u: {(t5 - t4) * 1000.0:.0f}ms | "
            f"Total: {(t5 - loop_start) * 1000.0:.0f}ms"
        )

        # 5mm / 1deg tolerance
        if abs(errx) < 0.005 and abs(erry) < 0.005 and abs(errt) < 0.0175:
            self.robot.routines.routine_blind_dock()
            return

def main():
    robot = RobotClient()
    robot.startup()
    try:
        autodock(robot)
    finally:
        robot.base.enable_freewheel_mode()
        robot.push_command()
        robot.stop()

