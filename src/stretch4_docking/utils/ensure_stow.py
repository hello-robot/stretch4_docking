STOW_LINEAR_TOL_M = 0.02
STOW_ANGULAR_TOL_RAD = 0.1
STOW_TOLERANCES = {
    'lift': STOW_LINEAR_TOL_M,
    'arm': STOW_LINEAR_TOL_M,
    'wrist_yaw': STOW_ANGULAR_TOL_RAD,
    'wrist_pitch': STOW_ANGULAR_TOL_RAD,
    'wrist_roll': STOW_ANGULAR_TOL_RAD,
}


def joints_out_of_stow(robot):
    # Targets come from the mounted tool's own stow config, since a tablet or
    # calibration tool stows nowhere near where the nil tool does. Grippers are
    # left out: they carry a stow entry but not a comparable joint position.
    stow_cfg = robot.end_of_arm.params.get('stow', {})
    off = {}
    for joint, tolerance in STOW_TOLERANCES.items():
        if joint not in stow_cfg:
            continue
        subsystem = getattr(robot, joint, None)
        if subsystem is None:
            subsystem = getattr(robot.end_of_arm, joint, None)
        if subsystem is None:
            continue  # this tool doesn't carry the joint
        pos, target = subsystem.status['pos'], stow_cfg[joint]
        if abs(pos - target) > tolerance:
            off[joint] = (pos, target)
    return off


def ensure_stowed(robot):
    # Docking servos the base with the arm out of the picture: an extended arm
    # or a raised lift is both a tipping hazard and a lidar occlusion, so square
    # that away before driving anywhere.
    robot.pull_status()
    off = joints_out_of_stow(robot)
    if not off:
        return True

    robot.stow()
    robot.pull_status()
    off = joints_out_of_stow(robot)
    if off:
        logger.error("Stow did not take ({}), refusing to dock".format(
            ", ".join(f"{j} at {pos:.3f}, want {target:.3f}" for j, (pos, target) in off.items())))
        return False
    return True
