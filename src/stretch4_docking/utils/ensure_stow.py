from loguru import logger


STOW_LINEAR_TOL_M = 0.02
STOW_ANGULAR_TOL_RAD = 0.2
STOW_TOLERANCES = {
    'lift': STOW_LINEAR_TOL_M,
    'arm': STOW_LINEAR_TOL_M,
    'wrist_yaw': STOW_ANGULAR_TOL_RAD,
    'wrist_pitch': STOW_ANGULAR_TOL_RAD,
    'wrist_roll': STOW_ANGULAR_TOL_RAD,
}

JOINT_STATE_NAMES = {
    'lift': ('lift_joint',),
    'arm': ('arm_l1_joint', 'arm_l2_joint', 'arm_l3_joint', 'arm_l4_joint'),
    'wrist_yaw': ('wrist_yaw_joint',),
    'wrist_pitch': ('wrist_pitch_joint',),
    'wrist_roll': ('wrist_roll_joint',),
}


def stow_targets(tool=None):
    """The current tool's stow configuration, read from RobotParams"""
    from stretch4_body.core.robot_params import RobotParams

    _, params = RobotParams.get_params()
    tool = tool or params['robot']['tool']
    return dict(params.get(tool, {}).get('stow', {}))


def positions_from_joint_state(name_to_position):
    """Stow-relevant joint positions from a ROS /joint_states name -> position mapping."""
    positions = {}
    for joint, names in JOINT_STATE_NAMES.items():
        values = [name_to_position[name] for name in names if name in name_to_position]
        if len(values) == len(names):
            positions[joint] = sum(values)
    return positions


def positions_from_robot(robot):
    """Stow-relevant joint positions from a live RobotClient."""
    positions = {}
    for joint in STOW_TOLERANCES:
        subsystem = getattr(robot, joint, None)
        if subsystem is None:
            subsystem = getattr(getattr(robot, 'end_of_arm', None), joint, None)
        if subsystem is None:
            continue  # this tool doesn't carry the joint
        positions[joint] = subsystem.status['pos']
    return positions


def joints_out_of_stow(positions, targets=None):
    """Which of `positions` sit outside stow tolerance, as {joint: (position, target)}.

    Joints missing from either the tool's stow config or `positions` are skipped: grippers carry
    a stow entry but no comparable joint position, and not every tool carries every joint.
    """
    if targets is None:
        targets = stow_targets()

    off = {}
    for joint, tolerance in STOW_TOLERANCES.items():
        if joint not in targets or positions.get(joint) is None:
            continue
        position, target = positions[joint], targets[joint]
        if abs(position - target) > tolerance:
            off[joint] = (position, target)
    return off


def describe(off):
    """Render joints_out_of_stow output for a log line or error message."""
    return ", ".join(
        f"{joint} at {position:.3f}, want {target:.3f}"
        for joint, (position, target) in sorted(off.items())
    )


def ensure_stowed(robot):
    """Stow if any joint is out of place, then confirm it took. False if it did not."""
    robot.pull_status()
    targets = stow_targets()
    if not joints_out_of_stow(positions_from_robot(robot), targets):
        return True

    robot.stow()
    robot.pull_status()
    off = joints_out_of_stow(positions_from_robot(robot), targets)
    if off:
        logger.error(f"Stow did not take ({describe(off)}), refusing to dock")
        return False
    return True
