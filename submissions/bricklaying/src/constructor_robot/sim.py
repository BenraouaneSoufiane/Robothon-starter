from __future__ import annotations

import math
import time
from dataclasses import dataclass
from importlib import resources

from .assignment import AssignedTask
from .assignment import WorkSplit
from .plan import ConstructionPlan
from .video import VideoRecorder


ASSET_PACKAGE = "constructor_robot.assets"
ASSET_NAME = "construction_site.xml"
STORAGE_POSITION = (-2.0, -1.5)
HUMAN_ZONE = (0.45, 0.55, 0.8, 0.6)


@dataclass(frozen=True)
class RobotWaypoint:
    task_id: str
    label: str
    position: tuple[float, float]


def robot_waypoints(plan: ConstructionPlan, split: WorkSplit) -> list[RobotWaypoint]:
    waypoints: list[RobotWaypoint] = []
    for assignment in split.robot_tasks:
        zone = plan.zones[assignment.task.zone]
        waypoints.append(RobotWaypoint(assignment.task.id, assignment.task.title, zone.position))
    return waypoints


def run_simulation(
    plan: ConstructionPlan,
    split: WorkSplit,
    render: bool = False,
    speed: float = 1.0,
    video: str | None = None,
) -> None:
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is not installed. Install it with: pip install -e '.[sim]'"
        ) from exc

    with resources.as_file(resources.files(ASSET_PACKAGE).joinpath(ASSET_NAME)) as asset_path:
        model = mujoco.MjModel.from_xml_path(str(asset_path))
        data = mujoco.MjData(model)
        waypoints = robot_waypoints(plan, split)

        viewer = None
        if render:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(model, data)
        recorder = VideoRecorder(model, video, camera="site_overview") if video else None

        try:
            _set_robot_pose(model, data, STORAGE_POSITION)
            _set_cargo_pose(model, data, (-2.15, -1.5, 0.13))
            _set_arm(model, data, shoulder=-0.45, elbow=-0.25, wrist=0.25, gripper_open=True)
            _sync(model, data, viewer, recorder)

            for assignment in split.robot_tasks:
                zone = plan.zones[assignment.task.zone]
                _perform_robot_task(
                    model,
                    data,
                    assignment,
                    zone.position,
                    speed=speed,
                    viewer=viewer,
                    recorder=recorder,
                )
                print(f"robot completed: {assignment.task.title} ({assignment.task.id})")
            if recorder is not None:
                print(f"video saved: {recorder.path} ({recorder.frame_count} frames)")
            if viewer is not None:
                print("simulation complete; close the MuJoCo viewer window to exit")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(model.opt.timestep)
        finally:
            if recorder is not None:
                recorder.close()
            if viewer is not None:
                viewer.close()


def _perform_robot_task(
    model: object,
    data: object,
    assignment: AssignedTask,
    target: tuple[float, float],
    speed: float,
    viewer: object | None,
    recorder: VideoRecorder | None,
) -> None:
    if assignment.task.required_skill in {"carry_materials", "deliver_tools"}:
        _move_robot_to(model, data, STORAGE_POSITION, speed=speed, viewer=viewer, recorder=recorder)
        _pickup_cargo(model, data, viewer=viewer, recorder=recorder)
        _move_robot_to(model, data, target, speed=speed, viewer=viewer, recorder=recorder, carry_cargo=True)
        _place_cargo(model, data, target, viewer=viewer, recorder=recorder)
        return

    _move_robot_to(model, data, target, speed=speed, viewer=viewer, recorder=recorder)
    _animate_work_sweep(model, data, viewer=viewer, recorder=recorder)


def _pickup_cargo(
    model: object,
    data: object,
    viewer: object | None,
    recorder: VideoRecorder | None,
) -> None:
    for pose in (
        (-0.15, -0.95, 0.35, True),
        (0.28, -1.35, 0.15, True),
        (0.28, -1.35, 0.15, False),
        (-0.25, -0.65, 0.25, False),
    ):
        _animate_arm_to(model, data, *pose, viewer=viewer, recorder=recorder, cargo_attached=not pose[3])


def _place_cargo(
    model: object,
    data: object,
    target: tuple[float, float],
    viewer: object | None,
    recorder: VideoRecorder | None,
) -> None:
    for pose in (
        (-0.25, -0.65, 0.25, False),
        (0.3, -1.25, 0.2, False),
    ):
        _animate_arm_to(model, data, *pose, viewer=viewer, recorder=recorder, cargo_attached=True)
    _set_cargo_pose(model, data, (target[0] + 0.15, target[1] - 0.18, 0.13))
    _animate_arm_to(model, data, 0.3, -1.25, 0.2, True, viewer=viewer, recorder=recorder, cargo_attached=False)
    _animate_arm_to(model, data, -0.45, -0.25, 0.25, True, viewer=viewer, recorder=recorder, cargo_attached=False)


def _animate_work_sweep(
    model: object,
    data: object,
    viewer: object | None,
    recorder: VideoRecorder | None,
) -> None:
    for pose in (
        (-0.25, -0.95, 0.35, True),
        (0.15, -1.1, -0.2, True),
        (-0.05, -0.75, 0.55, True),
        (-0.45, -0.25, 0.25, True),
    ):
        _animate_arm_to(model, data, *pose, viewer=viewer, recorder=recorder, cargo_attached=False)


def _animate_arm_to(
    model: object,
    data: object,
    shoulder: float,
    elbow: float,
    wrist: float,
    gripper_open: bool,
    viewer: object | None,
    recorder: VideoRecorder | None,
    cargo_attached: bool,
) -> None:
    current = _arm_state(model, data)
    target = (shoulder, elbow, wrist, 0.04 if gripper_open else 0.0)
    for index in range(70):
        alpha = (index + 1) / 70
        eased = alpha * alpha * (3 - 2 * alpha)
        values = tuple(start + (end - start) * eased for start, end in zip(current, target, strict=True))
        _set_arm(
            model,
            data,
            shoulder=values[0],
            elbow=values[1],
            wrist=values[2],
            finger_gap=values[3],
        )
        if cargo_attached:
            _set_cargo_pose(model, data, _gripper_world_position(model, data))
        _sync(model, data, viewer, recorder)


def _move_robot_to(
    model: object,
    data: object,
    target: tuple[float, float],
    speed: float,
    viewer: object | None,
    recorder: VideoRecorder | None,
    carry_cargo: bool = False,
) -> None:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")
    qpos_start = model.jnt_qposadr[joint_id]
    path = _safe_path((float(data.qpos[qpos_start]), float(data.qpos[qpos_start + 1])), target)

    for target_x, target_y in path:
        for _ in range(450):
            x = data.qpos[qpos_start]
            y = data.qpos[qpos_start + 1]
            dx = target_x - x
            dy = target_y - y
            distance = math.hypot(dx, dy)
            if distance < 0.04:
                break

            step = min(0.025 * speed, distance)
            data.qpos[qpos_start] = x + step * dx / distance
            data.qpos[qpos_start + 1] = y + step * dy / distance
            data.qpos[qpos_start + 2] = 0.18
            yaw = math.atan2(dy, dx)
            data.qpos[qpos_start + 3 : qpos_start + 7] = _yaw_quaternion(yaw)
            if carry_cargo:
                _set_cargo_pose(model, data, _gripper_world_position(model, data))
            _sync(model, data, viewer, recorder)


def _safe_path(start: tuple[float, float], target: tuple[float, float]) -> list[tuple[float, float]]:
    if not _segment_crosses_human_zone(start, target):
        return [target]

    center_x, center_y, half_x, half_y = HUMAN_ZONE
    min_x = center_x - half_x - 0.35
    max_x = center_x + half_x + 0.35
    lower_y = center_y - half_y - 0.35
    upper_y = center_y + half_y + 0.35
    lower = [(min_x, lower_y), (max_x, lower_y), target]
    upper = [(min_x, upper_y), (max_x, upper_y), target]
    return min((lower, upper), key=lambda path: _path_length(start, path))


def _segment_crosses_human_zone(start: tuple[float, float], target: tuple[float, float]) -> bool:
    center_x, center_y, half_x, half_y = HUMAN_ZONE
    min_x = center_x - half_x
    max_x = center_x + half_x
    min_y = center_y - half_y
    max_y = center_y + half_y
    for index in range(31):
        alpha = index / 30
        x = start[0] + (target[0] - start[0]) * alpha
        y = start[1] + (target[1] - start[1]) * alpha
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return True
    return False


def _path_length(start: tuple[float, float], path: list[tuple[float, float]]) -> float:
    total = 0.0
    current = start
    for waypoint in path:
        total += math.dist(current, waypoint)
        current = waypoint
    return total


def _set_robot_pose(model: object, data: object, position: tuple[float, float]) -> None:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot_free")
    qpos_start = model.jnt_qposadr[joint_id]
    data.qpos[qpos_start : qpos_start + 7] = [position[0], position[1], 0.18, 1, 0, 0, 0]


def _set_cargo_pose(model: object, data: object, position: tuple[float, float, float]) -> None:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cargo_free")
    qpos_start = model.jnt_qposadr[joint_id]
    data.qpos[qpos_start : qpos_start + 7] = [position[0], position[1], position[2], 1, 0, 0, 0]


def _set_arm(
    model: object,
    data: object,
    shoulder: float,
    elbow: float,
    wrist: float,
    gripper_open: bool | None = None,
    finger_gap: float | None = None,
) -> None:
    gap = finger_gap if finger_gap is not None else (0.04 if gripper_open else 0.0)
    joint_values = {
        "shoulder": shoulder,
        "elbow": elbow,
        "wrist": wrist,
        "left_finger": gap,
        "right_finger": gap,
    }
    for joint_name, value in joint_values.items():
        _set_joint_qpos(model, data, joint_name, value)
    for actuator_name, value in {
        "shoulder_motor": shoulder,
        "elbow_motor": elbow,
        "wrist_motor": wrist,
        "left_finger_motor": gap,
        "right_finger_motor": gap,
    }.items():
        _set_ctrl(model, data, actuator_name, value)


def _arm_state(model: object, data: object) -> tuple[float, float, float, float]:
    return (
        _joint_qpos(model, data, "shoulder"),
        _joint_qpos(model, data, "elbow"),
        _joint_qpos(model, data, "wrist"),
        _joint_qpos(model, data, "left_finger"),
    )


def _joint_qpos(model: object, data: object, joint_name: str) -> float:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def _set_joint_qpos(model: object, data: object, joint_name: str, value: float) -> None:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    data.qpos[model.jnt_qposadr[joint_id]] = value


def _set_ctrl(model: object, data: object, actuator_name: str, value: float) -> None:
    import mujoco

    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    data.ctrl[actuator_id] = value


def _gripper_world_position(model: object, data: object) -> tuple[float, float, float]:
    import mujoco

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
    return tuple(float(value) for value in data.site_xpos[site_id])


def _yaw_quaternion(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def _sync(
    model: object,
    data: object,
    viewer: object | None,
    recorder: VideoRecorder | None,
) -> None:
    import mujoco

    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    if recorder is not None:
        recorder.record(data)
    if viewer is not None:
        viewer.sync()
        time.sleep(model.opt.timestep)
