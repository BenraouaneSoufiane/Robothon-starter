from __future__ import annotations

import math
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from .video import VideoRecorder


SUBMISSION_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SUBMISSION_ROOT.parents[1]
HUMANOID_MODEL = REPO_ROOT / "assets" / "Master" / "ff_master_ultra.xml"
DEFAULT_BRICK_COUNT = 4
BRICK_SIZE = (0.115, 0.055, 0.06)
WALL_ORIGIN = (0.72, -0.42, BRICK_SIZE[2])
PALLET_POSITION = (-0.3, -1.25, 0.025)
PALLET_SIZE = (0.42, 0.12, 0.025)
PALLET_TOP_Z = PALLET_POSITION[2] + PALLET_SIZE[2]
MORTAR_TRAY_POSITION = (-0.08, -0.92, 0.02)
MORTAR_TRAY_SIZE = (0.24, 0.15, 0.02)
MORTAR_BED_SIZE = (0.2, 0.1, 0.012)
MORTAR_BED_POSITION = (
    MORTAR_TRAY_POSITION[0],
    MORTAR_TRAY_POSITION[1],
    MORTAR_TRAY_POSITION[2] + MORTAR_TRAY_SIZE[2] + MORTAR_BED_SIZE[2],
)
MORTAR_PATCH_SIZE = (0.006, BRICK_SIZE[1] * 0.86, BRICK_SIZE[2] * 0.76)
MORTAR_PATCH_LOCAL_X = BRICK_SIZE[0] + MORTAR_PATCH_SIZE[0]
MORTAR_DIP_Z = MORTAR_BED_POSITION[2] + MORTAR_BED_SIZE[2] + BRICK_SIZE[2] * 0.42
STACK_FRONT_CLEARANCE = 0.005
STACK_ORIGIN = (
    PALLET_POSITION[0],
    PALLET_POSITION[1] + PALLET_SIZE[1] - BRICK_SIZE[1] - STACK_FRONT_CLEARANCE,
    PALLET_TOP_Z + BRICK_SIZE[2],
)
STACK_ROW_SPACING = BRICK_SIZE[0] * 2 + 0.04
STACK_ROW_OFFSETS = (0.0, -STACK_ROW_SPACING, STACK_ROW_SPACING)
STACK_LAYER_HEIGHT = BRICK_SIZE[2] * 2 + 0.012
BRICK_GAP = 0.012
COURSE_GAP = 0.014
ACTIVE_GRIP_BODY = "right_wrist_roll_link"
ACTIVE_TOOL_HAND_BODY = "right_tool_hand"
ACTIVE_TOOL_HAND_FREEJOINT = "right_tool_hand_free"
ACTIVE_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
HAND_JOINT_ORDER = (
    "hand_thumb_0_joint",
    "hand_thumb_1_joint",
    "hand_thumb_2_joint",
    "hand_index_0_joint",
    "hand_index_1_joint",
    "hand_middle_0_joint",
    "hand_middle_1_joint",
    "hand_ring_0_joint",
    "hand_ring_1_joint",
)
PALM_DOWN_WORLD_AXIS = (0.0, 0.0, -1.0)
PALM_LOCAL_AXIS = (0.0, 1.0, 0.0)
PALM_CONTACT_LOCAL_OFFSET = (0.12, 0.025, 0.0)
BRICK_HAND_VISUAL_COMPRESSION = 0.006
TOUCH_DEMO_SPEED_SCALE = 0.35
TOUCH_SETTLE_FRAMES = 600
MORTAR_PASTE_FRAMES = 170
WALL_PLACE_APPROACH_Z = 0.16
WALL_PRESS_FRAMES = 90
BRICK_TOUCH_OVERTRAVEL = 0.0
TOUCH_BASE_FORWARD_OFFSET = -0.34
TOUCH_BASE_SIDE_OFFSET = 0.32
TOUCH_BASE_HEIGHT_ABOVE_BRICK = 0.29
PICKUP_STEP_HORIZONTAL_ADVANCE = 0.06
PICKUP_VERTICAL_CONTACT_OVERTRAVEL = 0.08
PICKUP_STEP_COUNT = 3
_ACTIVE_RECORDER: VideoRecorder | None = None

TOUCH_CROUCH_POSE = {
    "left_hip_pitch_joint": -1.45,
    "right_hip_pitch_joint": -1.45,
    "left_knee_joint": 2.62,
    "right_knee_joint": 2.62,
    "left_ankle_pitch_joint": -0.86,
    "right_ankle_pitch_joint": -0.86,
    "waist_pitch_joint": 0.5,
}

# Three visible pickup steps:
# 1. upper arm straight from the shoulder,
# 2. forearm horizontal at a 90 degree elbow bend,
# 3. hand vertical at a 90 degree wrist bend.
BRICK_PICKUP_ARM_POSES = {
    "upper_arm_straight": {
        "waist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.34,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0.0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    },
    "forearm_horizontal": {
        "waist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.34,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": math.pi / 2,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    },
    "hand_vertical": {
        "waist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": -0.34,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": math.pi / 2,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": -math.pi / 2,
        "right_wrist_yaw_joint": 0.0,
    },
}


@dataclass(frozen=True)
class BrickStep:
    brick_name: str
    mortar_name: str
    stack_position: tuple[float, float, float]
    wall_position: tuple[float, float, float]
    course: int
    index: int


@dataclass(frozen=True)
class CarriedBrick:
    brick_name: str
    grip_local_offset: tuple[float, float, float]
    yaw: float = math.pi / 2


@dataclass(frozen=True)
class MortarPatch:
    side_name: str
    side_sign: int


def run_bricklaying_demo(
    render: bool = False,
    speed: float = 1.0,
    video: str | None = None,
    brick_count: int = DEFAULT_BRICK_COUNT,
) -> None:
    global _ACTIVE_RECORDER

    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is not installed. Install it with: pip install -e '.[sim]'"
        ) from exc

    if not HUMANOID_MODEL.exists():
        raise RuntimeError(
            "FF Master humanoid MuJoCo model was not found. Expected it at "
            f"{HUMANOID_MODEL}."
        )
    if brick_count < 1:
        raise ValueError("brick_count must be at least 1")

    steps = brick_steps(brick_count)
    scene_file = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=HUMANOID_MODEL.parent,
        prefix="constructor_bricklaying_",
        suffix=".xml",
        delete=False,
    )
    try:
        scene_path = Path(scene_file.name)
        scene_file.write(_scene_xml(steps))
        scene_file.close()
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        _set_g1_pose(model, data, _touch_robot_base_position(steps[0].stack_position), yaw=0.0)
        _set_full_body_pose(model, data, _pose("touch_ready"))
        _reset_scene_props(model, data, steps)

        viewer = None
        if render:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(model, data)
        video_stride = max(1, round(speed))
        recorder = (
            VideoRecorder(model, video, camera="bricklaying_overview", sample_stride=video_stride)
            if video
            else None
        )
        previous_recorder = _ACTIVE_RECORDER
        _ACTIVE_RECORDER = recorder

        try:
            _sync(model, data, viewer)
            for index, step in enumerate(steps, start=1):
                _perform_brick_step(model, data, step, speed=speed, viewer=viewer)
                print(f"laid brick {index}/{len(steps)}: course {step.course + 1}, slot {step.index + 1}")
            print(
                "run summary: "
                f"humanoid={HUMANOID_MODEL.name}, "
                f"bricks={len(steps)}, "
                f"mortar_patches={len(steps) * 2}, "
                f"arm_ik_joints={len(ACTIVE_ARM_JOINTS)}, "
                f"tool_hand_joints={len(HAND_JOINT_ORDER)}"
            )
            if viewer is not None:
                print("bricklaying course complete; close the MuJoCo viewer window after your screenshot")
            else:
                print("bricklaying course complete")
            if recorder is not None:
                print(f"video saved: {recorder.path} ({recorder.frame_count} frames)")
            if viewer is not None:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(model.opt.timestep)
        finally:
            _ACTIVE_RECORDER = previous_recorder
            if recorder is not None:
                recorder.close()
            if viewer is not None:
                viewer.close()
    finally:
        scene_file.close()
        Path(scene_file.name).unlink(missing_ok=True)


def brick_steps(count: int) -> tuple[BrickStep, ...]:
    steps: list[BrickStep] = []
    for number in range(count):
        course = number // 4
        index = number % 4
        stagger = 0.5 if course % 2 else 0.0
        wall_y = WALL_ORIGIN[1] + (index + stagger) * (BRICK_SIZE[1] * 2 + BRICK_GAP)
        wall_z = WALL_ORIGIN[2] + course * (BRICK_SIZE[2] * 2 + COURSE_GAP)
        stack_row = number % len(STACK_ROW_OFFSETS)
        stack_col = number // 3
        stack_x = STACK_ORIGIN[0] + STACK_ROW_OFFSETS[stack_row]
        stack_z = STACK_ORIGIN[2] + stack_col * STACK_LAYER_HEIGHT
        steps.append(
            BrickStep(
                brick_name=f"brick_{number:02d}",
                mortar_name=f"mortar_{number:02d}",
                stack_position=(stack_x, STACK_ORIGIN[1], stack_z),
                wall_position=(WALL_ORIGIN[0], wall_y, wall_z),
                course=course,
                index=index,
            )
        )
    return tuple(steps)


def _touch_robot_base_position(brick_position: tuple[float, float, float]) -> tuple[float, float, float]:
    brick_top = brick_position[2] + BRICK_SIZE[2]
    return (
        brick_position[0] + TOUCH_BASE_FORWARD_OFFSET,
        brick_position[1] + TOUCH_BASE_SIDE_OFFSET,
        brick_top + TOUCH_BASE_HEIGHT_ABOVE_BRICK,
    )


def _perform_brick_step(
    model: object,
    data: object,
    step: BrickStep,
    speed: float,
    viewer: object | None,
) -> None:
    touch_speed = speed * TOUCH_DEMO_SPEED_SCALE
    top_grasp_position = _brick_top_grasp_position(step.stack_position)

    _move_brick(model, data, step.brick_name, step.stack_position, viewer=viewer)
    for pose_name in ("pickup_upper_arm_straight", "pickup_forearm_horizontal"):
        _transition_pose(model, data, _active_arm_pose(pose_name), speed=touch_speed, viewer=viewer)
    _transition_to_top_down_pickup_pose(
        model,
        data,
        top_grasp_position,
        _active_arm_pose("pickup_hand_vertical"),
        speed=touch_speed,
        viewer=viewer,
    )
    grip_local_offset = _top_contact_brick_local_offset()
    carried_brick = CarriedBrick(step.brick_name, grip_local_offset)
    _transition_pose(model, data, _active_hand_pose(open_hand=False), speed=touch_speed, viewer=viewer, carried_brick=carried_brick)
    mortar_patches: list[MortarPatch] = []
    _paste_brick_side(
        model,
        data,
        carried_brick,
        side_name=f"{step.mortar_name}_first",
        side_sign=1,
        attached_patches=mortar_patches,
        speed=touch_speed,
        viewer=viewer,
    )
    mortar_patches.append(MortarPatch(f"{step.mortar_name}_first", 1))
    carried_brick = CarriedBrick(step.brick_name, grip_local_offset, yaw=carried_brick.yaw + math.pi)
    _flip_gripped_brick(
        model,
        data,
        carried_brick,
        attached_patches=mortar_patches,
        speed=touch_speed,
        viewer=viewer,
    )
    _paste_brick_side(
        model,
        data,
        carried_brick,
        side_name=f"{step.mortar_name}_second",
        side_sign=-1,
        attached_patches=mortar_patches,
        speed=touch_speed,
        viewer=viewer,
    )
    mortar_patches.append(MortarPatch(f"{step.mortar_name}_second", -1))
    _place_brick_in_wall(
        model,
        data,
        step,
        carried_brick,
        attached_patches=mortar_patches,
        speed=touch_speed,
        viewer=viewer,
    )
    _pause_with_placed_brick(
        model,
        data,
        step,
        carried_brick,
        frames=max(60, round(TOUCH_SETTLE_FRAMES / max(speed, 0.1))),
        viewer=viewer,
        attached_patches=mortar_patches,
    )


def _brick_top_grasp_position(brick_position: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        brick_position[0],
        brick_position[1],
        brick_position[2] + BRICK_SIZE[2] - BRICK_TOUCH_OVERTRAVEL,
    )


def _scene_xml(steps: tuple[BrickStep, ...]) -> str:
    bricks = "\n".join(_free_box(step.brick_name, step.stack_position, BRICK_SIZE, "brick_mat", mass=2.2) for step in steps)
    mortar_patches = "\n".join(
        _free_box(
            f"{step.mortar_name}_{side}_side",
            (0.0, 0.0, -0.4 - index * 0.05),
            MORTAR_PATCH_SIZE,
            "mortar_mat",
            mass=0.01,
        )
        for index, (step, side) in enumerate((step, side) for step in steps for side in ("first", "second"))
    )
    return textwrap.dedent(
        f"""\
        <mujoco model="ff_master_bricklaying_process">
          <include file="ff_master_ultra.xml"/>
          <option integrator="Euler" solver="CG" iterations="80" tolerance="1e-8"/>

          <statistic center="0.55 0 0.7" extent="1.4"/>
          <visual>
            <headlight diffuse="0.7 0.7 0.7" ambient="0.25 0.25 0.25" specular="0.4 0.4 0.4"/>
            <global azimuth="125" elevation="-18" offwidth="1280" offheight="720"/>
          </visual>

          <asset>
            <texture name="site_grid" type="2d" builtin="checker" rgb1="0.36 0.37 0.33" rgb2="0.27 0.29 0.27" width="512" height="512"/>
            <material name="floor_mat" texture="site_grid" texrepeat="7 7" reflectance="0.08"/>
            <material name="brick_mat" rgba="0.66 0.24 0.14 1"/>
            <material name="mortar_mat" rgba="0.78 0.75 0.67 1"/>
            <material name="guide_mat" rgba="0.92 0.82 0.55 0.24"/>
            <material name="pallet_mat" rgba="0.38 0.26 0.14 1"/>
            <material name="tray_mat" rgba="0.22 0.23 0.22 1"/>
          </asset>

          <worldbody>
            <geom name="construction_floor" type="plane" size="2.2 2.2 0.05" material="floor_mat"/>
            <light name="work_light" pos="0 -1.5 2.8" dir="0 0 -1" diffuse="0.8 0.8 0.75"/>
            <camera name="bricklaying_overview" pos="-0.75 -2.75 1.65" xyaxes="0.97 -0.24 0 0.13 0.53 0.84" fovy="58"/>
            <camera name="hand_closeup" pos="-0.45 -1.22 0.58" xyaxes="0.99 -0.13 0 0.04 0.31 0.95" fovy="42"/>

            <body name="brick_pallet" pos="{PALLET_POSITION[0]} {PALLET_POSITION[1]} {PALLET_POSITION[2]}">
              <geom type="box" size="{PALLET_SIZE[0]} {PALLET_SIZE[1]} {PALLET_SIZE[2]}" material="pallet_mat"/>
            </body>
            <body name="mortar_tray" pos="{MORTAR_TRAY_POSITION[0]} {MORTAR_TRAY_POSITION[1]} {MORTAR_TRAY_POSITION[2]}">
              <geom type="box" size="{MORTAR_TRAY_SIZE[0]} {MORTAR_TRAY_SIZE[1]} {MORTAR_TRAY_SIZE[2]}" material="tray_mat"/>
            </body>
            <body name="mortar_bed" pos="{MORTAR_BED_POSITION[0]} {MORTAR_BED_POSITION[1]} {MORTAR_BED_POSITION[2]}">
              <geom type="box" size="{MORTAR_BED_SIZE[0]} {MORTAR_BED_SIZE[1]} {MORTAR_BED_SIZE[2]}" material="mortar_mat"/>
            </body>
            <body name="wall_course_guide" pos="{WALL_ORIGIN[0]} {WALL_ORIGIN[1] + 0.18} {WALL_ORIGIN[2] - BRICK_SIZE[2] - 0.01}">
              <geom type="box" size="{BRICK_SIZE[0] * 1.08} 0.36 0.01" material="guide_mat"/>
            </body>
            {_tool_hand_xml()}
            {bricks}
            {mortar_patches}
          </worldbody>
          <actuator>
            {_tool_hand_actuator_xml()}
          </actuator>
        </mujoco>
        """
    )


def _tool_hand_xml() -> str:
    digits = []
    specs = (
        ("thumb", 0.055, -0.035, -0.3),
        ("index", 0.026, -0.014, 0.0),
        ("middle", 0.026, 0.014, 0.0),
        ("ring", 0.022, 0.039, 0.12),
    )
    for name, length, y_offset, yaw in specs:
        thumb_tip = ""
        if name == "thumb":
            thumb_tip = f"""
                <body name="right_tool_{name}_pad" pos="{length * 0.72} 0 0">
                  <joint name="right_hand_{name}_2_joint" type="hinge" axis="0 1 0" range="-0.9 0.9" damping="0.08"/>
                  <geom name="right_tool_{name}_pad" type="capsule" fromto="0 0 0 {length * 0.48} 0 0" size="0.006" rgba="0.96 0.82 0.58 1"/>
                </body>
            """
        digits.append(
            f"""
            <body name="right_tool_{name}_base" pos="0.075 {y_offset} -0.008" euler="0 0 {yaw}">
              <joint name="right_hand_{name}_0_joint" type="hinge" axis="0 1 0" range="-0.75 0.75" damping="0.08"/>
              <geom name="right_tool_{name}_proximal" type="capsule" fromto="0 0 0 {length} 0 0" size="0.009" rgba="0.92 0.92 0.88 1"/>
              <body name="right_tool_{name}_tip" pos="{length} 0 0">
                <joint name="right_hand_{name}_1_joint" type="hinge" axis="0 1 0" range="-0.9 0.9" damping="0.08"/>
                <geom name="right_tool_{name}_distal" type="capsule" fromto="0 0 0 {length * 0.72} 0 0" size="0.007" rgba="0.96 0.82 0.58 1"/>
                {thumb_tip}
              </body>
            </body>
            """
        )
    return textwrap.dedent(
        f"""
        <body name="{ACTIVE_TOOL_HAND_BODY}" pos="0 0 -0.5">
          <freejoint name="{ACTIVE_TOOL_HAND_FREEJOINT}"/>
          <geom name="right_tool_palm" type="box" size="0.045 0.044 0.015" rgba="0.18 0.19 0.2 1"/>
          {''.join(digits)}
        </body>
        """
    )


def _tool_hand_actuator_xml() -> str:
    actuators = []
    for joint_suffix in HAND_JOINT_ORDER:
        joint_name = f"right_{joint_suffix}"
        actuators.append(
            f'<position name="{joint_name}" joint="{joint_name}" kp="12" dampratio="1" ctrlrange="-0.9 0.9"/>'
        )
    return "\n".join(actuators)


def _free_box(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    material: str,
    mass: float,
) -> str:
    return (
        f'<body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">'
        f'<freejoint name="{name}_free"/>'
        f'<geom name="{name}_geom" type="box" size="{size[0]} {size[1]} {size[2]}" material="{material}" '
        f'mass="{mass}" contype="0" conaffinity="0"/>'
        "</body>"
    )


def _pose(name: str) -> dict[str, float]:
    base = {
        "left_hip_pitch_joint": 0.0,
        "left_hip_roll_joint": 0.0,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.0,
        "left_ankle_pitch_joint": 0.0,
        "left_ankle_roll_joint": 0.0,
        "right_hip_pitch_joint": 0.0,
        "right_hip_roll_joint": 0.0,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.0,
        "right_ankle_pitch_joint": 0.0,
        "right_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.18,
        "left_shoulder_pitch_joint": 0.26,
        "left_shoulder_roll_joint": 0.34,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 1.18,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.26,
        "right_shoulder_roll_joint": -0.34,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 1.18,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }
    base.update(_hand_pose(open_hand=True))

    variants = {
        "ready": {},
        "touch_ready": {
            **TOUCH_CROUCH_POSE,
            "waist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.26,
            "right_shoulder_roll_joint": -0.34,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 1.18,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        },
        "pickup_upper_arm_straight": {
            **TOUCH_CROUCH_POSE,
            **BRICK_PICKUP_ARM_POSES["upper_arm_straight"],
        },
        "pickup_forearm_horizontal": {
            **TOUCH_CROUCH_POSE,
            **BRICK_PICKUP_ARM_POSES["forearm_horizontal"],
        },
        "pickup_hand_vertical": {
            **TOUCH_CROUCH_POSE,
            **BRICK_PICKUP_ARM_POSES["hand_vertical"],
        },
        "reach_stack": {
            "waist_yaw_joint": -0.45,
            "waist_pitch_joint": 0.26,
            "right_shoulder_pitch_joint": 0.72,
            "right_shoulder_roll_joint": -0.82,
            "right_shoulder_yaw_joint": -0.28,
            "right_elbow_joint": 1.55,
            "right_wrist_pitch_joint": -0.45,
        },
        "grasp": {
            "waist_yaw_joint": -0.45,
            "waist_pitch_joint": 0.28,
            "right_shoulder_pitch_joint": 0.75,
            "right_shoulder_roll_joint": -0.82,
            "right_shoulder_yaw_joint": -0.28,
            "right_elbow_joint": 1.6,
            "right_wrist_pitch_joint": -0.35,
            **_hand_pose(open_hand=False),
        },
        "carry": {
            "waist_yaw_joint": 0.0,
            "waist_pitch_joint": 0.12,
            "right_shoulder_pitch_joint": 0.1,
            "right_shoulder_roll_joint": -0.62,
            "right_shoulder_yaw_joint": -0.08,
            "right_elbow_joint": 1.35,
            "right_wrist_pitch_joint": -0.1,
            **_hand_pose(open_hand=False),
        },
        "mortar": {
            "waist_yaw_joint": 0.35,
            "waist_pitch_joint": 0.22,
            "left_shoulder_pitch_joint": 0.58,
            "left_shoulder_roll_joint": 0.7,
            "left_shoulder_yaw_joint": 0.2,
            "left_elbow_joint": 1.35,
            "left_wrist_pitch_joint": -0.35,
            "right_shoulder_pitch_joint": 0.18,
            "right_shoulder_roll_joint": -0.58,
            "right_elbow_joint": 1.22,
            **_hand_pose(open_hand=False),
        },
        "place": {
            "waist_yaw_joint": 0.42,
            "waist_pitch_joint": 0.34,
            "right_shoulder_pitch_joint": 0.78,
            "right_shoulder_roll_joint": -0.75,
            "right_shoulder_yaw_joint": 0.24,
            "right_elbow_joint": 1.58,
            "right_wrist_pitch_joint": -0.42,
            **_hand_pose(open_hand=False),
        },
        "release": {
            "waist_yaw_joint": 0.42,
            "waist_pitch_joint": 0.32,
            "right_shoulder_pitch_joint": 0.72,
            "right_shoulder_roll_joint": -0.75,
            "right_shoulder_yaw_joint": 0.24,
            "right_elbow_joint": 1.45,
            "right_wrist_pitch_joint": -0.25,
            **_hand_pose(open_hand=True),
        },
    }
    base.update(variants[name])
    return base


def _active_arm_pose(name: str) -> dict[str, float]:
    return {joint: value for joint, value in _pose(name).items() if joint in ACTIVE_ARM_JOINTS}


def _active_hand_pose(open_hand: bool) -> dict[str, float]:
    active_side = ACTIVE_GRIP_BODY.split("_", 1)[0]
    return {joint: value for joint, value in _hand_pose(open_hand).items() if joint.startswith(f"{active_side}_hand_")}


def _hand_pose(open_hand: bool) -> dict[str, float]:
    open_values = {
        "right": (-0.52, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    }
    closed_values = {
        "right": (-0.2, -0.58, -0.68, 0.48, 0.62, 0.5, 0.64, 0.44, 0.58),
    }
    selected = open_values if open_hand else closed_values
    pose: dict[str, float] = {}
    for side, values in selected.items():
        for joint_suffix, value in zip(HAND_JOINT_ORDER, values, strict=True):
            pose[f"{side}_{joint_suffix}"] = value
    return pose


def _transition_pose(
    model: object,
    data: object,
    target: dict[str, float],
    speed: float,
    viewer: object | None,
    carried_brick: CarriedBrick | None = None,
) -> None:
    current = {joint: _joint_qpos(model, data, joint) for joint in target}
    frames = max(8, round(38 / max(speed, 0.1)))
    for frame in range(frames):
        alpha = (frame + 1) / frames
        eased = alpha * alpha * (3 - 2 * alpha)
        for joint, value in target.items():
            blended = current[joint] + (value - current[joint]) * eased
            _set_joint_qpos(model, data, joint, blended)
            _set_ctrl(model, data, joint, blended)
        if carried_brick is not None:
            _move_carried_brick_to_grip(model, data, carried_brick)
        _sync(model, data, viewer)


def _transition_to_top_down_pickup_pose(
    model: object,
    data: object,
    top_grasp_position: tuple[float, float, float],
    posture: dict[str, float],
    speed: float,
    viewer: object | None,
) -> None:
    start = _body_local_to_world(model, data, ACTIVE_GRIP_BODY, PALM_CONTACT_LOCAL_OFFSET)
    target_position = _contact_adjusted_grasp_position(start, top_grasp_position)
    frames = max(8, round(38 / max(speed, 0.1)))
    for frame in range(frames):
        alpha = (frame + 1) / frames
        eased = alpha * alpha * (3 - 2 * alpha)
        target = tuple(start[index] + (target_position[index] - start[index]) * eased for index in range(3))
        _solve_active_grip_top_down_ik(model, data, target, posture=posture)
        _sync(model, data, viewer)


def _contact_adjusted_grasp_position(
    start: tuple[float, float, float],
    brick_top: tuple[float, float, float],
) -> tuple[float, float, float]:
    dx = brick_top[0] - start[0]
    dy = brick_top[1] - start[1]
    distance = math.hypot(dx, dy)
    horizontal_advance = PICKUP_STEP_HORIZONTAL_ADVANCE * PICKUP_STEP_COUNT
    if distance > 1e-6:
        dx = dx / distance * horizontal_advance
        dy = dy / distance * horizontal_advance
    else:
        dx = 0.0
        dy = 0.0
    return (
        brick_top[0] + dx,
        brick_top[1] + dy,
        brick_top[2] - PICKUP_VERTICAL_CONTACT_OVERTRAVEL,
    )


def _set_full_body_pose(model: object, data: object, pose: dict[str, float]) -> None:
    for joint, value in pose.items():
        _set_joint_qpos(model, data, joint, value)
        _set_ctrl(model, data, joint, value)


def _pause_with_gripped_brick(
    model: object,
    data: object,
    carried_brick: CarriedBrick,
    frames: int,
    viewer: object | None,
    attached_patches: list[MortarPatch] | None = None,
) -> None:
    for _ in range(frames):
        _move_carried_brick_to_grip(model, data, carried_brick)
        _update_mortar_patches(model, data, carried_brick, attached_patches or [])
        _sync(model, data, viewer)


def _pause_with_placed_brick(
    model: object,
    data: object,
    step: BrickStep,
    carried_brick: CarriedBrick,
    frames: int,
    viewer: object | None,
    attached_patches: list[MortarPatch],
) -> None:
    for _ in range(frames):
        _set_free_body_pose(model, data, f"{carried_brick.brick_name}_free", step.wall_position, yaw=carried_brick.yaw)
        _update_mortar_patches(model, data, carried_brick, attached_patches)
        _sync(model, data, viewer)


def _paste_brick_side(
    model: object,
    data: object,
    carried_brick: CarriedBrick,
    side_name: str,
    side_sign: int,
    attached_patches: list[MortarPatch],
    speed: float,
    viewer: object | None,
) -> None:
    above_bed = (MORTAR_BED_POSITION[0], MORTAR_BED_POSITION[1], MORTAR_DIP_Z + 0.12)
    dipped = (MORTAR_BED_POSITION[0], MORTAR_BED_POSITION[1], MORTAR_DIP_Z)
    _move_gripped_brick_center_to(
        model,
        data,
        carried_brick,
        above_bed,
        speed=speed,
        viewer=viewer,
        attached_patches=attached_patches,
    )
    _move_gripped_brick_center_to(
        model,
        data,
        carried_brick,
        dipped,
        speed=speed,
        viewer=viewer,
        attached_patches=attached_patches,
    )
    _pause_with_gripped_brick(
        model,
        data,
        carried_brick,
        frames=max(45, round(MORTAR_PASTE_FRAMES / max(speed, 0.1))),
        viewer=viewer,
        attached_patches=attached_patches,
    )
    _set_mortar_patch_pose(model, data, side_name, dipped, carried_brick.yaw, side_sign)
    _pause_with_gripped_brick(model, data, carried_brick, frames=45, viewer=viewer, attached_patches=attached_patches)
    _move_gripped_brick_center_to(
        model,
        data,
        carried_brick,
        above_bed,
        speed=speed,
        viewer=viewer,
        attached_patches=[*attached_patches, MortarPatch(side_name, side_sign)],
    )
    _set_mortar_patch_pose(model, data, side_name, above_bed, carried_brick.yaw, side_sign)


def _place_brick_in_wall(
    model: object,
    data: object,
    step: BrickStep,
    carried_brick: CarriedBrick,
    attached_patches: list[MortarPatch],
    speed: float,
    viewer: object | None,
) -> None:
    _move_g1_base_to(
        model,
        data,
        _touch_robot_base_position(step.wall_position),
        yaw=0.0,
        speed=speed,
        viewer=viewer,
        carried_brick=carried_brick,
        attached_patches=attached_patches,
    )
    above_wall = (
        step.wall_position[0],
        step.wall_position[1],
        step.wall_position[2] + WALL_PLACE_APPROACH_Z,
    )
    _move_gripped_brick_center_to(
        model,
        data,
        carried_brick,
        above_wall,
        speed=speed,
        viewer=viewer,
        attached_patches=attached_patches,
    )
    _move_gripped_brick_center_to(
        model,
        data,
        carried_brick,
        step.wall_position,
        speed=speed,
        viewer=viewer,
        attached_patches=attached_patches,
    )
    _pause_with_gripped_brick(
        model,
        data,
        carried_brick,
        frames=max(30, round(WALL_PRESS_FRAMES / max(speed, 0.1))),
        viewer=viewer,
        attached_patches=attached_patches,
    )
    _set_free_body_pose(model, data, f"{carried_brick.brick_name}_free", step.wall_position, yaw=carried_brick.yaw)
    _update_mortar_patches(model, data, carried_brick, attached_patches)
    _transition_pose(model, data, _active_hand_pose(open_hand=True), speed=speed, viewer=viewer)
    _move_grip_by_delta(
        model,
        data,
        delta=(-0.08, 0.0, 0.12),
        speed=speed,
        viewer=viewer,
    )


def _flip_gripped_brick(
    model: object,
    data: object,
    carried_brick: CarriedBrick,
    attached_patches: list[MortarPatch],
    speed: float,
    viewer: object | None,
) -> None:
    frames = max(16, round(46 / max(speed, 0.1)))
    start_yaw = carried_brick.yaw - math.pi
    current_center = _body_position(model, data, carried_brick.brick_name)
    for frame in range(frames):
        alpha = (frame + 1) / frames
        eased = alpha * alpha * (3 - 2 * alpha)
        yaw = start_yaw + (carried_brick.yaw - start_yaw) * eased
        _set_free_body_pose(model, data, f"{carried_brick.brick_name}_free", current_center, yaw=yaw)
        flipped_brick = CarriedBrick(carried_brick.brick_name, carried_brick.grip_local_offset, yaw=yaw)
        _update_mortar_patches(model, data, flipped_brick, attached_patches)
        _sync(model, data, viewer)


def _move_g1_base_to(
    model: object,
    data: object,
    target_position: tuple[float, float, float],
    yaw: float,
    speed: float,
    viewer: object | None,
    carried_brick: CarriedBrick | None = None,
    attached_patches: list[MortarPatch] | None = None,
) -> None:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    qpos_start = model.jnt_qposadr[joint_id]
    start_position = tuple(float(value) for value in data.qpos[qpos_start : qpos_start + 3])
    frames = max(18, round(70 / max(speed, 0.1)))
    for frame in range(frames):
        alpha = (frame + 1) / frames
        eased = alpha * alpha * (3 - 2 * alpha)
        position = tuple(
            start_position[index] + (target_position[index] - start_position[index]) * eased
            for index in range(3)
        )
        data.qpos[qpos_start : qpos_start + 7] = [position[0], position[1], position[2], *_yaw_quaternion(yaw)]
        if carried_brick is not None:
            _move_carried_brick_to_grip(model, data, carried_brick)
            _update_mortar_patches(model, data, carried_brick, attached_patches or [])
        _sync(model, data, viewer)


def _move_grip_by_delta(
    model: object,
    data: object,
    delta: tuple[float, float, float],
    speed: float,
    viewer: object | None,
) -> None:
    start_grip = _body_local_to_world(model, data, ACTIVE_GRIP_BODY, PALM_CONTACT_LOCAL_OFFSET)
    target = tuple(start_grip[index] + delta[index] for index in range(3))
    frames = max(12, round(34 / max(speed, 0.1)))
    for frame in range(frames):
        alpha = (frame + 1) / frames
        eased = alpha * alpha * (3 - 2 * alpha)
        grip_position = tuple(
            start_grip[index] + (target[index] - start_grip[index]) * eased
            for index in range(3)
        )
        _solve_active_grip_top_down_ik(model, data, grip_position)
        _sync(model, data, viewer)


def _move_gripped_brick_center_to(
    model: object,
    data: object,
    carried_brick: CarriedBrick,
    target_center: tuple[float, float, float],
    speed: float,
    viewer: object | None,
    attached_patches: list[MortarPatch] | None = None,
) -> None:
    start_center = _body_position(model, data, carried_brick.brick_name)
    start_grip = _body_position(model, data, ACTIVE_GRIP_BODY)
    delta = tuple(target_center[index] - start_center[index] for index in range(3))
    target_grip = tuple(start_grip[index] + delta[index] for index in range(3))
    frames = max(12, round(42 / max(speed, 0.1)))
    for frame in range(frames):
        alpha = (frame + 1) / frames
        eased = alpha * alpha * (3 - 2 * alpha)
        target = tuple(start_grip[index] + (target_grip[index] - start_grip[index]) * eased for index in range(3))
        _solve_active_grip_top_down_ik(model, data, target)
        _move_carried_brick_to_grip(model, data, carried_brick)
        _update_mortar_patches(model, data, carried_brick, attached_patches or [])
        _sync(model, data, viewer)


def _body_position(model: object, data: object, body_name: str) -> tuple[float, float, float]:
    import mujoco

    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return tuple(float(value) for value in data.xpos[body_id])


def _grip_to_brick_local_offset(
    model: object,
    data: object,
    brick_position: tuple[float, float, float],
) -> tuple[float, float, float]:
    return _world_to_body_local_offset(model, data, ACTIVE_GRIP_BODY, brick_position)


def _top_contact_brick_local_offset() -> tuple[float, float, float]:
    contact_depth = BRICK_SIZE[2] - BRICK_HAND_VISUAL_COMPRESSION
    return tuple(
        PALM_CONTACT_LOCAL_OFFSET[index] + PALM_LOCAL_AXIS[index] * contact_depth
        for index in range(3)
    )


def _move_brick_to_grip(
    model: object,
    data: object,
    brick_name: str,
    grip_local_offset: tuple[float, float, float],
) -> None:
    brick_position = _body_local_to_world(model, data, ACTIVE_GRIP_BODY, grip_local_offset)
    _set_free_body_pose(model, data, f"{brick_name}_free", brick_position, yaw=math.pi / 2)


def _move_carried_brick_to_grip(model: object, data: object, carried_brick: CarriedBrick) -> None:
    brick_position = _body_local_to_world(model, data, ACTIVE_GRIP_BODY, carried_brick.grip_local_offset)
    _set_free_body_pose(model, data, f"{carried_brick.brick_name}_free", brick_position, yaw=carried_brick.yaw)


def _set_mortar_patch_pose(
    model: object,
    data: object,
    side_name: str,
    brick_position: tuple[float, float, float],
    brick_yaw: float,
    side_sign: int,
) -> None:
    offset_x = side_sign * MORTAR_PATCH_LOCAL_X
    patch_position = (
        brick_position[0] + math.cos(brick_yaw) * offset_x,
        brick_position[1] + math.sin(brick_yaw) * offset_x,
        brick_position[2],
    )
    _set_free_body_pose(model, data, f"{side_name}_side_free", patch_position, yaw=brick_yaw)


def _update_mortar_patches(
    model: object,
    data: object,
    carried_brick: CarriedBrick,
    patches: list[MortarPatch],
) -> None:
    if not patches:
        return

    brick_position = _body_position(model, data, carried_brick.brick_name)
    for patch in patches:
        _set_mortar_patch_pose(model, data, patch.side_name, brick_position, carried_brick.yaw, patch.side_sign)


def _world_to_body_local_offset(
    model: object,
    data: object,
    body_name: str,
    world_position: tuple[float, float, float],
) -> tuple[float, float, float]:
    body_position = _body_position(model, data, body_name)
    rotation = _body_rotation(model, data, body_name)
    delta = tuple(world_position[index] - body_position[index] for index in range(3))
    return (
        rotation[0] * delta[0] + rotation[3] * delta[1] + rotation[6] * delta[2],
        rotation[1] * delta[0] + rotation[4] * delta[1] + rotation[7] * delta[2],
        rotation[2] * delta[0] + rotation[5] * delta[1] + rotation[8] * delta[2],
    )


def _body_local_to_world(
    model: object,
    data: object,
    body_name: str,
    local_offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    body_position = _body_position(model, data, body_name)
    rotation = _body_rotation(model, data, body_name)
    return (
        body_position[0]
        + rotation[0] * local_offset[0]
        + rotation[1] * local_offset[1]
        + rotation[2] * local_offset[2],
        body_position[1]
        + rotation[3] * local_offset[0]
        + rotation[4] * local_offset[1]
        + rotation[5] * local_offset[2],
        body_position[2]
        + rotation[6] * local_offset[0]
        + rotation[7] * local_offset[1]
        + rotation[8] * local_offset[2],
    )


def _body_rotation(model: object, data: object, body_name: str) -> tuple[float, ...]:
    import mujoco

    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return tuple(float(value) for value in data.xmat[body_id])


def _solve_active_grip_top_down_ik(
    model: object,
    data: object,
    target: tuple[float, float, float],
    posture: dict[str, float] | None = None,
) -> None:
    import mujoco
    import numpy as np

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ACTIVE_GRIP_BODY)
    dof_ids = tuple(_joint_dof_id(model, joint) for joint in ACTIVE_ARM_JOINTS)
    qpos_ids = tuple(_joint_qpos_id(model, joint) for joint in ACTIVE_ARM_JOINTS)
    lower = tuple(float(model.jnt_range[_joint_id(model, joint)][0]) for joint in ACTIVE_ARM_JOINTS)
    upper = tuple(float(model.jnt_range[_joint_id(model, joint)][1]) for joint in ACTIVE_ARM_JOINTS)
    target_array = np.array(target)
    desired_axis = np.array(PALM_DOWN_WORLD_AXIS)
    palm_axis = np.array(PALM_LOCAL_AXIS)
    local_offset = np.array(PALM_CONTACT_LOCAL_OFFSET)
    posture_values = np.array(
        [posture[joint] if posture and joint in posture else _joint_qpos(model, data, joint) for joint in ACTIVE_ARM_JOINTS]
    )
    posture_weights = np.diag(
        [
            0.05,
            0.05,
            0.05,
            0.16,
            0.08,
            0.18,
            0.08,
        ]
    )

    for _ in range(34):
        mujoco.mj_forward(model, data)
        rotation = data.xmat[body_id].reshape(3, 3)
        grasp_point = data.xpos[body_id] + rotation @ local_offset
        position_error = target_array - grasp_point
        hand_axis = rotation @ palm_axis
        orientation_error = np.cross(hand_axis, desired_axis)
        current_qpos = np.array([data.qpos[qpos_id] for qpos_id in qpos_ids])
        posture_error = posture_values - current_qpos

        if (
            float(np.linalg.norm(position_error)) < 0.005
            and float(np.linalg.norm(orientation_error)) < 0.035
            and (posture is None or abs(float(posture_error[3])) < 0.08)
        ):
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        point_offset = rotation @ local_offset
        point_jac = jacp - _skew(point_offset) @ jacr
        jac = np.vstack(
            (
                point_jac[:, dof_ids],
                0.35 * jacr[:, dof_ids],
                posture_weights,
            )
        )
        error = np.concatenate((position_error, 0.35 * orientation_error, posture_weights @ posture_error))
        damping = 0.045
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damping * damping * np.eye(jac.shape[0]), error)
        delta = np.clip(delta, -0.07, 0.07)

        for index, qpos_id in enumerate(qpos_ids):
            value = float(data.qpos[qpos_id] + delta[index])
            value = min(max(value, lower[index]), upper[index])
            data.qpos[qpos_id] = value
            _set_ctrl(model, data, ACTIVE_ARM_JOINTS[index], value)
    mujoco.mj_forward(model, data)


def _skew(vector: object) -> object:
    import numpy as np

    return np.array(
        (
            (0.0, -vector[2], vector[1]),
            (vector[2], 0.0, -vector[0]),
            (-vector[1], vector[0], 0.0),
        )
    )


def _set_g1_pose(model: object, data: object, position: tuple[float, float, float], yaw: float) -> None:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    qpos_start = model.jnt_qposadr[joint_id]
    data.qpos[qpos_start : qpos_start + 7] = [position[0], position[1], position[2], *_yaw_quaternion(yaw)]


def _reset_scene_props(model: object, data: object, steps: tuple[BrickStep, ...]) -> None:
    for step in steps:
        _set_free_body_pose(model, data, f"{step.brick_name}_free", step.stack_position, yaw=math.pi / 2)


def _move_brick(
    model: object,
    data: object,
    brick_name: str,
    position: tuple[float, float, float],
    viewer: object | None,
) -> None:
    _set_free_body_pose(model, data, f"{brick_name}_free", position, yaw=math.pi / 2)
    _sync(model, data, viewer)


def _set_free_body_pose(
    model: object,
    data: object,
    joint_name: str,
    position: tuple[float, float, float],
    yaw: float,
) -> None:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_start = model.jnt_qposadr[joint_id]
    data.qpos[qpos_start : qpos_start + 7] = [position[0], position[1], position[2], *_yaw_quaternion(yaw)]


def _joint_qpos(model: object, data: object, joint_name: str) -> float:
    return float(data.qpos[_joint_qpos_id(model, joint_name)])


def _set_joint_qpos(model: object, data: object, joint_name: str, value: float) -> None:
    data.qpos[_joint_qpos_id(model, joint_name)] = value


def _joint_id(model: object, joint_name: str) -> int:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"MuJoCo joint not found: {joint_name}")
    return joint_id


def _joint_qpos_id(model: object, joint_name: str) -> int:
    return int(model.jnt_qposadr[_joint_id(model, joint_name)])


def _joint_dof_id(model: object, joint_name: str) -> int:
    return int(model.jnt_dofadr[_joint_id(model, joint_name)])


def _set_ctrl(model: object, data: object, actuator_name: str, value: float) -> None:
    import mujoco

    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{actuator_name}")
    if actuator_id < 0:
        return
    data.ctrl[actuator_id] = value


def _yaw_quaternion(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]


def _sync(model: object, data: object, viewer: object | None) -> None:
    import mujoco

    _update_tool_hand_pose(model, data)
    mujoco.mj_forward(model, data)
    if _ACTIVE_RECORDER is not None:
        _ACTIVE_RECORDER.record(data)
    if viewer is not None:
        viewer.sync()
        time.sleep(model.opt.timestep)


def _update_tool_hand_pose(model: object, data: object) -> None:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ACTIVE_TOOL_HAND_FREEJOINT)
    if joint_id < 0:
        return

    wrist_position = _body_local_to_world(model, data, ACTIVE_GRIP_BODY, (0.02, 0.0, -0.02))
    wrist_rotation = _body_rotation(model, data, ACTIVE_GRIP_BODY)
    yaw = math.atan2(wrist_rotation[3], wrist_rotation[0])
    qpos_start = model.jnt_qposadr[joint_id]
    data.qpos[qpos_start : qpos_start + 7] = [wrist_position[0], wrist_position[1], wrist_position[2], *_yaw_quaternion(yaw)]
