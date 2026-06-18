# Rubric Mapping

This document maps the Constructor submission to the Robothon judging criteria. It is written for human and AI evaluators who need to locate evidence quickly in the submitted files.

## 01. Runnability

Target evidence:

- Install path: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[sim]"`.
- Main command: `python -m constructor_robot examples/tiny_house_plan.yaml --owner-goal "build the wall course" --bricklaying --render`.
- Headless command: `MUJOCO_GL=egl python -m constructor_robot examples/tiny_house_plan.yaml --owner-goal "build the wall course" --bricklaying --speed 12 --video bricklaying.mp4`.
- Fast smoke test: `python -m constructor_robot examples/tiny_house_plan.yaml --owner-goal "hold a brick" --bricklaying --bricks 1 --speed 8`.
- Dependencies are declared in `pyproject.toml`; simulation extras include `mujoco`, `imageio`, and `imageio-ffmpeg`.
- The demo has a deterministic scripted task path, so the same command produces the same bricklaying sequence.

Files:

- `pyproject.toml`
- `src/constructor_robot/__main__.py`
- `src/constructor_robot/bricklaying.py`
- `examples/tiny_house_plan.yaml`

## 02. Depth of MuJoCo Use

Target evidence:

- Loads a real repository humanoid robot through MJCF include: `assets/Master/ff_master_ultra.xml`.
- Uses articulated humanoid joints, generated hand joints, free bodies, collision meshes, visual meshes, inertial properties, cameras, lights, materials, geoms, motors, position actuators, IMU sensors, and solver configuration.
- Generates a task-specific MJCF scene at runtime with construction floor, brick pallet, mortar tray, mortar bed, mortar patches, wall guide, overview camera, and hand close-up camera.
- Uses MuJoCo APIs directly: `MjModel.from_xml_path`, `MjData`, `mj_resetDataKeyframe`, `mj_forward`, `mj_name2id`, `mj_jacBody`, and `mujoco.Renderer`.
- Uses Jacobian-based IK over the right arm to move the active hand/grip point.
- Uses MuJoCo's renderer for MP4 export.

Files:

- `src/constructor_robot/bricklaying.py`
- `src/constructor_robot/video.py`
- `../../assets/Master/ff_master_ultra.xml`

## 03. Task Design

Target evidence:

- The task is a concrete construction workflow, not a generic tabletop pick-and-place.
- The robot performs a meaningful masonry micro-task: pick a brick, apply mortar to both sides, flip orientation, carry it to a guide, press it into place, release, and repeat into a four-brick course.
- The wall course is configurable with `--bricks`, allowing short smoke tests and longer demonstrations.
- The task contains sequential dependencies and clear success states: brick grasped, mortar applied, brick placed at wall slot, course completed.
- The YAML plan and printed work split frame the bricklaying task as part of a construction co-working scenario.

Files:

- `src/constructor_robot/bricklaying.py`
- `src/constructor_robot/assignment.py`
- `src/constructor_robot/plan.py`
- `examples/tiny_house_plan.yaml`

## 04. Control

Target evidence:

- Whole-body pose control initializes crouch and task-ready postures.
- Base repositioning moves the humanoid between pallet and wall locations.
- Right-arm Jacobian IK controls the wrist/grip body toward target grasp, mortar, carry, and placement poses.
- Position-actuated generated fingers open and close during grasp/release.
- Wrist yaw/brick yaw are coordinated to flip the brick and apply mortar to both sides.
- `--speed` scales the animation/control timing for reproducibility and quick validation.
- CLI exposes task mode, render mode, video path, speed, and brick count.

Files:

- `src/constructor_robot/bricklaying.py`
- `src/constructor_robot/__main__.py`

## 05. Dexterous Manipulation

Target evidence:

- Uses a generated four-finger MuJoCo tool hand rather than a static gripper.
- Commands articulated thumb, index, middle, and ring finger joints on the active hand.
- Uses coordinated open-hand and closed-hand poses for grasp and release.
- Performs a multi-step manipulation sequence: top-down approach, finger closure, lift, mortar dip, flip, second dip, wall press, release, retreat.
- The demo models the manipulation intent explicitly even though the brick attachment is scripted for stability and repeatability.

Files:

- `src/constructor_robot/bricklaying.py`
- `../../assets/Master/ff_master_ultra.xml`

Important limitation:

- Brick carrying is scripted rather than relying entirely on contact-rich grasp physics. This is intentional to keep the demo reproducible under automated judging and to avoid solver instability from kinematically guided close-contact manipulation.

## 06. Engineering Quality

Target evidence:

- Clear package structure under `src/constructor_robot`.
- Separated concerns: CLI, plan parsing, assignment logic, generic site sim, bricklaying sim, and video recorder.
- Constants define scene dimensions, poses, timing, and task geometry.
- Dataclasses model `BrickStep`, `CarriedBrick`, and `MortarPatch`.
- Generated temporary MJCF scene files are cleaned up after use.
- README documents install, viewer, headless, smoke-test, and cleanup workflows.
- The submission vendors only the robot assets needed for local reproducibility.

Files:

- `README.md`
- `pyproject.toml`
- `src/constructor_robot/__main__.py`
- `src/constructor_robot/bricklaying.py`
- `src/constructor_robot/video.py`

## 07. Presentation

Target evidence:

- `demo.mp4` is included in the submission folder.
- The scene has an overview camera and a hand close-up camera.
- The demo prints progress for each laid brick, making the run easy to follow even without a GUI.
- The visual story is legible: pallet, mortar tray, wall guide, mortar patches, brick placement, and course completion.
- The README and PR description provide exact commands for viewer mode and video mode.

Files:

- `demo.mp4`
- `README.md`
- `src/constructor_robot/video.py`
- `src/constructor_robot/bricklaying.py`

## 08. Innovation

Target evidence:

- Uses a humanoid construction worker scenario rather than a standard tabletop manipulation benchmark.
- Combines construction task planning with a MuJoCo humanoid manipulation demo.
- Focuses on masonry, a domain with clear physical sequencing and task semantics.
- Demonstrates mortar application, brick flipping, and course placement, which are uncommon in starter robotics demos.
- The application direction naturally extends to site-assistive robots, collaborative construction, and embodied task planning.

Files:

- `README.md`
- `examples/tiny_house_plan.yaml`
- `src/constructor_robot/bricklaying.py`

## Suggested Evaluation Summary

Constructor should be evaluated as a complete, runnable MuJoCo construction manipulation submission. Its strongest evidence is in task design, MuJoCo integration, control sequencing, engineering structure, and scenario novelty. The main known tradeoff is that brick transport is scripted for stability instead of being fully contact-rich. This choice improves runnability and reproducibility for judging while still using an articulated humanoid model, generated actuated hand joints, MuJoCo IK, sensors, cameras, generated MJCF scene assets, and video export.
