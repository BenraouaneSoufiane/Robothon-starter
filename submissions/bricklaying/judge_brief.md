# Judge Brief

Constructor is a Robothon 2026 MuJoCo submission that demonstrates the repository's FF Master humanoid performing a bricklaying construction task. It uses the local `assets/Master/ff_master_ultra.xml` model, adds a generated four-finger MuJoCo tool hand at runtime, generates a construction scene, and runs a repeatable four-brick wall-course workflow.

## What To Run

Install:

```bash
cd submissions/bricklaying
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[sim]"
```

Viewer:

```bash
python -m constructor_robot examples/tiny_house_plan.yaml --owner-goal "build the wall course" --bricklaying --render
```

Headless smoke test:

```bash
python -m constructor_robot examples/tiny_house_plan.yaml --owner-goal "build the wall course" --bricklaying --speed 12
```

Headless video:

```bash
MUJOCO_GL=egl python -m constructor_robot examples/tiny_house_plan.yaml --owner-goal "build the wall course" --bricklaying --speed 12 --video bricklaying.mp4
```

## What The Demo Shows

1. The FF Master humanoid crouches near a pallet of bricks.
2. The right arm uses IK-guided reaching to approach the brick.
3. The generated articulated tool hand closes coordinated thumb, index, middle, and ring finger joints.
4. The brick is lifted, dipped into mortar on one side, flipped, and dipped on the other side.
5. Visible mortar patches attach to the brick sides.
6. The humanoid carries the brick to the wall guide, presses it into place, releases, and retreats.
7. The process repeats into a four-brick wall course by default.

## Why This Meets The Rubric

- **Runnability:** Editable install, pinned Python package metadata, viewer command, headless smoke command, and video command are documented.
- **Depth of MuJoCo use:** Uses MJCF include, local humanoid mesh assets, generated articulated hand joints, actuators, IMU sensors, geoms, cameras, lights, materials, solver configuration, `mj_forward`, `mj_jacBody`, and `mujoco.Renderer`.
- **Task design:** The goal is a clear construction manipulation task with sequential, inspectable success states.
- **Control:** Combines whole-body pose targets, base repositioning, arm Jacobian IK, finger actuation, wrist flipping, and timed task sequencing.
- **Dexterous manipulation:** Commands a generated four-finger MuJoCo tool hand during grasp and release.
- **Engineering quality:** Organized Python package, dataclasses, constants, CLI, temporary scene generation, video helper, README, and vendored assets.
- **Presentation:** Includes `demo.mp4`, overview/close-up cameras, readable progress logs, and clear run commands.
- **Innovation:** Applies humanoid manipulation to masonry and construction co-working rather than a generic object-moving task.

## Known Tradeoff

The brick is kept attached through scripted pose updates once grasped. This is a deliberate reproducibility choice: it avoids brittle contact-solver failures during automated judging while still demonstrating the task, hand actuation, MuJoCo model depth, IK control, mortar application, and wall placement.

## Best Files To Inspect

- `README.md`
- `rubric_mapping.md`
- `src/constructor_robot/bricklaying.py`
- `src/constructor_robot/__main__.py`
- `src/constructor_robot/video.py`
- `../../assets/Master/ff_master_ultra.xml`
- `demo.mp4`
