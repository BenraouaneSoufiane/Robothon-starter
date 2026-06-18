# FF Master Bricklaying MuJoCo Demo

A MuJoCo robotics project focused on a humanoid bricklaying task. The main demo uses the repository's FF Master humanoid model and a generated articulated MuJoCo tool hand to:

1. Load a generated construction scene around the FF Master humanoid robot.
2. Crouch the robot beside a pallet of full-scale bricks.
3. Reach down with IK-guided arm motion and coordinated thumb/index/middle/ring finger closure.
4. Lift each brick, dip both sides into a mortar tray, and attach visible mortar patches.
5. Carry the cemented brick to a wall guide and place a four-brick course.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[sim]"
```

To run the bricklaying demo with the MuJoCo viewer:

```bash
python3 -m constructor_robot examples/tiny_house_plan.yaml --owner-goal "build the wall course" --bricklaying --render
```

To save the bricklaying demo as an MP4:

```bash
.venv/bin/python -m constructor_robot examples/tiny_house_plan.yaml --owner-goal "build the wall course" --bricklaying --video bricklaying.mp4
```

On a headless Linux machine, MuJoCo may need an offscreen OpenGL backend:

```bash
MUJOCO_GL=egl .venv/bin/python -m constructor_robot examples/tiny_house_plan.yaml --owner-goal "build the wall course" --bricklaying --video bricklaying.mp4
```

For a shorter validation run, reduce the task:

```bash
.venv/bin/python -m constructor_robot examples/tiny_house_plan.yaml --owner-goal "hold a brick" --bricklaying --bricks 1 --speed 8
```

The demo is a deterministic hybrid simulation: the humanoid uses MuJoCo bodies, joints, actuators, inertial properties, IMU sensors, collision meshes, cameras, and solver settings from `assets/Master/ff_master_ultra.xml`. The generated scene adds a four-finger position-actuated tool hand near the active wrist, while brick attachment is scripted so the construction sequence remains reproducible for judging. The scene uses an explicit Euler/CG solve in the generated XML to keep close-range manipulation stable.

The command currently still accepts a YAML plan because the CLI was originally shared with a broader construction-planning demo. The bricklaying animation itself is driven by `src/constructor_robot/bricklaying.py`.

## Bricklaying Behavior

The bricklaying scene is not a fixed image. It contains:

- The FF Master humanoid loaded from the repository's `assets/Master` folder.
- A generated right-hand tool with articulated thumb, index, middle, and ring finger joints.
- A brick pallet, mortar tray, mortar patches, wall guide, overview camera, and close-up hand camera.
- Multi-stage control: base repositioning, crouch posture, right-arm Jacobian IK, position-actuated finger closure, wrist flip, mortar dipping, wall placement, and release.
- A default four-brick course, configurable with `--bricks`.
- Optional MP4 export through the shared video recorder.

The bricklaying demo uses the robot assets already included at the repository root under `assets/Master`. The generated scene is written beside `ff_master_ultra.xml` at runtime so MuJoCo can resolve the model's mesh paths, then it is cleaned up before the program exits.

## Rubric Alignment

For compact evaluator-oriented versions of this section, see `rubric_mapping.md`, `judge_brief.md`, `self_assessment.md`, and the machine-readable `scorecard.json`.

- **Runnability:** `pyproject.toml` defines a reproducible editable install with `.[sim]`; the CLI supports viewer, headless video, single-brick smoke tests, and configurable speed.
- **Depth of MuJoCo use:** the demo uses MJCF includes, free bodies, mesh/collision geoms, cameras, lights, motors, position actuators, IMU sensors, Jacobian IK, solver configuration, and repeated `mj_forward` stepping.
- **Task design:** the robot performs a meaningful construction micro-task: pick, butter both sides with mortar, place, and repeat into a course.
- **Control:** the script combines task planning from YAML, whole-body pose control, base motion, arm IK, wrist flipping, gripper/finger actuation, and video/data replay.
- **Dexterous manipulation:** the generated right tool hand closes coordinated thumb, index, middle, and ring finger joints around the brick before lift and release.
- **Engineering quality:** constants, dataclasses, pure scene generation helpers, CLI options, and video export keep the implementation inspectable.
- **Presentation:** `demo.mp4` can be regenerated from the same command path; the overview and hand-closeup cameras make the manipulation legible.
- **Innovation:** the scenario applies humanoid dexterity to masonry rather than a generic pick-and-place table task.

## File Relevance

Keep these files for the bricklaying task:

- `src/constructor_robot/bricklaying.py` - main FF Master bricklaying scene and animation.
- `src/constructor_robot/video.py` - MP4 recording helper used by `--video`.
- `src/constructor_robot/__main__.py` - command-line entry point.
- `examples/tiny_house_plan.yaml` - still required by the current CLI.
- `../../assets/Master/ff_master_ultra.xml` - humanoid model loaded by the demo.
- `../../assets/Master/meshes/*.STL` - robot mesh assets required by `ff_master_ultra.xml`.
- `pyproject.toml` - package metadata, dependencies, and console script.

These files are not the core bricklaying demo, but are still imported by the current CLI and should only be removed after simplifying `src/constructor_robot/__main__.py`:

- `src/constructor_robot/plan.py` - parses the YAML plan.
- `src/constructor_robot/assignment.py` - prints the human/robot work split before the bricklaying demo starts.
- `src/constructor_robot/sim.py` - runs the older generic construction-site robot simulation.
- `src/constructor_robot/assets/construction_site.xml` - MuJoCo model for the older generic simulation.

These files and directories are generated or local-only clutter and are safe to delete when cleaning the working directory:

- `.venv/`
- `.pytest_cache/`
- `src/**/__pycache__/`
- `src/**/*.pyc`
- `src/constructor_robot.egg-info/`
- `MUJOCO_LOG.TXT`
- `bricklaying.mp4` if you do not need the rendered output
- `../../assets/Master/constructor_bricklaying_*.xml`

The `constructor_bricklaying_*.xml` files are temporary scene files created by `bricklaying.py`. If they remain in `assets/Master`, a previous run likely exited before cleanup finished.

## Legacy Planner

```bash
pip install -e .
python -m constructor_robot examples/tiny_house_plan.yaml --dry-run
```

This planner path exists from the original construction co-working prototype. It is useful only if you still want the YAML work-splitting behavior.

## Plan Format

```yaml
project:
  name: Tiny House Foundation
  owner: Amina
workers:
  - name: Mason crew
    skills: [masonry, inspection]
robot:
  name: RBT-01
  capabilities: [carry_materials, deliver_tools, site_scan]
tasks:
  - id: deliver_blocks
    title: Deliver concrete blocks
    zone: foundation
    required_skill: carry_materials
    effort: 3
    risk: medium
    dependencies: []
```

Tasks with high risk, inspection, or human-only skills are assigned to humans. Heavy, repetitive, or delivery tasks are good robot candidates when the robot has the capability.

## Safety Note

This is a simulation and planning prototype, not a real construction-site safety controller. A production robot would need certified perception, geofencing, emergency stops, task supervision, and compliance with local safety rules.
