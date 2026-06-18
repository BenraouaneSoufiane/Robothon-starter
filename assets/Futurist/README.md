# FF Futurist URDF Asset

Humanoid robot asset package for **FF Futurist**.

## Contents

- `futurist.urdf` — robot description file (`robot name="futurist"`)
- `*_hull.stl` and `logo.stl` — mesh files referenced by the URDF

The URDF uses relative mesh filenames, so mesh files should stay in this same directory.

## Validation

From the repository root:

```bash
python examples/run_futurist_demo.py --check-assets
```

If every referenced mesh exists, generate a MuJoCo demo video:

```bash
python examples/run_futurist_demo.py
```

## PyBullet Example

```python
import pybullet as p

p.connect(p.GUI)
robot = p.loadURDF("futurist.urdf")
```

## ROS / RViz Example

```bash
ros2 run urdf_tutorial display.launch.py model:=futurist.urdf
```

## Notes

- The model includes legs, arms, hands, and two head joints.
- Mesh paths are plain relative filenames, not `package://` URLs.
- Materials are declared as URDF colors; no texture files are required.
