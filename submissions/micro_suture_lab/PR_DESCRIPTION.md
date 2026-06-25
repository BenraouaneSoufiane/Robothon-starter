## Micro Suture Lab

**Registration UUID:** `d4d194e1-f169-424e-ad86-ec2dc1953d52`

### Project Summary

Micro Suture Lab is a self-contained MuJoCo dexterity submission for FFAI Robothon 2026. A five-finger wrist/gantry hand grasps a surgical needle, passes it through three tissue hoops, draws the suture tail, cinches a knot while recovering from a deterministic slip disturbance, presses a verification button, and exports video and structured controller evidence.

### Key Innovations

- **Five-finger manipulation:** Ten proximal/distal finger joints coordinate a precision pinch with ring/little-finger stabilization.
- **Long-horizon suturing task:** Nine stages cover sterile scan, grasp, three hoop passes, thread draw, knot cinch, verification, and export.
- **Sensor-rich MuJoCo model:** 15 actuators, 19 frame-position/joint-position/touch sensors, free-joint task objects, and 34 geoms.
- **Residual control:** A deterministic stage planner applies bounded PID residual corrections with anti-windup and an open-loop ablation flag.
- **Robustness evidence:** Fixed-seed perturbation cases compare the residual controller with an open-loop baseline.
- **Optimization path:** `tune_gains.py` runs a compact CMA-ES sweep over the PID gains and exports tuning history.
- **Presentation evidence:** `make_plots.py` regenerates trajectory and robustness figures from JSON artifacts.

### Evidence to Regenerate

- `make quick` — short simulation/video smoke test
- `make demo` — full demonstration and core artifacts
- `make eval` — fixed-seed open-loop versus closed-loop robustness sweep
- `make plots` — trajectory, evaluation, and tuning figures
- `make test` — model, rollout, evaluation, and plotting tests

### Run Instructions

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r submissions/micro_suture_lab/requirements.txt
.venv/bin/python submissions/micro_suture_lab/run_micro_suture_lab.py
```

Quick smoke test:

```bash
.venv/bin/python submissions/micro_suture_lab/run_micro_suture_lab.py --quick
```

The convenience entrypoint `submissions/micro_suture_lab/run.sh` accepts the same arguments.

### Submission Files

- `micro_suture_scene.xml` — MuJoCo robot, surgical workcell, objects, actuators, and sensors
- `run_micro_suture_lab.py` — Controller, simulation, rendering, artifact export, and robustness evaluation
- `run.sh` — Shell entrypoint
- `requirements.txt` — Python dependencies
- `README.md` — Full project documentation and evidence disclosures
- `evaluation_report.json` — Rubric-aligned evidence index without self-assigned scores
- `registration.json` — Participant and registration metadata
- `artifacts/demo.mp4` — Full generated demonstration
- `artifacts/quick_demo.mp4` — Short generated demonstration
- `artifacts/trajectory.json` — Sampled actions, state, metrics, and all MuJoCo sensor values
- `artifacts/report.json` — Runtime result, model statistics, and aggregate metrics
- `artifacts/evaluation.json` — Open-loop/residual perturbation comparisons
- `artifacts/contact_timeline.json` — Grip/contact proxy and slip events
- `artifacts/policy_card.json` — Controller topology and closed-loop metrics
- `artifacts/trajectory_diagnostics.png` — Generated trajectory diagnostics
- `artifacts/evaluation_summary.png` — Generated robustness summary

