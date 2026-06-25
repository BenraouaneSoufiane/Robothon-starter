# Micro Suture Lab

**FFAI Robothon 2026 — Freestyle Category**

Micro Suture Lab is a self-contained MuJoCo dexterity submission in which a five-finger wrist/gantry hand grasps a surgical needle, passes it through three tissue hoops, draws a suture tail, recovers from a deterministic slip disturbance during knot cinch, and presses a verification button.

The submission is designed to be judged from regenerated evidence, not from hand-written scores: one runner produces video, trajectory, contact, policy, manifest, and robustness artifacts; a separate plot command turns those JSON files into diagnostic figures.

## Task Sequence

| # | Stage | Success signal |
|---:|---|---|
| 0 | Sterile scan | Frame-position, joint-position, and touch sensors are available |
| 1 | Approach and preload | Palm aligns to the free needle before contact |
| 2 | Five-finger needle grasp | Thumb/index/middle pinch is stabilized by ring and little fingers |
| 3 | Lift and confirm | Needle remains carried under gravity |
| 4 | Entry hoop pass | Needle tip reaches the entry target |
| 5 | Middle hoop pass | Needle follows the curved safe corridor |
| 6 | Exit hoop pass | Third hoop is cleared |
| 7 | Suture draw and knot cinch | Thread attaches after entry and slip impulse is recovered |
| 8 | Verification press | Button depth exceeds the completion threshold |
| 9 | Dataset export pose | Evidence artifacts are finalized |

## Technical Specifications

- 15 position actuators: gantry X/Y/Z, wrist yaw, ten finger joints, and button slide.
- Free-joint needle and suture-tail bodies with gravity, damping, frictional contact, and no weld on the needle grasp.
- MuJoCo timestep defaults to 1 ms.
- Contact model uses elliptic cones, multiccd, high-friction pad/needle pair overrides, and condim 6 fingertip pads.
- Sensors include frame-position targets, joint-position readings, and five fingertip touch sensors.
- Closed-loop controller combines a staged nominal plan with bounded PID residual feedback and anti-windup.
- Robustness evaluation runs fixed-seed closed-loop and open-loop rollouts through `stress_eval`.

## Reproducible Commands

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r submissions/micro_suture_lab/requirements.txt
.venv/bin/python submissions/micro_suture_lab/run_micro_suture_lab.py --quick
```

Equivalent Make targets:

```bash
make quick
make demo
make eval
make plots
make test
```

Optional gain tuning:

```bash
make tune
```

The default demo renders `18` seconds at `30` FPS to `artifacts/demo.mp4`. Quick mode renders `4` seconds at `20` FPS to `artifacts/quick_demo.mp4`. The runner records whether the backend was real MuJoCo rendering or a headless schematic fallback in `artifacts/report.json`.

## Generated Artifacts

```text
submissions/micro_suture_lab/
├── micro_suture_scene.xml
├── run_micro_suture_lab.py
├── make_plots.py
├── tune_gains.py
├── requirements.txt
├── registration.json
├── evaluation_report.json
├── PR_DESCRIPTION.md
└── artifacts/
    ├── demo.mp4
    ├── quick_demo.mp4
    ├── trajectory.json
    ├── report.json
    ├── evaluation.json
    ├── contact_timeline.json
    ├── policy_card.json
    ├── submission_manifest.json
    ├── trajectory_diagnostics.png
    ├── evaluation_summary.png
    └── gain_tuning.png
```

## Evaluation Method

`python3 submissions/micro_suture_lab/run_micro_suture_lab.py eval` runs paired rollouts with identical perturbation seeds. The closed-loop controller uses palm position feedback and bounded residual corrections; the open-loop baseline follows the same stage prior with feedback disabled. The exported `artifacts/evaluation.json` includes per-seed task scores, endpoint errors, hoop pass state, grasp/lift state, and aggregate success rates.

## Evidence Checklist

- Run `make quick` to confirm the model loads, simulates, renders, and writes core artifacts.
- Run `make eval` to regenerate the 40-seed open-loop versus closed-loop comparison.
- Run `make plots` to regenerate trajectory, evaluation, and gain-tuning figures.
- Run `make test` to check model loading, sensor/actuator inventory, rollout success signals, feedback-vs-open-loop evaluation shape, and plotting.

## License

This project is submitted for the FFAI Robothon 2026 competition.
