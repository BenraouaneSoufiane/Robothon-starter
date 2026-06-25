"""Generate diagnostic PNG plots from Micro Suture Lab artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def load(name):
    path = ART / name
    return json.loads(path.read_text()) if path.exists() else None


def plot_trajectory(traj, out):
    if not traj:
        return False
    t = np.array([r["time_s"] for r in traj])
    err_mm = np.array([r.get("err_m", r.get("closed_loop_servo_error_m", 0.0)) for r in traj]) * 1000.0
    corr_mm = np.array([np.linalg.norm(r.get("corr_m", r.get("residual_xyz_m", [0.0, 0.0, 0.0]))) for r in traj]) * 1000.0
    fingers = ("thumb", "index", "middle", "ring", "little")
    touch = np.array([[r.get("touches", {}).get(f, 0.0) for f in fingers] for r in traj])
    hoops = np.array([sum(r.get("hoops", r.get("hoops_passed", []))) for r in traj])
    height_mm = np.array([r.get("needle_height_m", 0.0) for r in traj]) * 1000.0
    grasp = np.array([int(r.get("grasp", r.get("grasp_confirmed", False))) for r in traj])

    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    ax[0].plot(t, err_mm, label="palm error |e|", color="#1f77b4")
    ax[0].plot(t, corr_mm, label="|feedback correction|", color="#d62728", alpha=0.75)
    ax[0].set_ylabel("mm")
    ax[0].legend()
    ax[0].set_title("Closed-loop palm tracking")
    ax[0].grid(alpha=0.3)

    for i, finger in enumerate(fingers):
        ax[1].plot(t, touch[:, i], label=finger)
    ax[1].set_ylabel("touch force")
    ax[1].legend(loc="upper right", ncol=5, fontsize=8)
    ax[1].set_title("Fingertip contact forces")
    ax[1].grid(alpha=0.3)

    ax[2].plot(t, height_mm, color="#2ca02c", label="needle grip height")
    ax[2].fill_between(t, height_mm.min(), height_mm, where=(grasp == 1), alpha=0.15, color="#2ca02c", label="grasp active")
    ax[2].set_ylabel("mm")
    ax[2].legend()
    ax[2].set_title("Needle height and grasp state")
    ax[2].grid(alpha=0.3)

    ax[3].step(t, hoops, where="post", color="#9467bd")
    ax[3].set_ylabel("hoops passed")
    ax[3].set_xlabel("time (s)")
    ax[3].set_yticks([0, 1, 2, 3])
    ax[3].set_title("Hoop pass progression")
    ax[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return True


def _new_eval_arrays(ev):
    closed = ev["rollouts"]["closed_loop"]
    open_loop = ev["rollouts"]["open_loop"]
    closed_err = np.array([r["endpoint_error_m"] for r in closed]) * 1000.0
    open_err = np.array([r["endpoint_error_m"] for r in open_loop]) * 1000.0
    return closed_err, open_err, ev["closed_loop"]["success_rate"], ev["open_loop"]["success_rate"], ev.get("seeds", len(closed))


def _old_eval_arrays(ev):
    rows = ev["rollouts"]
    closed_err = np.array([r["residual_policy_error_mm"] for r in rows])
    open_err = np.array([r["open_loop_error_mm"] for r in rows])
    closed_rate = float(np.mean([r["residual_policy_success"] for r in rows]))
    open_rate = float(np.mean([r["open_loop_success"] for r in rows]))
    return closed_err, open_err, closed_rate, open_rate, len(rows)


def plot_eval(ev, out):
    if not ev:
        return False
    if isinstance(ev.get("rollouts"), dict):
        closed_err, open_err, closed_rate, open_rate, seeds = _new_eval_arrays(ev)
    else:
        closed_err, open_err, closed_rate, open_rate, seeds = _old_eval_arrays(ev)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    hi = max(float(open_err.max()), float(closed_err.max()), 10.0)
    bins = np.linspace(0, hi, 24)
    ax[0].hist(open_err, bins=bins, alpha=0.6, label=f"open-loop ({open_err.mean():.1f}+/-{open_err.std():.1f} mm)", color="#d62728")
    ax[0].hist(closed_err, bins=bins, alpha=0.7, label=f"closed-loop ({closed_err.mean():.1f}+/-{closed_err.std():.1f} mm)", color="#2ca02c")
    ax[0].set_xlabel("endpoint error (mm)")
    ax[0].set_ylabel("rollouts")
    ax[0].set_title(f"Endpoint error across {seeds} seeds")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].bar(["open-loop", "closed-loop"], [open_rate, closed_rate], color=["#d62728", "#2ca02c"])
    ax[1].set_ylabel("task success rate")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Open-loop vs closed-loop success")
    for i, value in enumerate([open_rate, closed_rate]):
        ax[1].text(i, min(value + 0.03, 1.02), f"{value:.0%}", ha="center")
    ax[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return True


def plot_tuning(hist, out):
    if not hist:
        return False
    gens = np.array([h["gen"] for h in hist])
    best = np.array([-h["best_neg_score"] for h in hist])
    mean = np.array([-h["mean_neg_score"] for h in hist])
    sigma = np.array([h["sigma"] for h in hist])

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(gens, best, marker="o", label="best score")
    ax[0].plot(gens, mean, marker="s", label="population mean")
    ax[0].set_xlabel("generation")
    ax[0].set_ylabel("task score")
    ax[0].set_title("CMA-ES gain optimization")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(gens, sigma, marker="o", color="#9467bd")
    ax[1].set_xlabel("generation")
    ax[1].set_ylabel("sigma")
    ax[1].set_title("Search radius")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main():
    ART.mkdir(parents=True, exist_ok=True)
    generated = []
    if plot_trajectory(load("trajectory.json"), ART / "trajectory_diagnostics.png"):
        generated.append("trajectory_diagnostics.png")
    if plot_eval(load("evaluation.json"), ART / "evaluation_summary.png"):
        generated.append("evaluation_summary.png")
    if plot_tuning(load("tuning_history.json"), ART / "gain_tuning.png"):
        generated.append("gain_tuning.png")
    print(json.dumps({"generated": generated}, indent=2))
    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
