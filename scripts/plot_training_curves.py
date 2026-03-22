#!/usr/bin/env python3
"""Plot training curves: baseline vs robust (DOF lag domain randomization)."""
import argparse
import os

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(log_dir: str, tag: str):
    """Load scalar values from tensorboard events. Returns (steps, values)."""
    if not os.path.isdir(log_dir):
        return [], []
    ea = EventAccumulator(log_dir)
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline_dir",
        default="logs/x1_dh_stand/exported_data/2026-03-20_21-40-23baseline_1",
        help="Path to baseline run directory",
    )
    ap.add_argument(
        "--robust_dir",
        default="logs/x1_dh_stand/exported_data/2026-03-21_11-55-23lag_1",
        help="Path to robust (DOF lag) run directory",
    )
    ap.add_argument("--out_dir", default="results", help="Output directory")
    return ap.parse_args()


def main():
    args = parse_args()
    base = os.path.abspath(args.baseline_dir)
    robust = os.path.abspath(args.robust_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    tags = ["Train/mean_reward", "Train/mean_episode_length"]
    # Episode metrics if logged
    for ep_tag in ["Episode/lin_vel_mae", "Episode/timeout_rate", "Episode/fall_rate"]:
        tags.append(ep_tag)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Train/mean_reward
    tag = "Train/mean_reward"
    s_b, v_b = load_scalar(base, tag)
    s_r, v_r = load_scalar(robust, tag)
    ax = axes[0]
    if s_b:
        ax.plot(s_b, v_b, label="baseline", color="C0", linewidth=1.5)
    if s_r:
        ax.plot(s_r, v_r, label="robust (DOF lag aug)", color="C1", linewidth=1.5)
    ax.set_xlabel("Learning iteration")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Training: Mean Reward")
    ax.legend()
    ax.grid(True, alpha=0.25)

    # Train/mean_episode_length
    tag = "Train/mean_episode_length"
    s_b, v_b = load_scalar(base, tag)
    s_r, v_r = load_scalar(robust, tag)
    ax = axes[1]
    if s_b:
        ax.plot(s_b, v_b, label="baseline", color="C0", linewidth=1.5)
    if s_r:
        ax.plot(s_r, v_r, label="robust (DOF lag aug)", color="C1", linewidth=1.5)
    ax.set_xlabel("Learning iteration")
    ax.set_ylabel("Mean episode length")
    ax.set_title("Training: Episode Length")
    ax.legend()
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "training_curves.png")
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[plot_training] Saved: {out_path}")


if __name__ == "__main__":
    main()
