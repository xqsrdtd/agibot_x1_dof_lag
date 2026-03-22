#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def infer_lag_bin(row: pd.Series) -> str:
    # eval.py: "no lag" => enable_dof_lag = False (defaults still set dof_lag_min/max to 0)
    if not bool(row["enable_dof_lag"]):
        return "no"
    vmax = int(row["dof_lag_max"])
    # With fixed mode, dof_lag_max is what matters (always used).
    if vmax == 5:
        return "low"
    if vmax == 15:
        return "med"
    if vmax == 40:
        return "high"
    return f"other({vmax})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV produced by scripts/run_eval_sweep.sh")
    ap.add_argument("--baseline_run", required=True, help="baseline run folder name under logs/<experiment_name>/exported_data/")
    ap.add_argument("--robust_run", required=True, help="robust run folder name under logs/<experiment_name>/exported_data/")
    ap.add_argument("--out_dir", default="results", help="Output directory for plots and summary CSV")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"[analyze] Empty CSV: {args.csv}")

    df["run_kind"] = df["load_run"].apply(
        lambda x: "baseline" if str(x) == str(args.baseline_run) else ("robust" if str(x) == str(args.robust_run) else str(x))
    )
    df["lag_bin"] = df.apply(infer_lag_bin, axis=1)

    # success proxy:
    # legged_robot.py: timeout == survived full episode; non-timeout reset == fall/termination
    # eval.py exports timeout_rate / fall_rate
    df["success_rate"] = df["timeout_rate"].astype(float)

    keep_bins = ["no", "low", "med", "high"]
    df = df[df["lag_bin"].isin(keep_bins)].copy()
    if df.empty:
        raise SystemExit("[analyze] After binning, no rows left. Check dof_lag_max values used in the sweep.")

    metrics = ["success_rate", "lin_vel_mae", "fall_rate"]
    # Per (run_kind, lag_bin), we aggregate across seeds (each row in eval.csv is already averaged over episodes).
    out_rows: List[Dict] = []

    for run_kind in sorted(df["run_kind"].unique()):
        for lag_bin in keep_bins:
            sub = df[(df["run_kind"] == run_kind) & (df["lag_bin"] == lag_bin)]
            if sub.empty:
                continue
            out_rows.append(
                {
                    "run_kind": run_kind,
                    "lag_bin": lag_bin,
                    "success_rate_mean": sub["success_rate"].mean(),
                    "success_rate_std": sub["success_rate"].std(ddof=0),
                    "lin_vel_mae_mean": sub["lin_vel_mae"].mean(),
                    "lin_vel_mae_std": sub["lin_vel_mae"].std(ddof=0),
                    "fall_rate_mean": sub["fall_rate"].mean(),
                    "fall_rate_std": sub["fall_rate"].std(ddof=0),
                    "num_seeds": sub["seed"].nunique(),
                }
            )

    summary = pd.DataFrame(out_rows)
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "dof_lag_sweep_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[analyze] Wrote summary CSV: {summary_path}")

    # Plot ordering
    x_labels = keep_bins
    x = list(range(len(x_labels)))

    def plot_metric(metric_mean: str, metric_std: str, title: str, ylabel: str, out_path: str):
        plt.figure(figsize=(8, 4.8))
        for run_kind in ["baseline", "robust"]:
            sub = summary[summary["run_kind"] == run_kind].set_index("lag_bin").reindex(x_labels)
            means = sub[metric_mean].values.astype(float)
            stds = sub[metric_std].values.astype(float)
            plt.errorbar(x, means, yerr=stds, marker="o", capsize=3, linewidth=2, label=run_kind)
        plt.xticks(x, x_labels)
        plt.title(title)
        plt.xlabel("DOF lag bin")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"[analyze] Saved plot: {out_path}")

    plot_metric(
        "success_rate_mean",
        "success_rate_std",
        title="Success rate vs DOF lag",
        ylabel="success proxy = timeout_rate",
        out_path=os.path.join(args.out_dir, "dof_lag_sweep_success.png"),
    )
    plot_metric(
        "lin_vel_mae_mean",
        "lin_vel_mae_std",
        title="Tracking error vs DOF lag",
        ylabel="lin_vel_mae (lower is better)",
        out_path=os.path.join(args.out_dir, "dof_lag_sweep_tracking.png"),
    )


if __name__ == "__main__":
    main()

