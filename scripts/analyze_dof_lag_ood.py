#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV produced by scripts/run_eval_dof_lag_ood.sh")
    ap.add_argument("--out_dir", default="results", help="Output directory for plots and summary CSV")
    ap.add_argument("--baseline_run", default=None, help="Optional: run folder name to label as baseline.")
    ap.add_argument("--robust_run", default=None, help="Optional: run folder name to label as robust.")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"[analyze_ood] Empty CSV: {args.csv}")

    # Ensure numeric ordering.
    df["dof_lag_max"] = df["dof_lag_max"].astype(int)

    def label_run(load_run: str) -> str:
        if args.baseline_run is not None and str(load_run) == str(args.baseline_run):
            return "baseline"
        if args.robust_run is not None and str(load_run) == str(args.robust_run):
            return "robust"
        return str(load_run)

    df["run_label"] = df["load_run"].apply(label_run)

    # success proxy: use timeout_rate if in [0,1], else 1 - fall_rate (fall_rate is correct)
    df["timeout_rate_raw"] = df["timeout_rate"].astype(float)
    df["fall_rate"] = df["fall_rate"].astype(float).clip(0, 1)
    if df["timeout_rate_raw"].max() > 1.01:
        print(
            f"[analyze_ood] WARNING: timeout_rate max={df['timeout_rate_raw'].max():.2f} > 1. "
            "Using success_rate = 1 - fall_rate instead."
        )
        df["success_rate"] = 1.0 - df["fall_rate"]
    else:
        df["success_rate"] = df["timeout_rate_raw"].clip(0, 1)

    metrics = ["success_rate", "lin_vel_mae", "fall_rate"]
    if "ang_vel_mae" in df.columns:
        df["ang_vel_mae"] = df["ang_vel_mae"].astype(float)
        metrics.append("ang_vel_mae")
    summary = (
        df.groupby(["run_label", "dof_lag_max"], as_index=False)[metrics]
        .agg(["mean", "std", "count"])
    )

    # flatten columns
    summary.columns = [
        "_".join([c for c in col if c != ""]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    # Rename expected fields (more readable)
    # Example: success_rate_mean, lin_vel_mae_std, fall_rate_mean, ...
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "dof_lag_ood_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[analyze_ood] Wrote summary CSV: {summary_path}")

    lag_values = sorted(df["dof_lag_max"].unique().tolist())
    x = lag_values

    # Plot: success proxy vs dof_lag_max
    plt.figure(figsize=(8, 4.8))
    for run_label in sorted(df["run_label"].unique()):
        sub = summary[summary["run_label"] == run_label].copy()
        if sub.empty:
            continue
        sub = sub.set_index("dof_lag_max").reindex(x)
        means = sub["success_rate_mean"].astype(float).values
        stds = sub["success_rate_std"].astype(float).values
        plt.errorbar(x, means, yerr=stds, marker="o", capsize=3, linewidth=2, label=run_label)
    plt.xticks(x)
    plt.title("OOD success proxy vs DOF lag")
    plt.xlabel("DOF lag max (timesteps)")
    plt.ylabel("success rate (1 - fall_rate)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(args.out_dir, "dof_lag_ood_success.png")
    plt.savefig(out1, dpi=160)
    plt.close()
    print(f"[analyze_ood] Saved plot: {out1}")

    # Plot: tracking error vs dof_lag_max
    plt.figure(figsize=(8, 4.8))
    for run_label in sorted(df["run_label"].unique()):
        sub = summary[summary["run_label"] == run_label].copy()
        if sub.empty:
            continue
        sub = sub.set_index("dof_lag_max").reindex(x)
        means = sub["lin_vel_mae_mean"].astype(float).values
        stds = sub["lin_vel_mae_std"].astype(float).values
        plt.errorbar(x, means, yerr=stds, marker="o", capsize=3, linewidth=2, label=run_label)
    plt.xticks(x)
    plt.title("OOD tracking error vs DOF lag")
    plt.xlabel("DOF lag max (timesteps)")
    plt.ylabel("lin_vel_mae (lower is better)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(args.out_dir, "dof_lag_ood_tracking.png")
    plt.savefig(out2, dpi=160)
    plt.close()
    print(f"[analyze_ood] Saved plot: {out2}")

    # Plot: fall_rate vs dof_lag_max
    plt.figure(figsize=(8, 4.8))
    for run_label in sorted(df["run_label"].unique()):
        sub = summary[summary["run_label"] == run_label].copy()
        if sub.empty:
            continue
        sub = sub.set_index("dof_lag_max").reindex(x)
        means = sub["fall_rate_mean"].astype(float).values
        stds = sub["fall_rate_std"].astype(float).values
        plt.errorbar(x, means, yerr=stds, marker="o", capsize=3, linewidth=2, label=run_label)
    plt.xticks(x)
    plt.title("OOD fall rate vs DOF lag")
    plt.xlabel("DOF lag max (timesteps)")
    plt.ylabel("fall rate (lower is better)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out3 = os.path.join(args.out_dir, "dof_lag_ood_fall_rate.png")
    plt.savefig(out3, dpi=160)
    plt.close()
    print(f"[analyze_ood] Saved plot: {out3}")

    # Plot: ang_vel_mae vs dof_lag_max (if available)
    if "ang_vel_mae_mean" in summary.columns:
        plt.figure(figsize=(8, 4.8))
        for run_label in sorted(df["run_label"].unique()):
            sub = summary[summary["run_label"] == run_label].copy()
            if sub.empty:
                continue
            sub = sub.set_index("dof_lag_max").reindex(x)
            means = sub["ang_vel_mae_mean"].astype(float).values
            stds = sub["ang_vel_mae_std"].astype(float).values
            plt.errorbar(x, means, yerr=stds, marker="o", capsize=3, linewidth=2, label=run_label)
        plt.xticks(x)
        plt.title("OOD angular velocity tracking error vs DOF lag")
        plt.xlabel("DOF lag max (timesteps)")
        plt.ylabel("ang_vel_mae (lower is better)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        out4 = os.path.join(args.out_dir, "dof_lag_ood_ang_vel_mae.png")
        plt.savefig(out4, dpi=160)
        plt.close()
        print(f"[analyze_ood] Saved plot: {out4}")

    # Write report-ready table (Markdown + LaTeX)
    _write_report_table(summary, args.out_dir)


def _write_report_table(summary, out_dir):
    """Write compact report table with ID (lag=0) comparison."""
    id_rows = summary[summary["dof_lag_max"] == 0]
    ood_rows = summary[summary["dof_lag_max"] > 0].copy()
    lag_values = sorted(ood_rows["dof_lag_max"].unique().tolist())

    lines = [
        "# DOF Lag OOD Evaluation Summary",
        "",
        "## In-Distribution (lag=0)",
        "| Model | Success | lin_vel_mae | fall_rate |",
        "|-------|---------|-------------|-----------|",
    ]
    for _, row in id_rows.iterrows():
        s = row.get("success_rate_mean", 0) * 100
        m = row.get("lin_vel_mae_mean", 0)
        f = row.get("fall_rate_mean", 0) * 100
        lines.append(f"| {row['run_label']} | {s:.1f}% | {m:.3f} | {f:.1f}% |")

    lines.extend([
        "",
        "## OOD: Success rate (%) by DOF lag",
        "| lag | baseline | robust |",
        "|-----|----------|--------|",
    ])
    for lag in lag_values:
        b = summary[(summary["run_label"] == "baseline") & (summary["dof_lag_max"] == lag)]
        r = summary[(summary["run_label"] == "robust") & (summary["dof_lag_max"] == lag)]
        bv = b["success_rate_mean"].values[0] * 100 if not b.empty else "-"
        rv = r["success_rate_mean"].values[0] * 100 if not r.empty else "-"
        bv = f"{bv:.1f}" if isinstance(bv, (int, float)) else bv
        rv = f"{rv:.1f}" if isinstance(rv, (int, float)) else rv
        lines.append(f"| {lag} | {bv} | {rv} |")

    out_path = os.path.join(out_dir, "dof_lag_ood_report.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[analyze_ood] Wrote report: {out_path}")


if __name__ == "__main__":
    main()

