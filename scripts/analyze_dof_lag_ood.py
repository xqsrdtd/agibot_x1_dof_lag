#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_run_aliases(items):
    out = {}
    if not items:
        return out
    for it in items:
        if "=" not in it:
            raise SystemExit(f"[analyze_ood] Bad --run_alias (need folder=label): {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="One or more eval CSVs (from eval.py / run_eval_dof_lag_ood.sh); merged for multi-curve plots.",
    )
    ap.add_argument("--out_dir", default="results", help="Output directory for plots and summary CSV")
    ap.add_argument(
        "--out_stem",
        default="dof_lag_ood",
        help="Stem for outputs, e.g. dof_lag_ood_three -> dof_lag_ood_three_tracking.png",
    )
    ap.add_argument(
        "--run_alias",
        action="append",
        default=[],
        metavar="FOLDER=LABEL",
        help="Legend label for a load_run folder (repeatable), e.g. 2026-03-20_21-40-23baseline_1=baseline",
    )
    ap.add_argument("--baseline_run", default=None, help="Optional: run folder name to label as baseline.")
    ap.add_argument("--robust_run", default=None, help="Optional: run folder name to label as robust.")
    ap.add_argument(
        "--omit_dof_lag_max",
        action="append",
        default=[],
        type=int,
        metavar="LAG",
        help="Drop rows with this dof_lag_max before plots/summary (repeatable), e.g. omit tail lag=120.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    alias_map = _parse_run_aliases(args.run_alias)

    frames = []
    for path in args.csv:
        if not os.path.isfile(path):
            raise SystemExit(f"[analyze_ood] CSV not found: {path}")
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise SystemExit(f"[analyze_ood] Empty CSV(s): {args.csv}")

    # Ensure numeric ordering.
    df["dof_lag_max"] = df["dof_lag_max"].astype(int)
    if args.omit_dof_lag_max:
        omit = set(int(x) for x in args.omit_dof_lag_max)
        before = len(df)
        df = df[~df["dof_lag_max"].isin(omit)].copy()
        print(f"[analyze_ood] Omitted dof_lag_max in {omit}: {before} -> {len(df)} rows")

    def label_run(load_run: str) -> str:
        s = str(load_run)
        if s in alias_map:
            return alias_map[s]
        if args.baseline_run is not None and s == str(args.baseline_run):
            return "baseline"
        if args.robust_run is not None and s == str(args.robust_run):
            return "robust"
        return s

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
    # reset_index() keeps run_label / dof_lag_max as columns (as_index=False can drop them with multi-agg on some pandas)
    summary = (
        df.groupby(["run_label", "dof_lag_max"])[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # flatten columns
    summary.columns = [
        "_".join([c for c in col if c != ""]) if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    # Rename expected fields (more readable)
    # Example: success_rate_mean, lin_vel_mae_std, fall_rate_mean, ...
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, f"{args.out_stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[analyze_ood] Wrote summary CSV: {summary_path}")

    lag_values = sorted(df["dof_lag_max"].unique().tolist())

    def _plot_series(ax_x, means, stds, label):
        means = np.asarray(means, dtype=float)
        stds = np.asarray(stds, dtype=float)
        ok = np.isfinite(means)
        if not np.any(ok):
            return
        ax_x = np.asarray(ax_x, dtype=float)[ok]
        yerr = np.where(np.isfinite(stds[ok]), stds[ok], 0.0)
        plt.errorbar(ax_x, means[ok], yerr=yerr, marker="o", capsize=3, linewidth=2, label=label)

    # Plot: success proxy vs dof_lag_max
    plt.figure(figsize=(9, 5.0))
    for run_label in sorted(df["run_label"].unique()):
        sub = summary[summary["run_label"] == run_label].copy()
        if sub.empty:
            continue
        sub = sub.set_index("dof_lag_max").reindex(lag_values)
        means = sub["success_rate_mean"].astype(float).values
        stds = sub["success_rate_std"].astype(float).values
        _plot_series(lag_values, means, stds, run_label)
    plt.xticks(lag_values)
    plt.title("OOD success proxy vs DOF lag")
    plt.xlabel("DOF lag max (timesteps)")
    plt.ylabel("success rate (1 - fall_rate)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out1 = os.path.join(args.out_dir, f"{args.out_stem}_success.png")
    plt.savefig(out1, dpi=160)
    plt.close()
    print(f"[analyze_ood] Saved plot: {out1}")

    # Plot: tracking error vs dof_lag_max
    plt.figure(figsize=(9, 5.0))
    for run_label in sorted(df["run_label"].unique()):
        sub = summary[summary["run_label"] == run_label].copy()
        if sub.empty:
            continue
        sub = sub.set_index("dof_lag_max").reindex(lag_values)
        means = sub["lin_vel_mae_mean"].astype(float).values
        stds = sub["lin_vel_mae_std"].astype(float).values
        _plot_series(lag_values, means, stds, run_label)
    plt.xticks(lag_values)
    plt.title("OOD tracking error vs DOF lag")
    plt.xlabel("DOF lag max (timesteps)")
    plt.ylabel("lin_vel_mae (lower is better)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out2 = os.path.join(args.out_dir, f"{args.out_stem}_tracking.png")
    plt.savefig(out2, dpi=160)
    plt.close()
    print(f"[analyze_ood] Saved plot: {out2}")

    # Plot: fall_rate vs dof_lag_max
    plt.figure(figsize=(9, 5.0))
    for run_label in sorted(df["run_label"].unique()):
        sub = summary[summary["run_label"] == run_label].copy()
        if sub.empty:
            continue
        sub = sub.set_index("dof_lag_max").reindex(lag_values)
        means = sub["fall_rate_mean"].astype(float).values
        stds = sub["fall_rate_std"].astype(float).values
        _plot_series(lag_values, means, stds, run_label)
    plt.xticks(lag_values)
    plt.title("OOD fall rate vs DOF lag")
    plt.xlabel("DOF lag max (timesteps)")
    plt.ylabel("fall rate (lower is better)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out3 = os.path.join(args.out_dir, f"{args.out_stem}_fall_rate.png")
    plt.savefig(out3, dpi=160)
    plt.close()
    print(f"[analyze_ood] Saved plot: {out3}")

    # Plot: ang_vel_mae vs dof_lag_max (if available)
    if "ang_vel_mae_mean" in summary.columns:
        plt.figure(figsize=(9, 5.0))
        for run_label in sorted(df["run_label"].unique()):
            sub = summary[summary["run_label"] == run_label].copy()
            if sub.empty:
                continue
            sub = sub.set_index("dof_lag_max").reindex(lag_values)
            means = sub["ang_vel_mae_mean"].astype(float).values
            stds = sub["ang_vel_mae_std"].astype(float).values
            _plot_series(lag_values, means, stds, run_label)
        plt.xticks(lag_values)
        plt.title("OOD angular velocity tracking error vs DOF lag")
        plt.xlabel("DOF lag max (timesteps)")
        plt.ylabel("ang_vel_mae (lower is better)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        out4 = os.path.join(args.out_dir, f"{args.out_stem}_ang_vel_mae.png")
        plt.savefig(out4, dpi=160)
        plt.close()
        print(f"[analyze_ood] Saved plot: {out4}")

    # Write report-ready table (Markdown + LaTeX)
    _write_report_table(summary, args.out_dir, args.out_stem)


def _write_report_table(summary, out_dir, out_stem: str):
    """Write compact report table with ID (lag=0) comparison."""
    id_rows = summary[summary["dof_lag_max"] == 0]
    ood_rows = summary[summary["dof_lag_max"] > 0].copy()
    lag_vals = sorted(ood_rows["dof_lag_max"].unique().tolist())
    models = sorted(summary["run_label"].unique().tolist())

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

    hdr = "| lag | " + " | ".join(models) + " |"
    sep = "|-----|" + "|".join(["--------"] * len(models)) + "|"
    lines.extend(["", "## OOD: Success rate (%) by DOF lag", hdr, sep])
    for lag in lag_vals:
        cells = []
        for name in models:
            sub = summary[(summary["run_label"] == name) & (summary["dof_lag_max"] == lag)]
            if sub.empty:
                cells.append("-")
            else:
                v = float(sub["success_rate_mean"].values[0]) * 100
                cells.append(f"{v:.1f}")
        lines.append("| " + str(lag) + " | " + " | ".join(cells) + " |")

    out_path = os.path.join(out_dir, f"{out_stem}_report.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[analyze_ood] Wrote report: {out_path}")


if __name__ == "__main__":
    main()

