#!/usr/bin/env python3
"""
Merge all per-run eval CSVs (eval.py / run_eval_dof_lag_ood.sh output) and plot curves together.

Excludes: *_summary.csv, dof_lag_ood_eval_combined.csv (duplicate rows), tiny/debug sweeps.
Edit MERGE_FILES / RUN_ALIASES when you add new runs.

Requires pandas + matplotlib (system python3 may lack them; use a conda env Python if needed).
"""
import os
import subprocess
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(REPO_ROOT, "results")

# Raw eval CSVs to merge (same schema as eval.py). Order only affects merge order, not legend.
MERGE_FILES = [
    "dof_lag_ood_eval_baseline_1.csv",
    "dof_lag_ood_eval_robust_1.csv",
    "eval_weakrl_lag_sweep.csv",
    "eval_weakrl_adv_pos_lag_sweep.csv",
    "eval_weakrl_adv_abs_lag_sweep.csv",
    "eval_weakrl_adv_signed_lag_sweep.csv",
    "eval_ddpo_ft_lag_sweep.csv",
    "eval_ddpo_ft_align_lag_sweep.csv",
]

# dof_lag_max values to exclude from merged plots (e.g. lag=120 eval looks bad; keep lag=100).
OMIT_DOF_LAG_MAX = [120]

# Exact `load_run` folder name -> legend label (analyze_dof_lag_ood --run_alias)
RUN_ALIASES = {
    "2026-03-20_21-40-23baseline_1": "baseline",
    "2026-03-21_11-55-23lag_1": "dr_lag",
    "2026-04-03_16-39-03_weakrl_seed0_env2048": "weakrl",
    "2026-04-03_23-26-29_weakrl_adv_pos_seed0": "weakrl_adv_pos",
    "2026-04-04_06-34-59_weakrl_adv_abs_seed0": "weakrl_adv_abs",
    "2026-04-04_14-11-17_weakrl_adv_signed_seed0": "weakrl_adv_signed",
    "2026-04-04_21-39-05_dppo_ft_lr1e6_seed0_env2048": "weakrl_pos_dppo_ft",
    "2026-04-05_18-19-44_dppo_ft_align_0_env2048": "weakrl_pos_dppo_ft_align",
}


def main():
    out_merged = os.path.join(RESULTS, "eval_all_runs_merged.csv")
    out_stem = "dof_lag_ood_all_eval"

    frames = []
    missing = []
    for name in MERGE_FILES:
        path = os.path.join(RESULTS, name)
        if not os.path.isfile(path):
            missing.append(name)
            continue
        frames.append(pd.read_csv(path))

    if missing:
        print(f"[plot_all_eval] WARNING: missing files (skipped): {missing}", file=sys.stderr)

    if not frames:
        raise SystemExit("[plot_all_eval] No CSVs found; check MERGE_FILES and results/.")

    merged = pd.concat(frames, ignore_index=True)
    os.makedirs(RESULTS, exist_ok=True)
    merged.to_csv(out_merged, index=False)
    print(f"[plot_all_eval] Wrote {out_merged} ({len(merged)} rows)")

    # Runs present in data but not aliased (warn once)
    present = set(merged["load_run"].astype(str).unique())
    unmapped = sorted(present - set(RUN_ALIASES.keys()))
    if unmapped:
        print(
            "[plot_all_eval] WARNING: load_run values without RUN_ALIASES "
            f"(legend will use full folder name): {unmapped}",
            file=sys.stderr,
        )

    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts", "analyze_dof_lag_ood.py"),
        "--csv",
        out_merged,
        "--out_dir",
        RESULTS,
        "--out_stem",
        out_stem,
    ]
    for folder, label in RUN_ALIASES.items():
        cmd.extend(["--run_alias", f"{folder}={label}"])
    for lag in OMIT_DOF_LAG_MAX:
        cmd.extend(["--omit_dof_lag_max", str(lag)])

    print("[plot_all_eval] Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f"[plot_all_eval] Done. Plots: {RESULTS}/{out_stem}_*.png")


if __name__ == "__main__":
    main()
