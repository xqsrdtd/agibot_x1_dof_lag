#!/usr/bin/env python3
"""
Fix episode-metric columns in eval CSVs affected by the old evaluate() bug in eval.py
(summing n * batch_mean into totals but dividing by num_episodes when n > remaining).

Only rows that actually show inflated aggregates are rescaled (multiply metrics by
num_episodes / num_envs). Detection heuristic (for num_envs > num_episodes):
  - timeout_rate > 1.01  OR  active_cmd_ratio > 1.05
Rows where the run accumulated episodes in smaller waves (so rates stayed in [0,1] and
active_cmd_ratio <= 1) are left unchanged — a blanket scale would wrongly shrink them.

Re-run eval with the fixed eval.py for ground truth.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from typing import List

METRIC_KEYS = [
    "lin_vel_mae",
    "ang_vel_mae",
    "active_cmd_ratio",
    "tau2_mean",
    "power_abs_mean",
    "timeout_rate",
    "fall_rate",
]

REQUIRED_META = ("num_envs", "num_episodes")


def row_needs_rescale(row: dict) -> bool:
    ne = int(row["num_envs"])
    nep = int(row["num_episodes"])
    if ne <= nep:
        return False
    tr = float(row["timeout_rate"])
    ac = float(row["active_cmd_ratio"])
    return tr > 1.01 or ac > 1.05


def rescale_row(row: dict) -> bool:
    if not row_needs_rescale(row):
        return False
    factor = int(row["num_episodes"]) / float(int(row["num_envs"]))
    for k in METRIC_KEYS:
        if k in row and row[k] != "":
            row[k] = str(float(row[k]) * factor)
    return True


def process_file(path: str, dry_run: bool) -> dict:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames or not all(k in fieldnames for k in REQUIRED_META):
            return {"path": path, "skipped": True, "reason": "missing columns"}
        if not all(k in fieldnames for k in METRIC_KEYS):
            return {"path": path, "skipped": True, "reason": "missing metric columns"}
        rows = list(reader)

    n_scaled = 0
    for row in rows:
        if rescale_row(row):
            n_scaled += 1

    if dry_run:
        return {"path": path, "skipped": False, "dry_run": True, "rows": len(rows), "scaled_rows": n_scaled}

    if n_scaled == 0:
        return {"path": path, "skipped": False, "rows": len(rows), "scaled_rows": 0, "note": "no rows matched bug signature"}

    bak = path + ".pre_rowwise_rescale_bak"
    shutil.copy2(path, bak)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {"path": path, "skipped": False, "backup": bak, "rows": len(rows), "scaled_rows": n_scaled}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "paths",
        nargs="*",
        help="CSV files (default: scan results/ for eval-style CSVs)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change; do not write files",
    )
    args = p.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..", "..")
    results_dir = os.path.abspath(os.path.join(root, "results"))

    if args.paths:
        paths = [os.path.abspath(x) for x in args.paths]
    else:
        paths = []
        if os.path.isdir(results_dir):
            for name in sorted(os.listdir(results_dir)):
                if not name.endswith(".csv"):
                    continue
                full = os.path.join(results_dir, name)
                with open(full, newline="") as f:
                    header = f.readline()
                if "active_cmd_ratio" in header and "num_envs" in header:
                    paths.append(full)

    reports: List[dict] = []
    for path in paths:
        reports.append(process_file(path, args.dry_run))

    for r in reports:
        if r.get("skipped"):
            print(f"[skip] {r['path']}: {r.get('reason')}")
        elif r.get("dry_run"):
            print(f"[dry-run] {r['path']}: would scale {r['scaled_rows']}/{r['rows']} rows")
        elif r.get("scaled_rows", 0) == 0:
            print(f"[ok] {r['path']}: no rows scaled ({r.get('note', '')})")
        else:
            print(f"[ok] {r['path']}: scaled {r['scaled_rows']}/{r['rows']} rows -> backup {r['backup']}")


if __name__ == "__main__":
    main()
