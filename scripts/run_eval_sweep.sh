#!/usr/bin/env bash
set -euo pipefail

# Eval sweep for DOF lag robustness study.
# This script runs:
#   baseline vs robust  ×  (no/low/med/high) lag  ×  seeds
# and appends all results into a single CSV.

TASK="x1_dh_stand"
BASELINE_RUN=""
ROBUST_RUN=""
LOAD_RUNS=()

SEEDS="0,1,2"
NUM_ENVS=512
NUM_EPISODES=200
CHECKPOINT=-1
OUT_CSV="results/dof_lag_sweep_eval.csv"
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_eval_sweep.sh \
    --load_run <folder_name> \
    [--seeds 0,1,2] \
    [--num_envs 512] \
    [--num_episodes 200] \
    [--checkpoint -1] \
    [--out results/dof_lag_sweep_eval.csv] \
    [--overwrite]

Or compare two specific policies (backward compatible):
  bash scripts/run_eval_sweep.sh \
    --baseline_run <baseline_folder_name> \
    --robust_run <robust_folder_name> \
    [--seeds 0,1,2] \
    [--num_envs 512] \
    [--num_episodes 200] \
    [--checkpoint -1] \
    [--out results/dof_lag_sweep_eval.csv] \
    [--overwrite]
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline_run) BASELINE_RUN="${2:-}"; shift 2 ;;
    --robust_run) ROBUST_RUN="${2:-}"; shift 2 ;;
    --load_run) LOAD_RUNS+=("${2:-}"); shift 2 ;;
    --load_runs) IFS=',' read -r -a tmp <<< "${2:-}"; for x in "${tmp[@]}"; do LOAD_RUNS+=("$x"); done; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --num_envs) NUM_ENVS="${2:-}"; shift 2 ;;
    --num_episodes) NUM_EPISODES="${2:-}"; shift 2 ;;
    --checkpoint)
      if [[ -z "${2:-}" || "${2}" == --* ]]; then
        echo "[run_eval_sweep] ERROR: --checkpoint needs a number or -1; got: '${2:-}'"
        echo "  Omit --checkpoint to use latest (default -1)."
        exit 1
      fi
      CHECKPOINT="${2}"
      shift 2
      ;;
    --out) OUT_CSV="${2:-}"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[run_eval_sweep] Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ ${#LOAD_RUNS[@]} -eq 0 ]]; then
  if [[ -n "${BASELINE_RUN}" ]]; then LOAD_RUNS+=("${BASELINE_RUN}"); fi
  if [[ -n "${ROBUST_RUN}" ]]; then LOAD_RUNS+=("${ROBUST_RUN}"); fi
fi

if [[ ${#LOAD_RUNS[@]} -eq 0 ]]; then
  echo "[run_eval_sweep] Missing --load_run(s) or (--baseline_run and/or --robust_run)."
  usage
  exit 1
fi

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

if [[ "${OUT_CSV}" != /* ]]; then
  OUT_CSV="${REPO_ROOT}/${OUT_CSV}"
fi

if [[ "${OVERWRITE}" -eq 1 ]]; then
  rm -f "${OUT_CSV}"
fi

mkdir -p "$(dirname "${OUT_CSV}")"

eval_one() {
  local load_run="$1"
  local seed="$2"
  local enable_dof_lag="$3" # 0 or 1
  local dof_lag_min="$4"
  local dof_lag_max="$5"

  local cmd=(
    python "${REPO_ROOT}/humanoid/scripts/eval.py"
    --task "${TASK}"
    --headless
    --load_run "${load_run}"
    --checkpoint "${CHECKPOINT}"
    --num_envs "${NUM_ENVS}"
    --num_episodes "${NUM_EPISODES}"
    --seed "${seed}"
    --out "${OUT_CSV}"
  )

  if [[ "${enable_dof_lag}" -eq 1 ]]; then
    cmd+=(
      --enable_dof_lag
      --dof_lag_min "${dof_lag_min}"
      --dof_lag_max "${dof_lag_max}"
      --dof_lag_mode fixed
    )
  fi

  echo "[run_eval_sweep] Running: load_run=${load_run} seed=${seed} enable_dof_lag=${enable_dof_lag} min=${dof_lag_min} max=${dof_lag_max}"
  "${cmd[@]}"
}

run_for_model() {
  local load_run="$1"
  for seed in "${SEED_ARR[@]}"; do
    # no lag
    eval_one "${load_run}" "${seed}" 0 0 0

    # low / med / high (fixed uses max timesteps)
    eval_one "${load_run}" "${seed}" 1 0 5
    eval_one "${load_run}" "${seed}" 1 5 15
    eval_one "${load_run}" "${seed}" 1 15 40
  done
}

echo "[run_eval_sweep] Output CSV: ${OUT_CSV}"
echo "[run_eval_sweep] Seeds: ${SEEDS}"

for lr in "${LOAD_RUNS[@]}"; do
  run_for_model "${lr}"
done

echo "[run_eval_sweep] Done. CSV appended: ${OUT_CSV}"

