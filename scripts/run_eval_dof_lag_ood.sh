#!/usr/bin/env bash
set -euo pipefail

# OOD eval for DOF lag upper bound.
# Evaluate one or more trained runs under a custom list of dof_lag_max values.
# Runs eval.py with --headless (no Isaac viewer). To debug with viewer: EVAL_WITH_VIEWER=1 bash scripts/run_eval_dof_lag_ood.sh ...

TASK="x1_dh_stand"
LOAD_RUNS=()

SEEDS="0,1,2"
NUM_ENVS=512
NUM_EPISODES=200
CHECKPOINT=-1
# Optional: path to a single .pt (same as eval.py --model_path). If set, ignores --load_run(s) and --checkpoint.
MODEL_PATH=""

# default OOD list: includes 0 (no lag), then in-range (5/15/40) and beyond (60/80)
DOF_LAG_MAX_LIST="0,5,15,40,60,80"
DOF_LAG_MIN=0

OUT_CSV="results/dof_lag_ood_eval.csv"
OVERWRITE=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_eval_dof_lag_ood.sh \
    --load_run <folder_name> \
    [--checkpoint <N>]   # loads model_N.pt; default -1 = latest in folder
    [--seeds 0,1,2] \
    [--num_envs 512] \
    [--num_episodes 200] \
    [--dof_lag_max_list 0,5,15,40,60,80] \
    [--dof_lag_min 0] \
    [--out results/dof_lag_ood_eval.csv] \
    [--overwrite]

Or multiple runs (each uses the same --checkpoint):
  bash scripts/run_eval_dof_lag_ood.sh \
    --load_runs a,b,c \
    --checkpoint 3500 \
    --dof_lag_max_list 0,10,20,40,60

Or single model file (best checkpoint path from tensorboard):
  bash scripts/run_eval_dof_lag_ood.sh \
    --model_path logs/x1_dh_stand/exported_data/<run>/model_3500.pt \
    --dof_lag_max_list 0,5,15,40,60,80
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --load_run) LOAD_RUNS+=("${2:-}"); shift 2 ;;
    --load_runs)
      IFS=',' read -r -a tmp <<< "${2:-}"
      for x in "${tmp[@]}"; do LOAD_RUNS+=("$x"); done
      shift 2
      ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --num_envs) NUM_ENVS="${2:-}"; shift 2 ;;
    --num_episodes) NUM_EPISODES="${2:-}"; shift 2 ;;
    --checkpoint)
      if [[ -z "${2:-}" ]]; then
        echo "[run_eval_dof_lag_ood] ERROR: --checkpoint needs a value (integer, or -1 for latest model_*.pt)."
        echo "  Do not leave --checkpoint empty; omit the whole --checkpoint line to use -1 (latest)."
        exit 1
      fi
      if [[ "${2}" == --* ]]; then
        echo "[run_eval_dof_lag_ood] ERROR: --checkpoint was not given a number; next token is '${2}'."
        echo "  (Common mistake: line break after --checkpoint so the next flag was eaten.)"
        echo "  Example: --checkpoint 3500   or   omit --checkpoint for latest."
        exit 1
      fi
      CHECKPOINT="${2}"
      shift 2
      ;;
    --model_path) MODEL_PATH="${2:-}"; shift 2 ;;
    --dof_lag_max_list) DOF_LAG_MAX_LIST="${2:-}"; shift 2 ;;
    --dof_lag_min) DOF_LAG_MIN="${2:-}"; shift 2 ;;
    --out) OUT_CSV="${2:-}"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[run_eval_dof_lag_ood] Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]] && [[ ${#LOAD_RUNS[@]} -eq 0 ]]; then
  echo "[run_eval_dof_lag_ood] Missing --load_run(s) or --model_path"
  usage
  exit 1
fi

IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"
IFS=',' read -r -a DOF_MAX_ARR <<< "${DOF_LAG_MAX_LIST}"

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
    --num_envs "${NUM_ENVS}"
    --num_episodes "${NUM_EPISODES}"
    --seed "${seed}"
    --out "${OUT_CSV}"
  )

  if [[ -n "${MODEL_PATH}" ]]; then
    cmd+=(--model_path "${MODEL_PATH}")
  else
    cmd+=(--load_run "${load_run}" --checkpoint "${CHECKPOINT}")
  fi

  if [[ "${enable_dof_lag}" -eq 1 ]]; then
    cmd+=(
      --enable_dof_lag
      --dof_lag_min "${dof_lag_min}"
      --dof_lag_max "${dof_lag_max}"
      --dof_lag_mode fixed
    )
  fi

  echo "[run_eval_dof_lag_ood] load_run=${load_run} model_path=${MODEL_PATH:-} checkpoint=${CHECKPOINT} seed=${seed} enable=${enable_dof_lag} min=${dof_lag_min} max=${dof_lag_max}"
  "${cmd[@]}"
}

run_ood_loop() {
  local lr="$1"
  for seed in "${SEED_ARR[@]}"; do
    for dof_max in "${DOF_MAX_ARR[@]}"; do
      if [[ "${dof_max}" == "0" ]]; then
        eval_one "${lr}" "${seed}" 0 0 0
      else
        eval_one "${lr}" "${seed}" 1 "${DOF_LAG_MIN}" "${dof_max}"
      fi
    done
  done
}

if [[ -n "${MODEL_PATH}" ]]; then
  echo "[run_eval_dof_lag_ood] Using --model_path (single checkpoint file)"
  run_ood_loop ""
else
  for lr in "${LOAD_RUNS[@]}"; do
    run_ood_loop "${lr}"
  done
fi

echo "[run_eval_dof_lag_ood] Done. CSV appended: ${OUT_CSV}"

