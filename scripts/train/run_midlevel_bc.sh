#!/usr/bin/env bash

set -euo pipefail

gpu_index=${1:?usage: run_midlevel_bc.sh GPU_INDEX}
readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly checkpoint="${MIDLEVEL_BC_OUTPUT:-outputs/checkpoints/midlevel_bc.pt}"
readonly validation_report="${MIDLEVEL_BC_VALIDATION_REPORT:-outputs/evaluations/midlevel_bc_validation.json}"
readonly validation_details="${MIDLEVEL_BC_VALIDATION_DETAILS:-outputs/evaluations/midlevel_bc_validation.npz}"
readonly training_log="${MIDLEVEL_BC_TRAINING_LOG:-outputs/logs/midlevel_bc_training.log}"
readonly validation_log="${MIDLEVEL_BC_VALIDATION_LOG:-outputs/logs/midlevel_bc_validation.log}"

cd "$repo_root"
export PYTHONUNBUFFERED=1

read -r gpu_memory_mib gpu_utilization < <(
  nvidia-smi \
    --id="$gpu_index" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    | tr -d ','
)
if [[ "${MIDLEVEL_ALLOW_BUSY_GPU:-0}" != "1" ]] \
  && (( gpu_memory_mib > 512 || gpu_utilization > 10 )); then
  printf 'selected GPU is busy: gpu=%s memory_mib=%s utilization=%s%%\n' \
    "$gpu_index" "$gpu_memory_mib" "$gpu_utilization" >&2
  exit 2
fi
printf 'gpu=%s memory_mib=%s utilization=%s%% status=accepted\n' \
  "$gpu_index" "$gpu_memory_mib" "$gpu_utilization"

if [[ "${MIDLEVEL_REGENERATE_TASKS:-0}" == "1" ]]; then
  printf '%s\n' \
    'MIDLEVEL_REGENERATE_TASKS is disabled: the 393216-task training set uses' \
    'a balanced core plus audited targeted local-speed augmentation. Run' \
    'scripts/tools/run_midlevel_local_speed_expansion_8gpu.sh separately.' >&2
  exit 3
fi

overwrite_args=()
if [[ "${MIDLEVEL_BC_OVERWRITE:-0}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

mkdir -p \
  "$(dirname "$checkpoint")" \
  "$(dirname "$validation_report")" \
  "$(dirname "$validation_details")" \
  "$(dirname "$training_log")" \
  "$(dirname "$validation_log")"
CUDA_VISIBLE_DEVICES="$gpu_index" conda run --no-capture-output -n pool \
  python scripts/train/train_midlevel_bc.py \
    --tasks outputs/tasks/midlevel_two_ball_train.npz \
    --validation-tasks outputs/tasks/midlevel_two_ball_validation.npz \
    --epochs "${MIDLEVEL_BC_EPOCHS:-800}" \
    --batch-size "${MIDLEVEL_BC_BATCH_SIZE:-2048}" \
    --learning-rate "${MIDLEVEL_BC_LEARNING_RATE:-1e-3}" \
    --final-learning-rate "${MIDLEVEL_BC_FINAL_LEARNING_RATE:-3e-5}" \
    --angle-weight "${MIDLEVEL_BC_ANGLE_WEIGHT:-1.0}" \
    --speed-weight "${MIDLEVEL_BC_SPEED_WEIGHT:-8.0}" \
    --seed 0 \
    --device cuda:0 \
    --output "$checkpoint" \
    "${overwrite_args[@]}" \
  2>&1 | tee "$training_log"

CUDA_VISIBLE_DEVICES="$gpu_index" conda run --no-capture-output -n pool \
  python scripts/tools/evaluate_midlevel_bc.py \
    "$checkpoint" \
    --tasks outputs/tasks/midlevel_two_ball_validation.npz \
    --backend mujoco-warp \
    --device cuda:0 \
    --physics-device cuda:0 \
    --num-envs "${MIDLEVEL_EVAL_ENVS:-4096}" \
    --chunk-steps 64 \
    --check-interval-steps 8192 \
    --report-output "$validation_report" \
    --details-output "$validation_details" \
    "${overwrite_args[@]}" \
  2>&1 | tee "$validation_log"

printf 'midlevel_bc status=COMPLETE checkpoint=%s validation=%s\n' \
  "$checkpoint" "$validation_report"
