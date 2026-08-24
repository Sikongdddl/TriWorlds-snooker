#!/usr/bin/env bash
set -euo pipefail

label=${1:?usage: run_midlevel_v12_residual_variant.sh LABEL INITIALIZER UPDATES LEARNING_RATE PHYSICAL_WEIGHT INTERVAL_WEIGHT HARD_METRIC HARD_WEIGHT SEED [BASELINE_DETAILS_LABEL]}
initializer=${2:?missing BC initializer}
updates=${3:?missing residual updates}
learning_rate=${4:?missing learning rate}
physical_weight=${5:?missing physical loss weight}
interval_weight=${6:?missing success-interval loss weight}
hard_metric=${7:?missing hard-task metric}
hard_weight=${8:?missing hard-task weight}
seed=${9:?missing seed}
baseline_details_label=${10:-}

case "$label" in
  *[!a-zA-Z0-9_]*)
    printf 'invalid label: %s\n' "$label" >&2
    exit 2
    ;;
esac
if [[ -n "$baseline_details_label" ]]; then
  case "$baseline_details_label" in
    *[!a-zA-Z0-9_]*)
      printf 'invalid baseline details label: %s\n' \
        "$baseline_details_label" >&2
      exit 2
      ;;
  esac
fi

output="outputs/checkpoints/midlevel_two_ball_td3_her_v12_${label}"
training_log="outputs/logs/midlevel_v12_${label}.log"
evaluation_log="outputs/evaluations/midlevel_v12_${label}.log"
details_output="outputs/evaluations/midlevel_v12_${label}.details.npz"

for path in "${output}.residual_only.zip" "$details_output"; do
  if [[ -e "$path" ]]; then
    printf 'refusing to overwrite existing output: %s\n' "$path" >&2
    exit 2
  fi
done

mkdir -p outputs/checkpoints outputs/logs outputs/evaluations
printf 'variant=%s initializer=%s updates=%s learning_rate=%s physical_weight=%s interval_weight=%s hard_metric=%s hard_weight=%s seed=%s status=START time=%s\n' \
  "$label" "$initializer" "$updates" "$learning_rate" \
  "$physical_weight" "$interval_weight" "$hard_metric" "$hard_weight" \
  "$seed" "$(date --iso-8601=seconds)"

python scripts/train/train_midlevel_two_ball_sac_her.py \
  --tasks outputs/tasks/midlevel_two_ball_train.npz \
  --offline-speed-curves \
    outputs/diagnostics/midlevel_speed_perturbations_196608.npz \
  --bc-validation-tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --bc-training-mode canonical \
  --residual-only \
  --initialize-from-bc "$initializer" \
  --bc-hindsight-fraction 0.0 \
  --bc-angle-weight 1.0 \
  --bc-speed-weight 8.0 \
  --bc-physical-loss-weight "$physical_weight" \
  --bc-physical-distance-scale-m 0.05 \
  --bc-max-validation-speed-mae-mps 0.0210965 \
  --bc-max-validation-speed-p95-mps 0.0784771 \
  --max-speed-residual-mps 0.03 \
  --offline-residual-warmup-updates "$updates" \
  --offline-residual-learning-rate "$learning_rate" \
  --offline-actor-batch-size 2048 \
  --offline-actor-margin-loss-weight 0.0 \
  --offline-actor-success-margin-m 0.05 \
  --offline-actor-success-interval-loss-weight "$interval_weight" \
  --offline-actor-success-interval-scale-mps 0.01 \
  --offline-actor-middle-pocket-weight 1.0 \
  --offline-actor-hard-task-weight "$hard_weight" \
  --offline-actor-hard-task-quantile 0.75 \
  --offline-actor-hard-task-metric "$hard_metric" \
  --backend mujoco-warp \
  --device cuda:0 \
  --physics-device cuda:0 \
  --num-envs 1 \
  --total-timesteps 1 \
  --seed "$seed" \
  --output "$output" \
  2>&1 | tee "$training_log"

python scripts/tools/evaluate_midlevel_two_ball_td3_her.py \
  "${output}.residual_only.zip" \
  --tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --backend mujoco-warp \
  --device cuda:0 \
  --physics-device cuda:0 \
  --num-envs 1024 \
  --chunk-steps 16 \
  --check-interval-steps 2048 \
  --details-output "$details_output" \
  2>&1 | tee "$evaluation_log"

if [[ -n "$baseline_details_label" ]]; then
  baseline_log="outputs/evaluations/midlevel_v12_baseline_${baseline_details_label}.log"
  baseline_details="outputs/evaluations/midlevel_v12_baseline_${baseline_details_label}.details.npz"
  if [[ -e "$baseline_details" ]]; then
    printf 'refusing to overwrite existing output: %s\n' \
      "$baseline_details" >&2
    exit 2
  fi
  python scripts/tools/evaluate_midlevel_two_ball_td3_her.py \
    "$initializer" \
    --tasks outputs/tasks/midlevel_two_ball_validation.npz \
    --backend mujoco-warp \
    --device cuda:0 \
    --physics-device cuda:0 \
    --num-envs 1024 \
    --chunk-steps 16 \
    --check-interval-steps 2048 \
    --details-output "$baseline_details" \
    2>&1 | tee "$baseline_log"
fi

printf 'variant=%s status=COMPLETE time=%s checkpoint=%s details=%s\n' \
  "$label" "$(date --iso-8601=seconds)" \
  "${output}.residual_only.zip" "$details_output"
