#!/usr/bin/env bash
set -euo pipefail

label=${1:?usage: run_midlevel_v11_residual_variant.sh LABEL INITIALIZER UPDATES LEARNING_RATE MARGIN_WEIGHT MIDDLE_WEIGHT HARD_WEIGHT HARD_QUANTILE SEED}
initializer=${2:?missing BC initializer}
updates=${3:?missing residual updates}
learning_rate=${4:?missing learning rate}
margin_weight=${5:?missing margin weight}
middle_weight=${6:?missing middle-pocket weight}
hard_weight=${7:?missing hard-task weight}
hard_quantile=${8:?missing hard-task quantile}
seed=${9:?missing seed}

case "$label" in
  *[!a-zA-Z0-9_]*)
    printf 'invalid label: %s\n' "$label" >&2
    exit 2
    ;;
esac

output="outputs/checkpoints/midlevel_two_ball_td3_her_v11_residual_${label}"
training_log="outputs/logs/midlevel_v11_residual_${label}.log"
evaluation_log="outputs/evaluations/midlevel_v11_residual_${label}.log"

mkdir -p outputs/checkpoints outputs/logs outputs/evaluations
printf 'variant=%s initializer=%s updates=%s learning_rate=%s margin_weight=%s middle_weight=%s hard_weight=%s hard_quantile=%s seed=%s status=START time=%s\n' \
  "$label" "$initializer" "$updates" "$learning_rate" "$margin_weight" \
  "$middle_weight" "$hard_weight" "$hard_quantile" "$seed" \
  "$(date --iso-8601=seconds)"

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
  --bc-physical-loss-weight 1.0 \
  --bc-physical-distance-scale-m 0.05 \
  --bc-max-validation-speed-mae-mps 0.0210965 \
  --bc-max-validation-speed-p95-mps 0.0784771 \
  --max-speed-residual-mps 0.03 \
  --offline-residual-warmup-updates "$updates" \
  --offline-residual-learning-rate "$learning_rate" \
  --offline-actor-batch-size 2048 \
  --offline-actor-margin-loss-weight "$margin_weight" \
  --offline-actor-success-margin-m 0.05 \
  --offline-actor-middle-pocket-weight "$middle_weight" \
  --offline-actor-hard-task-weight "$hard_weight" \
  --offline-actor-hard-task-quantile "$hard_quantile" \
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
  2>&1 | tee "$evaluation_log"

printf 'variant=%s status=COMPLETE time=%s checkpoint=%s\n' \
  "$label" "$(date --iso-8601=seconds)" "${output}.residual_only.zip"
