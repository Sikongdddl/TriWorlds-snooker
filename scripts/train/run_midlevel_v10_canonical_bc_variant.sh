#!/usr/bin/env bash
set -euo pipefail

label=${1:?usage: run_midlevel_v10_canonical_bc_variant.sh LABEL EPOCHS BATCH_SIZE LEARNING_RATE FINAL_LEARNING_RATE SEED}
epochs=${2:?missing epochs}
batch_size=${3:?missing batch size}
learning_rate=${4:?missing learning rate}
final_learning_rate=${5:?missing final learning rate}
seed=${6:?missing seed}

case "$label" in
  *[!a-zA-Z0-9_]*)
    printf 'invalid label: %s\n' "$label" >&2
    exit 2
    ;;
esac

output="outputs/checkpoints/midlevel_two_ball_td3_her_v10_canonical_${label}"
training_log="outputs/logs/midlevel_v10_canonical_${label}.log"
evaluation_log="outputs/evaluations/midlevel_v10_canonical_${label}.log"

mkdir -p outputs/checkpoints outputs/logs outputs/evaluations
printf 'variant=%s epochs=%s batch_size=%s learning_rate=%s final_learning_rate=%s seed=%s status=START time=%s\n' \
  "$label" "$epochs" "$batch_size" "$learning_rate" \
  "$final_learning_rate" "$seed" "$(date --iso-8601=seconds)"

python scripts/train/train_midlevel_two_ball_td3_her.py \
  --tasks outputs/tasks/midlevel_two_ball_train.npz \
  --bc-validation-tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --bc-training-mode canonical \
  --bc-only \
  --bc-epochs "$epochs" \
  --bc-batch-size "$batch_size" \
  --bc-learning-rate "$learning_rate" \
  --bc-final-learning-rate "$final_learning_rate" \
  --bc-angle-weight 1.0 \
  --bc-speed-weight 8.0 \
  --bc-max-validation-speed-mae-mps 0.0210965 \
  --bc-max-validation-speed-p95-mps 0.0784771 \
  --backend mujoco-warp \
  --device cuda:0 \
  --physics-device cuda:0 \
  --num-envs 1 \
  --total-timesteps 1 \
  --seed "$seed" \
  --output "$output" \
  2>&1 | tee "$training_log"

python scripts/tools/evaluate_midlevel_two_ball_td3_her.py \
  "${output}.bc_only.zip" \
  --tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --backend mujoco-warp \
  --device cuda:0 \
  --physics-device cuda:0 \
  --num-envs 1024 \
  --chunk-steps 16 \
  --check-interval-steps 2048 \
  2>&1 | tee "$evaluation_log"

printf 'variant=%s status=COMPLETE time=%s checkpoint=%s\n' \
  "$label" "$(date --iso-8601=seconds)" "${output}.bc_only.zip"
