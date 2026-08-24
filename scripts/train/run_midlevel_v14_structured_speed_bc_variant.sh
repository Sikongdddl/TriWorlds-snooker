#!/usr/bin/env bash
set -euo pipefail

angle_mode=${1:?usage: run_midlevel_v14_structured_speed_bc_variant.sh ANGLE_MODE SEED SPEED_REFERENCE}
seed=${2:?missing seed}
speed_reference=${3:-outputs/checkpoints/midlevel_two_ball_td3_her_v10_canonical_e800_b2048_s2.bc_only.zip}
case "$angle_mode" in
  reference|zero) ;;
  *)
    printf 'invalid angle mode: %s\n' "$angle_mode" >&2
    exit 2
    ;;
esac
case "$seed" in
  0|1|2) ;;
  *)
    printf 'formal structured BC seed must be 0, 1, or 2: %s\n' "$seed" >&2
    exit 2
    ;;
esac

readonly output="outputs/checkpoints/midlevel_v14_structured_208896_${angle_mode}_s${seed}.zip"
readonly training_log="outputs/logs/midlevel_v14_structured_208896_${angle_mode}_s${seed}.log"
readonly evaluation_log="outputs/evaluations/midlevel_v14_structured_208896_${angle_mode}_s${seed}.log"
readonly details="outputs/evaluations/midlevel_v14_structured_208896_${angle_mode}_s${seed}.npz"

mkdir -p outputs/checkpoints outputs/logs outputs/evaluations
if [[ -f "$output" && -f "$details" ]]; then
  printf 'structured_bc mode=%s seed=%s status=ALREADY_COMPLETE checkpoint=%s\n' \
    "$angle_mode" "$seed" "$output"
  exit 0
fi
if [[ -f "$output" ]]; then
  printf 'structured_bc mode=%s seed=%s status=REUSE_CHECKPOINT pending_evaluation=true\n' \
    "$angle_mode" "$seed"
else
  # Locked on the fixed 12,288-task development set.  The fresh seed-7 test
  # library remains sealed and is not opened anywhere in this runner.
  python scripts/train/train_midlevel_structured_speed_bc.py \
    --tasks outputs/tasks/midlevel_two_ball_train.npz \
    --offline-speed-curves outputs/diagnostics/midlevel_speed_perturbations_208896.npz \
    --development-tasks outputs/tasks/midlevel_two_ball_validation.npz \
    --angle-reference outputs/checkpoints/midlevel_two_ball_td3_her_v10_canonical_e800_b2048_s2.bc_only.zip \
    --speed-reference "$speed_reference" \
    --angle-mode "$angle_mode" \
    --seed "$seed" \
    --expected-task-count 208896 \
    --epochs 60 \
    --batch-size 4096 \
    --learning-rate 3e-4 \
    --final-learning-rate 3e-5 \
    --speed-weight 1.0 \
    --speed-error-scale-mps 0.005 \
    --canonical-anchor-weight 64.0 \
    --middle-pocket-weight 2.0 \
    --sensitivity-weight-minimum 0.5 \
    --sensitivity-weight-maximum 4.0 \
    --sensitivity-loss-weight 0.2 \
    --sensitivity-distance-scale-m 0.05 \
    --max-grad-norm 1.0 \
    --device cuda:0 \
    --output "$output" \
    2>&1 | tee "$training_log"
fi

python scripts/tools/evaluate_midlevel_two_ball_td3_her.py \
  "$output" \
  --tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --backend mujoco-warp \
  --device cuda:0 \
  --physics-device cuda:0 \
  --num-envs 4096 \
  --chunk-steps 64 \
  --check-interval-steps 8192 \
  --details-output "$details" \
  2>&1 | tee "$evaluation_log"

printf 'structured_bc mode=%s seed=%s speed_reference=%s status=COMPLETE checkpoint=%s\n' \
  "$angle_mode" "$seed" "$speed_reference" "$output"
