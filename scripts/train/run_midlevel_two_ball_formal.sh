#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

date --iso-8601=seconds
if [[ "${MIDLEVEL_REGENERATE_TASKS:-0}" == "1" ]]; then
  echo "phase=generate_train_and_validation"
  conda run --no-capture-output -n pool python scripts/tools/generate_midlevel_tasks.py \
    --backend mujoco-warp \
    --physics-device cuda:0 \
    --num-worlds 4096 \
    --chunk-steps 64 \
    --check-interval-steps 8192 \
    --train-count 61440 \
    --validation-count 6144 \
    --seed 0 \
    --replay-check -1 \
    --train-output outputs/tasks/midlevel_two_ball_train.npz \
    --validation-output outputs/tasks/midlevel_two_ball_validation.npz
else
  echo "phase=use_existing_validated_task_libraries"
fi

echo "phase=behavior_cloning_and_td3_her"
conda run --no-capture-output -n pool python scripts/train/train_midlevel_two_ball_td3_her.py \
  --tasks outputs/tasks/midlevel_two_ball_train.npz \
  --backend mujoco-warp \
  --physics-device cuda:0 \
  --device cuda:0 \
  --num-envs 4096 \
  --buffer-size 327680 \
  --batch-size 1024 \
  --gradient-steps 64 \
  --learning-starts 0 \
  --actor-learning-rate 1e-5 \
  --critic-learning-rate 3e-4 \
  --critic-warmup-updates 8192 \
  --critic-probe-delta-weight 1.0 \
  --critic-probe-ranking-weight 0.0 \
  --critic-probe-ranking-margin 0.10 \
  --critic-probe-minimum-reward-difference 0.05 \
  --critic-supervision-batch-size 256 \
  --critic-probe-holdout-fraction 0.20 \
  --critic-probe-holdout-seed 20000 \
  --critic-action-center-scale-mps 0.03 \
  --critic-min-pairwise-ranking-agreement 0.0 \
  --critic-min-candidate-selection-count 128 \
  --critic-min-candidate-improvement-precision 0.75 \
  --critic-min-candidate-improvement-precision-lower-95 0.65 \
  --critic-min-candidate-reward-improvement 0.002 \
  --critic-min-candidate-safe-improvement 0.0 \
  --critic-min-candidate-joint-success-improvement 0.0 \
  --critic-max-candidate-failure-increase 0.0 \
  --actor-update-interval 8 \
  --actor-learning-starts 16384 \
  --actor-candidate-supervision-weight 1.0 \
  --actor-physical-probe-supervision-weight 0.0 \
  --actor-candidate-min-q-improvement 0.10 \
  --actor-candidate-min-safe-q 1.5 \
  --actor-candidate-max-critic-disagreement 0.25 \
  --actor-candidate-offsets-mps -0.03 -0.01 0.0 0.01 0.03 \
  --residual-l2-weight 0.02 \
  --max-speed-residual-mps 0.03 \
  --residual-exploration-initial-std 0.35 \
  --residual-exploration-final-std 0.05 \
  --residual-exploration-decay-timesteps 65536 \
  --her-ratio 0.10 \
  --success-replay-ratio 0.20 \
  --failure-replay-ratio 0.20 \
  --local-probe-replay-ratio 0.25 \
  --local-probe-task-count 16384 \
  --local-probe-offsets-mps -0.03 -0.01 0.0 0.01 0.03 \
  --bc-epochs 600 \
  --bc-batch-size 2048 \
  --bc-learning-rate 1e-3 \
  --bc-final-learning-rate 3e-5 \
  --bc-angle-weight 1.0 \
  --bc-speed-weight 8.0 \
  --bc-validation-tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --bc-max-validation-speed-mae-mps 0.025 \
  --bc-max-validation-speed-p95-mps 0.09 \
  --bc-regularization-initial-weight 1.0 \
  --bc-regularization-final-weight 1.0 \
  --bc-regularization-decay-actor-updates 1 \
  --bc-regularization-batch-size 1024 \
  --bc-regularization-residual-weight 0.25 \
  --chunk-steps 64 \
  --check-interval-steps 8192 \
  --total-timesteps 65536 \
  --seed 0 \
  --checkpoint-every 16384 \
  --output outputs/checkpoints/midlevel_two_ball_td3_her_v4

echo "phase=evaluate_validation"
conda run --no-capture-output -n pool python scripts/tools/evaluate_midlevel_two_ball_td3_her.py \
  outputs/checkpoints/midlevel_two_ball_td3_her_v4.zip \
  --tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --backend mujoco-warp \
  --physics-device cuda:0 \
  --num-envs 1024 \
  --chunk-steps 64 \
  --check-interval-steps 8192

date --iso-8601=seconds
echo "phase=complete"
