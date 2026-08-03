#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONUNBUFFERED=1

date --iso-8601=seconds
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

echo "phase=behavior_cloning_and_ppo"
conda run --no-capture-output -n pool python scripts/train/train_midlevel_two_ball_ppo.py \
  --tasks outputs/tasks/midlevel_two_ball_train.npz \
  --backend mujoco-warp \
  --physics-device cuda:0 \
  --num-envs 4096 \
  --batch-size 1024 \
  --n-epochs 2 \
  --learning-rate 3e-4 \
  --angle-action-std 0.05 \
  --speed-action-std 0.25 \
  --bc-epochs 100 \
  --bc-batch-size 1024 \
  --bc-learning-rate 1e-3 \
  --bc-angle-weight 4.0 \
  --chunk-steps 64 \
  --check-interval-steps 8192 \
  --total-timesteps 1000000 \
  --seed 0 \
  --checkpoint-every 8192 \
  --output outputs/checkpoints/midlevel_two_ball_ppo

echo "phase=evaluate_validation"
conda run --no-capture-output -n pool python scripts/tools/evaluate_midlevel_two_ball_ppo.py \
  outputs/checkpoints/midlevel_two_ball_ppo.zip \
  --tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --backend mujoco-warp \
  --physics-device cuda:0 \
  --num-envs 1024 \
  --chunk-steps 64 \
  --check-interval-steps 8192

date --iso-8601=seconds
echo "phase=complete"
