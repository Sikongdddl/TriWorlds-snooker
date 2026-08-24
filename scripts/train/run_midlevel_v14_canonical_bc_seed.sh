#!/usr/bin/env bash
set -euo pipefail

seed=${1:?usage: run_midlevel_v14_canonical_bc_seed.sh SEED}
case "$seed" in
  0|1|2) ;;
  *)
    printf 'formal canonical BC seed must be 0, 1, or 2: %s\n' "$seed" >&2
    exit 2
    ;;
esac

readonly output="outputs/checkpoints/midlevel_v14_canonical_208896_s${seed}"
readonly training_log="outputs/logs/midlevel_v14_canonical_208896_s${seed}.log"
readonly evaluation_log="outputs/evaluations/midlevel_v14_canonical_208896_s${seed}.log"
readonly details="outputs/evaluations/midlevel_v14_canonical_208896_s${seed}.npz"

python - <<'PY'
import json
import numpy as np

with np.load("outputs/tasks/midlevel_two_ball_train.npz", allow_pickle=False) as archive:
    metadata = json.loads(str(archive["metadata"].item()))
if int(metadata["task_count"]) != 208_896:
    raise RuntimeError(
        f"formal BC requires 208896 tasks, got {metadata['task_count']}"
    )
print(
    "canonical_bc_library_check=PASS "
    f"tasks={metadata['task_count']} hash={metadata['content_sha256']}",
    flush=True,
)
PY

mkdir -p outputs/checkpoints outputs/logs outputs/evaluations
if [[ -f "${output}.bc_only.zip" && -f "$details" ]]; then
  printf 'canonical_bc seed=%s status=ALREADY_COMPLETE checkpoint=%s\n' \
    "$seed" "${output}.bc_only.zip"
  exit 0
fi
if [[ -f "${output}.bc_only.zip" ]]; then
  printf 'canonical_bc seed=%s status=REUSE_CHECKPOINT pending_evaluation=true\n' \
    "$seed"
else
  python scripts/train/train_midlevel_two_ball_td3_her.py \
    --tasks outputs/tasks/midlevel_two_ball_train.npz \
    --bc-validation-tasks outputs/tasks/midlevel_two_ball_validation.npz \
    --bc-training-mode canonical \
    --bc-only \
    --bc-epochs 800 \
    --bc-batch-size 2048 \
    --bc-learning-rate 1e-3 \
    --bc-final-learning-rate 3e-5 \
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
fi

python scripts/tools/evaluate_midlevel_two_ball_td3_her.py \
  "${output}.bc_only.zip" \
  --tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --backend mujoco-warp \
  --device cuda:0 \
  --physics-device cuda:0 \
  --num-envs 4096 \
  --chunk-steps 64 \
  --check-interval-steps 8192 \
  --details-output "$details" \
  2>&1 | tee "$evaluation_log"

printf 'canonical_bc seed=%s status=COMPLETE checkpoint=%s\n' \
  "$seed" "${output}.bc_only.zip"
