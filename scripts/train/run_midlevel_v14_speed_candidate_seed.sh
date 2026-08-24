#!/usr/bin/env bash
set -euo pipefail

seed=${1:?usage: run_midlevel_v14_speed_candidate_seed.sh SEED}
case "$seed" in
  0|1|2) ;;
  *)
    printf 'formal speed candidate seed must be 0, 1, or 2: %s\n' "$seed" >&2
    exit 2
    ;;
esac

readonly checkpoint="outputs/checkpoints/midlevel_v14_canonical_208896_s${seed}.bc_only.zip"
readonly angle_reference="outputs/checkpoints/midlevel_two_ball_td3_her_v10_canonical_e800_b2048_s2.bc_only.zip"
readonly details="outputs/evaluations/midlevel_v14_speed_candidate_reference_angle_s${seed}.npz"
readonly evaluation_log="outputs/evaluations/midlevel_v14_speed_candidate_reference_angle_s${seed}.log"

if [[ ! -f "$checkpoint" ]]; then
  printf 'missing expanded canonical speed candidate: %s\n' "$checkpoint" >&2
  exit 2
fi
mkdir -p outputs/evaluations
if [[ -f "$details" ]]; then
  printf 'speed_candidate seed=%s status=ALREADY_COMPLETE details=%s\n' \
    "$seed" "$details"
  exit 0
fi

python scripts/tools/evaluate_midlevel_two_ball_td3_her.py \
  "$checkpoint" \
  --angle-checkpoint "$angle_reference" \
  --tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --backend mujoco-warp \
  --device cuda:0 \
  --physics-device cuda:0 \
  --num-envs 4096 \
  --chunk-steps 64 \
  --check-interval-steps 8192 \
  --details-output "$details" \
  2>&1 | tee "$evaluation_log"

printf 'speed_candidate seed=%s angle_reference=%s status=COMPLETE details=%s\n' \
  "$seed" "$angle_reference" "$details"
