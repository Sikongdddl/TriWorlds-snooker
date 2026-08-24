#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_midlevel_v14_structured_control.sh MODE SPEED_REFERENCE}
speed_reference=${2:?missing selected expanded speed checkpoint}
case "$mode" in
  reference|zero) ;;
  *)
    printf 'invalid structured control mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac

readonly angle_reference="outputs/checkpoints/midlevel_two_ball_td3_her_v10_canonical_e800_b2048_s2.bc_only.zip"
readonly details="outputs/evaluations/midlevel_v14_structured_control_${mode}.npz"
readonly evaluation_log="outputs/evaluations/midlevel_v14_structured_control_${mode}.log"

mkdir -p outputs/evaluations
if [[ -f "$details" ]]; then
  printf 'structured_control mode=%s status=ALREADY_COMPLETE details=%s\n' \
    "$mode" "$details"
  exit 0
fi

angle_args=()
if [[ "$mode" == reference ]]; then
  angle_args+=(--angle-checkpoint "$angle_reference")
else
  angle_args+=(--force-zero-angle)
fi

python scripts/tools/evaluate_midlevel_two_ball_td3_her.py \
  "$speed_reference" \
  --tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --backend mujoco-warp \
  --device cuda:0 \
  --physics-device cuda:0 \
  --num-envs 4096 \
  --chunk-steps 64 \
  --check-interval-steps 8192 \
  --details-output "$details" \
  "${angle_args[@]}" \
  2>&1 | tee "$evaluation_log"

printf 'structured_control mode=%s speed_reference=%s status=COMPLETE details=%s\n' \
  "$mode" "$speed_reference" "$details"
