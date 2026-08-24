#!/usr/bin/env bash
set -euo pipefail

label=${1:?usage: run_midlevel_v13_safe_candidate_variant.sh LABEL INITIALIZER SEED STEP ENSEMBLE POCKET_HEADS POSITIVE_WEIGHT SELECTION_WEIGHT SELECTION_TARGET MIN_PROBABILITY MAX_DISAGREEMENT UNCERTAINTY_SCALE RESIDUAL_PENALTY HIDDEN_PRESET}
initializer=${2:?missing BC initializer}
seed=${3:?missing seed}
step_mps=${4:?missing candidate step}
ensemble_size=${5:?missing ensemble size}
pocket_heads=${6:?missing pocket head count}
positive_weight=${7:?missing positive weight}
selection_weight=${8:?missing selection loss weight}
selection_target=${9:?missing selection target}
min_probability=${10:?missing minimum probability}
max_disagreement=${11:?missing maximum disagreement}
uncertainty_scale=${12:?missing uncertainty scale}
residual_penalty=${13:?missing residual penalty}
hidden_preset=${14:?missing hidden preset}

case "$label" in
  *[!a-zA-Z0-9_]*)
    printf 'invalid label: %s\n' "$label" >&2
    exit 2
    ;;
esac
case "$hidden_preset" in
  medium) hidden_sizes=(256 256 128) ;;
  large) hidden_sizes=(512 512 256) ;;
  *)
    printf 'invalid hidden preset: %s\n' "$hidden_preset" >&2
    exit 2
    ;;
esac

output="outputs/checkpoints/midlevel_two_ball_td3_her_v13_${label}"
training_log="outputs/logs/midlevel_v13_${label}.log"
evaluation_log="outputs/evaluations/midlevel_v13_${label}.log"
details_output="outputs/evaluations/midlevel_v13_${label}.details.npz"

for path in "${output}.residual_only.zip" "$details_output"; do
  if [[ -e "$path" ]]; then
    printf 'refusing to overwrite existing output: %s\n' "$path" >&2
    exit 2
  fi
done

mkdir -p outputs/checkpoints outputs/logs outputs/evaluations
printf 'variant=%s initializer=%s seed=%s step_mps=%s ensemble=%s pocket_heads=%s positive_weight=%s selection_weight=%s selection_target=%s min_probability=%s max_disagreement=%s uncertainty_scale=%s residual_penalty=%s hidden_preset=%s status=START time=%s\n' \
  "$label" "$initializer" "$seed" "$step_mps" "$ensemble_size" \
  "$pocket_heads" "$positive_weight" "$selection_weight" \
  "$selection_target" "$min_probability" "$max_disagreement" \
  "$uncertainty_scale" "$residual_penalty" "$hidden_preset" \
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
  --bc-physical-loss-weight 0.0 \
  --bc-max-validation-speed-mae-mps 0.0210965 \
  --bc-max-validation-speed-p95-mps 0.0784771 \
  --max-speed-residual-mps 0.12 \
  --offline-safe-candidate-classifier \
  --safe-candidate-step-mps "$step_mps" \
  --safe-candidate-updates 4096 \
  --safe-candidate-learning-rate 3.0e-4 \
  --safe-candidate-batch-size 2048 \
  --safe-candidate-ensemble-size "$ensemble_size" \
  --safe-candidate-pocket-heads "$pocket_heads" \
  --safe-candidate-hidden-sizes "${hidden_sizes[@]}" \
  --safe-candidate-positive-weight "$positive_weight" \
  --safe-candidate-selection-loss-weight "$selection_weight" \
  --safe-candidate-selection-target "$selection_target" \
  --safe-candidate-unknown-weight 0.25 \
  --safe-candidate-label-tolerance-mps "$(awk -v step="$step_mps" 'BEGIN { print step / 2.0 }')" \
  --safe-candidate-min-probability "$min_probability" \
  --safe-candidate-min-improvement 0.0 \
  --safe-candidate-max-disagreement "$max_disagreement" \
  --safe-candidate-uncertainty-scale "$uncertainty_scale" \
  --safe-candidate-residual-penalty "$residual_penalty" \
  --backend mujoco-warp \
  --device cuda:0 \
  --physics-device cuda:0 \
  --num-envs 1 \
  --total-timesteps 1 \
  --seed "$seed" \
  --output "$output" \
  2>&1 | tee "$training_log"

python scripts/tools/evaluate_midlevel_two_ball_sac_her.py \
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

printf 'variant=%s status=COMPLETE time=%s checkpoint=%s details=%s\n' \
  "$label" "$(date --iso-8601=seconds)" \
  "${output}.residual_only.zip" "$details_output"
