#!/usr/bin/env bash

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

BASE_SHARD_DIR=outputs/tasks/midlevel_balanced_shards/v1_seed0
EXTENSION_SHARD_DIR=outputs/tasks/midlevel_balanced_shards/extension_to_196608
EXTENSION_LOG_DIR=outputs/logs/midlevel_balanced_collection_extension_196608
POST_LOG_DIR=outputs/logs/midlevel_balanced_postprocess_196608
FIRST_WAVE_PID=${FIRST_WAVE_PID:-}

TRAIN_STAGED=outputs/tasks/midlevel_two_ball_train.balanced_196608_unvalidated.npz
VALIDATION_STAGED=outputs/tasks/midlevel_two_ball_validation.balanced_12288_unvalidated.npz
MERGE_MANIFEST=outputs/tasks/midlevel_two_ball_balanced_196608_merge_manifest.json

mkdir -p "$EXTENSION_SHARD_DIR" "$EXTENSION_LOG_DIR" "$POST_LOG_DIR"
exec > >(tee -a "$POST_LOG_DIR/controller.log") 2>&1

timestamp() {
  date -Iseconds
}

base_missing_count() {
  local missing=0
  local shard_index
  for shard_index in 0 1 2 3 4 5 6 7; do
    [[ -f "$BASE_SHARD_DIR/train_shard_${shard_index}.npz" ]] \
      || missing=$((missing + 1))
    [[ -f "$BASE_SHARD_DIR/validation_shard_${shard_index}.npz" ]] \
      || missing=$((missing + 1))
  done
  printf '%s\n' "$missing"
}

first_wave_active() {
  if [[ -n "$FIRST_WAVE_PID" ]] && kill -0 "$FIRST_WAVE_PID" 2>/dev/null; then
    return 0
  fi
  pgrep -f \
    'generate_midlevel_tasks.py.*midlevel_balanced_shards/v1_seed0' \
    >/dev/null
}

wait_for_idle_gpus() {
  local busy
  local memory_used
  local utilization
  while true; do
    busy=0
    while IFS=',' read -r memory_used utilization; do
      memory_used=${memory_used// /}
      utilization=${utilization// /}
      if (( memory_used > 512 || utilization > 10 )); then
        busy=$((busy + 1))
      fi
    done < <(
      nvidia-smi \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    )
    if (( busy == 0 )); then
      return
    fi
    printf 'controller_wait busy_gpus=%s timestamp=%s\n' \
      "$busy" "$(timestamp)"
    sleep 30
  done
}

ACTIVE_CHILD_PIDS=()

terminate_active_children() {
  local pid
  trap - HUP INT TERM
  for pid in "${ACTIVE_CHILD_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  exit 130
}

trap terminate_active_children HUP INT TERM

printf 'controller_tmux_start target_train=196608 target_validation=12288 timestamp=%s\n' \
  "$(timestamp)"

while true; do
  missing=$(base_missing_count)
  if (( missing == 0 )); then
    break
  fi
  if ! first_wave_active; then
    printf 'controller_abort reason=first_wave_stopped missing=%s timestamp=%s\n' \
      "$missing" "$(timestamp)"
    exit 2
  fi
  printf 'controller_wait first_wave_active=1 missing_shards=%s timestamp=%s\n' \
    "$missing" "$(timestamp)"
  sleep 30
done

printf 'controller_first_wave status=PASS train=61440 validation=6144 timestamp=%s\n' \
  "$(timestamp)"

existing=0
for shard_index in 0 1 2 3 4 5 6 7; do
  for split in train validation; do
    output="$EXTENSION_SHARD_DIR/${split}_shard_${shard_index}.npz"
    if [[ -e "$output" || -e "${output%.npz}.unvalidated.npz" ]]; then
      existing=$((existing + 1))
    fi
  done
done
if (( existing != 0 )); then
  printf 'controller_abort reason=extension_outputs_exist count=%s timestamp=%s\n' \
    "$existing" "$(timestamp)"
  exit 3
fi

wait_for_idle_gpus

TRAIN_COUNTS=(16848 16848 16848 16848 16848 16848 16848 17232)
VALIDATION_COUNTS=(756 756 756 756 756 756 756 852)
TRAIN_SEEDS=(20000 20002 20004 20006 20008 20010 20012 7809)
VALIDATION_CLI_SEEDS=(30000 30002 30004 30006 30008 30010 30012 1006)

for shard_index in 0 1 2 3 4 5 6 7; do
  train_output="$EXTENSION_SHARD_DIR/train_shard_${shard_index}.npz"
  validation_output="$EXTENSION_SHARD_DIR/validation_shard_${shard_index}.npz"
  log_output="$EXTENSION_LOG_DIR/gpu_${shard_index}.log"
  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES="$shard_index" conda run --no-capture-output -n pool \
      python scripts/tools/generate_midlevel_tasks.py \
        --backend mujoco-warp \
        --split train \
        --physics-device cuda:0 \
        --num-worlds 4096 \
        --chunk-steps 64 \
        --check-interval-steps 8192 \
        --train-count "${TRAIN_COUNTS[$shard_index]}" \
        --seed "${TRAIN_SEEDS[$shard_index]}" \
        --train-output "$train_output" \
        --replay-check -1
    CUDA_VISIBLE_DEVICES="$shard_index" conda run --no-capture-output -n pool \
      python scripts/tools/generate_midlevel_tasks.py \
        --backend mujoco-warp \
        --split validation \
        --physics-device cuda:0 \
        --num-worlds 4096 \
        --chunk-steps 64 \
        --check-interval-steps 8192 \
        --validation-count "${VALIDATION_COUNTS[$shard_index]}" \
        --seed "${VALIDATION_CLI_SEEDS[$shard_index]}" \
        --validation-output "$validation_output" \
        --replay-check -1
  ) >"$log_output" 2>&1 &
  ACTIVE_CHILD_PIDS+=("$!")
  printf 'extension_launched shard=%s gpu=%s pid=%s train=%s train_seed=%s validation=%s validation_schedule_seed=%s log=%s\n' \
    "$shard_index" "$shard_index" "${ACTIVE_CHILD_PIDS[-1]}" \
    "${TRAIN_COUNTS[$shard_index]}" "${TRAIN_SEEDS[$shard_index]}" \
    "${VALIDATION_COUNTS[$shard_index]}" \
    "$((VALIDATION_CLI_SEEDS[$shard_index] + 1))" "$log_output"
done

extension_failed=0
while true; do
  active=0
  for pid in "${ACTIVE_CHILD_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      active=$((active + 1))
    fi
  done
  printf 'extension_active=%s timestamp=%s\n' "$active" "$(timestamp)"
  if (( active == 0 )); then
    break
  fi
  sleep 30
done
for shard_index in 0 1 2 3 4 5 6 7; do
  if wait "${ACTIVE_CHILD_PIDS[$shard_index]}"; then
    printf 'extension_done shard=%s status=PASS\n' "$shard_index"
  else
    status=$?
    printf 'extension_done shard=%s status=FAIL exit=%s\n' \
      "$shard_index" "$status"
    extension_failed=1
  fi
done
ACTIVE_CHILD_PIDS=()
if (( extension_failed != 0 )); then
  printf 'controller_abort reason=extension_failed timestamp=%s\n' "$(timestamp)"
  exit 4
fi

TRAIN_SHARDS=()
VALIDATION_SHARDS=()
for shard_index in 0 1 2 3 4 5 6 7; do
  TRAIN_SHARDS+=("$BASE_SHARD_DIR/train_shard_${shard_index}.npz")
  VALIDATION_SHARDS+=("$BASE_SHARD_DIR/validation_shard_${shard_index}.npz")
done
for shard_index in 0 1 2 3 4 5 6 7; do
  TRAIN_SHARDS+=("$EXTENSION_SHARD_DIR/train_shard_${shard_index}.npz")
  VALIDATION_SHARDS+=("$EXTENSION_SHARD_DIR/validation_shard_${shard_index}.npz")
done

conda run --no-capture-output -n pool \
  python scripts/tools/merge_midlevel_task_shards.py \
    --train-shards "${TRAIN_SHARDS[@]}" \
    --validation-shards "${VALIDATION_SHARDS[@]}" \
    --train-count 196608 \
    --validation-count 12288 \
    --train-output "$TRAIN_STAGED" \
    --validation-output "$VALIDATION_STAGED" \
    --manifest-output "$MERGE_MANIFEST" \
    >"$POST_LOG_DIR/merge.log" 2>&1
printf 'controller_merge status=PASS train=196608 validation=12288 timestamp=%s\n' \
  "$(timestamp)"

conda run --no-capture-output -n pool \
  python scripts/tools/audit_midlevel_task_difficulty.py \
    "$TRAIN_STAGED" "$VALIDATION_STAGED" \
    --require-balanced \
    --output outputs/tasks/midlevel_two_ball_balanced_196608_staged.audit.json \
    >"$POST_LOG_DIR/staged_audit.log" 2>&1
printf 'controller_staged_audit status=PASS timestamp=%s\n' "$(timestamp)"

for gpu_index in 0 1 2 3 4 5 6 7; do
  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES="$gpu_index" conda run --no-capture-output -n pool \
      python scripts/tools/validate_midlevel_task_batch_range.py \
        "$TRAIN_STAGED" \
        --start-task "$((gpu_index * 24576))" \
        --max-tasks 24576 \
        --num-worlds 4096 \
        --physics-device cuda:0 \
        --chunk-steps 64 \
        --check-interval-steps 8192 \
        --report-output "$POST_LOG_DIR/train_range_${gpu_index}.json"
    if (( gpu_index < 3 )); then
      CUDA_VISIBLE_DEVICES="$gpu_index" conda run --no-capture-output -n pool \
        python scripts/tools/validate_midlevel_task_batch_range.py \
          "$VALIDATION_STAGED" \
          --start-task "$((gpu_index * 4096))" \
          --max-tasks 4096 \
          --num-worlds 4096 \
          --physics-device cuda:0 \
          --chunk-steps 64 \
          --check-interval-steps 8192 \
          --report-output "$POST_LOG_DIR/validation_range_${gpu_index}.json"
    fi
  ) >"$POST_LOG_DIR/replay_gpu_${gpu_index}.log" 2>&1 &
  ACTIVE_CHILD_PIDS+=("$!")
  printf 'controller_replay_launched gpu=%s pid=%s train_start=%s train_count=24576\n' \
    "$gpu_index" "${ACTIVE_CHILD_PIDS[-1]}" "$((gpu_index * 24576))"
done

replay_failed=0
for gpu_index in 0 1 2 3 4 5 6 7; do
  if wait "${ACTIVE_CHILD_PIDS[$gpu_index]}"; then
    printf 'controller_replay_done gpu=%s status=PASS\n' "$gpu_index"
  else
    status=$?
    printf 'controller_replay_done gpu=%s status=FAIL exit=%s\n' \
      "$gpu_index" "$status"
    replay_failed=1
  fi
done
ACTIVE_CHILD_PIDS=()
if (( replay_failed != 0 )); then
  printf 'controller_abort reason=full_replay_failed timestamp=%s\n' "$(timestamp)"
  exit 5
fi

TRAIN_REPORTS=()
for gpu_index in 0 1 2 3 4 5 6 7; do
  TRAIN_REPORTS+=("$POST_LOG_DIR/train_range_${gpu_index}.json")
done
VALIDATION_REPORTS=(
  "$POST_LOG_DIR/validation_range_0.json"
  "$POST_LOG_DIR/validation_range_1.json"
  "$POST_LOG_DIR/validation_range_2.json"
)

conda run --no-capture-output -n pool \
  python scripts/tools/publish_midlevel_task_libraries.py \
    --train-staged "$TRAIN_STAGED" \
    --validation-staged "$VALIDATION_STAGED" \
    --train-reports "${TRAIN_REPORTS[@]}" \
    --validation-reports "${VALIDATION_REPORTS[@]}" \
    --manifest-output outputs/tasks/midlevel_two_ball_difficulty_grid_v1.json \
    >"$POST_LOG_DIR/publish.log" 2>&1

conda run --no-capture-output -n pool \
  python scripts/tools/audit_midlevel_task_difficulty.py \
    --require-balanced \
    --output outputs/tasks/midlevel_two_ball_difficulty_grid_v1.audit.json \
    >"$POST_LOG_DIR/published_audit.log" 2>&1

printf 'controller_complete status=PASS train=196608 validation=12288 timestamp=%s\n' \
  "$(timestamp)"
