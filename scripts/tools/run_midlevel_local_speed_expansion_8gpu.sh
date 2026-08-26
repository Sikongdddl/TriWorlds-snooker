#!/usr/bin/env bash

set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly source_train="outputs/tasks/midlevel_two_ball_train.npz"
readonly validation="outputs/tasks/midlevel_two_ball_validation.npz"
readonly shard_dir="outputs/tasks/midlevel_local_speed_shards/v1_seed20260826"
readonly log_dir="outputs/logs/midlevel_local_speed_expansion_v1"
readonly augmentation="outputs/tasks/midlevel_two_ball_train.local_speed_196608.npz"
readonly provenance="outputs/tasks/midlevel_two_ball_train.local_speed_196608.provenance.npz"
readonly staged_train="outputs/tasks/midlevel_two_ball_train.targeted_393216_unvalidated.npz"
readonly merge_manifest="outputs/tasks/midlevel_two_ball_targeted_393216_merge_manifest.json"
readonly publish_manifest="outputs/tasks/midlevel_two_ball_targeted_local_speed_v1.json"
readonly global_seed=20260826
readonly global_task_count=196608
readonly shard_count=8
readonly tasks_per_shard=24576
readonly replay_tasks_per_gpu=49152

cd "$repo_root"
mkdir -p "$shard_dir" "$log_dir"
exec > >(tee -a "$log_dir/controller.log") 2>&1

timestamp() {
  date -Iseconds
}

active_pids=()

terminate_children() {
  local child_pid
  trap - HUP INT TERM
  for child_pid in "${active_pids[@]:-}"; do
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
      kill -TERM "$child_pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  exit 130
}

trap terminate_children HUP INT TERM

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

if [[ ! -f "$source_train" || ! -f "$validation" ]]; then
  printf 'controller_abort reason=missing_core_or_validation timestamp=%s\n' \
    "$(timestamp)"
  exit 2
fi

for output in \
  "$augmentation" \
  "$provenance" \
  "$staged_train" \
  "$merge_manifest" \
  "$publish_manifest"; do
  if [[ -e "$output" ]]; then
    printf 'controller_abort reason=output_exists path=%s timestamp=%s\n' \
      "$output" "$(timestamp)"
    exit 3
  fi
done
printf 'controller_start core=196608 augmentation=196608 target=393216 groups=49152 timestamp=%s\n' \
  "$(timestamp)"

final_shard_pairs=0
staged_shard_pairs=0
partial_shard_state=0
for shard_index in 0 1 2 3 4 5 6 7; do
  task_output="$shard_dir/local_speed_shard_${shard_index}.npz"
  provenance_output="$shard_dir/local_speed_shard_${shard_index}.provenance.npz"
  task_staged="$shard_dir/local_speed_shard_${shard_index}.unvalidated.npz"
  provenance_staged="$shard_dir/local_speed_shard_${shard_index}.provenance.unvalidated.npz"
  task_final_exists=0
  provenance_final_exists=0
  task_staged_exists=0
  provenance_staged_exists=0
  [[ -f "$task_output" ]] && task_final_exists=1
  [[ -f "$provenance_output" ]] && provenance_final_exists=1
  [[ -f "$task_staged" ]] && task_staged_exists=1
  [[ -f "$provenance_staged" ]] && provenance_staged_exists=1
  if (( task_final_exists != provenance_final_exists \
      || task_staged_exists != provenance_staged_exists \
      || (task_final_exists == 1 && task_staged_exists == 1) )); then
    partial_shard_state=1
  fi
  final_shard_pairs=$((final_shard_pairs + task_final_exists))
  staged_shard_pairs=$((staged_shard_pairs + task_staged_exists))
done
if (( partial_shard_state != 0 )); then
  printf 'controller_abort reason=partial_shard_state timestamp=%s\n' "$(timestamp)"
  exit 4
fi

if (( final_shard_pairs == shard_count && staged_shard_pairs == 0 )); then
  printf 'controller_resume stage=merge finalized_shards=%s timestamp=%s\n' \
    "$final_shard_pairs" "$(timestamp)"
elif (( final_shard_pairs == 0 && staged_shard_pairs == shard_count )); then
  printf 'controller_resume stage=provenance_audit staged_shards=%s timestamp=%s\n' \
    "$staged_shard_pairs" "$(timestamp)"
  for shard_index in 0 1 2 3 4 5 6 7; do
    task_output="$shard_dir/local_speed_shard_${shard_index}.npz"
    provenance_output="$shard_dir/local_speed_shard_${shard_index}.provenance.npz"
    conda run --no-capture-output -n pool \
      python scripts/tools/generate_midlevel_local_speed_tasks.py \
        --source "$source_train" \
        --global-task-count "$global_task_count" \
        --global-seed "$global_seed" \
        --shard-index "$shard_index" \
        --shard-count "$shard_count" \
        --output "$task_output" \
        --provenance-output "$provenance_output" \
        --resume-unvalidated \
        >"$log_dir/resume_shard_${shard_index}.log" 2>&1
    printf 'provenance_audit shard=%s status=PASS timestamp=%s\n' \
      "$shard_index" "$(timestamp)"
  done
elif (( final_shard_pairs == 0 && staged_shard_pairs == 0 )); then
  wait_for_idle_gpus
  active_pids=()
  for shard_index in 0 1 2 3 4 5 6 7; do
    task_output="$shard_dir/local_speed_shard_${shard_index}.npz"
    provenance_output="$shard_dir/local_speed_shard_${shard_index}.provenance.npz"
    (
      set -euo pipefail
      CUDA_VISIBLE_DEVICES="$shard_index" conda run --no-capture-output -n pool \
        python scripts/tools/generate_midlevel_local_speed_tasks.py \
          --source "$source_train" \
          --global-task-count "$global_task_count" \
          --global-seed "$global_seed" \
          --shard-index "$shard_index" \
          --shard-count "$shard_count" \
          --output "$task_output" \
          --provenance-output "$provenance_output" \
          --physics-device cuda:0 \
          --num-worlds 4096 \
          --chunk-steps 64 \
          --check-interval-steps 8192
    ) >"$log_dir/generation_gpu_${shard_index}.log" 2>&1 &
    active_pids+=("$!")
    printf 'generation_launched shard=%s gpu=%s pid=%s tasks=%s timestamp=%s\n' \
      "$shard_index" "$shard_index" "${active_pids[-1]}" \
      "$tasks_per_shard" "$(timestamp)"
  done

  generation_failed=0
  for shard_index in 0 1 2 3 4 5 6 7; do
    if wait "${active_pids[$shard_index]}"; then
      printf 'generation_done shard=%s status=PASS timestamp=%s\n' \
        "$shard_index" "$(timestamp)"
    else
      exit_status=$?
      printf 'generation_done shard=%s status=FAIL exit=%s timestamp=%s\n' \
        "$shard_index" "$exit_status" "$(timestamp)"
      generation_failed=1
    fi
  done
  active_pids=()
  if (( generation_failed != 0 )); then
    printf 'controller_abort reason=generation_failed timestamp=%s\n' "$(timestamp)"
    exit 4
  fi
else
  printf 'controller_abort reason=incomplete_shard_set final=%s staged=%s timestamp=%s\n' \
    "$final_shard_pairs" "$staged_shard_pairs" "$(timestamp)"
  exit 4
fi

task_shards=()
provenance_shards=()
for shard_index in 0 1 2 3 4 5 6 7; do
  task_shards+=("$shard_dir/local_speed_shard_${shard_index}.npz")
  provenance_shards+=("$shard_dir/local_speed_shard_${shard_index}.provenance.npz")
done
conda run --no-capture-output -n pool \
  python scripts/tools/merge_midlevel_local_speed_shards.py \
    --core "$source_train" \
    --shards "${task_shards[@]}" \
    --provenance-shards "${provenance_shards[@]}" \
    --augmentation-output "$augmentation" \
    --provenance-output "$provenance" \
    --training-output "$staged_train" \
    --manifest-output "$merge_manifest" \
    >"$log_dir/merge.log" 2>&1
printf 'controller_merge status=PASS training=393216 timestamp=%s\n' "$(timestamp)"

wait_for_idle_gpus
active_pids=()
for gpu_index in 0 1 2 3 4 5 6 7; do
  (
    set -euo pipefail
    CUDA_VISIBLE_DEVICES="$gpu_index" conda run --no-capture-output -n pool \
      python scripts/tools/validate_midlevel_task_batch_range.py \
        "$staged_train" \
        --targeted-training-distribution \
        --start-task "$((gpu_index * replay_tasks_per_gpu))" \
        --max-tasks "$replay_tasks_per_gpu" \
        --num-worlds 4096 \
        --physics-device cuda:0 \
        --chunk-steps 64 \
        --check-interval-steps 8192 \
        --report-output "$log_dir/train_range_${gpu_index}.json"
  ) >"$log_dir/replay_gpu_${gpu_index}.log" 2>&1 &
  active_pids+=("$!")
  printf 'replay_launched gpu=%s pid=%s start=%s count=%s timestamp=%s\n' \
    "$gpu_index" "${active_pids[-1]}" \
    "$((gpu_index * replay_tasks_per_gpu))" "$replay_tasks_per_gpu" \
    "$(timestamp)"
done

replay_failed=0
for gpu_index in 0 1 2 3 4 5 6 7; do
  if wait "${active_pids[$gpu_index]}"; then
    printf 'replay_done gpu=%s status=PASS timestamp=%s\n' \
      "$gpu_index" "$(timestamp)"
  else
    exit_status=$?
    printf 'replay_done gpu=%s status=FAIL exit=%s timestamp=%s\n' \
      "$gpu_index" "$exit_status" "$(timestamp)"
    replay_failed=1
  fi
done
active_pids=()
if (( replay_failed != 0 )); then
  printf 'controller_abort reason=replay_failed timestamp=%s\n' "$(timestamp)"
  exit 5
fi

replay_reports=()
for gpu_index in 0 1 2 3 4 5 6 7; do
  replay_reports+=("$log_dir/train_range_${gpu_index}.json")
done
conda run --no-capture-output -n pool \
  python scripts/tools/publish_midlevel_targeted_training_library.py \
    --training-staged "$staged_train" \
    --core "$source_train" \
    --augmentation "$augmentation" \
    --provenance "$provenance" \
    --merge-manifest "$merge_manifest" \
    --replay-reports "${replay_reports[@]}" \
    --validation "$validation" \
    --training-output "$source_train" \
    --manifest-output "$publish_manifest" \
    >"$log_dir/publish.log" 2>&1
printf 'controller_complete training=393216 validation=12288 manifest=%s timestamp=%s\n' \
  "$publish_manifest" "$(timestamp)"
