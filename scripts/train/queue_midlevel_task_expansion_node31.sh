#!/usr/bin/env bash
set -euo pipefail

readonly lock_path="/tmp/triworlds_midlevel_task_expansion_12288.lock"
readonly poll_seconds=30
readonly confirmation_seconds=10

exec 9>"$lock_path"
if ! flock -n 9; then
  printf 'another task-expansion queue already holds %s\n' "$lock_path" >&2
  exit 2
fi

idle_gpu() {
  nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
    | awk -F, '
        {
          gpu=$1; memory=$2; utilization=$3
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", gpu)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", memory)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", utilization)
          # gsub() leaves GNU awk values string-typed.  Coerce explicitly so
          # e.g. "15" MiB is not compared lexicographically with "128".
          if (selected == "" && memory + 0 <= 128 && utilization + 0 <= 5) {
            selected=gpu
          }
        }
        END {
          if (selected != "") {
            print selected
          }
        }
      '
}

printf 'task_expansion status=WAITING_FOR_IDLE_GPU time=%s\n' \
  "$(date --iso-8601=seconds)"
while true; do
  gpu_index=$(idle_gpu)
  if [[ -n "$gpu_index" ]]; then
    printf 'task_expansion candidate_gpu=%s confirming_for=%ss time=%s\n' \
      "$gpu_index" "$confirmation_seconds" "$(date --iso-8601=seconds)"
    sleep "$confirmation_seconds"
    confirmed_gpu=$(idle_gpu)
    if [[ "$confirmed_gpu" == "$gpu_index" ]]; then
      break
    fi
    printf 'task_expansion candidate_gpu=%s no_longer_idle time=%s\n' \
      "$gpu_index" "$(date --iso-8601=seconds)"
  fi
  sleep "$poll_seconds"
done

case "$gpu_index" in
  0|1|2|3|4|5|6|7) ;;
  *)
    printf 'invalid GPU index selected: %s\n' "$gpu_index" >&2
    exit 2
    ;;
esac

printf 'task_expansion status=START gpu=%s time=%s\n' \
  "$gpu_index" "$(date --iso-8601=seconds)"
exec env CUDA_VISIBLE_DEVICES="$gpu_index" bwrap \
  --bind / / \
  --dev-bind /dev /dev \
  --proc /proc \
  --tmpfs /home \
  --dir /home/ubuntu \
  --ro-bind /data/home/haoyiwei/mujoco-billiards /home/ubuntu/mujoco-billiards \
  --bind /data/home/haoyiwei/TriWorlds-snooker /home/ubuntu/TriWorlds-snooker \
  /bin/bash -lc \
  'set -euo pipefail
   source /data/home/haoyiwei/miniconda3/etc/profile.d/conda.sh
   conda activate pool
   cd /home/ubuntu/TriWorlds-snooker
   exec bash scripts/train/run_midlevel_task_expansion_12288.sh'
