#!/usr/bin/env bash
set -euo pipefail

# Run this queue inside tmux on the login node.  Every GPU child is itself a
# named tmux session containing a persistent SSH connection to node31.
readonly repo="/data/home/haoyiwei/TriWorlds-snooker"
readonly status_dir="outputs/status/midlevel_v14"
readonly fresh_test_seal="outputs/tasks/midlevel_two_ball_test_seed7_12288.seal.json"
readonly poll_seconds=30
mkdir -p "$status_dir" outputs/logs

if [[ -f "$status_dir/development_round.done" ]]; then
  printf 'v14_queue status=ALREADY_COMPLETE\n'
  exit 0
fi

source /data/home/haoyiwei/miniconda3/etc/profile.d/conda.sh
conda activate pool
cd "$repo"

task_count() {
  python - <<'PY'
import json
import numpy as np

with np.load("outputs/tasks/midlevel_two_ball_train.npz", allow_pickle=False) as archive:
    print(int(json.loads(str(archive["metadata"].item()))["task_count"]))
PY
}

wait_for_expanded_library() {
  while true; do
    count=$(task_count)
    if [[ "$count" == 208896 ]]; then
      printf 'v14_queue expanded_library=READY time=%s\n' \
        "$(date --iso-8601=seconds)"
      return
    fi
    if [[ "$count" != 196608 ]]; then
      printf 'v14_queue unexpected_task_count=%s\n' "$count" >&2
      exit 2
    fi
    printf 'v14_queue expanded_library=WAIT count=%s time=%s\n' \
      "$count" "$(date --iso-8601=seconds)"
    sleep "$poll_seconds"
  done
}

wait_for_fresh_test_seal() {
  while [[ ! -f "$fresh_test_seal" ]]; do
    printf 'v14_queue fresh_test_seal=WAIT path=%s time=%s\n' \
      "$fresh_test_seal" "$(date --iso-8601=seconds)"
    sleep "$poll_seconds"
  done
  python - "$fresh_test_seal" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    seal = json.load(source)
if seal.get("version") != "midlevel-fresh-test-seal-v2":
    raise RuntimeError("fresh-test seal has an unexpected version")
if seal.get("status") != "sealed_before_formal_development_selection":
    raise RuntimeError("fresh-test seal has an unexpected status")
if int(seal["test"]["task_count"]) != 12_288:
    raise RuntimeError("fresh-test seal has an unexpected task count")
if seal["policy"]["forbidden_uses"] != [
    "training",
    "early_stopping",
    "hyperparameter_selection",
    "seed_selection",
    "model_selection",
]:
    raise RuntimeError("fresh-test seal policy changed unexpectedly")
print(
    "v14_queue fresh_test_seal=VERIFIED "
    f"content_sha256={seal['test']['content_sha256']}",
    flush=True,
)
PY
}

wait_for_idle_gpu() {
  local gpu=$1
  local confirmations=0
  local memory utilization
  while (( confirmations < 2 )); do
    read -r memory utilization < <(
      ssh node31 \
        "nvidia-smi --id=$gpu --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits" \
        | tr -d ','
    )
    if (( memory <= 128 && utilization <= 5 )); then
      confirmations=$((confirmations + 1))
    else
      confirmations=0
      printf 'v14_queue gpu=%s status=BUSY memory_mib=%s util=%s time=%s\n' \
        "$gpu" "$memory" "$utilization" "$(date --iso-8601=seconds)"
    fi
    if (( confirmations < 2 )); then
      sleep 10
    fi
  done
  printf 'v14_queue gpu=%s status=IDLE_CONFIRMED time=%s\n' \
    "$gpu" "$(date --iso-8601=seconds)"
}

launch_gpu_session() {
  local name=$1
  local gpu=$2
  local remote_command=$3
  local done_path="$status_dir/${name}.done"
  local tmux_log="outputs/logs/${name}.tmux.log"
  if [[ -f "$done_path" ]]; then
    printf 'v14_queue session=%s status=ALREADY_COMPLETE\n' "$name"
    return
  fi
  if tmux has-session -t "$name" 2>/dev/null; then
    printf 'v14_queue session=%s status=ALREADY_RUNNING\n' "$name"
    return
  fi
  wait_for_idle_gpu "$gpu"
  tmux new-session -d -s "$name" \
    "cd '$repo' && set -o pipefail && ssh node31 'cd $repo && $remote_command' 2>&1 | tee '$tmux_log' && touch '$done_path'"
  printf 'v14_queue session=%s gpu=%s status=START time=%s\n' \
    "$name" "$gpu" "$(date --iso-8601=seconds)"
}

wait_for_session() {
  local name=$1
  local done_path="$status_dir/${name}.done"
  while [[ ! -f "$done_path" ]]; do
    if ! tmux has-session -t "$name" 2>/dev/null; then
      printf 'v14_queue session=%s status=FAILED log=outputs/logs/%s.tmux.log\n' \
        "$name" "$name" >&2
      exit 2
    fi
    sleep "$poll_seconds"
  done
  printf 'v14_queue session=%s status=COMPLETE time=%s\n' \
    "$name" "$(date --iso-8601=seconds)"
}

wait_for_expanded_library
wait_for_fresh_test_seal

# Canonical-only BC establishes a three-seed expanded-data baseline while the
# three new physical curve batches are collected on independent GPUs.
launch_gpu_session midlevel_v14_canonical_s0 0 \
  "bash scripts/train/launch_midlevel_v14_canonical_bc_node31.sh 0 0"
launch_gpu_session midlevel_v14_canonical_s1 2 \
  "bash scripts/train/launch_midlevel_v14_canonical_bc_node31.sh 2 1"
launch_gpu_session midlevel_v14_canonical_s2 3 \
  "bash scripts/train/launch_midlevel_v14_canonical_bc_node31.sh 3 2"
launch_gpu_session midlevel_v14_curve_196608 4 \
  "bash scripts/train/launch_midlevel_v14_speed_curve_tail_node31.sh 4 196608"
launch_gpu_session midlevel_v14_curve_200704 5 \
  "bash scripts/train/launch_midlevel_v14_speed_curve_tail_node31.sh 5 200704"
launch_gpu_session midlevel_v14_curve_204800 6 \
  "bash scripts/train/launch_midlevel_v14_speed_curve_tail_node31.sh 6 204800"

for name in \
  midlevel_v14_canonical_s0 \
  midlevel_v14_canonical_s1 \
  midlevel_v14_canonical_s2
do
  wait_for_session "$name"
done

# Compare the three speed mappings under one identical frozen angle before
# selecting an initializer.  The raw canonical results above remain the formal
# three-seed pure-BC baseline, but are intentionally not used for speed choice.
launch_gpu_session midlevel_v14_speed_candidate_s0 0 \
  "bash scripts/train/launch_midlevel_v14_speed_candidate_node31.sh 0 0"
launch_gpu_session midlevel_v14_speed_candidate_s1 2 \
  "bash scripts/train/launch_midlevel_v14_speed_candidate_node31.sh 2 1"
launch_gpu_session midlevel_v14_speed_candidate_s2 3 \
  "bash scripts/train/launch_midlevel_v14_speed_candidate_node31.sh 3 2"

# The fixed-angle candidate evaluations use GPUs 0/2/3 while the three tail
# curve batches continue independently on 4/5/6.
for name in \
  midlevel_v14_curve_196608 \
  midlevel_v14_curve_200704 \
  midlevel_v14_curve_204800
do
  wait_for_session "$name"
done

bash scripts/train/merge_midlevel_v14_speed_curves.sh \
  2>&1 | tee outputs/logs/midlevel_v14_curve_merge.log
touch "$status_dir/curve_merge.done"

for name in \
  midlevel_v14_speed_candidate_s0 \
  midlevel_v14_speed_candidate_s1 \
  midlevel_v14_speed_candidate_s2
do
  wait_for_session "$name"
done

readonly initializer_report="outputs/evaluations/midlevel_v14_speed_initializer.json"
python scripts/tools/select_midlevel_v14_canonical_initializer.py \
  --details-dir outputs/evaluations \
  --output "$initializer_report" \
  2>&1 | tee outputs/logs/midlevel_v14_speed_initializer.log
speed_reference=$(python - "$initializer_report" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    print(json.load(source)["selected_checkpoint"])
PY
)
printf 'v14_queue speed_reference=%s status=LOCKED_FROM_DEVELOPMENT\n' \
  "$speed_reference"

# All structured jobs share the expanded-data speed initializer.  The frozen
# angle reference remains v10 seed 2 inside the runner.
launch_gpu_session midlevel_v14_structured_reference_s0 0 \
  "bash scripts/train/launch_midlevel_v14_structured_speed_bc_node31.sh 0 reference 0 $speed_reference"
launch_gpu_session midlevel_v14_structured_reference_s1 2 \
  "bash scripts/train/launch_midlevel_v14_structured_speed_bc_node31.sh 2 reference 1 $speed_reference"
launch_gpu_session midlevel_v14_structured_reference_s2 3 \
  "bash scripts/train/launch_midlevel_v14_structured_speed_bc_node31.sh 3 reference 2 $speed_reference"
launch_gpu_session midlevel_v14_structured_zero_s0 4 \
  "bash scripts/train/launch_midlevel_v14_structured_speed_bc_node31.sh 4 zero 0 $speed_reference"
launch_gpu_session midlevel_v14_structured_zero_s1 5 \
  "bash scripts/train/launch_midlevel_v14_structured_speed_bc_node31.sh 5 zero 1 $speed_reference"
launch_gpu_session midlevel_v14_structured_zero_s2 6 \
  "bash scripts/train/launch_midlevel_v14_structured_speed_bc_node31.sh 6 zero 2 $speed_reference"

# Two untrained hybrid controls isolate the effect of supervised hindsight
# from the effect of retaining versus zeroing the angle.  They use exactly the
# selected canonical speed output and never update an Actor.
launch_gpu_session midlevel_v14_control_zero 7 \
  "bash scripts/train/launch_midlevel_v14_structured_control_node31.sh 7 zero $speed_reference"
launch_gpu_session midlevel_v14_control_reference 1 \
  "bash scripts/train/launch_midlevel_v14_structured_control_node31.sh 1 reference $speed_reference"

for name in \
  midlevel_v14_structured_reference_s0 \
  midlevel_v14_structured_reference_s1 \
  midlevel_v14_structured_reference_s2 \
  midlevel_v14_structured_zero_s0 \
  midlevel_v14_structured_zero_s1 \
  midlevel_v14_structured_zero_s2 \
  midlevel_v14_control_reference \
  midlevel_v14_control_zero
do
  wait_for_session "$name"
done

python scripts/tools/analyze_midlevel_v14_development.py \
  --details-dir outputs/evaluations \
  --output outputs/evaluations/midlevel_v14_development_summary.json \
  2>&1 | tee outputs/logs/midlevel_v14_development_analysis.log
touch "$status_dir/development_round.done"
printf 'v14_queue status=DEVELOPMENT_COMPLETE time=%s\n' \
  "$(date --iso-8601=seconds)"
