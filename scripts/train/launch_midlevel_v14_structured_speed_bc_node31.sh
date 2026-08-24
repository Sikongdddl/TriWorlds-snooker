#!/usr/bin/env bash
set -euo pipefail

gpu_index=${1:?usage: launch_midlevel_v14_structured_speed_bc_node31.sh GPU_INDEX ANGLE_MODE SEED SPEED_REFERENCE}
angle_mode=${2:?missing angle mode}
seed=${3:?missing seed}
speed_reference=${4:-outputs/checkpoints/midlevel_two_ball_td3_her_v10_canonical_e800_b2048_s2.bc_only.zip}
case "$gpu_index" in
  0|1|2|3|4|5|6|7) ;;
  *)
    printf 'invalid GPU index: %s\n' "$gpu_index" >&2
    exit 2
    ;;
esac

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
   exec bash scripts/train/run_midlevel_v14_structured_speed_bc_variant.sh "$@"' \
  bash "$angle_mode" "$seed" "$speed_reference"
