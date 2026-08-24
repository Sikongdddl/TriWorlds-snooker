#!/usr/bin/env bash
set -euo pipefail

gpu_index=${1:?usage: launch_midlevel_policy_slot_audit_node31.sh GPU_INDEX [audit arguments...]}
shift
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
   exec python scripts/tools/audit_midlevel_mujoco_warp_slots.py "$@"' \
  bash "$@"
