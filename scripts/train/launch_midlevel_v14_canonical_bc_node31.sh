#!/usr/bin/env bash
set -euo pipefail

gpu_index=${1:?usage: launch_midlevel_v14_canonical_bc_node31.sh GPU_INDEX SEED}
seed=${2:?missing seed}
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
   exec bash scripts/train/run_midlevel_v14_canonical_bc_seed.sh "$@"' \
  bash "$seed"
