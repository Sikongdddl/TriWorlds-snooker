#!/usr/bin/env bash
set -euo pipefail

readonly train="outputs/tasks/midlevel_two_ball_train.npz"
readonly development="outputs/tasks/midlevel_two_ball_validation.npz"
readonly test="outputs/tasks/midlevel_two_ball_test_seed7_12288.npz"
readonly seal="outputs/tasks/midlevel_two_ball_test_seed7_12288.seal.json"

source /data/home/haoyiwei/miniconda3/etc/profile.d/conda.sh
conda activate pool
cd /data/home/haoyiwei/TriWorlds-snooker

if [[ -f "$seal" ]]; then
  printf 'fresh_test_seal status=ALREADY_COMPLETE path=%s\n' "$seal"
  exit 0
fi
while true; do
  count=$(python - "$train" <<'PY'
import json
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as archive:
    print(int(json.loads(str(archive["metadata"].item()))["task_count"]))
PY
)
  if [[ "$count" == 208896 && -f "$test" ]]; then
    break
  fi
  printf 'fresh_test_seal status=WAIT train_count=%s test_published=%s time=%s\n' \
    "$count" "$([[ -f "$test" ]] && printf true || printf false)" \
    "$(date --iso-8601=seconds)"
  sleep 30
done

python scripts/tools/seal_midlevel_fresh_test.py \
  --train "$train" \
  --development "$development" \
  --test "$test" \
  --output "$seal"
printf 'fresh_test_seal status=COMPLETE path=%s\n' "$seal"
