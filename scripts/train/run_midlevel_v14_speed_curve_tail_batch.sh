#!/usr/bin/env bash
set -euo pipefail

batch_start=${1:?usage: run_midlevel_v14_speed_curve_tail_batch.sh BATCH_START}
case "$batch_start" in
  196608|200704|204800) ;;
  *)
    printf 'unexpected 208896-tail batch start: %s\n' "$batch_start" >&2
    exit 2
    ;;
esac

readonly output_dir="outputs/diagnostics/midlevel_speed_perturbations_208896_tail_batches"
printf -v batch_name 'batch_%06d.npz' "$batch_start"
readonly output="$output_dir/$batch_name"
mkdir -p "$output_dir" outputs/logs

python - <<'PY'
import json
import numpy as np

with np.load("outputs/tasks/midlevel_two_ball_train.npz", allow_pickle=False) as archive:
    metadata = json.loads(str(archive["metadata"].item()))
if int(metadata["task_count"]) != 208_896:
    raise RuntimeError(
        f"tail curves require 208896 tasks, got {metadata['task_count']}"
    )
print(
    "tail_curve_library_check=PASS "
    f"tasks={metadata['task_count']} hash={metadata['content_sha256']}",
    flush=True,
)
PY

if [[ -f "$output" ]]; then
  python - "$output" "$batch_start" <<'PY'
import json
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as archive:
    metadata = json.loads(str(archive["metadata"].item()))
    indices = np.asarray(archive["task_indices"], dtype=np.int64)
start = int(sys.argv[2])
if not np.array_equal(indices, np.arange(start, start + 4096)):
    raise RuntimeError("existing tail batch has incorrect task indices")
if int(metadata["source_task_library_count"]) != 208_896:
    raise RuntimeError("existing tail batch has incorrect source task count")
print(f"tail_curve_batch status=ALREADY_COMPLETE start={start}", flush=True)
PY
  exit 0
fi

python scripts/tools/collect_midlevel_speed_perturbation_batch.py \
  --tasks outputs/tasks/midlevel_two_ball_train.npz \
  --batch-start "$batch_start" \
  --num-worlds 4096 \
  --offsets-mps -0.03 -0.02 -0.01 0.0 0.01 0.02 0.03 \
  --physics-device cuda:0 \
  --chunk-steps 64 \
  --check-interval-steps 8192 \
  --output "$output"

printf 'tail_curve_batch status=COMPLETE start=%s output=%s\n' \
  "$batch_start" "$output"
