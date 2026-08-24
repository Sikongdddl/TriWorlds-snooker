#!/usr/bin/env bash
set -euo pipefail

readonly base_tasks="outputs/tasks/midlevel_two_ball_train.npz"
readonly addition_tasks="outputs/tasks/midlevel_two_ball_train_add_seed6_12288.npz"
readonly staged_addition="outputs/tasks/midlevel_two_ball_train_add_seed6_12288.unvalidated.npz"
readonly backup_tasks="outputs/tasks/midlevel_two_ball_train_196608_seed0_plus_seed2_plus_seed4.npz"
readonly expected_base_hash="0b984f8fe9662fd8afa81f0f105e63d9aeda99a73aad3f8baf6a7370b74d87db"

check_library() {
  local path=$1
  local expected_count=$2
  local expected_per_pocket=$3
  local expected_seed=$4
  local expected_hash=${5:-}
  python - "$path" "$expected_count" "$expected_per_pocket" \
    "$expected_seed" "$expected_hash" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

path = Path(sys.argv[1])
expected_count = int(sys.argv[2])
expected_per_pocket = int(sys.argv[3])
expected_seed = int(sys.argv[4])
expected_hash = sys.argv[5]

with np.load(path, allow_pickle=False) as archive:
    metadata = json.loads(str(archive["metadata"].item()))
    pockets = np.asarray(archive["pocket_indices"], dtype=np.int64)
    candidate_seeds = np.asarray(archive["candidate_seeds"], dtype=np.uint64)

counts = np.bincount(pockets, minlength=6).astype(int).tolist()
failures: list[str] = []
if len(pockets) != expected_count:
    failures.append(f"task count {len(pockets)} != {expected_count}")
if counts != [expected_per_pocket] * 6:
    failures.append(
        f"pocket counts {counts} != {[expected_per_pocket] * 6}"
    )
if int(metadata["generation_seed"]) != expected_seed:
    failures.append(
        f"generation seed {metadata['generation_seed']} != {expected_seed}"
    )
if int(metadata["task_count"]) != expected_count:
    failures.append(
        f"metadata task count {metadata['task_count']} != {expected_count}"
    )
identities = set(
    zip(map(int, pockets), map(int, candidate_seeds), strict=True)
)
if len(identities) != expected_count:
    failures.append("(pocket, candidate_seed) task identities are not unique")
if expected_hash and metadata["content_sha256"] != expected_hash:
    failures.append(
        f"content hash {metadata['content_sha256']} != {expected_hash}"
    )
if failures:
    raise RuntimeError(f"{path}: " + "; ".join(failures))
print(
    f"library_check=PASS path={path} tasks={expected_count} "
    f"per_pocket={expected_per_pocket} seed={expected_seed} "
    f"content_sha256={metadata['content_sha256']}",
    flush=True,
)
PY
}

mkdir -p outputs/tasks outputs/logs

if [[ ! -f "$base_tasks" ]]; then
  printf 'missing base task library: %s\n' "$base_tasks" >&2
  exit 2
fi

current_count=$(python - "$base_tasks" <<'PY'
import json
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as archive:
    print(int(json.loads(str(archive["metadata"].item()))["task_count"]))
PY
)

if [[ "$current_count" == 208896 ]]; then
  check_library "$base_tasks" 208896 34816 0
  printf 'task_expansion status=ALREADY_COMPLETE output=%s\n' "$base_tasks"
  exit 0
fi
if [[ "$current_count" != 196608 ]]; then
  printf 'unexpected base task count: %s (expected 196608)\n' \
    "$current_count" >&2
  exit 2
fi
check_library "$base_tasks" 196608 32768 0 "$expected_base_hash"

if [[ ! -f "$addition_tasks" ]]; then
  resume_args=()
  if [[ -f "$staged_addition" ]]; then
    resume_args+=(--resume-unvalidated)
    printf 'task_expansion resuming_staged=%s\n' "$staged_addition"
  fi
  python scripts/tools/generate_midlevel_tasks.py \
    --split train \
    --train-count 12288 \
    --seed 6 \
    --train-output "$addition_tasks" \
    --backend mujoco-warp \
    --physics-device cuda:0 \
    --num-worlds 4096 \
    --chunk-steps 64 \
    --check-interval-steps 8192 \
    --replay-check -1 \
    "${resume_args[@]}"
else
  printf 'task_expansion reusing_validated_addition=%s\n' "$addition_tasks"
fi

check_library "$addition_tasks" 12288 2048 6
# The long generation may overlap a user-initiated library update.  Recheck
# the exact base immediately before the destructive publication step.
check_library "$base_tasks" 196608 32768 0 "$expected_base_hash"

python scripts/tools/append_midlevel_tasks.py \
  "$base_tasks" \
  "$addition_tasks" \
  --output "$base_tasks" \
  --backup "$backup_tasks" \
  --physics-device cuda:0 \
  --num-worlds 4096 \
  --chunk-steps 64 \
  --check-interval-steps 8192

check_library "$base_tasks" 208896 34816 0
check_library "$backup_tasks" 196608 32768 0 "$expected_base_hash"
printf 'task_expansion status=COMPLETE output=%s tasks=208896 per_pocket=34816 backup=%s\n' \
  "$base_tasks" "$backup_tasks"
