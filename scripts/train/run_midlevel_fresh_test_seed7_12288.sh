#!/usr/bin/env bash
set -euo pipefail

# This library is generated independently and sealed for final-only use before
# the formal development-round selection.  Do not pass it to training, early
# stopping, hyperparameter selection, or development evaluation.  It is
# reserved for one final deterministic test.
readonly test_tasks="outputs/tasks/midlevel_two_ball_test_seed7_12288.npz"
readonly staged_test="outputs/tasks/midlevel_two_ball_test_seed7_12288.unvalidated.npz"

check_test_library() {
  python - "$test_tasks" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

path = Path(sys.argv[1])
with np.load(path, allow_pickle=False) as archive:
    metadata = json.loads(str(archive["metadata"].item()))
    pockets = np.asarray(archive["pocket_indices"], dtype=np.int64)
    seeds = np.asarray(archive["candidate_seeds"], dtype=np.uint64)

counts = np.bincount(pockets, minlength=6).astype(int).tolist()
failures: list[str] = []
if len(pockets) != 12_288:
    failures.append(f"task count {len(pockets)} != 12288")
if counts != [2_048] * 6:
    failures.append(f"pocket counts {counts} != {[2_048] * 6}")
if int(metadata["generation_seed"]) != 7:
    failures.append(f"generation seed {metadata['generation_seed']} != 7")
if int(metadata["task_count"]) != 12_288:
    failures.append(f"metadata task count {metadata['task_count']} != 12288")
identities = set(zip(map(int, pockets), map(int, seeds), strict=True))
if len(identities) != 12_288:
    failures.append("(pocket, candidate_seed) task identities are not unique")
if failures:
    raise RuntimeError(f"{path}: " + "; ".join(failures))
print(
    "fresh_test_check=PASS "
    f"path={path} tasks=12288 per_pocket=2048 seed=7 "
    f"content_sha256={metadata['content_sha256']}",
    flush=True,
)
PY
}

mkdir -p outputs/tasks outputs/logs

if [[ -f "$test_tasks" ]]; then
  check_test_library
  printf 'fresh_test status=ALREADY_COMPLETE output=%s generation_validated=true pending_cross_split_seal=true\n' \
    "$test_tasks"
  exit 0
fi

resume_args=()
if [[ -f "$staged_test" ]]; then
  resume_args+=(--resume-unvalidated)
  printf 'fresh_test resuming_staged=%s\n' "$staged_test"
fi

python scripts/tools/generate_midlevel_tasks.py \
  --split train \
  --train-count 12288 \
  --seed 7 \
  --train-output "$test_tasks" \
  --backend mujoco-warp \
  --physics-device cuda:0 \
  --num-worlds 4096 \
  --chunk-steps 64 \
  --check-interval-steps 8192 \
  --replay-check -1 \
  "${resume_args[@]}"

check_test_library
printf 'fresh_test status=COMPLETE output=%s generation_validated=true pending_cross_split_seal=true\n' \
  "$test_tasks"
