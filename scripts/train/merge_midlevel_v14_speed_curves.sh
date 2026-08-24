#!/usr/bin/env bash
set -euo pipefail

readonly old_dir="outputs/diagnostics/midlevel_speed_perturbations_196608_batches"
readonly tail_dir="outputs/diagnostics/midlevel_speed_perturbations_208896_tail_batches"
readonly output="outputs/diagnostics/midlevel_speed_perturbations_208896.npz"

shopt -s nullglob
old_batches=("$old_dir"/batch_*.npz)
tail_batches=("$tail_dir"/batch_*.npz)
if [[ ${#old_batches[@]} != 48 ]]; then
  printf 'expected 48 archived curve batches, found %s\n' \
    "${#old_batches[@]}" >&2
  exit 2
fi
if [[ ${#tail_batches[@]} != 3 ]]; then
  printf 'expected 3 new tail curve batches, found %s\n' \
    "${#tail_batches[@]}" >&2
  exit 2
fi

python scripts/tools/merge_midlevel_speed_perturbation_batches.py \
  "${old_batches[@]}" \
  "${tail_batches[@]}" \
  --tasks outputs/tasks/midlevel_two_ball_train.npz \
  --expected-task-count 208896 \
  --center-stop-tolerance 0.005 \
  --output "$output"

python - "$output" <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "src"))
from snooker_env.midlevel_offline_curves import OfflineSpeedCurveDataset

curves = OfflineSpeedCurveDataset.load(Path(sys.argv[1]))
report = curves.report()
if report["task_count"] != 208_896 or report["offset_count"] != 7:
    raise RuntimeError(f"unexpected merged curve report: {report}")
print(f"merged_curve_validation=PASS report={report}", flush=True)
PY
