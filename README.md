# Snooker MuJoCo

MuJoCo scaffold for studying billiards/snooker manipulation with a future dual-arm mobile robot. The current repository focuses on the physics and software interfaces around a cue, balls, table contacts, and a layered policy pipeline before adding full robot control.

The project is intentionally small: plain MJCF models, the native `mujoco` Python API, NumPy, and executable smoke tests.

## Current Status

Implemented:

- The original 9-foot table XML from `hideboz/mujoco-billiards`, including its checker floor, four lights, SDF pockets, rails, and materials.
- A source-built `libsdf_plugin.so` providing the table's custom hollow-cylinder and trapezoid SDFs.
- The source `mujoco-billiards` ball class, all 15 numbered textures, and its marked cue-ball construction.
- A dynamic cue with a Sketchfab visual mesh and primitive physical geoms.
- Dynamic cue ball and object balls with free joints.
- An articulated LIFT robot scaffold in the default scene.
- Robot-free mid-level training scene for cue/ball physics experiments.
- Single-step two-ball environment and deterministic direct behavior-cloning policy.
- A 12-D residual joint-position Gymnasium environment with differential-IK nominal control.
- The imported Gento Skye URDF, 14-D role-aware differential IK, and matching PPO residual checkpoint.
- A physical Gento cue grasp: side approach, vertically closing fingers, and solid palm guards without weld constraints.
- High/mid/low policy interface scaffold.
- Smoke tests for model loading, cue/ball contacts, spin response, and policy interfaces.
- Render scripts for table, cue stroke, robot scaffold, and spin-response comparison videos.

Not implemented yet:

- AC-One robot model.
- Final dual-arm controller.
- Mobile base control.
- Full snooker rules and automatic removal of pocketed balls.
- Real-table system identification for cloth/cushion/cue-tip parameters.
- Vision or domain randomization.

## Demo Commands

Open the default scene:

```bash
python scripts/tools/view_scene.py
```

Inspect the compiled model:

```bash
python scripts/tools/inspect_model.py
```

Run the robot-free mid-level cue/ball smoke test:

```bash
python scripts/smoke_tests/run_midlevel_env_smoke.py
```

Render spin-response comparison videos:

```bash
MUJOCO_GL=egl python scripts/render/render_spin_response_comparison.py \
  --kind both \
  --width 1920 \
  --height 720 \
  --output-dir outputs/videos_midlevel
```

Generated videos are written under `outputs/`, which is intentionally ignored by git.

## Installation

Use the existing Conda environment, install dependencies, then build the source SDF plugin against that environment's MuJoCo:

```bash
conda activate pool
pip install -r requirements.txt
python scripts/assets/build_mujoco_billiards_sdf_plugin.py
```

For headless rendering on Linux, set an appropriate MuJoCo GL backend:

```bash
export MUJOCO_GL=egl
```

Run repository commands only after `conda activate pool`. Rebuild the plugin whenever MuJoCo is upgraded because it links against the active package's headers and shared library.

The rebuilt plugin uses exact Euclidean distances and analytical gradients for
the three active table SDFs. Validate its callback ABI, distances, gradients,
parameter rejection, and all source-plugin defaults with:

```bash
python scripts/smoke_tests/run_mujoco_billiards_sdf_source_smoke.py
```

The `pool` environment also uses the editable MuJoCo Warp checkout at
`/home/ubuntu/mujoco_warp`. Its table adapter uses the same analytical SDFs and
a bounded representative sphere/trapezoid contact manifold. Run the GPU parity
checks with:

```bash
python scripts/smoke_tests/run_mujoco_warp_parity_smoke.py
python scripts/smoke_tests/run_mujoco_warp_trapezoid_sdf_smoke.py
```

## Repository Layout

```text
snooker_mujoco/
├── assets/                 # downloaded and converted visual assets
├── models/                 # MJCF scene and reusable model includes
├── scripts/
│   ├── assets/             # asset conversion/extraction utilities
│   ├── render/             # image/video render scripts
│   ├── smoke_tests/        # physics and interface checks
│   └── tools/              # model inspection and viewer launch
├── src/snooker_env/        # scene helpers and policy pipeline interfaces
├── requirements.txt
└── README.md
```

Large downloaded archives and generated videos are ignored:

```text
assets/*.zip
outputs/
```

## Scenes and Models

Main scene:

```text
models/scene_pool_asset.xml
```

This scene combines:

- `models/mujoco_billiards/billiard-table-definitions.xml`: byte-identical source table definition used directly for visuals and contacts.
- `models/mujoco_billiards_integration.xml`: non-physical named sites used only for project pocket-event reporting.
- `../mujoco-billiards/ball-definitions.xml`: source ball class and numbered materials, included directly from the sibling source checkout.
- `models/balls_physics.xml`: source-style marked cue ball and textured 15-ball rack, with project-stable body and joint names.
- `models/cue_physics.xml`: dynamic cue body with visual mesh and primitive collision.
- `models/lift_articulated.xml`: articulated LIFT robot scaffold.
- `models/grip_constraints.xml`: first-pass soft constraints between LIFT TCP sites and cue grip sites.

Robot-free mid-level training scene:

```text
models/midlevel_train_scene.xml
```

This scene keeps the same source table, floor, lights, and cameras, but removes robot control from the loop. It contains a cue, cue ball, and one object ball, and is used by `MidLevelCueEnv` to test cue/ball dynamics directly.

## Coordinate Convention

All physics quantities use SI units:

- Length: meters.
- Mass: kilograms.
- Time: seconds.

World frame:

- `+X`: table short direction.
- `+Y`: table long direction.
- `+Z`: up.

Cue frame:

- Cue local `+X` points from butt to tip.
- `cue_tip_site` is at the positive local-X end.
- Mid-level cue commands are expressed in world coordinates.

## Policy Architecture

The software scaffold follows a three-layer split.

High level:

- Selects semantic shot policies from table and ball state.
- Emits a sequence of `SkillCommand`s.

Mid level:

- Converts each semantic shot command into executable cue commands.
- Current semantic skills:
  - `PotShotPolicy`
  - `SafetyShotPolicy`
  - `PositionShotPolicy`
  - `BreakShotPolicy`

Low level:

- Tracks timed cue pose and velocity commands with a dual-arm differential-IK nominal controller.
- Adds bounded 12-D joint-position residuals and executes them through MuJoCo position actuators.
- Provides a Gymnasium environment and PPO training entry point for residual joint-level RL.

The core mid-level contract is:

```text
SkillCommand + SceneState -> tuple[CueCommand, ...]
CueCommand = cue pose + cue velocity + duration + optional debug label
```

`CueCommand` deliberately does not include gripper force:

- Gripper force belongs to future low-level tool manipulation and sim-to-real robustness.

Run the low-level residual environment smoke test:

```bash
python scripts/smoke_tests/run_lowlevel_residual_env_smoke.py
```

The Gento scene keeps robot-right fixed as the forward support/direction hand
and drives the cue axially with robot-left as the rear speed hand. It uses the
source table collision geometry, adds robot/table collision proxies, and keeps
the 20 mm cue shaft above the 1.090 m rail top. Build the six additional SDF
shapes into the ignored project-local plugin directory, then validate both the
nominal IK and imported 14-D PPO policy:

```bash
python scripts/assets/build_mujoco_billiards_sdf_plugin.py \
  --source /path/to/mujoco-billiards \
  --target-dir local_plugins
python scripts/smoke_tests/run_gento_role_ik_smoke.py
MUJOCO_GL=egl python scripts/render/render_gento_role_ik_validation.py
```

The video, contact-frame PNG, and metric JSON are written under
`outputs/gento_dev_midlevel/`.

Start a PPO training run:

```bash
python scripts/train/train_lowlevel_residual.py --total-timesteps 100000
```

## Mid-Level Two-Ball Direct Behavior Cloning

The learned mid-level policy is a robot-free inverse shot model. Its normalized
observation is:

```text
[cue_x, cue_y, object_x, object_y, pocket_x, pocket_y, stop_x, stop_y]
```

The action contains a ghost-ball angle residual in `[-15, 15]` degrees and cue
speed in `[0.3, 2.5]` m/s. A deterministic executor keeps the cue horizontal,
hits the cue-ball center, and simulates the exact source table at a 10 us
timestep until the required balls remain below the linear and surface-angular
stop thresholds for 0.2 s.

Generate the independent balanced core libraries (196,608 train and 12,288 validation):

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> python scripts/tools/generate_midlevel_tasks.py \
  --backend mujoco-warp \
  --physics-device cuda:0 \
  --num-worlds 1024
```

Task generation uses a short MJWarp prefilter followed by the full exact
stopping rollout, then replays every accepted task before saving. Each task
stores the feasible generating direction and speed as its BC label, together
with physics and content hashes. Loading rejects stale or modified libraries.

Generation is stratified on the two physical distances that dominate basic
shot difficulty:

| Band | Cue ball to object ball | Object ball to pocket |
| --- | ---: | ---: |
| short | 0.30–0.48 m | 0.18–0.34 m |
| medium | 0.48–0.68 m | 0.34–0.52 m |
| long | 0.68–0.90 m | 0.52–0.70 m |

The Cartesian product creates nine cells. The balanced core and validation are
generated with independent seeds and are balanced across every pocket/cell
combination; their per-cell counts differ by at most one. Aggregating a cell by
its harder axis gives approximately `1/9` easy, `3/9` medium, and `5/9` hard
tasks, so short finishing shots cannot dominate either split.
Audit any libraries with:

```bash
python scripts/tools/audit_midlevel_task_difficulty.py --require-balanced
```

The formal training library adds 196,608 targeted tasks to the balanced core,
for 393,216 tasks total. It selects 49,152 distinct source geometries with
extra weight on corner pockets and long cue-to-object distances. Every geometry
is replayed at `-0.02`, `-0.01`, `+0.01`, and `+0.02` m/s around its source
speed, retaining the four tasks only as a complete stable group. Run the
resumable eight-GPU collection, merge, full-library replay, and atomic publish
pipeline in tmux with:

```bash
tmux new-session -d -s midlevel_local_speed_393216 \
  'bash scripts/tools/run_midlevel_local_speed_expansion_8gpu.sh'
```

The training split must cover every pocket/cell combination but is intentionally
non-uniform after targeted augmentation. Validation remains strictly balanced
at 12,288 tasks.

Train exactly one Actor with one seed and one supervised stage:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> python scripts/train/train_midlevel_bc.py \
  --tasks outputs/tasks/midlevel_two_ball_train.npz \
  --validation-tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --device cuda:0 \
  --seed 0 \
  --output outputs/checkpoints/midlevel_bc.pt
```

When an existing task library has been physically replayed successfully and
only its stored implementation fingerprints are stale, add
`--allow-task-fingerprint-mismatch`. Backend type and shot timing are never
overridden by this option, and both old and active fingerprints are recorded in
the checkpoint manifest.

The Actor expands the raw observation into 47 deterministic geometry features
and feeds them through a `512, 512, 256` MLP. Both action dimensions are
trained together by weighted action reconstruction. No second training stage,
extra physical probes, replay buffer, or model/seed selection is used. The
fixed validation library is reported after training but does not select among
checkpoints.

Run physical validation with:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> python scripts/tools/evaluate_midlevel_bc.py \
  outputs/checkpoints/midlevel_bc.pt \
  --tasks outputs/tasks/midlevel_two_ball_validation.npz \
  --backend mujoco-warp \
  --device cuda:0 \
  --physics-device cuda:0
```

The physical report includes aggregate, per-pocket, per-cell, and
`easy`/`medium`/`hard` success and stop-error metrics. Evaluation rejects an
unbalanced validation library before launching physics.

For one checked-GPU training and validation run, select one idle physical GPU
explicitly:

```bash
bash scripts/train/run_midlevel_bc.sh <gpu-index>
```

Task regeneration is deliberately separate from training because the formal
library combines a balanced core with audited targeted augmentation;
`MIDLEVEL_REGENERATE_TASKS=1` is therefore rejected by the training wrapper.

`BCCheckpointMidLevelPolicy` adapts the `.pt` checkpoint to the existing
`SkillCommand + SceneState -> CueCommand` pipeline contract.

## Validation and Smoke Tests

Model and scene checks:

```bash
python scripts/tools/inspect_model.py
python scripts/smoke_tests/run_physics_smoke.py
python scripts/smoke_tests/run_initial_rack_smoke.py
python scripts/smoke_tests/run_collision_calibration.py
```

The collision calibration checks head-on ball transfer, source-cylinder cushion restitution, cloth rolling resistance, and middle-pocket entry. The physics step is 10 µs, matching the source one-ball scene's requirement for its stiff rail contacts; low-level control remains at 10 ms through 1000 MuJoCo substeps.

Render baseline-versus-calibrated collision comparisons:

```bash
MUJOCO_GL=egl python scripts/render/render_collision_comparisons.py --scenario all
```

Robot/cue scaffold checks:

```bash
python scripts/smoke_tests/run_dual_grip_scaffold.py
python scripts/smoke_tests/run_grip_constraint_smoke.py
python scripts/smoke_tests/run_guided_grip_stroke.py
```

Policy interface checks:

```bash
python scripts/smoke_tests/pipeline_smoke.py
python scripts/smoke_tests/run_midlevel_env_smoke.py
python scripts/smoke_tests/run_midlevel_reward_smoke.py
python scripts/smoke_tests/run_midlevel_two_ball_env_smoke.py
python scripts/smoke_tests/run_midlevel_bc_training_smoke.py
python scripts/smoke_tests/run_midlevel_difficulty_smoke.py
python scripts/smoke_tests/run_midlevel_mujoco_warp_generation_filter_smoke.py
python scripts/smoke_tests/run_midlevel_mujoco_warp_slot_order_smoke.py \
  outputs/tasks/midlevel_two_ball_validation.npz
```

Cue spin response checks:

```bash
python scripts/smoke_tests/run_spin_response_sweep.py
python scripts/smoke_tests/search_draw_shot.py
```

Current physics finding:

- Left/right side spin produces clearly distinguishable lateral and angular responses.
- High/low vertical cue offsets produce measurable angular velocity differences.
- A clean, realistic draw-shot rollback is not yet validated. Earlier high-power rollback videos were rejected as a validation artifact because the cue continued pushing after ball-ball contact.

## Rendering

Render the default scene:

```bash
MUJOCO_GL=egl python scripts/render/render_scene.py
```

Render table and robot showcase videos:

```bash
MUJOCO_GL=egl python scripts/render/render_pool_asset_videos.py \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --seconds 6
```

Render a robot-free mid-level stroke:

```bash
MUJOCO_GL=egl python scripts/render/render_midlevel_env_stroke.py
```

Render high/center/low and left/center/right spin comparisons:

```bash
MUJOCO_GL=egl python scripts/render/render_spin_response_comparison.py --kind both
```

## Asset Workflow

The active table definition is an exact copy of
`../mujoco-billiards/billiard-table-definitions.xml`; its source SDF
geometry is used for both rendering and contacts. Both scenes directly include
`../mujoco-billiards/ball-definitions.xml` and load its textures from
the source `img/` directory. Build and install the matching plugin with
`scripts/assets/build_mujoco_billiards_sdf_plugin.py`.

The cue is still the localized Sketchfab mesh. Regenerate only that asset with:

```bash
python scripts/assets/extract_cue_visual_asset.py
```

## Known Limitations

- The 9-foot pool table is not a regulation 12-foot snooker table.
- The custom SDF plugin is installed inside the `pool` environment and must be rebuilt after MuJoCo upgrades.
- Pocket entry events are detected, but pocketed balls are not removed automatically.
- Cloth and cushion contact parameters come from the source table; cue-tip tuning remains project-specific.
- LIFT mesh collision is disabled; simplified robot collision proxies still need to be added.
- LIFT pose is a scaffold, not a solved IK posture.
- `run_guided_grip_stroke.py` uses direct Jacobian qpos updates as a smoke test, not a final controller.
- Robot-free `MidLevelCueEnv` directly controls the cue free joint and should be treated as an ideal low-level executor.

## Attribution

The active table design and ball assets come from the sibling checkout at
`../mujoco-billiards`; the retained attribution copy under
`assets/table/mujoco_billiards` comes from
[`hideboz/mujoco-billiards`](https://github.com/hideboz/mujoco-billiards),
licensed under MIT. See `assets/table/mujoco_billiards/LICENSE`.

The cue under `assets/cue/sketchfab_pool_table_traditional` remains derived
from **Pool Table Traditional** by fizyman under CC-BY-4.0; the original license
is retained at `assets/table/pool_table_traditional/license.txt`.
