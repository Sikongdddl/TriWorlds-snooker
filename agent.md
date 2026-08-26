# Agent Notes

This project is `/home/ubuntu/TriWorlds-snooker`, a MuJoCo scaffold for robot billiards/snooker simulation.

## Environment

Use the existing conda environment:

```bash
conda activate pool
```

The active table requires the custom source SDF plugin. Rebuild it after replacing or upgrading MuJoCo:

```bash
python scripts/assets/build_mujoco_billiards_sdf_plugin.py
```

## Core Commands

Run these from project root:

```bash
python scripts/tools/inspect_model.py
python scripts/smoke_tests/run_midlevel_env_smoke.py
python scripts/smoke_tests/run_midlevel_two_ball_env_smoke.py
python scripts/smoke_tests/run_midlevel_bc_training_smoke.py
python scripts/smoke_tests/pipeline_smoke.py
```

Rendering requires EGL/GPU access:

```bash
MUJOCO_GL=egl python scripts/render/render_midlevel_env_stroke.py
```

## Script Layout

- `scripts/assets`: asset conversion/extraction.
- `scripts/tools`: reusable developer tools, such as model inspection and viewer launch.
- `scripts/render`: image/video generation.
- `scripts/smoke_tests`: validation scripts for physics, constraints, and policy interfaces.

## Model Layout

- `../mujoco-billiards/ball-definitions.xml`: ball class and numbered materials loaded directly by both active scenes; textures resolve from its `img/` directory.
- `assets/table/mujoco_billiards`: retained source attribution; it is not the active ball-texture path.
- `assets/table/pool_table_traditional`: legacy Sketchfab source retained for the cue asset and its license.
- `assets/cue/sketchfab_pool_table_traditional`: localized Sketchfab cue visual mesh.
- `models/mujoco_billiards/billiard-table-definitions.xml`: exact source table XML, including checker floor, four lights, SDF pockets, rails, and collision geometry.
- `models/mujoco_billiards_integration.xml`: project-only named pocket sites; it does not alter source rendering or contacts.
- `models/cue_physics.xml`: dynamic cue body; Sketchfab visual mesh plus primitive `cue_shaft` and `cue_tip`.
- `models/balls_physics.xml`: source-class marked cue ball and full numbered rack with project-stable names.
- `models/midlevel_train_scene.xml`: robot-free midlevel training scene with the source table, cue, cue ball, and one object ball.
- `models/midlevel_balls.xml`: two-ball asset for midlevel training.
- `models/lift_articulated.xml`: articulated LIFT scaffold.
- `models/grip_constraints.xml`: current soft equality constraints from LIFT TCPs to cue grip sites.

## Pipeline Architecture

Source modules are intentionally flat under `src/snooker_env`:

- `pipeline.py`: high/mid/low orchestration.
- `pipeline_types.py`: shared dataclasses and enums.
- `pipeline_high_level.py`: high-level strategy placeholder.
- `pipeline_mid_level.py`: semantic shot policies.
- `pipeline_low_level.py`: low-level controller placeholders.
- `pipeline_system.py`: non-RL body positioning/recovery.
- `midlevel_env.py`: robot-free ideal cue-control MuJoCo environment.
- `midlevel_two_ball_env.py`: single-step two-ball execution and reward.
- `midlevel_bc.py`: direct behavior-cloning Actor and checkpoint adapter.

High level selects semantic midlevel shot policy calls. Midlevel policies output low-level cue command trajectories:

```text
SkillCommand + SceneState -> tuple[CueCommand, ...]
CueCommand = cue pose + cue velocity + optional debug label
```

`CueCommand.debug_label` is diagnostics only. The low-level controller should execute the numeric pose and velocity fields without depending on which midlevel policy produced the command. Command duration is fixed by the executor's `action_repeat`, not produced by the policy.

Current midlevel policies:

- `PotShotPolicy`
- `SafetyShotPolicy`
- `PositionShotPolicy`
- `BreakShotPolicy`

## Midlevel Direct Behavior Cloning

The learned two-ball policy uses one supervised stage and one seed. It maps the
8-D shot observation directly to the normalized ghost-ball angle residual and
cue speed. Training reconstructs the feasible action stored with every task;
the fixed validation split reports reconstruction metrics but never selects a
different seed or checkpoint.

Use `scripts/train/train_midlevel_bc.py` for a single run, or
`scripts/train/run_midlevel_bc.sh <gpu-index>` for checked-GPU training followed
by one physical validation. Regenerate task libraries whenever their active
model or backend fingerprint is stale.

Previously generated task arrays may be reused with
`--allow-task-fingerprint-mismatch` only after representative physical replay
shows equivalent outcomes. The option never bypasses backend-type or shot-timing
differences, and the checkpoint records both fingerprint sets.

New task libraries must balance the 3×3 cue-object/object-pocket distance grid
defined in `midlevel_difficulty.py` across every named pocket. Train and
validation use independent generation seeds. Run
`scripts/tools/audit_midlevel_task_difficulty.py --require-balanced` before
training and retain the per-difficulty validation metrics in the BC report.

## Cue/Table Constraint

Cue/table feasibility is a simulator-layer hard constraint, not just a reward penalty.

`MidLevelCueEnv` projects requested cue poses upward if the cue would penetrate the table or rails. The executed cue pose is the feasible projected pose. The result records:

- `constraint_projection_count`
- `min_cue_table_clearance`

Learned policies should use the actual executed result. Projection count is diagnostic; it should not be treated as the only mechanism preventing impossible actions.

## Important Caveats

- The default scene uses a 9-foot pool table, not regulation snooker dimensions.
- The source table requires the installed `libsdf_plugin.so`; model loading fails if the plugin is absent or incompatible.
- Pocket regions and entry events exist, but ball removal is not implemented.
- The source table uses `+Y` as its long axis, `+X` as its short axis, and `z=1.05` m as the cloth surface.
- Current midlevel env assumes an ideal cue controller, not the dual-arm robot.
- The full LIFT scene still uses scaffold grip constraints and not solved IK/control.
- Avoid training policies that rely on cue penetration or arbitrary free-joint teleportation.

## Current Good Smoke Tests

These should be run in `pool`:

```bash
python scripts/smoke_tests/run_midlevel_env_smoke.py
python scripts/smoke_tests/run_midlevel_two_ball_env_smoke.py
python scripts/smoke_tests/run_midlevel_bc_training_smoke.py
python scripts/smoke_tests/run_midlevel_difficulty_smoke.py
```

Expected midlevel smoke characteristics:

- cue tip contacts cue ball
- cue ball contacts object ball
- no NaN/Inf
- no numerical explosion
- `min_cue_table_clearance` remains positive after projection
