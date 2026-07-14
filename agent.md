# Agent Notes

This project is `/home/ubuntu/jrWork/snooker_mujoco`, a MuJoCo scaffold for robot billiards/snooker simulation.

## Environment

Use the existing conda environment:

```bash
CONDA_NO_PLUGINS=true conda run -n metaworld python ...
```

Verified in `metaworld`:

- Python 3.10.4
- NumPy 1.24.4
- MuJoCo 3.5.0
- imageio 2.20.0
- torch 2.2.2+cu121
- CUDA works only outside the sandbox / with elevated execution.

Inside the sandbox, `torch.cuda.is_available()` may be false even when the host GPU is available. Use elevated execution for GPU checks or rendering.

Do not use the `mujoco` conda environment for RL work unless it is updated: it can run MuJoCo smoke tests, but torch/imageio were missing when last checked.

## Core Commands

Run these from project root:

```bash
CONDA_NO_PLUGINS=true conda run -n metaworld python scripts/tools/inspect_model.py
CONDA_NO_PLUGINS=true conda run -n metaworld python scripts/smoke_tests/run_midlevel_env_smoke.py
CONDA_NO_PLUGINS=true conda run -n metaworld python scripts/smoke_tests/midlevel_curriculum_smoke.py
CONDA_NO_PLUGINS=true conda run -n metaworld python scripts/smoke_tests/pipeline_smoke.py
```

Rendering requires EGL/GPU access:

```bash
MUJOCO_GL=egl CONDA_NO_PLUGINS=true conda run -n metaworld python scripts/render/render_midlevel_env_stroke.py
```

## Script Layout

- `scripts/assets`: asset conversion/extraction.
- `scripts/tools`: reusable developer tools, such as model inspection and viewer launch.
- `scripts/render`: image/video generation.
- `scripts/smoke_tests`: validation scripts for physics, constraints, and policy interfaces.

## Model Layout

Visual assets and MuJoCo physics stay separate.

- `assets/table/pool_table_traditional`: downloaded Sketchfab table asset.
- `assets/cue/sketchfab_pool_table_traditional`: localized Sketchfab cue visual mesh.
- `models/pool_table_asset.xml`: visual table wrapper.
- `models/table_physics.xml`: hidden playfield/cushion/pocket proxy physics.
- `models/cue_physics.xml`: dynamic cue body; Sketchfab visual mesh plus primitive `cue_shaft` and `cue_tip`.
- `models/balls_physics.xml`: full scene billiard balls.
- `models/midlevel_train_scene.xml`: robot-free midlevel training scene with visual table, table physics proxy, cue, cue ball, and one object ball.
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
- `midlevel_rl.py`: three-stage curriculum-learning interfaces.

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

Do not expose internal stages as high-level calls. The stages are internal curriculum components.

## Midlevel RL Curriculum

All semantic midlevel policies should follow this internal curriculum:

1. `impact_parameter_inference`
   - Learns cue direction, cue speed, contact offset, cue elevation, spin intent, tolerance.
2. `cue_setup_trajectory_generation`
   - Learns setup pose, align pose, and stroke-start pose.
3. `stroke_trajectory_generation`
   - Learns executable `CueCommand` sequence: setup, align, stroke-start, stroke, follow-through.

`midlevel_rl.py` currently provides framework-free interfaces plus a scripted baseline. SAC training code should wrap these interfaces rather than changing the high/mid/low boundary.

## Cue/Table Constraint

Cue/table feasibility is a simulator-layer hard constraint, not just a reward penalty.

`MidLevelCueEnv` projects requested cue poses upward if the cue would penetrate the table or rails. The executed cue pose is the feasible projected pose. The result records:

- `constraint_projection_count`
- `min_cue_table_clearance`

RL should observe the actual executed result. Projection count is diagnostic; it should not be treated as the only mechanism preventing impossible actions.

## Important Caveats

- The default scene uses an 8-foot pool table asset, not regulation snooker dimensions.
- Table collision is still an approximate hidden proxy.
- Pocket visuals exist, but pocket sensors / pocket wells / ball removal are not implemented.
- Ball/cloth/cushion/cue-tip friction and restitution are placeholders, not calibrated.
- Current midlevel env assumes an ideal cue controller, not the dual-arm robot.
- The full LIFT scene still uses scaffold grip constraints and not solved IK/control.
- Avoid training policies that rely on cue penetration or arbitrary free-joint teleportation.

## Current Good Smoke Tests

These passed in `metaworld`:

```bash
CONDA_NO_PLUGINS=true conda run -n metaworld python scripts/smoke_tests/run_midlevel_env_smoke.py
CONDA_NO_PLUGINS=true conda run -n metaworld python scripts/smoke_tests/midlevel_curriculum_smoke.py
```

Expected midlevel smoke characteristics:

- cue tip contacts cue ball
- cue ball contacts object ball
- no NaN/Inf
- no numerical explosion
- `min_cue_table_clearance` remains positive after projection
