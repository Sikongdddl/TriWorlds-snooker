# MuJoCo Pool/Snooker Scene Scaffold

This project is the working scaffold for a future dual-arm mobile robot billiards/snooker simulation. The default scene uses the downloaded Sketchfab `Pool Table Traditional` glTF package as the visual base, with the LIFT robot visual asset placed beside the table.

It still intentionally does not include the AC-One robot, dual-arm control, mobile base control, Gymnasium wrappers, full snooker rules, perception, or domain randomization.

## Default Scene

Default model:

```text
models/scene_pool_asset.xml
```

This scene is generated from:

```text
assets/table/pool_table_traditional/scene.gltf
```

The glTF meshes are converted to MuJoCo-referenced OBJ files under:

```text
assets/table/pool_table_traditional/mujoco_full/
```

The table, pockets, net bags, and hanging light come from the downloaded asset. The default scene includes an articulated LIFT model compiled from URDF. Static glTF balls are replaced by dynamic MuJoCo bodies. The cue uses the Sketchfab cue as a visual mesh attached to a dynamic MuJoCo body, while collision/contact remains on primitive geoms so the scene can be used for physics smoke tests and later training.

Visual mesh and collision geometry stay separate:

- `models/table_physics.xml`: hidden playfield, cushion, and pocket-site proxies.
- `models/balls_physics.xml`: dynamic cue ball and rack balls.
- `models/cue_physics.xml`: dynamic cue with Sketchfab visual mesh, primitive `cue_shaft`/`cue_tip`, `cue_left_grip_site`, and `cue_right_grip_site`.
- `models/lift_articulated.xml`: articulated LIFT MJCF exported from `lift_fixed.urdf`, with position actuators and TCP sites.

## Coordinates and Units

All values use SI units where MuJoCo physics is involved.

- World X axis: table long direction.
- World Y axis: table short direction.
- World Z axis: up.
- The visual asset is converted from glTF coordinates into this convention.

The current visual table follows the 8-foot pool table asset dimensions. It is not yet rescaled into a regulation snooker table.

## Install

```bash
cd /home/ubuntu/jrWork/snooker_mujoco
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Convert the downloaded glTF asset into MuJoCo OBJ/MJCF assets:

```bash
python scripts/assets/convert_pool_table_asset.py
```

Extract the Sketchfab cue as a local visual-only mesh for the dynamic MuJoCo cue:

```bash
python scripts/assets/extract_cue_visual_asset.py
```

Inspect the default model:

```bash
python scripts/tools/inspect_model.py
```

Open the default viewer:

```bash
python scripts/tools/view_scene.py
```

Render images and an orbit video:

```bash
MUJOCO_GL=egl python scripts/render/render_scene.py
```

Render several pool-table showcase videos:

```bash
MUJOCO_GL=egl python scripts/render/render_pool_asset_videos.py --width 1280 --height 720 --fps 30 --seconds 6
```

Render a robot-free mid-level pot shot video:

```bash
MUJOCO_GL=egl python scripts/render/render_midlevel_env_stroke.py
```

Run a physics smoke test for the current cue/ball chain:

```bash
python scripts/smoke_tests/run_physics_smoke.py
```

Check the current dual-grip scaffold alignment:

```bash
python scripts/smoke_tests/run_dual_grip_scaffold.py
```

Run a guided dual-grip stroke scaffold:

```bash
python scripts/smoke_tests/run_guided_grip_stroke.py
```

Run the high/mid/low-level policy pipeline smoke test:

```bash
python scripts/smoke_tests/pipeline_smoke.py
```

Run the robot-free mid-level shot policy environment:

```bash
python scripts/smoke_tests/run_midlevel_env_smoke.py
```

Run the three-stage mid-level curriculum interface smoke test:

```bash
python scripts/smoke_tests/midlevel_curriculum_smoke.py
```

Check whether cue-tip offsets produce distinguishable cue-ball spin responses:

```bash
python scripts/smoke_tests/run_spin_response_sweep.py
```

Script layout:

- `scripts/assets`: asset conversion/extraction scripts.
- `scripts/tools`: repeated developer tools such as model inspection and viewer launch.
- `scripts/render`: image/video generation scripts.
- `scripts/smoke_tests`: validation scripts for physics, constraints, and policy interfaces.

## Pipeline Scaffold

The initial training/control architecture lives in flat modules under `src/snooker_env` and mirrors the three-layer split used in `booster_gym`:

- High level: game strategy selects a sequence of semantic mid-level shot policy calls from table/ball/optional vision state.
- Mid level: each shot policy converts one `SkillCommand` into a low-level command trajectory.
- Low level: tool manipulation tracks cue pose and cue velocity commands with robot joints.

The mid-level interface is intentionally uniform:

```text
SkillCommand + SceneState -> tuple[CueCommand, ...]
CueCommand = cue pose + cue velocity + optional debug label
```

The optional `debug_label` is for logs and training diagnostics only. Low-level control should not depend on the semantic shot policy that produced a command. Command duration is fixed by the executor's `action_repeat`, not produced by the policy.

The first supported mid-level shot policies are:

- `PotShotPolicy`: offensive potting shot.
- `SafetyShotPolicy`: defensive safety shot.
- `PositionShotPolicy`: potting-style shot with cue-ball position objective.
- `BreakShotPolicy`: high-power opening/break shot.

Impact-parameter inference, cue setup, backswing, stroke, and follow-through are internal stages of each policy. They are not exposed as separate high-level calls.

For mid-level training, `models/midlevel_train_scene.xml` provides a robot-free scene with the visual table, table physics proxy, cue, cue ball, and one object ball. `MidLevelCueEnv` executes `CueCommand` trajectories directly on the cue free joint as an idealized low-level controller. This controller enforces table/cushion feasibility at the simulator layer: requested cue poses that would penetrate the table or rails are projected upward into the feasible action space, and `constraint_projection_count` records how often this happened. This lets mid-level policies train shot selection and command-trajectory generation before adding dual-arm robot tracking, without learning physically impossible cue motions.

Mid-level RL training uses a fixed three-stage curriculum in `midlevel_rl.py`:

1. `impact_parameter_inference`: learn cue direction, cue speed, contact offset, cue elevation, and spin intent.
2. `cue_setup_trajectory_generation`: learn pre-shot setup, alignment, and stroke-start poses.
3. `stroke_trajectory_generation`: learn the executable `CueCommand` sequence for backswing, impact, and follow-through.

These stages are internal to each semantic shot policy. High level still calls `PotShotPolicy`, `SafetyShotPolicy`, `PositionShotPolicy`, or `BreakShotPolicy`; it does not call the curriculum stages directly.

Non-RL system behaviors are separated in `pipeline_system.py`:

- `GeometricBodyPositioningPlanner`: walking-to-shot preparation scaffold.
- `DefaultRecoveryPlanner`: shooting-to-walking recovery scaffold.

The current implementations are deliberately scripted placeholders. They define the contracts for future planning, RL, VLM, impedance control, and residual joint-level learning without introducing a training framework dependency yet.

## Main Models

- `models/scene_pool_asset.xml`: current default scene for future development.
- `models/pool_table_asset.xml`: converted Sketchfab pool-table visual asset plus hidden table proxy.
- `models/table_physics.xml`: hidden table collision and pocket proxy layer.
- `models/midlevel_train_scene.xml`: robot-free mid-level training scene with cue, cue ball, and one object ball.
- `models/midlevel_balls.xml`: two-ball asset for mid-level policy training.
- `models/balls_physics.xml`: dynamic billiard balls.
- `models/cue_physics.xml`: dynamic cue body, Sketchfab cue visual mesh, primitive physics geoms, and grip sites.
- `models/lift_articulated.xml`: articulated LIFT model compiled from URDF and included in the default scene.
- `models/grip_constraints.xml`: stage-1 soft equality constraints from LIFT TCP sites to cue grip sites.

## Current Limits

- Table collision is still an approximate hidden proxy.
- LIFT mesh collision is disabled in the default scene; simplified robot collision proxies still need to be added.
- The LIFT base pose and lift-column height are a scaffold pose chosen to put TCP sites near the cue grip sites. It is not a solved robot IK posture yet.
- `run_guided_grip_stroke.py` uses direct Jacobian IK qpos updates as a scaffold smoke test. It is not the final RL controller.
- Pockets are visually present and have placeholder sites, but not yet event sensors or ball-removal logic.
- Cue-table collision is disabled in the first smoke-test contact mask so the cue can pass over the rail and hit the ball; cue-ball and ball-table contacts are active.
- Cloth/cushion/ball/cue-tip dynamics are not calibrated.
- The asset is an 8-foot pool table, not a regulation 12-foot snooker table.

## Asset Attribution

`assets/table/pool_table_traditional` is based on "Pool Table Traditional" by fizyman, licensed under CC-BY-4.0. See `assets/table/pool_table_traditional/license.txt`.
