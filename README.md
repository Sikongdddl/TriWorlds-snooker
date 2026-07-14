# Snooker MuJoCo

MuJoCo scaffold for studying billiards/snooker manipulation with a future dual-arm mobile robot. The current repository focuses on the physics and software interfaces around a cue, balls, table contacts, and a layered policy pipeline before adding full robot control.

The project is intentionally small: plain MJCF models, the native `mujoco` Python API, NumPy, and executable smoke tests.

## Current Status

Implemented:

- A pool/snooker-style MuJoCo scene using the downloaded Sketchfab **Pool Table Traditional** visual asset.
- Hidden primitive collision proxies for the table bed, cushions, cue, and balls.
- A dynamic cue with a Sketchfab visual mesh and primitive physical geoms.
- Dynamic cue ball and object balls with free joints.
- An articulated LIFT robot scaffold in the default scene.
- Robot-free mid-level training scene for cue/ball physics experiments.
- High/mid/low policy interface scaffold.
- Smoke tests for model loading, cue/ball contacts, spin response, and policy interfaces.
- Render scripts for table, cue stroke, robot scaffold, and spin-response comparison videos.

Not implemented yet:

- AC-One robot model.
- Final dual-arm controller.
- Mobile base control.
- Gymnasium/RL training loop.
- Full snooker rules.
- Ball pocket sensors/removal.
- Calibrated cloth/cushion/cue-tip dynamics.
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

Create an environment and install the minimal runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For headless rendering on Linux, set an appropriate MuJoCo GL backend:

```bash
export MUJOCO_GL=egl
```

The development machine used for this repo also has a Conda environment named `metaworld` with MuJoCo, NumPy, imageio, OpenCV, and PyTorch. The project itself does not require that environment.

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

- `models/pool_table_asset.xml`: converted Sketchfab table visual asset.
- `models/table_physics.xml`: hidden table-bed, cushion, and pocket-site proxy layer.
- `models/balls_physics.xml`: dynamic ball bodies for the full scene.
- `models/cue_physics.xml`: dynamic cue body with visual mesh and primitive collision.
- `models/lift_articulated.xml`: articulated LIFT robot scaffold.
- `models/grip_constraints.xml`: first-pass soft constraints between LIFT TCP sites and cue grip sites.

Robot-free mid-level training scene:

```text
models/midlevel_train_scene.xml
```

This scene keeps the visual table and physical proxies, but removes robot control from the loop. It contains a cue, cue ball, and one object ball, and is used by `MidLevelCueEnv` to test cue/ball dynamics directly.

## Coordinate Convention

All physics quantities use SI units:

- Length: meters.
- Mass: kilograms.
- Time: seconds.

World frame:

- `+X`: table long direction.
- `+Y`: table short direction.
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

- Tracks cue pose and velocity commands with robot joints.
- Future work: dual-arm impedance control, passive tool fixtures, and residual joint-level RL.

The core mid-level contract is:

```text
SkillCommand + SceneState -> tuple[CueCommand, ...]
CueCommand = cue pose + cue velocity + optional debug label
```

`CueCommand` deliberately does not include gripper force or per-command duration:

- Gripper force belongs to future low-level tool manipulation and sim-to-real robustness.
- Command duration is fixed by the executor's `action_repeat`, not learned by the mid-level policy.

## Mid-Level RL Plan

The current training target is not full robot control. The first RL task is a robot-free cue-control environment where the policy learns cue-level shot execution.

For offensive pot shots, the intended low-dimensional action manifold is:

```text
cue speed
tip offset y
tip offset z
```

This keeps the action space aligned with the final low-level command interface while avoiding a full 3D pose/quaternion action space. The deterministic parts of cue placement are constraints of the skill manifold, not an extra policy adapter.

The staged curriculum scaffold in `src/snooker_env/midlevel_rl.py` is:

1. `impact_parameter_inference`: cue direction, cue speed, contact offset, elevation, spin intent.
2. `cue_setup_trajectory_generation`: setup pose, align pose, stroke-start pose.
3. `stroke_trajectory_generation`: executable `CueCommand` sequence.

These stages are internal to each semantic skill. High-level policy still calls `PotShotPolicy`, `SafetyShotPolicy`, `PositionShotPolicy`, or `BreakShotPolicy`.

## Validation and Smoke Tests

Model and scene checks:

```bash
python scripts/tools/inspect_model.py
python scripts/smoke_tests/run_physics_smoke.py
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
python scripts/smoke_tests/midlevel_curriculum_smoke.py
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

The downloaded Sketchfab zip is ignored by git. After placing it under `assets/`, regenerate converted assets with:

```bash
python scripts/assets/convert_pool_table_asset.py
python scripts/assets/extract_cue_visual_asset.py
```

Visual and physical geometry are intentionally separate:

- Visual meshes come from the asset package.
- MuJoCo collision remains primitive and independently tunable.

This keeps the scene useful for physics tests even if the visual asset changes.

## Known Limitations

- The table is based on an 8-foot pool table asset, not a regulation 12-foot snooker table.
- Table collision is an approximate hidden proxy.
- Pocket sites exist, but event sensors and ball removal are not implemented.
- Cloth, cushion, ball, and cue-tip contact parameters are placeholders.
- LIFT mesh collision is disabled; simplified robot collision proxies still need to be added.
- LIFT pose is a scaffold, not a solved IK posture.
- `run_guided_grip_stroke.py` uses direct Jacobian qpos updates as a smoke test, not a final controller.
- Robot-free `MidLevelCueEnv` directly controls the cue free joint and should be treated as an ideal low-level executor.

## Attribution

The table asset under `assets/table/pool_table_traditional` is based on **Pool Table Traditional** by fizyman, licensed under CC-BY-4.0. See:

```text
assets/table/pool_table_traditional/license.txt
```
