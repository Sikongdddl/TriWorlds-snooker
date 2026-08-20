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
- A 12-D residual joint-position Gymnasium environment with differential-IK nominal control.
- High/mid/low policy interface scaffold.
- A PoolTool-based nine-ball high-level environment aligned to the MuJoCo table convention.
- A two-player, six-action DQN baseline with turn-taking and frozen-opponent self-play cloning.
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

Reproduce a PoolTool event-based billiards simulation:

```bash
python -m pip install pooltool-billiards --extra-index-url https://archive.panda3d.org/
python scripts/pooltool/reproduce_pooltool_example.py
```

PoolTool brings a larger dependency stack, including Panda3D, Numba, SciPy, and newer NumPy releases. Prefer installing it in a separate environment if you also use this repository for MuJoCo/RL training.

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
│   ├── pooltool/           # PoolTool reproduction experiments
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
- `models/table_physics.xml`: hidden cloth, rounded cushion, pocket sensor, and catch-bin physics layer.
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

- `+X`: table short direction.
- `+Y`: table long direction.
- `+Z`: up.
- Table center is near the world origin.
- The current American pool table play area is `1.27 m x 2.54 m`, so the playable plane spans approximately `X=[-0.635, 0.635]` and `Y=[-1.27, 1.27]`.

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

Start a PPO training run:

```bash
python scripts/train/train_lowlevel_residual.py --total-timesteps 100000
```

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

## PoolTool Reference Environment

PoolTool is an event-based billiards simulator with a higher-fidelity billiards rules/physics stack than the current MuJoCo scaffold. It is useful as a reference environment for high-level shot planning before connecting strategy to robot execution.

The local high-level wrapper does not use PoolTool's default 7-foot table. `PoolToolSinglePlayerEnv` builds a custom 9-foot American pool table aligned to the MuJoCo/mid-level convention:

- Play area: `1.27 m x 2.54 m`.
- PoolTool internal frame: corner origin, short side along `x`, long side along `y`.
- Project world frame: centered origin, short side along `X`, long side along `Y`, cloth at `Z=1.05 m`.
- Pocket centers in world coordinates: corners `(+/-0.675, +/-1.310) m`, middles `(+/-0.717426, 0) m`.
- Ball radius/mass: `0.0285 m`, `0.165 kg`; resting center height is `1.0785 m`.

Use `PoolToolSinglePlayerEnv.pool_to_world_xy()` and `world_to_pool_xy()` when moving data between PoolTool simulation and the rest of the project. `PoolToolDQNEnv.encode_state()` already emits ball positions in centered world meters.

Install it as an optional dependency:

```bash
python -m pip install pooltool-billiards --extra-index-url https://archive.panda3d.org/
```

Using a separate Python environment is recommended because PoolTool may upgrade NumPy/SciPy and other scientific packages.

The PoolTool hello-world API is:

```python
import pooltool as pt

table = pt.Table.default()
balls = pt.get_rack(pt.GameType.NINEBALL, table)
cue = pt.Cue(cue_ball_id="cue")
system = pt.System(table=table, balls=balls, cue=cue)
system.cue.set_state(V0=8, phi=pt.aim.at_ball(system, "1"))
pt.simulate(system, inplace=True)
pt.show(system)
```

PoolTool cue parameters map naturally to high-level shot candidates:

- `V0`: cue impact speed in m/s.
- `phi`: table-plane shot direction in degrees.
- `theta`: cue inclination in degrees.
- `a`: side spin, with positive/negative corresponding to left/right side.
- `b`: top/bottom spin, with positive top and negative bottom.

Run the local reproduction script:

```bash
python scripts/pooltool/reproduce_pooltool_example.py \
  --speed 8.0 \
  --side-spin 0.0 \
  --top-spin 0.0 \
  --output outputs/pooltool/pooltool_example_summary.json
```

Open PoolTool's GUI after simulation:

```bash
python scripts/pooltool/reproduce_pooltool_example.py --show
```

This script writes a JSON summary of cue parameters, events, final ball positions, linear velocities, and angular velocities. It intentionally stays outside the MuJoCo robot-control stack for now.

Run the first high-level clearance planner:

```bash
python scripts/pooltool/run_clearance_planner.py --game-type example
```

Run a scripted break, then continue with the heuristic clearance planner:

```bash
python scripts/pooltool/run_clearance_planner.py \
  --game-type nineball \
  --legal-mode any \
  --break-rack \
  --break-speed 10
```

The planner writes three rollout artifacts by default:

```text
outputs/pooltool/clearance_plan.json
outputs/pooltool/clearance_rollout.html
outputs/pooltool/clearance_rollout.msgpack
```

`clearance_rollout.html` is a static top-down report with one section per shot. It shows ball trajectories, selected target/pocket, cue parameters found by the internal solver, top candidate scores, and the first PoolTool events.

Render the same rollout as a top-down MP4:

```bash
python scripts/pooltool/render_pooltool_rollout_video.py \
  --rollout outputs/pooltool/clearance_rollout.msgpack \
  --plan outputs/pooltool/clearance_plan.json \
  --output outputs/pooltool/clearance_rollout.mp4
```

`clearance_rollout.msgpack` stores the full PoolTool `MultiSystem`. Open it in the native PoolTool GUI:

```bash
python scripts/pooltool/show_pooltool_rollout.py outputs/pooltool/clearance_rollout.msgpack
```

Inside the GUI, use `n`/`p` to switch shots. Press `Enter` to toggle PoolTool's parallel visualization mode, where the saved shots are overlaid.

The planner's high-level action has two supported forms:

```text
ShotAction = target_ball_id + target_pocket_id
ShotAction = target_ball_id + target_pocket_id + cue_landing_cell
```

The optional `cue_landing_cell` is a regular table-plane grid cell for the final cue-ball position. With the default `8 x 4` landing grid, the DQN action space is:

```text
9 balls * 6 pockets * 32 cue landing cells = 1728 actions
```

This is the full network output space. During training, `PoolToolDQNEnv` keeps the landing-cell action space open by default. It still samples feasible cue parameters for each direct-pot `(ball, pocket)` pair and stores the reached cue-ball landing cells in a local SQLite cache keyed by the grid-discretized table state plus the target ball/pocket and solver-grid settings. If the policy selects a landing cell that is not in the cached reachable set, the environment gives a negative reward. Use `--mask-unreachable-landing-actions` only for ablation runs that should hard-mask those landing cells during exploration.

Internally, `PoolToolSinglePlayerEnv` searches for cue parameters that can realize the requested action, simulates them in PoolTool, and returns whether the target ball entered the requested pocket and, when requested, whether the cue ball stopped in the requested landing cell. The search currently samples speed, small aim offsets, side spin, and top/bottom spin around direct-pot ghost-ball geometry. If no sampled cue command solves a requested `(ball, pocket, cue_landing_cell)` action, the DQN environment treats the action as unsolved and gives a negative reward.

`HeuristicClearancePlanner` still ranks simple `(ball, pocket)` actions with depth-limited lookahead. Precise multi-rail position play, combinations, safeties, and full professional shot solving are not implemented yet.

The scripted break is not a high-level policy action. It is an episode setup helper that applies a strong break shot to scatter a rack before the `(ball, pocket)` planner takes over.

Train the DQN position-play action space:

```bash
python scripts/pooltool/train_dqn_high_level.py \
  --episodes 10000 \
  --break-speed 10 \
  --randomize-break \
  --break-speed-range 8 12 \
  --break-phi-jitter-degrees 2 \
  --device cuda
```

Precompute the opening-state landing mask table from randomized breaks:

```bash
python scripts/pooltool/precompute_landing_masks.py \
  --samples 1000 \
  --break-speed-range 8 12 \
  --break-phi-jitter-degrees 2 \
  --cache outputs/pooltool/landing_mask_cache.sqlite
```

Use the old 54-action ball/pocket space for compatibility:

```bash
python scripts/pooltool/train_dqn_high_level.py --no-cue-landing
```

### Two-Player High-Level DQN

The current two-player task uses ordered nine-ball targeting. The environment determines the lowest legal object ball, while each policy chooses one of the six pockets:

```text
observation = cue-ball XY + nine object-ball XY/pocketed states
action      = pocket_id in {lb, lc, lt, rb, rc, rt}
```

The action space deliberately has no geometric potability mask. All six pockets remain visible to the policy, so learning which pocket is appropriate is part of the high-level task rather than information leaked by the environment.

Every selected pocket produces a physical table transition:

- If the direct-pot solver finds a valid shot, PoolTool executes that solution.
- If no clear pot path exists, the environment executes a `best_effort_direct` shot along the requested pocket's raw ghost-ball line.
- A pot keeps the current player at the table.
- A miss or foul switches players, preserving the physical final positions of the cue ball and object balls.
- A cue-ball scratch is restored using a deterministic ball-in-hand placement before the next player acts.
- The rack ends when all legal object balls are pocketed or `--max-turns` is reached.

This best-effort transition is important: an infeasible pocket choice is not a no-op and does not simply discard the turn. The acting player still strikes the cue ball, and the opponent continues from the resulting table state.

Reproduce the 10,000-episode corrected experiment:

```bash
python scripts/pooltool/train_two_player_dqn_high_level.py \
  --episodes 10000 \
  --max-turns 60 \
  --learning-starts 1000 \
  --batch-size 64 \
  --buffer-size 50000 \
  --lr 3e-4 \
  --train-reward-scale 0.1 \
  --train-reward-clip 20 \
  --self-play-clone \
  --clone-window 100 \
  --clone-min-episodes 300 \
  --clone-win-rate 0.6 \
  --clone-reward-advantage 5.0 \
  --initial-opponent-epsilon 1.0 \
  --opponent-epsilon 0.05 \
  --epsilon-decay-steps 20000 \
  --break-speed 10 \
  --seed 109 \
  --device cpu \
  --log-interval 100 \
  --save-interval 250 \
  --output outputs/pooltool/two_player_corrected_10000eps_seed109.json \
  --checkpoint outputs/pooltool/two_player_corrected_10000eps_seed109.pt
```

With `--self-play-clone`, only the active player's Q network is optimized. The opponent begins as a random six-pocket policy, is replaced by a frozen copy of the active network after the configured win-rate and reward-advantage thresholds are both met, and is periodically refreshed as the active policy improves.

The seed-109 corrected run produced the following final-window diagnostics:

```text
last 1000 clear rate      0.998
last 1000 p0 win rate     0.507
last 1000 p1 win rate     0.491
last 1000 mean turns      11.20
final active updates      79346
valid action count        always 6
```

The training log records each shot's path type, cue-ball displacement, foul, pot/miss reason, player switch, action count, epsilon, and scores. This makes it possible to detect accidental no-op transitions or action-mask leakage.

Plot a completed run:

```bash
python scripts/pooltool/plot_two_player_dqn_training.py \
  --input outputs/pooltool/two_player_corrected_10000eps_seed109.json \
  --output outputs/pooltool/two_player_corrected_10000eps_seed109_curves.png \
  --window 200 \
  --title "Corrected Two-player DQN, seed=109"
```

Evaluate both saved policies and write a renderable PoolTool rollout:

```bash
python scripts/pooltool/evaluate_two_player_dqn_high_level.py \
  --checkpoint outputs/pooltool/two_player_corrected_10000eps_seed109.pt \
  --seed 109 \
  --max-turns 60 \
  --epsilon 0.05 \
  --output outputs/pooltool/two_player_corrected_10000eps_seed109_rollout.json \
  --multisystem-output outputs/pooltool/two_player_corrected_10000eps_seed109_rollout.msgpack
```

Render the rollout as an annotated MP4:

```bash
python scripts/pooltool/render_pooltool_rollout_video.py \
  --rollout outputs/pooltool/two_player_corrected_10000eps_seed109_rollout.msgpack \
  --plan outputs/pooltool/two_player_corrected_10000eps_seed109_rollout.json \
  --output outputs/pooltool/two_player_corrected_10000eps_seed109_rollout.mp4 \
  --width 1280 \
  --height 720 \
  --fps 30
```

Training logs, checkpoints, plots, and videos live under `outputs/` and are intentionally ignored by git.

### Discrete Value-Iteration Baseline

The direct-pot heuristic can also be wrapped as a finite tabular MDP:

```text
state  = cue grid cell + one grid/pocketed cell per object ball
action = target_ball_id + target_pocket_id
reward = pot reward + clear bonus - foul/miss/speed penalties
```

The transition model is generated by PoolTool simulation. For each discrete state, the code evaluates high-level `(ball, pocket)` actions with the same internal direct-shot solver, records the resulting next discrete state, and then runs value iteration.

Run the small one-ball smoke case:

```bash
python scripts/pooltool/run_discrete_value_iteration.py \
  --game-type example \
  --x-bins 6 \
  --y-bins 3 \
  --max-depth 3 \
  --max-states 32
```

Run the nineball break-and-clear baseline:

```bash
python scripts/pooltool/run_discrete_value_iteration.py \
  --game-type nineball \
  --legal-mode any \
  --break-rack \
  --break-speed 10 \
  --x-bins 8 \
  --y-bins 4 \
  --action-prune 0 \
  --max-depth 0 \
  --max-states 0 \
  --prune-blocked-actions
```

`--action-prune 0` means no score-based or beam-style action pruning is used. `--prune-blocked-actions` only removes actions that have no direct-pot ghost-ball geometry or have a blocked cue/object/pocket path. `--max-depth 0` and `--max-states 0` mean expansion continues until the reachable discrete frontier closes. This is the clean value-iteration setting, but it can be slow for nineball because every transition calls PoolTool simulation.

For faster development, use a state cap without reintroducing action pruning:

```bash
python scripts/pooltool/run_discrete_value_iteration.py \
  --game-type nineball \
  --legal-mode any \
  --break-rack \
  --break-speed 10 \
  --x-bins 8 \
  --y-bins 4 \
  --action-prune 0 \
  --prune-blocked-actions \
  --max-states 64 \
  --log-interval 10
```

This writes:

```text
outputs/pooltool/discrete_value_iteration_plan.json
outputs/pooltool/discrete_value_iteration_rollout.html
outputs/pooltool/discrete_value_iteration_rollout.msgpack
```

The script logs transition-graph expansion and value-iteration progress:

```text
expand_graph: expanded=60 states=64 transitions=1896 frontier=4 depth=3
value_iteration: iteration=1 states=64 max_delta=19.968000
```

The script supports online local refitting when rollout reaches a discretized state that was not covered by a budget-limited expansion. This keeps capped experiments usable while preserving the action space as full discrete `(ball, pocket)` enumeration.

## Validation and Smoke Tests

Model and scene checks:

```bash
python scripts/tools/inspect_model.py
python scripts/smoke_tests/run_physics_smoke.py
python scripts/smoke_tests/run_initial_rack_smoke.py
python scripts/smoke_tests/run_collision_calibration.py
```

The collision calibration checks head-on ball transfer, cushion restitution,
cloth rolling resistance, and middle-pocket capture. The default physics step
is 0.25 ms; low-level control remains at 10 ms through 40 MuJoCo substeps.

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
python scripts/smoke_tests/midlevel_curriculum_smoke.py
python scripts/smoke_tests/run_pooltool_highlevel_smoke.py
python scripts/smoke_tests/run_pooltool_break_clearance_smoke.py
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
