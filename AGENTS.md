# Repository Guidelines

## Project Structure & Module Organization

- `src/snooker_env/` contains the Python package: scene loading, shared dataclasses, high/mid/low policy layers, MuJoCo controllers, and Gymnasium environments.
- `models/` contains composable MJCF/XML definitions for the table, balls, cue, constraints, and LIFT robot. `models/mujoco_billiards/billiard-table-definitions.xml` is an exact vendor copy; do not reformat it. Active scenes load ball definitions and textures directly from `/home/ubuntu/mujoco-billiards`.
- `assets/` stores licensed visual meshes and textures. The active table intentionally uses the source SDF geoms for both appearance and collision.
- `scripts/smoke_tests/` holds executable integration checks; `scripts/render/` creates diagnostic videos; `scripts/tools/` provides model inspection and viewing; `scripts/train/` contains RL entry points.
- Generated media belongs under `outputs/` and must remain untracked.

## Build, Test, and Development Commands

Run all repository commands in the `pool` Conda environment:

```bash
conda activate pool
```

GPU workloads must run on the `node31` compute node rather than on the local/login node. Check GPU utilization first, prefer an idle GPU, and select it explicitly with `CUDA_VISIBLE_DEVICES`. `node31` and the local node share the same filesystem, so do not copy or synchronize the repository between them. Use this workflow:

```bash
ssh node31
cd TriWorlds-snooker/
conda activate pool
CUDA_VISIBLE_DEVICES=<gpu-index> python <script.py> [args...]
```

Inside the launched process, the selected physical GPU is exposed as `cuda:0`; continue to pass `--physics-device cuda:0` to repository scripts when applicable.

Useful checks:

```bash
python scripts/assets/build_mujoco_billiards_sdf_plugin.py # build/install source SDF plugin
python scripts/tools/inspect_model.py                       # compile and summarize MJCF
python scripts/smoke_tests/run_physics_smoke.py             # basic contact physics
python scripts/smoke_tests/pipeline_smoke.py                # policy contracts
python scripts/smoke_tests/run_midlevel_env_smoke.py        # cue/ball rollout
python scripts/smoke_tests/run_lowlevel_residual_env_smoke.py # Gym environment
```

Use `MUJOCO_GL=egl` for headless rendering, for example `MUJOCO_GL=egl python scripts/render/render_scene.py`. Launch residual PPO training with `python scripts/train/train_lowlevel_residual.py --total-timesteps 100000`.

## Coding Style & Naming Conventions

Use Python 3 type hints, four-space indentation, `snake_case` for functions/modules, and `PascalCase` for classes. Prefer small dataclasses and protocols for cross-layer contracts. Keep MuJoCo names explicit and stable (for example, `cue_tip_site` and `left_arm_joint1`). Physics arrays should use NumPy `float64`, SI units, and MuJoCo's `wxyz` quaternion convention. The source table's long axis is world `+Y`, its short axis is `+X`, and the cloth surface is at `z=1.05` m. No formatter is configured; match surrounding code.

## Testing Guidelines

Tests are standalone smoke scripts rather than pytest suites, and no coverage threshold is defined. Name new checks `run_<feature>_smoke.py` when possible. Cover model loading, finite state, expected contacts, and relevant constraints. Run the narrow test first, then related physics and pipeline checks. Install the custom plugin in `pool` before loading either scene.

## Commit & Pull Request Guidelines

Recent commits use short, lowercase, descriptive subjects such as `add velocity-aware dual-arm cue control`. Keep each commit focused and avoid committing generated videos or downloaded archives. Pull requests should explain the behavior change, list commands run, link relevant issues, and include output metrics or rendered screenshots/videos for physics or visual changes. Call out model-parameter changes and known limitations explicitly.
