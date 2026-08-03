# Gento Role-Aware IK Training

This independent experiment trains a 14-D PPO residual on top of Gento's
differential IK in `models/gento_side_grasp_scene.xml`.

- Robot-right is the fixed front support hand. It anchors position and
  orientation, defining the cue direction.
- Robot-left is the rear speed hand. It alone receives the axial stroke-speed
  feed-forward command.
- Both grippers use physical upper/lower finger contact; there is no welded cue
  equality.
- The source table, robot collision proxies, cue shaft, and palm guards all
  participate in collision detection.

The imported 100k-step checkpoint is stored at
`assets/policies/gento_role_ik_residual_final.zip`.

```bash
python scripts/smoke_tests/run_gento_role_ik_smoke.py
python experiments/gento_role_ik/evaluate.py --episodes 20
python experiments/gento_role_ik/train.py --total-timesteps 100000
MUJOCO_GL=egl python scripts/render/render_gento_role_ik_validation.py
```

New checkpoints, Monitor logs, and evaluation JSON files are generated under
`experiments/gento_role_ik/artifacts/` and remain untracked.
