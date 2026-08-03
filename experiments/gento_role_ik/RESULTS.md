# Gento Role-Aware IK Migration Results

The original 100,352-step PPO checkpoint was migrated with the Gento URDF and
revalidated against the `dev-midlevel` table frame (`+Y` long axis, cloth at
`z=1.05 m`). The deterministic default-command rollout completed successfully.

| Metric | Zero-residual IK | Imported PPO |
|---|---:|---:|
| Success | yes | yes |
| Peak cue-ball speed | 0.08112 m/s | 0.09415 m/s |
| Cue-ball forward displacement | 37.23 mm | 45.57 mm |
| Maximum direction error | 0.00249 rad | 0.00529 rad |
| Maximum front support error | 2.55 mm | 2.69 mm |
| Maximum rear grip error | 1.62 mm | 1.13 mm |
| Maximum robot/table penetration | 0 | 0 |
| Maximum cue/palm penetration | 0 | 0 |
| Maximum cue/table contact tolerance | 0.0291 mm | 0.0266 mm |

The cue center is `z=1.101 m`; its 10 mm collision radius leaves 1 mm nominal
clearance above the 1.090 m rail top. MuJoCo's soft-contact tolerance accounts
for the sub-0.03 mm contact distances in the table row above.
