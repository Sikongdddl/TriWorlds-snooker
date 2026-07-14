# Sketchfab Cue Visual Asset

This directory contains the localized visual mesh for the cue extracted from the downloaded Sketchfab `Pool Table Traditional` glTF asset.

- `pool_cue_visual_local.obj`: visual mesh centered on the MuJoCo cue body.
- `cue_mat_baseColor.png`: base color texture used by `models/cue_physics.xml`.

The active physics model is still defined by primitive geoms in `models/cue_physics.xml`: `cue_shaft` and `cue_tip`. The mesh here is visual-only and has no collision or mass.
