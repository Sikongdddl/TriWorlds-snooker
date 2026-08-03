# mujoco-billiards assets

The numbered ball textures in `img/` were copied from
[`hideboz/mujoco-billiards`](https://github.com/hideboz/mujoco-billiards) at
commit `6d7632fb18cf7ee86207bf35ef9e5c346d945f8e`.

`models/mujoco_billiards/billiard-table-definitions.xml` is a byte-identical
copy of the source table. The active scenes use its materials, checker floor,
four lights, custom SDF pockets and cushions, and native `+Y` long-axis frame.
The ball model keeps the source radius (28.5 mm), mass (165 g), contact
parameters, and numbered textures. Install the source-built SDF plugin with
`python scripts/assets/build_mujoco_billiards_sdf_plugin.py` before loading a
scene.

The imported work is provided under the MIT License; see `LICENSE`.
