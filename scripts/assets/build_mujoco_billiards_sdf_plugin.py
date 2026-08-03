"""Build and install mujoco-billiards' SDF plugin into the active environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_SOURCE = Path("/home/ubuntu/mujoco-billiards")

EXTRA_SDF_SOURCES = (
    "chopped_cylinder.cc",
    "cone.cc",
    "half_hollow_cylinder.cc",
    "hollow_cylinder.cc",
    "sdf.cc",
    "trapezoid.cc",
    "vertical_capped_cylinder.cc",
)

EXTRA_REGISTER_SOURCE = r'''#include <mujoco/mjplugin.h>
#include "chopped_cylinder.h"
#include "cone.h"
#include "half_hollow_cylinder.h"
#include "hollow_cylinder.h"
#include "trapezoid.h"
#include "vertical_capped_cylinder.h"

namespace mujoco::plugin::sdf {
mjPLUGIN_LIB_INIT(sdf_extra) {
  ChoppedCylinder::RegisterPlugin();
  Cone::RegisterPlugin();
  HalfHollowCylinder::RegisterPlugin();
  HollowCylinder::RegisterPlugin();
  Trapezoid::RegisterPlugin();
  VerticalCappedCylinder::RegisterPlugin();
}
}
'''


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile source SDF shapes against the active MuJoCo Python package."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Install into a project-local directory instead of site-packages.",
    )
    args = parser.parse_args()

    source_dir = args.source.resolve()
    sdf_dir = source_dir / "sdf"
    sources = sorted(sdf_dir.glob("*.cc"))
    if not sources:
        raise FileNotFoundError(f"No SDF C++ sources found under {sdf_dir}")

    package_spec = importlib.util.find_spec("mujoco")
    if package_spec is None or package_spec.origin is None:
        raise RuntimeError("MuJoCo is not installed in the active Python environment.")
    package_dir = Path(package_spec.origin).resolve().parent
    mujoco_version = importlib.metadata.version("mujoco")
    include_dir = package_dir / "include"
    plugin_dir = package_dir / "plugin"
    library_candidates = sorted(package_dir.glob("libmujoco.so.*"))
    if not include_dir.is_dir() or not library_candidates:
        raise RuntimeError(f"Active MuJoCo package is missing build files: {package_dir}")

    library = library_candidates[-1]
    target_dir = (
        args.target_dir.resolve() if args.target_dir is not None else plugin_dir
    )
    target = target_dir / "libsdf_plugin.so"
    backup = target_dir / f"libsdf_plugin.so.stock-{mujoco_version}"
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mujoco-billiards-sdf-") as temp_dir:
        temp_path = Path(temp_dir)
        output = temp_path / "libsdf_plugin.so"
        build_sources = sources
        if args.target_dir is not None:
            # The Python wheel already loads MuJoCo's stock SDF library. A
            # second registration of bolt/bowl/gear/nut/torus is fatal, so a
            # project-local library contains only the six additional shapes
            # required by mujoco-billiards.
            register_extra = temp_path / "register_extra.cc"
            register_extra.write_text(EXTRA_REGISTER_SOURCE, encoding="utf-8")
            build_sources = [sdf_dir / name for name in EXTRA_SDF_SOURCES]
            build_sources.append(register_extra)
        command = [
            "g++",
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-shared",
            *(str(path) for path in build_sources),
            f"-I{sdf_dir}",
            f"-I{include_dir}",
            f"-L{package_dir}",
            "-Wl,-rpath,$ORIGIN/..",
            "-Wl,--no-undefined",
            f"-l:{library.name}",
            "-o",
            str(output),
        ]
        print("Building:", " ".join(command))
        subprocess.run(command, check=True)
        if args.target_dir is None and target.exists() and not backup.exists():
            shutil.copy2(target, backup)
            print(f"Backed up stock plugin to {backup}")
        shutil.copy2(output, target)
        target.chmod(0o755)

    print(f"Installed custom SDF plugin to {target}")
    print("Start a fresh Python process before loading the billiards XML.")


if __name__ == "__main__":
    main()
