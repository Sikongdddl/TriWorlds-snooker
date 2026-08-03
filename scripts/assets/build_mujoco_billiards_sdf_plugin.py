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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile source SDF shapes against the active MuJoCo Python package."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
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
    target = plugin_dir / "libsdf_plugin.so"
    backup = plugin_dir / f"libsdf_plugin.so.stock-{mujoco_version}"
    plugin_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mujoco-billiards-sdf-") as temp_dir:
        output = Path(temp_dir) / "libsdf_plugin.so"
        command = [
            "g++",
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-shared",
            *(str(path) for path in sources),
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
        if target.exists() and not backup.exists():
            shutil.copy2(target, backup)
            print(f"Backed up stock plugin to {backup}")
        shutil.copy2(output, target)
        target.chmod(0o755)

    print(f"Installed custom SDF plugin to {target}")
    print("Start a fresh Python process before loading the billiards XML.")


if __name__ == "__main__":
    main()
