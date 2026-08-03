#!/usr/bin/env python3
"""Check staged task-library retention and atomic validated publication."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()
TOOLS = ROOT / "scripts" / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_midlevel_two_ball_ppo_env_smoke import _fixture_task  # noqa: E402
from snooker_env.midlevel_tasks import (  # noqa: E402
    TaskValidationReport,
    TwoBallTaskDataset,
)
from snooker_env.midlevel_two_ball import TwoBallShotSimulator  # noqa: E402


def _load_generator_module():
    path = TOOLS / "generate_midlevel_tasks.py"
    spec = importlib.util.spec_from_file_location(
        "midlevel_task_generator_for_publish_smoke",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load task generator from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_generator_module()
    simulator = TwoBallShotSimulator()
    task = _fixture_task(simulator)
    generated = TwoBallTaskDataset.from_tasks(
        [task],
        simulator,
        generation_seed=7,
    )
    previous = TwoBallTaskDataset.from_tasks(
        [task],
        simulator,
        generation_seed=99,
    )

    module.generate_task_dataset = lambda *args, **kwargs: generated

    with tempfile.TemporaryDirectory(
        prefix="midlevel-task-publish-smoke-"
    ) as directory:
        output = Path(directory) / "tasks.npz"
        staged = Path(directory) / "tasks.unvalidated.npz"
        previous.save(output)
        previous_hash = TwoBallTaskDataset.load(
            output,
            simulator=simulator,
        ).content_sha256()

        module.validate_task_dataset = lambda *args, **kwargs: TaskValidationReport(
            checked_count=1,
            passed_count=0,
            max_stop_replay_error=0.1,
            failures=("deliberate replay failure",),
        )
        try:
            module._generate(
                "smoke",
                1,
                7,
                output,
                simulator,
                -1,
                1,
                "cpu",
                "cuda:0",
                1,
                1,
                1,
                128,
                1_024,
                8.0,
                1.5,
                2_000,
            )
        except RuntimeError as error:
            if "deliberate replay failure" not in str(error):
                raise
        else:
            raise RuntimeError("A failed replay unexpectedly published its library.")
        if not staged.is_file():
            raise RuntimeError("Failed replay did not retain the unvalidated library.")
        if (
            TwoBallTaskDataset.load(output, simulator=simulator).content_sha256()
            != previous_hash
        ):
            raise RuntimeError("Failed replay replaced the previous validated library.")

        module.validate_task_dataset = lambda *args, **kwargs: TaskValidationReport(
            checked_count=1,
            passed_count=1,
            max_stop_replay_error=0.0,
            failures=(),
        )
        module._generate(
            "smoke",
            1,
            7,
            output,
            simulator,
            -1,
            1,
            "cpu",
            "cuda:0",
            1,
            1,
            1,
            128,
            1_024,
            8.0,
            1.5,
            2_000,
        )
        published = TwoBallTaskDataset.load(output, simulator=simulator)
        if published.content_sha256() != generated.content_sha256():
            raise RuntimeError("Validated library was not published atomically.")
        if staged.exists():
            raise RuntimeError("Published library left a stale staging archive.")

    print("midlevel_task_staged_publish=PASS")


if __name__ == "__main__":
    main()
