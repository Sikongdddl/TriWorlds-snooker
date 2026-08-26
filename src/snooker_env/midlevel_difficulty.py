"""Distance-stratified difficulty definitions for two-ball shot tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


TASK_DIFFICULTY_VERSION = "two-distance-grid-v1"
DISTANCE_BAND_NAMES = ("short", "medium", "long")
CUE_OBJECT_DISTANCE_RANGES_M = (
    (0.30, 0.48),
    (0.48, 0.68),
    (0.68, 0.90),
)
OBJECT_POCKET_DISTANCE_RANGES_M = (
    (0.18, 0.34),
    (0.34, 0.52),
    (0.52, 0.70),
)
DIFFICULTY_LEVEL_NAMES = ("easy", "medium", "hard")
TASK_DIFFICULTY_CELL_COUNT = len(DISTANCE_BAND_NAMES) ** 2
_DISTANCE_TOLERANCE_M = 1.0e-9


@dataclass(frozen=True)
class TaskDifficultyCell:
    """One cell in the cue-object × object-pocket distance grid."""

    index: int
    cue_object_band: int
    object_pocket_band: int

    @property
    def name(self) -> str:
        return (
            f"cue_{DISTANCE_BAND_NAMES[self.cue_object_band]}__"
            f"object_{DISTANCE_BAND_NAMES[self.object_pocket_band]}"
        )

    @property
    def level(self) -> int:
        return max(self.cue_object_band, self.object_pocket_band)

    @property
    def level_name(self) -> str:
        return DIFFICULTY_LEVEL_NAMES[self.level]

    @property
    def cue_object_range_m(self) -> tuple[float, float]:
        return CUE_OBJECT_DISTANCE_RANGES_M[self.cue_object_band]

    @property
    def object_pocket_range_m(self) -> tuple[float, float]:
        return OBJECT_POCKET_DISTANCE_RANGES_M[self.object_pocket_band]


TASK_DIFFICULTY_CELLS = tuple(
    TaskDifficultyCell(
        index=(cue_band * len(DISTANCE_BAND_NAMES) + object_band),
        cue_object_band=cue_band,
        object_pocket_band=object_band,
    )
    for cue_band in range(len(DISTANCE_BAND_NAMES))
    for object_band in range(len(DISTANCE_BAND_NAMES))
)


def difficulty_cell(index: int) -> TaskDifficultyCell:
    """Return a validated difficulty-cell definition."""

    normalized = int(index)
    if not 0 <= normalized < TASK_DIFFICULTY_CELL_COUNT:
        raise IndexError(f"Unknown task difficulty cell: {index}")
    return TASK_DIFFICULTY_CELLS[normalized]


def task_distance_arrays(
    cue_positions: np.ndarray,
    object_positions: np.ndarray,
    pocket_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two physical distances that define task difficulty."""

    cue = np.asarray(cue_positions, dtype=np.float64)
    object_ball = np.asarray(object_positions, dtype=np.float64)
    pockets = np.asarray(pocket_positions, dtype=np.float64)
    if cue.shape != object_ball.shape or cue.shape != pockets.shape:
        raise ValueError(
            "Cue, object, and pocket position arrays must have identical shapes."
        )
    if cue.shape[-1:] != (2,):
        raise ValueError("Task difficulty positions must end in an XY dimension.")
    cue_object = np.linalg.norm(cue - object_ball, axis=-1)
    object_pocket = np.linalg.norm(object_ball - pockets, axis=-1)
    if not np.all(np.isfinite(cue_object)) or not np.all(
        np.isfinite(object_pocket)
    ):
        raise FloatingPointError("Task difficulty distances are non-finite.")
    return cue_object, object_pocket


def _distance_band_indices(
    distances: np.ndarray,
    ranges: Sequence[tuple[float, float]],
    *,
    label: str,
) -> np.ndarray:
    minimum = float(ranges[0][0])
    maximum = float(ranges[-1][1])
    if np.any(distances < minimum - _DISTANCE_TOLERANCE_M) or np.any(
        distances > maximum + _DISTANCE_TOLERANCE_M
    ):
        observed_minimum = float(np.min(distances))
        observed_maximum = float(np.max(distances))
        raise ValueError(
            f"{label} distance is outside the difficulty grid "
            f"[{minimum}, {maximum}] m: observed "
            f"[{observed_minimum}, {observed_maximum}] m."
        )
    boundaries = np.asarray(
        [distance_range[1] for distance_range in ranges[:-1]],
        dtype=np.float64,
    )
    return np.searchsorted(boundaries, distances, side="right").astype(np.int8)


def task_difficulty_indices(
    cue_positions: np.ndarray,
    object_positions: np.ndarray,
    pocket_positions: np.ndarray,
) -> np.ndarray:
    """Classify one or more tasks into the nine distance cells."""

    cue_object, object_pocket = task_distance_arrays(
        cue_positions,
        object_positions,
        pocket_positions,
    )
    cue_bands = _distance_band_indices(
        cue_object,
        CUE_OBJECT_DISTANCE_RANGES_M,
        label="Cue-object",
    )
    object_bands = _distance_band_indices(
        object_pocket,
        OBJECT_POCKET_DISTANCE_RANGES_M,
        label="Object-pocket",
    )
    return (
        cue_bands * len(DISTANCE_BAND_NAMES) + object_bands
    ).astype(np.int8)


def task_difficulty_index(
    cue_position: np.ndarray,
    object_position: np.ndarray,
    pocket_position: np.ndarray,
) -> int:
    """Classify one task into a distance cell."""

    indices = task_difficulty_indices(
        np.asarray(cue_position, dtype=np.float64)[None, :2],
        np.asarray(object_position, dtype=np.float64)[None, :2],
        np.asarray(pocket_position, dtype=np.float64)[None, :2],
    )
    return int(indices[0])


def difficulty_profile(
    cue_positions: np.ndarray,
    object_positions: np.ndarray,
    pocket_positions: np.ndarray,
) -> dict[str, object]:
    """Return JSON-ready distance statistics and cell counts."""

    cue_object, object_pocket = task_distance_arrays(
        cue_positions,
        object_positions,
        pocket_positions,
    )
    indices = task_difficulty_indices(
        cue_positions,
        object_positions,
        pocket_positions,
    )
    count = int(indices.size)
    cell_counts = np.bincount(
        indices.astype(np.int64),
        minlength=TASK_DIFFICULTY_CELL_COUNT,
    )
    level_counts = np.zeros(len(DIFFICULTY_LEVEL_NAMES), dtype=np.int64)
    for cell in TASK_DIFFICULTY_CELLS:
        level_counts[cell.level] += cell_counts[cell.index]

    def distance_summary(values: np.ndarray) -> dict[str, float]:
        percentiles = np.percentile(values, (0, 5, 25, 50, 75, 95, 100))
        return {
            name: float(value)
            for name, value in zip(
                ("min", "p05", "p25", "p50", "p75", "p95", "max"),
                percentiles,
                strict=True,
            )
        }

    return {
        "version": TASK_DIFFICULTY_VERSION,
        "task_count": count,
        "cue_object_ranges_m": [
            {
                "name": name,
                "minimum": distance_range[0],
                "maximum": distance_range[1],
            }
            for name, distance_range in zip(
                DISTANCE_BAND_NAMES,
                CUE_OBJECT_DISTANCE_RANGES_M,
                strict=True,
            )
        ],
        "object_pocket_ranges_m": [
            {
                "name": name,
                "minimum": distance_range[0],
                "maximum": distance_range[1],
            }
            for name, distance_range in zip(
                DISTANCE_BAND_NAMES,
                OBJECT_POCKET_DISTANCE_RANGES_M,
                strict=True,
            )
        ],
        "distance_statistics_m": {
            "cue_object": distance_summary(cue_object),
            "object_pocket": distance_summary(object_pocket),
        },
        "cells": {
            cell.name: {
                "index": cell.index,
                "difficulty": cell.level_name,
                "count": int(cell_counts[cell.index]),
                "fraction": float(cell_counts[cell.index] / count),
            }
            for cell in TASK_DIFFICULTY_CELLS
        },
        "difficulty_levels": {
            level_name: {
                "count": int(level_counts[level]),
                "fraction": float(level_counts[level] / count),
            }
            for level, level_name in enumerate(DIFFICULTY_LEVEL_NAMES)
        },
    }
