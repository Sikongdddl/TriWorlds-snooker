"""Contact and pocket events shared by MuJoCo snooker environments."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ContactEvent:
    """One newly-begun physical contact or pocket-entry event."""

    time: float
    kind: str
    primary_name: str
    secondary_name: str
    position: np.ndarray
    normal: np.ndarray
    normal_force: float


class CollisionEventMonitor:
    """Classify contact begins and detect balls that fall into pocket regions."""

    def __init__(self, model: mujoco.MjModel, *, pocket_z: float = 0.745) -> None:
        self.model = model
        self.pocket_z = float(pocket_z)
        self.geom_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or f"geom_{index}"
            for index in range(model.ngeom)
        )
        self.ball_geom_ids = {
            index
            for index, name in enumerate(self.geom_names)
            if name == "cue_ball_geom" or name.startswith("object_ball_") and name.endswith("_geom")
        }
        self.ball_bodies = tuple(
            (name, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
            for name in ("cue_ball",) + tuple(f"object_ball_{index}" for index in range(16))
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
        )
        self.pocket_sites = tuple(
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, index) or f"site_{index}", index)
            for index in range(model.nsite)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, index) or "").startswith("pocket_")
        )
        self.reset()

    def reset(self) -> None:
        self.events: list[ContactEvent] = []
        self.pocketed_balls: set[str] = set()
        self._active_pairs: set[tuple[int, int]] = set()
        self._last_pair_time: dict[tuple[int, int], float] = {}

    def _kind(self, geom1: int, geom2: int) -> str:
        names = {self.geom_names[geom1], self.geom_names[geom2]}
        ball_count = int(geom1 in self.ball_geom_ids) + int(geom2 in self.ball_geom_ids)
        if "cue_tip" in names and ball_count:
            return "cue_tip_ball"
        if "cue_shaft" in names and ball_count:
            return "cue_shaft_ball"
        if ball_count == 2:
            return "ball_ball"
        if ball_count and any(name.startswith("cushion_") for name in names):
            return "ball_cushion"
        if ball_count and "playfield_collision" in names:
            return "ball_cloth"
        if ball_count and any(name.startswith("pocket_catch_") for name in names):
            return "ball_pocket_floor"
        return "other"

    def scan(self, data: mujoco.MjData) -> tuple[ContactEvent, ...]:
        """Record contacts that began this simulation step and new pocket entries."""

        new_events: list[ContactEvent] = []
        current_pairs: set[tuple[int, int]] = set()
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = tuple(sorted((int(contact.geom1), int(contact.geom2))))
            current_pairs.add(pair)
            if pair in self._active_pairs:
                continue
            last_time = self._last_pair_time.get(pair, -np.inf)
            self._last_pair_time[pair] = float(data.time)
            if data.time - last_time < 0.01:
                continue
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, data, index, force)
            event = ContactEvent(
                time=float(data.time),
                kind=self._kind(*pair),
                primary_name=self.geom_names[pair[0]],
                secondary_name=self.geom_names[pair[1]],
                position=np.asarray(contact.pos, dtype=np.float64).copy(),
                normal=np.asarray(contact.frame[:3], dtype=np.float64).copy(),
                normal_force=float(force[0]),
            )
            self.events.append(event)
            new_events.append(event)
        self._active_pairs = current_pairs

        for ball_name, body_id in self.ball_bodies:
            if ball_name in self.pocketed_balls:
                continue
            ball_position = data.xpos[body_id]
            if ball_position[2] >= self.pocket_z:
                continue
            for pocket_name, site_id in self.pocket_sites:
                radius = float(self.model.site_size[site_id, 0]) + 0.03
                if np.linalg.norm(ball_position[:2] - data.site_xpos[site_id, :2]) <= radius:
                    event = ContactEvent(
                        time=float(data.time),
                        kind="ball_pocket",
                        primary_name=ball_name,
                        secondary_name=pocket_name,
                        position=ball_position.copy(),
                        normal=np.zeros(3, dtype=np.float64),
                        normal_force=0.0,
                    )
                    self.pocketed_balls.add(ball_name)
                    self.events.append(event)
                    new_events.append(event)
                    break
        return tuple(new_events)
