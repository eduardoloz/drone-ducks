"""A kinematic toy world for closed-loop tests and the `--source toy` demo. NOT a flight simulator.

Modelled: a drone that follows body-frame velocity commands with a first-order lag (roughly how
PX4's velocity loop looks from outside), a level forward-looking pinhole camera, and one person
walking on flat ground. Ignored: attitude, wind, camera tilt while accelerating, motion blur.
PX4 SITL + Gazebo (the sim/ track) is the real simulator; this exists so a controller change can
be checked in milliseconds, in CI, without PX4.

    world = ToyWorld(person_n=9.0, person_e=3.0)
    history = run_closed_loop(world, FaceFollowController(), seconds=20)
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .controller import FaceFollowController, TargetObservation, VelocityCommand
from .detectors.base import FaceDetector
from .face import Box, Face


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class ToyWorld:
    hfov_deg: float = 62.2
    width: int = 640
    height: int = 480
    face_width_m: float = 0.16
    face_height_m: float = 1.6     # face centre above the ground
    tau_s: float = 0.35            # velocity response time constant (PX4-ish)
    min_face_px: float = 8.0       # smaller faces count as "not detected"
    cam_pitch_deg: float = 0.0     # camera tilted down by this much (0 = level)
    # drone state (north/east metres, altitude metres, yaw degrees NED: 0 = north, 90 = east)
    north: float = 0.0
    east: float = 0.0
    altitude: float = 3.0
    yaw_deg: float = 0.0
    v_fwd: float = 0.0
    v_right: float = 0.0
    v_down: float = 0.0
    yaw_rate: float = 0.0
    # person
    person_n: float = 8.0
    person_e: float = 0.0
    person_vn: float = 0.0
    person_ve: float = 0.0
    visible: bool = True
    t: float = 0.0

    @property
    def focal_px(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    def step(self, cmd: VelocityCommand, dt: float) -> None:
        a = dt / (self.tau_s + dt) if self.tau_s > 0 else 1.0
        self.v_fwd += (cmd.forward_m_s - self.v_fwd) * a
        self.v_right += (cmd.right_m_s - self.v_right) * a
        self.v_down += (cmd.down_m_s - self.v_down) * a
        self.yaw_rate += (cmd.yawspeed_deg_s - self.yaw_rate) * a
        psi = math.radians(self.yaw_deg)
        self.north += (self.v_fwd * math.cos(psi) - self.v_right * math.sin(psi)) * dt
        self.east += (self.v_fwd * math.sin(psi) + self.v_right * math.cos(psi)) * dt
        self.altitude -= self.v_down * dt
        self.yaw_deg = wrap_deg(self.yaw_deg + self.yaw_rate * dt)
        self.person_n += self.person_vn * dt
        self.person_e += self.person_ve * dt
        self.t += dt

    def face_in_body(self) -> Tuple[float, float, float]:
        """Face position relative to the camera: (forward, right, down) metres."""
        dn, de = self.person_n - self.north, self.person_e - self.east
        psi = math.radians(self.yaw_deg)
        return (dn * math.cos(psi) + de * math.sin(psi), -dn * math.sin(psi) + de * math.cos(psi),
                self.altitude - self.face_height_m)

    @property
    def distance_m(self) -> float:
        return math.hypot(self.person_n - self.north, self.person_e - self.east)

    @property
    def bearing_deg(self) -> float:
        fwd, right, _ = self.face_in_body()
        return math.degrees(math.atan2(right, fwd))

    @property
    def speed_m_s(self) -> float:
        return math.sqrt(self.v_fwd ** 2 + self.v_right ** 2 + self.v_down ** 2)

    def face_in_camera(self) -> Tuple[float, float, float]:
        """Like face_in_body but in the (down-tilted) camera frame."""
        fwd, right, down = self.face_in_body()
        th = math.radians(self.cam_pitch_deg)
        return fwd * math.cos(th) + down * math.sin(th), right, -fwd * math.sin(th) + down * math.cos(th)

    def face_box(self) -> Optional[Box]:
        if not self.visible:
            return None
        fwd, right, down = self.face_in_camera()
        if fwd <= 0.2:
            return None
        f = self.focal_px
        u, v = self.width / 2.0 + f * right / fwd, self.height / 2.0 + f * down / fwd
        w = f * self.face_width_m / fwd
        h = 1.25 * w
        if w < self.min_face_px or not (0 <= u <= self.width and 0 <= v <= self.height):
            return None
        return (u - w / 2, v - h / 2, u + w / 2, v + h / 2)

    def observe(self, stamp: Optional[float] = None) -> Optional[TargetObservation]:
        box = self.face_box()
        if box is None:
            return None
        return TargetObservation.from_box(box, self.width, self.height, self.t if stamp is None else stamp)

    def render(self, sprite: Optional[np.ndarray] = None, sprite_margin: float = 0.6) -> np.ndarray:
        """BGR frame: sky, ground, and the face (a real face crop if given, else a cartoon)."""
        import cv2

        H, W = self.height, self.width
        img = np.empty((H, W, 3), np.uint8)
        horizon = int(np.clip(H / 2 - self.focal_px * math.tan(math.radians(self.cam_pitch_deg)), 1, H - 1))
        img[:horizon] = np.linspace(235, 170, horizon, dtype=np.uint8)[:, None, None] * np.array([1, 0.9, 0.75])
        img[horizon:] = (70, 110, 80)
        box = self.face_box()
        if box is None:
            return img
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # a simple torso under the face so the scene looks like a person
        cv2.rectangle(img, (int(cx - bw), int(y2 + 0.2 * bh)), (int(cx + bw), int(y2 + 3.5 * bh)), (90, 60, 160), -1)
        if sprite is not None:
            sw, sh = bw * (1 + 2 * sprite_margin), bh * (1 + 2 * sprite_margin)
            sx1, sy1 = int(round(cx - sw / 2)), int(round(cy - sh / 2))
            tile = cv2.resize(sprite, (max(2, int(round(sw))), max(2, int(round(sh)))), interpolation=cv2.INTER_AREA)
            ax1, ay1 = max(0, sx1), max(0, sy1)
            ax2, ay2 = min(W, sx1 + tile.shape[1]), min(H, sy1 + tile.shape[0])
            if ax2 > ax1 and ay2 > ay1:
                img[ay1:ay2, ax1:ax2] = tile[ay1 - sy1:ay2 - sy1, ax1 - sx1:ax2 - sx1]
        else:
            cv2.ellipse(img, (int(cx), int(cy)), (int(bw / 2), int(bh / 2)), 0, 0, 360, (150, 180, 220), -1)
            for ex in (-0.2, 0.2):
                cv2.circle(img, (int(cx + ex * bw), int(cy - 0.1 * bh)), max(1, int(0.06 * bw)), (40, 40, 40), -1)
            cv2.ellipse(img, (int(cx), int(cy + 0.22 * bh)), (int(0.18 * bw), max(1, int(0.05 * bh))), 0, 0, 180,
                        (60, 60, 160), max(1, int(0.03 * bw)))
        return img


def make_sprite(image: np.ndarray, face: Face, margin: float = 0.6) -> np.ndarray:
    """Crop a face with `margin` x its size of context on every side (for ToyWorld.render)."""
    h, w = image.shape[:2]
    mx, my = margin * face.w, margin * face.h
    x1, y1 = int(max(0, face.x1 - mx)), int(max(0, face.y1 - my))
    x2, y2 = int(min(w, face.x2 + mx)), int(min(h, face.y2 + my))
    return image[y1:y2, x1:x2].copy()


class OracleDetector(FaceDetector):
    """Returns the toy world's true face box (+ optional pixel noise and random dropouts)."""

    name = "oracle"

    def __init__(self, world: ToyWorld, noise_px: float = 0.0, dropout: float = 0.0, seed: int = 0) -> None:
        self.world, self.noise_px, self.dropout = world, noise_px, dropout
        self.rng = random.Random(seed)

    def detect(self, frame: np.ndarray) -> List[Face]:
        box = self.world.face_box()
        if box is None or self.rng.random() < self.dropout:
            return []
        n = [self.rng.gauss(0.0, self.noise_px) for _ in range(4)] if self.noise_px else [0.0] * 4
        return [Face(box[0] + n[0], box[1] + n[1], box[2] + n[2], box[3] + n[3], 0.9)]

    def info(self):
        return {"backend": self.name, "model": "ground truth", "device": "none"}


def run_closed_loop(world: ToyWorld, controller: FaceFollowController, seconds: float, dt: float = 1 / 15,
                    latency_s: float = 0.0,
                    on_step: Optional[Callable[[ToyWorld, object], None]] = None) -> List[dict]:
    """Close the loop observation -> controller -> world without images. Returns per-step history.

    latency_s delays each observation (capture-to-command delay of a real perception pipeline);
    the observation keeps its capture stamp, as the real pipeline does.
    """
    pending: deque = deque()
    history = []
    steps = int(round(seconds / dt))
    for _ in range(steps):
        pending.append((world.t + latency_s, world.observe()))
        obs = None
        while pending and pending[0][0] <= world.t + 1e-9:
            obs = pending.popleft()[1]
        out = controller.step(obs, world.t)
        if on_step:
            on_step(world, out)
        history.append({"t": world.t, "mode": out.mode, "action": out.action, "distance": world.distance_m,
                        "bearing": world.bearing_deg, "speed": world.speed_m_s, "seen": obs is not None,
                        "cmd": out.command})
        world.step(out.command, dt)
    return history
