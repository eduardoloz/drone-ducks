"""Face-follow controller: one target box in, one body-frame velocity command out.

This module is *pure*: no camera, no MAVSDK, no clock. You pass the observation and the current
time in, you get a command out, so every behaviour can be unit-tested (tests/test_controller.py).

What it does each step
----------------------
1. Decide whether we have a usable target: a fresh (not stale), finite, in-frame observation.
2. If yes (mode TRACKING), turn the face position/size into errors:
     bearing   = how far left/right of centre the face is, in degrees     -> yaw rate
     distance  = estimated from face width (geometry.distance_from_width) -> forward speed
     elevation = how far below the desired image row the face is, deg    -> down speed (optional)
   Each error goes through a soft deadband, a proportional gain and a saturation limit.
   Forward speed is scaled down when the face is near the image edge, so we turn first and
   approach second (charging forward at an off-centre face is how you lose it).
3. If no target: command a hover (all zeros). Mode IDLE (never had one), LOST (had one, waiting
   for it to come back), and after `lost_timeout_s` (or `acquire_timeout_s` from IDLE) TIMEOUT,
   which carries an `action` ("land", "rtl" or "hold") for the bridge to execute. TIMEOUT latches
   until reset(): we never resume chasing a face after deciding to land.
4. Rate-limit (slew) the command so it changes smoothly. Losing the target bypasses the slew
   limiter and goes straight to hover; PX4 decelerates within its own acceleration limits.

Sign conventions (MAVSDK VelocityBodyYawspeed, PX4 body frame FRD):
  forward_m_s  + = fly forward          right_m_s + = fly right
  down_m_s     + = descend (NED!)       yawspeed_deg_s + = turn clockwise seen from above (right)
So: face right of centre -> positive yaw rate; face far away -> positive forward;
face below the desired row -> positive down (descend), only if vertical control is enabled.

PX4 owns the real control loops (attitude, rate, velocity). These are just setpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from . import geometry

IDLE = "IDLE"
TRACKING = "TRACKING"
LOST = "LOST"
TIMEOUT = "TIMEOUT"
LOST_ACTIONS = ("land", "rtl", "hold")


@dataclass(frozen=True)
class TargetObservation:
    """The followed face, normalised to the image: (0, 0) top-left, (1, 1) bottom-right.

    stamp is time.monotonic() when the *frame was captured* (not when detection finished), so
    the controller can refuse to act on stale frames.
    """

    cx: float
    cy: float
    w: float
    h: float
    stamp: float
    # Which way the face points (jetson.headpose, + = toward the camera's right). None = unknown
    # (body-only steering, no landmarks, face too small). Only used if k_head_lateral > 0.
    head_yaw_deg: Optional[float] = None

    @classmethod
    def from_box(cls, box: Tuple[float, float, float, float], frame_w: int, frame_h: int,
                 stamp: float, head_yaw_deg: Optional[float] = None) -> "TargetObservation":
        x1, y1, x2, y2 = box
        return cls(cx=(x1 + x2) / 2.0 / frame_w, cy=(y1 + y2) / 2.0 / frame_h,
                   w=(x2 - x1) / frame_w, h=(y2 - y1) / frame_h, stamp=stamp, head_yaw_deg=head_yaw_deg)


@dataclass(frozen=True)
class VelocityCommand:
    """Body-frame velocity setpoint, same fields as MAVSDK's VelocityBodyYawspeed."""

    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    down_m_s: float = 0.0
    yawspeed_deg_s: float = 0.0

    def is_zero(self, eps: float = 1e-9) -> bool:
        return all(abs(v) <= eps for v in self.as_tuple())

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.forward_m_s, self.right_m_s, self.down_m_s, self.yawspeed_deg_s)

    def __str__(self) -> str:
        return (f"fwd {self.forward_m_s:+.2f} m/s  right {self.right_m_s:+.2f} m/s  "
                f"down {self.down_m_s:+.2f} m/s  yaw {self.yawspeed_deg_s:+.1f} deg/s")


HOVER = VelocityCommand()


@dataclass(frozen=True)
class ControlOutput:
    command: VelocityCommand
    mode: str
    reason: str
    action: Optional[str] = None          # set only in TIMEOUT: "land" | "rtl" | "hold"
    bearing_deg: Optional[float] = None
    elevation_deg: Optional[float] = None
    distance_m: Optional[float] = None
    lost_for_s: Optional[float] = None
    head_yaw_deg: Optional[float] = None  # smoothed head yaw the lateral term used (None = unknown)


@dataclass
class FollowConfig:
    # Camera model. hfov must match the real lens (see geometry.CAMERAS); vfov is derived if None.
    hfov_deg: float = 62.2
    aspect: float = 4.0 / 3.0
    vfov_deg: Optional[float] = None
    face_width_m: float = geometry.FACE_WIDTH_M  # calibrate: stand at a known distance, adjust

    # Where we want the face: distance in metres, image row (0 top .. 1 bottom).
    target_distance_m: float = 4.0
    min_distance_m: float = 2.5      # closer than this: never fly forward
    target_cy: float = 0.5

    # Proportional gains.
    k_yaw: float = 1.2               # deg/s of yaw rate per deg of bearing error
    k_forward: float = 0.5           # m/s per m of distance error
    k_vertical: float = 0.03         # m/s per deg of elevation error
    k_lateral: float = 0.0           # m/s per deg of bearing error (0 = centre with yaw only)
    # Stay in front of the person: m/s sideways per deg of (smoothed) head yaw. 0 = off (default).
    # Face points to the camera's right -> fly right, while yaw keeps them centred (an orbit).
    # Shares max_lateral_m_s with k_lateral. Sideways flight is blind: keep the limit small.
    k_head_lateral: float = 0.0
    head_yaw_deadband_deg: float = 8.0
    head_yaw_tau_s: float = 0.5      # smoothing time constant of the head-yaw estimate
    head_yaw_hold_s: float = 0.5     # unknown yaw this long -> estimate decays to 0 (no stale orbits)

    # Soft deadbands (error inside the band -> 0, outside -> error minus band; no jump at the edge).
    yaw_deadband_deg: float = 3.0
    distance_deadband_m: float = 0.5
    vertical_deadband_deg: float = 4.0

    # Saturation limits. Keep them small for first flights.
    max_yaw_rate_deg_s: float = 30.0
    max_forward_m_s: float = 1.0
    max_backward_m_s: float = 0.5    # backing up is blind (no rear camera): keep it slow
    max_vertical_m_s: float = 0.5
    max_lateral_m_s: float = 0.5

    # Slew limits (how fast each command may change).
    max_accel_m_s2: float = 1.0
    max_yaw_accel_deg_s2: float = 60.0
    max_dt_s: float = 0.25           # a stalled loop must not unlock a huge jump

    # Vertical control is OFF by default: hold altitude, never descend toward a person.
    enable_vertical: bool = False
    min_altitude_m: float = 2.0
    max_altitude_m: float = 6.0

    # Target loss.
    max_obs_age_s: float = 0.5       # observations older than this count as "no target"
    lost_timeout_s: float = 5.0      # hover this long waiting for the face to come back...
    acquire_timeout_s: float = 30.0  # ...or this long if we never saw one at all
    lost_action: str = "land"        # then: "land" | "rtl" | "hold"

    def __post_init__(self) -> None:
        if not 0 < self.hfov_deg < 180:
            raise ValueError("hfov_deg must be in (0, 180)")
        if self.lost_action not in LOST_ACTIONS:
            raise ValueError(f"lost_action must be one of {LOST_ACTIONS}")
        if self.min_distance_m > self.target_distance_m:
            raise ValueError("min_distance_m must not exceed target_distance_m")
        for name in ("max_yaw_rate_deg_s", "max_forward_m_s", "max_backward_m_s", "max_vertical_m_s",
                     "max_lateral_m_s", "max_accel_m_s2", "max_yaw_accel_deg_s2", "max_obs_age_s",
                     "lost_timeout_s", "acquire_timeout_s", "face_width_m", "max_dt_s", "k_head_lateral",
                     "head_yaw_deadband_deg", "head_yaw_tau_s", "head_yaw_hold_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.min_altitude_m >= self.max_altitude_m:
            raise ValueError("min_altitude_m must be below max_altitude_m")

    @property
    def vfov(self) -> float:
        return self.vfov_deg if self.vfov_deg is not None else geometry.vfov_from_hfov(self.hfov_deg,
                                                                                      self.aspect)


def soft_deadband(error: float, band: float) -> float:
    """0 inside +-band, error minus band outside (continuous, so no jump at the edge)."""
    if abs(error) <= band:
        return 0.0
    return error - math.copysign(band, error)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def slew(previous: float, target: float, max_rate: float, dt: float) -> float:
    """Move from previous toward target by at most max_rate * dt."""
    step = max_rate * dt
    return previous + clamp(target - previous, -step, step)


class FaceFollowController:
    def __init__(self, config: Optional[FollowConfig] = None) -> None:
        self.config = config or FollowConfig()
        self.reset()

    def reset(self) -> None:
        self._last_now: Optional[float] = None
        self._first_now: Optional[float] = None
        self._last_seen: Optional[float] = None
        self._timed_out = False
        self._timeout_reason = ""
        self._cmd = HOVER
        self._head_yaw: Optional[float] = None    # smoothed head yaw, deg
        self._head_yaw_at: Optional[float] = None  # stamp of the last raw head-yaw sample

    def _update_head_yaw(self, raw: Optional[float], stamp: float, dt: float) -> Optional[float]:
        c = self.config
        if raw is not None and math.isfinite(raw):
            if self._head_yaw is None or c.head_yaw_tau_s <= 0:
                self._head_yaw = raw
            elif dt > 0:
                a = dt / (c.head_yaw_tau_s + dt)
                self._head_yaw += a * (raw - self._head_yaw)
            self._head_yaw_at = stamp
        elif self._head_yaw is not None and (self._head_yaw_at is None
                                             or stamp - self._head_yaw_at > c.head_yaw_hold_s):
            self._head_yaw = None
        return self._head_yaw

    @property
    def last_command(self) -> VelocityCommand:
        return self._cmd

    # ------------------------------------------------------------------ helpers
    def _check(self, obs: Optional[TargetObservation], now: float) -> Tuple[bool, str]:
        if obs is None:
            return False, "no target"
        values = (obs.cx, obs.cy, obs.w, obs.h, obs.stamp)
        if not all(math.isfinite(v) for v in values):
            return False, "invalid observation (NaN/inf)"
        if not (0.0 < obs.w <= 1.0 and 0.0 < obs.h <= 1.0):
            return False, "invalid box size"
        if not (0.0 <= obs.cx <= 1.0 and 0.0 <= obs.cy <= 1.0):
            return False, "box centre outside the image"
        age = now - obs.stamp
        if age < -0.05:
            return False, "observation from the future (clock mismatch?)"
        if age > self.config.max_obs_age_s:
            return False, f"stale observation ({age:.2f} s old)"
        return True, ""

    def _dt(self, now: float) -> float:
        if self._last_now is None:
            return 0.0
        return clamp(now - self._last_now, 0.0, self.config.max_dt_s)

    # ------------------------------------------------------------------ main entry point
    def step(self, obs: Optional[TargetObservation], now: float,
             altitude_m: Optional[float] = None) -> ControlOutput:
        """Compute the command for this instant.

        obs:        the followed face (or None when nothing is tracked)
        now:        current time on the same clock as obs.stamp (time.monotonic())
        altitude_m: height above home if known; only used when enable_vertical is True
        """
        c = self.config
        if self._first_now is None:
            self._first_now = now
        dt = self._dt(now)
        self._last_now = now if self._last_now is None else max(now, self._last_now)

        if self._timed_out:
            self._cmd = HOVER
            return ControlOutput(HOVER, TIMEOUT, self._timeout_reason, action=c.lost_action,
                                 lost_for_s=self._lost_for(now))

        ok, why = self._check(obs, now)
        if not ok:
            return self._no_target(now, why)

        self._last_seen = obs.stamp
        bearing = geometry.bearing_deg(obs.cx, c.hfov_deg)
        elevation = geometry.bearing_deg(obs.cy, c.vfov) - geometry.bearing_deg(c.target_cy, c.vfov)
        distance = geometry.distance_from_width(obs.w, c.hfov_deg, c.face_width_m)
        notes = []

        yaw = clamp(c.k_yaw * soft_deadband(bearing, c.yaw_deadband_deg),
                    -c.max_yaw_rate_deg_s, c.max_yaw_rate_deg_s)
        head = self._update_head_yaw(obs.head_yaw_deg, obs.stamp, dt)
        right = c.k_lateral * soft_deadband(bearing, c.yaw_deadband_deg)
        if c.k_head_lateral > 0 and head is not None:
            right += c.k_head_lateral * soft_deadband(head, c.head_yaw_deadband_deg)
        right = clamp(right, -c.max_lateral_m_s, c.max_lateral_m_s)

        forward = c.k_forward * soft_deadband(distance - c.target_distance_m, c.distance_deadband_m)
        if forward > 0:
            # Turn first, approach second: full speed only when the face is near the centre.
            forward *= max(0.0, 1.0 - abs(bearing) / (c.hfov_deg / 2.0))
        if distance < c.min_distance_m and forward > 0:
            forward = 0.0
        if distance < c.min_distance_m:
            notes.append("too close")
        forward = clamp(forward, -c.max_backward_m_s, c.max_forward_m_s)

        down = 0.0
        if c.enable_vertical:
            if altitude_m is None or not math.isfinite(altitude_m):
                notes.append("vertical off: no altitude")
            else:
                down = clamp(c.k_vertical * soft_deadband(elevation, c.vertical_deadband_deg),
                             -c.max_vertical_m_s, c.max_vertical_m_s)
                if down > 0 and altitude_m <= c.min_altitude_m:
                    down = 0.0
                    notes.append("at min altitude")
                elif down < 0 and altitude_m >= c.max_altitude_m:
                    down = 0.0
                    notes.append("at max altitude")

        target = VelocityCommand(forward, right, down, yaw)
        prev = self._cmd
        self._cmd = VelocityCommand(
            slew(prev.forward_m_s, target.forward_m_s, c.max_accel_m_s2, dt),
            slew(prev.right_m_s, target.right_m_s, c.max_accel_m_s2, dt),
            slew(prev.down_m_s, target.down_m_s, c.max_accel_m_s2, dt),
            slew(prev.yawspeed_deg_s, target.yawspeed_deg_s, c.max_yaw_accel_deg_s2, dt),
        )
        reason = f"bearing {bearing:+.1f} deg, dist {distance:.1f} m"
        if notes:
            reason += " (" + ", ".join(notes) + ")"
        if c.k_head_lateral > 0 and head is not None:
            reason += f", head {head:+.0f} deg"
        return ControlOutput(self._cmd, TRACKING, reason, bearing_deg=bearing, elevation_deg=elevation,
                             distance_m=distance, lost_for_s=0.0, head_yaw_deg=head)

    def _lost_for(self, now: float) -> Optional[float]:
        return None if self._last_seen is None else max(0.0, now - self._last_seen)

    def _no_target(self, now: float, why: str) -> ControlOutput:
        c = self.config
        self._cmd = HOVER  # hover immediately; PX4 limits the deceleration
        self._head_yaw = None
        if self._last_seen is None:
            waited = now - self._first_now
            if waited >= c.acquire_timeout_s:
                return self._timeout(now, f"no face found in {waited:.0f} s")
            return ControlOutput(HOVER, IDLE, f"hover: waiting for a face ({why})")
        lost_for = now - self._last_seen
        if lost_for >= c.lost_timeout_s:
            return self._timeout(now, f"target lost for {lost_for:.1f} s")
        return ControlOutput(HOVER, LOST, f"hover: target lost {lost_for:.1f} s ({why})",
                             lost_for_s=lost_for)

    def _timeout(self, now: float, why: str) -> ControlOutput:
        self._timed_out = True
        self._timeout_reason = f"{why} -> {self.config.lost_action}"
        return ControlOutput(HOVER, TIMEOUT, self._timeout_reason, action=self.config.lost_action,
                             lost_for_s=self._lost_for(now))
