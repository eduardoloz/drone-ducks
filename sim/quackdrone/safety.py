"""Safety logic that sits between your code and PX4. Pure functions and small state machines,
so all of it is unit-tested without a simulator.

Layers of protection, outermost first (see docs/ARCHITECTURE.md, "Failsafe matrix"):
  1. PX4 itself: geofence (GF_*), battery (BAT_*/COM_LOW_BAT_ACT), offboard-loss (COM_OF_LOSS_T,
     COM_OBL_RC_ACT), datalink loss (NAV_DLL_ACT), and the safety pilot's RC switch on real hardware.
  2. quackdrone (this module): clamps every command to Limits, keeps a soft fence and altitude band,
     stops the drone when velocity commands stop arriving (CommandWatchdog), and lands early on low
     battery (BatteryMonitor). It never fights PX4: once PX4 changes mode on its own we stop sending.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

from .geometry import all_finite, body_to_ned, clamp, limit_norm, ned_to_body


@dataclass(frozen=True)
class Limits:
    """Limits quackdrone enforces on every command (PX4's MPC_* limits still apply on top)."""

    max_speed_xy: float = 2.0      # m/s, horizontal velocity commands
    max_speed_z: float = 1.0       # m/s, vertical velocity commands
    max_yaw_rate: float = 60.0     # deg/s
    max_step_m: float = 5.0        # longest single relative move (move/climb)
    min_alt_m: float = 0.5         # metres above the takeoff point, while flying
    max_alt_m: float = 10.0
    fence_radius_m: float = 30.0   # horizontal soft fence around the takeoff point; 0 = off


@dataclass(frozen=True)
class VelocityCommand:
    """Same fields and units as MAVSDK's VelocityBodyYawspeed (see geometry.py for the frame)."""

    forward_m_s: float = 0.0
    right_m_s: float = 0.0
    down_m_s: float = 0.0
    yawspeed_deg_s: float = 0.0

    def is_zero(self) -> bool:
        return not (self.forward_m_s or self.right_m_s or self.down_m_s or self.yawspeed_deg_s)


ZERO_VELOCITY = VelocityCommand()


def clamp_velocity_body(cmd: VelocityCommand, limits: Limits, *, yaw_deg: float,
                        pos_ned: tuple[float, float, float], home_ne: Optional[tuple[float, float]],
                        ground_d: Optional[float]) -> tuple[VelocityCommand, list[str]]:
    """Return (safe command, notes). Notes say which limits changed the command.

    Raises ValueError for NaN/inf input: a controller that outputs NaN is broken, and silently
    turning NaN into 0 would hide that.
    """
    f, r, d, w = cmd.forward_m_s, cmd.right_m_s, cmd.down_m_s, cmd.yawspeed_deg_s
    if not all_finite(f, r, d, w):
        raise ValueError(f"non-finite velocity command {cmd}")
    notes: list[str] = []

    f2, r2 = limit_norm(f, r, limits.max_speed_xy)
    if (f2, r2) != (f, r):
        notes.append("max_speed_xy")
    f, r = f2, r2
    if abs(d) > limits.max_speed_z:
        d = math.copysign(limits.max_speed_z, d)
        notes.append("max_speed_z")
    if abs(w) > limits.max_yaw_rate:
        w = math.copysign(limits.max_yaw_rate, w)
        notes.append("max_yaw_rate")

    if ground_d is not None:
        alt = ground_d - pos_ned[2]
        if alt >= limits.max_alt_m and d < 0:
            d = 0.0
            notes.append("max_alt")
        elif alt <= limits.min_alt_m and d > 0:
            d = 0.0
            notes.append("min_alt")

    if limits.fence_radius_m > 0 and home_ne is not None:
        dn, de = pos_ned[0] - home_ne[0], pos_ned[1] - home_ne[1]
        dist = math.hypot(dn, de)
        if dist >= limits.fence_radius_m:
            vn, ve = body_to_ned(f, r, yaw_deg)
            un, ue = dn / dist, de / dist
            outward = vn * un + ve * ue
            if outward > 0:
                vn, ve = vn - outward * un, ve - outward * ue
                f, r = ned_to_body(vn, ve, yaw_deg)
                notes.append("fence")

    return VelocityCommand(f, r, d, w), notes


def clamp_position_target(n: float, e: float, d: float, limits: Limits, *,
                          home_ne: Optional[tuple[float, float]],
                          ground_d: Optional[float]) -> tuple[tuple[float, float, float], list[str]]:
    """Clamp an absolute NED target into the altitude band and the soft fence."""
    if not all_finite(n, e, d):
        raise ValueError(f"non-finite position target {(n, e, d)}")
    notes: list[str] = []
    if ground_d is not None:
        alt = ground_d - d
        alt2 = clamp(alt, limits.min_alt_m, limits.max_alt_m)
        if alt2 != alt:
            notes.append("alt_band")
            d = ground_d - alt2
    if limits.fence_radius_m > 0 and home_ne is not None:
        dn, de = n - home_ne[0], e - home_ne[1]
        dist = math.hypot(dn, de)
        if dist > limits.fence_radius_m:
            k = limits.fence_radius_m / dist
            n, e = home_ne[0] + dn * k, home_ne[1] + de * k
            notes.append("fence")
    return (n, e, d), notes


class CommandWatchdog:
    """Dead-man switch for velocity control.

    Why it is needed: MAVSDK re-sends the *last* offboard setpoint at 20 Hz on its own thread, so
    PX4's own offboard-loss failsafe only fires if the whole process dies. If your perception loop
    hangs or crashes inside a still-running process, a velocity setpoint would be flown forever.
    This watchdog turns "no new command for timeout_s" into: brake (zero velocity), then once the
    drone has slowed down, hold position.

    States: IDLE (not in velocity control) -> ACTIVE (fresh commands) -> BRAKING -> HOLDING.
    """

    IDLE, ACTIVE, BRAKING, HOLDING = "idle", "active", "braking", "holding"

    def __init__(self, timeout_s: float = 0.5, brake_max_s: float = 2.0, settle_speed: float = 0.2):
        self.timeout_s = timeout_s
        self.brake_max_s = brake_max_s
        self.settle_speed = settle_speed
        self.state = self.IDLE
        self.last_cmd = 0.0
        self.brake_start = 0.0

    def command(self, now: float) -> None:
        """A fresh velocity command was sent."""
        self.state = self.ACTIVE
        self.last_cmd = now

    def release(self) -> None:
        """Velocity control ended for another reason (position command, landing, failsafe)."""
        self.state = self.IDLE

    def brake(self, now: float) -> None:
        """Caller already sent zero velocity (Drone.stop()): go straight to BRAKING."""
        self.state = self.BRAKING
        self.brake_start = now

    def update(self, now: float, speed: float) -> Optional[str]:
        """Call periodically. Returns "brake" or "hold" when the caller must act, else None."""
        if self.state == self.ACTIVE and now - self.last_cmd > self.timeout_s:
            self.state = self.BRAKING
            self.brake_start = now
            return "brake"
        if self.state == self.BRAKING and (speed < self.settle_speed
                                           or now - self.brake_start > self.brake_max_s):
            self.state = self.HOLDING
            return "hold"
        return None


class BatteryMonitor:
    """Edge-triggered battery thresholds with hysteresis. update() returns "warn" once when the
    charge first drops below warn_pct, "act" once below act_pct, else None. Unknown charge
    (None/NaN, e.g. no battery telemetry yet) never triggers anything."""

    def __init__(self, warn_pct: float = 30.0, act_pct: float = 20.0, hysteresis_pct: float = 3.0):
        if act_pct > warn_pct:
            raise ValueError("act_pct must be <= warn_pct")
        self.warn_pct, self.act_pct, self.hyst = warn_pct, act_pct, hysteresis_pct
        self.warned = False
        self.acted = False

    def update(self, pct: Optional[float]) -> Optional[str]:
        if pct is None or not math.isfinite(pct):
            return None
        if pct > self.warn_pct + self.hyst:
            self.warned = self.acted = False  # recharged / battery swapped
        if not self.acted and pct <= self.act_pct:
            self.acted = self.warned = True
            return "act"
        if not self.warned and pct <= self.warn_pct:
            self.warned = True
            return "warn"
        return None


def scaled(limits: Limits, **changes) -> Limits:
    """Copy of `limits` with some fields changed: scaled(Limits(), max_speed_xy=0.5)."""
    return replace(limits, **changes)
