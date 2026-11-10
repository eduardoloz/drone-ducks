"""Drone: the one class most code needs.

    from quackdrone import Drone

    with Drone("udpin://0.0.0.0:14540") as drone:   # PX4 SITL. Real drone: "serial:///dev/ttyTHS1:921600"
        drone.takeoff(2.0)                            # arm + Offboard + climb, blocks until there
        drone.move(forward=1.0)                       # position control, blocks until reached
        drone.turn(90)                                # + = clockwise (right)
        for _ in range(50):                           # velocity control, e.g. from a vision loop
            drone.set_velocity_body(forward_m_s=0.5, yawspeed_deg_s=10)
            time.sleep(0.1)
    # leaving the block lands (also on an exception or Ctrl-C)

What it does for you (details in docs/ARCHITECTURE.md):
  * PX4 owns all control loops; we only send position/velocity/yaw setpoints in Offboard mode.
  * Waits for PX4's estimator to be ready (`health.is_armable`) before arming.
  * Streams a setpoint before switching to Offboard (PX4 rule).
  * Clamps every command to `Limits` (speed, altitude band, soft fence).
  * Velocity watchdog: no new velocity command for `cmd_timeout_s` -> brake, then hold position.
  * Never fights PX4: if PX4 leaves Offboard on its own (failsafe, geofence, pilot takeover) we stop
    commanding and motion calls raise ControlLost until you call resume_control() or land().
  * Lands early when the battery drops below `battery_land_pct` (PX4's own battery failsafe stays
    as the backstop).

Threads: your thread calls Drone methods; MAVSDK delivers telemetry on its own thread into a
TelemetryCache; one "quackdrone-supervisor" thread (20 Hz) runs the watchdog and failsafe checks.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, fields
from typing import Callable, Optional

from .backends.base import Backend
from .errors import ControlLost, NotReady, QuackDroneError
from .geometry import body_to_ned, wrap_deg
from .safety import (BatteryMonitor, CommandWatchdog, Limits, VelocityCommand, clamp_position_target,
                     clamp_velocity_body)
from .telemetry import Telemetry

DEFAULT_CONNECTION = "udpin://0.0.0.0:14540"


@dataclass
class DroneConfig:
    """Timing and policy knobs. Units: metres, seconds, degrees, percent."""

    takeoff_alt_m: float = 1.5
    pos_tolerance_m: float = 0.15      # "reached" = within this distance ...
    yaw_tolerance_deg: float = 4.0     # ... and this heading error ...
    settle_speed_m_s: float = 0.15     # ... and slower than this ...
    settle_time_s: float = 0.3         # ... for this long
    move_timeout_s: float = 25.0       # position commands give up (return False) after this
    takeoff_timeout_s: float = 30.0
    ready_timeout_s: float = 90.0      # estimator convergence after PX4 boot takes ~10-30 s
    cmd_timeout_s: float = 0.5         # velocity watchdog (dead-man switch)
    link_timeout_s: float = 3.0        # no telemetry for this long = link lost
    offboard_engage_timeout_s: float = 3.0
    battery_warn_pct: float = 30.0
    battery_land_pct: float = 20.0
    low_battery_action: str = "land"   # "land" or "warn"
    land_on_exit: bool = True          # `with Drone(...)` lands when the block ends
    supervisor_hz: float = 20.0


@dataclass(frozen=True)
class Event:
    """Something the supervisor or a command noticed. kind is one of:
    takeoff, land, limit, timeout, cmd_timeout, hold, px4_override, link_lost, link_ok,
    low_battery, px4 (a PX4 status text), exception, supervisor_error."""

    t: float
    kind: str
    message: str
    data: dict = field(default_factory=dict)


def make_backend(connection: str) -> Backend:
    """Pick a backend from the connection URL. "fake://" (or "fake://fast") needs no PX4."""
    if connection.startswith("fake://"):
        from .backends.fake import FakeBackend
        return FakeBackend.fast() if connection.rstrip("/").endswith("fast") else FakeBackend()
    from .backends.mavsdk_native import MavsdkBackend
    return MavsdkBackend(connection)


def _pick(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (d or {}).items() if k in names})


class Drone:
    def __init__(self, connection: str = DEFAULT_CONNECTION, *, limits: Optional[Limits] = None,
                 config: Optional[DroneConfig] = None, backend: Optional[Backend] = None,
                 log: Optional[Callable[[str], None]] = print):
        self.connection = connection
        self.limits = limits or Limits()
        self.config = config or DroneConfig()
        self.backend = backend or make_backend(connection)
        self.cache = self.backend.cache
        self._log = log
        self._send_lock = threading.RLock()
        self._in_control = False
        self._seen_offboard = False
        self._control_since = 0.0
        self._lost_reason: Optional[str] = None
        self.override_mode: Optional[str] = None   # mode PX4 switched to on its own, if any
        self._ground_d: Optional[float] = None      # local-NED down of the takeoff point
        self._home_ne: Optional[tuple[float, float]] = None
        self._target: Optional[list[float]] = None  # last position setpoint [n, e, d, yaw]
        self.last_velocity: VelocityCommand = VelocityCommand()
        self._watchdog = CommandWatchdog(self.config.cmd_timeout_s, settle_speed=0.2)
        self._battery = BatteryMonitor(self.config.battery_warn_pct, self.config.battery_land_pct)
        self._events: deque[Event] = deque(maxlen=500)
        self._listeners: list[Callable[[Event], None]] = []
        self._status_seq = 0
        self._link_lost = False
        self._last_limit_event = 0.0
        self._stop_evt = threading.Event()
        self._supervisor: Optional[threading.Thread] = None
        self._connected = False

    @classmethod
    def from_dict(cls, d: dict, **kw) -> "Drone":
        """Build from a flat dict (e.g. TypeFly's robot_info "extra"): picks `connection` plus any
        Limits / DroneConfig field names it recognises and ignores the rest."""
        return cls(d.get("connection", DEFAULT_CONNECTION), limits=_pick(Limits, d),
                   config=_pick(DroneConfig, d), **kw)

    # ================================================================== lifecycle
    def connect(self, timeout_s: float = 30.0) -> "Drone":
        if self._connected:
            return self
        self.backend.connect(timeout_s)
        self._connected = True
        self._stop_evt.clear()
        self._supervisor = threading.Thread(target=self._supervise, name="quackdrone-supervisor",
                                            daemon=True)
        self._supervisor.start()
        deadline = time.monotonic() + 5.0
        while self.cache.age("north_m") == math.inf and time.monotonic() < deadline:
            time.sleep(0.05)  # first position telemetry
        self._say(f"connected via {self.backend.describe()}")
        return self

    def wait_ready(self, timeout_s: Optional[float] = None) -> None:
        """Block until PX4 has local + global + home position and reports armable."""
        timeout_s = self.config.ready_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout_s
        nudged = False
        while time.monotonic() < deadline:
            t = self.telemetry
            if t.ready_to_arm:
                return
            # A previous session that ended in OFFBOARD leaves PX4 there; nothing is armable until
            # setpoints flow again. Switching to HOLD resets that and is harmless otherwise.
            if (not nudged and not t.armed and t.flight_mode == "OFFBOARD"
                    and t.local_position_ok and t.global_position_ok):
                nudged = True
                try:
                    self.backend.hold()
                except QuackDroneError as e:
                    self._say(f"switch to HOLD failed: {e}")
            time.sleep(0.2)
        t = self.telemetry
        raise NotReady(f"PX4 not ready to arm after {timeout_s:.0f} s (local={t.local_position_ok} "
                       f"global={t.global_position_ok} home={t.home_position_ok} armable={t.armable}); "
                       f"last PX4 message: {t.status_text!r}")

    def close(self, land: Optional[bool] = None) -> None:
        """Land if we are flying (unless land=False), then release the link.

        "Flying" means armed + in the air + PX4 still in OFFBOARD, even if we had already given up
        control (e.g. after a link hiccup): closing the link would otherwise trigger PX4's
        offboard-loss failsafe (Return mode, which climbs to RTL_RETURN_ALT). If PX4 is in any
        other mode, a failsafe or the pilot owns the drone and we leave it alone."""
        land = self.config.land_on_exit if land is None else land
        try:
            t = self.telemetry
            if land and self._connected and t.armed and t.in_air and (self._in_control or t.flight_mode == "OFFBOARD"):
                self.land(wait=True)
        finally:
            self._stop_evt.set()
            if self._supervisor:
                self._supervisor.join(timeout=2.0)
            if self._connected:
                self.backend.close()
            self._connected = False

    def __enter__(self) -> "Drone":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._event("exception", f"{exc_type.__name__}: {exc} -> landing")
        self.close()

    # ================================================================== state
    @property
    def telemetry(self) -> Telemetry:
        return self.cache.snapshot()

    @property
    def in_control(self) -> bool:
        """True while we own Offboard control (after takeoff/resume_control, before land/failsafe)."""
        return self._in_control

    @property
    def altitude_m(self) -> float:
        """Metres above the takeoff point (NaN before the first takeoff)."""
        if self._ground_d is None:
            return math.nan
        return self._ground_d - self.telemetry.down_m

    @property
    def home_ned(self) -> Optional[tuple[float, float, float]]:
        """Takeoff point in local NED, or None before the first takeoff."""
        if self._ground_d is None or self._home_ne is None:
            return None
        return self._home_ne[0], self._home_ne[1], self._ground_d

    def events(self, kind: Optional[str] = None) -> list[Event]:
        return [e for e in self._events if kind is None or e.kind == kind]

    def on_event(self, callback: Callable[[Event], None]) -> None:
        """callback(event) runs on the thread that noticed it (often the supervisor): keep it short."""
        self._listeners.append(callback)

    # ================================================================== flight
    def takeoff(self, altitude_m: Optional[float] = None, timeout_s: Optional[float] = None) -> bool:
        """Arm, enter Offboard and climb to altitude_m above the current spot. Returns True once
        there (False on timeout). Raises ArmingDenied / OffboardRejected / NotReady."""
        self._require_connected()
        alt = self.config.takeoff_alt_m if altitude_m is None else float(altitude_m)
        alt = min(max(alt, self.limits.min_alt_m), self.limits.max_alt_m)
        t = self.telemetry
        if t.armed and t.in_air:
            raise QuackDroneError("already flying: call resume_control() to take over instead")
        self.wait_ready()
        t = self.telemetry
        n, e, d, yaw = t.north_m, t.east_m, t.down_m, t.yaw_deg
        with self._send_lock:
            self._ground_d, self._home_ne = d, (n, e)
            self.backend.arm()
            try:
                self.backend.set_position_ned(n, e, d, yaw)  # PX4 rule: stream before switching
                self.backend.start_offboard()
            except Exception:
                try:
                    self.backend.disarm()
                except Exception:
                    pass
                raise
            self._take_control()
            self._target = [n, e, d, yaw]
        self._event("takeoff", f"armed, Offboard on, climbing to {alt:.1f} m")
        return self.goto_ned(n, e, d - alt, yaw, timeout_s=timeout_s or self.config.takeoff_timeout_s)

    def resume_control(self) -> None:
        """Take (back) control of a drone that is already flying, holding where it is. Use after a
        PX4 failsafe you have dealt with, or to adopt a drone a crashed script left in the air."""
        self._require_connected()
        t = self.telemetry
        if not (t.armed and t.in_air):
            raise QuackDroneError("resume_control() needs a flying drone; use takeoff()")
        with self._send_lock:
            self.backend.set_position_ned(t.north_m, t.east_m, t.down_m, t.yaw_deg)
            self.backend.start_offboard()
            self._take_control()
            self._target = [t.north_m, t.east_m, t.down_m, t.yaw_deg]
            if self._ground_d is None:  # adopting a drone: assume it is takeoff_alt above ground
                self._ground_d = t.down_m + self.config.takeoff_alt_m
                self._home_ne = (t.north_m, t.east_m)
        self._event("hold", f"resumed control in Offboard at {self.altitude_m:.1f} m")

    def land(self, wait: bool = True, timeout_s: float = 60.0, reason: str = "landing") -> bool:
        """Switch PX4 to LAND (works whether or not we are in control). Returns True once on the
        ground and disarmed (or immediately if wait=False)."""
        self._require_connected()
        t = self.telemetry
        if not t.armed and not t.in_air:
            self._release_control(reason)
            return True
        with self._send_lock:
            was_in_control = self._in_control
            self._release_control(reason)
            if was_in_control:
                try:
                    self.backend.stop_offboard()  # stop MAVSDK re-sending our last setpoint
                except QuackDroneError as e:
                    self._say(f"offboard stop before landing: {e}")
            self.backend.land()
        self._event("land", "PX4 LAND mode")
        if not wait:
            return True
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            t = self.telemetry
            if not t.in_air and not t.armed:
                return True
            time.sleep(0.2)
        return not self.telemetry.in_air

    def return_to_launch(self) -> None:
        """Hand over to PX4's Return mode (climbs to RTL_RETURN_ALT first: 60 m by default!)."""
        self._require_connected()
        with self._send_lock:
            was_in_control = self._in_control
            self._release_control("return to launch")
            if was_in_control:
                try:
                    self.backend.stop_offboard()
                except QuackDroneError:
                    pass
            self.backend.return_to_launch()
        self._event("land", "PX4 RETURN mode")

    # ------------------------------------------------------------------ position control
    def goto_ned(self, north_m: float, east_m: float, down_m: float, yaw_deg: Optional[float] = None,
                 *, wait: bool = True, timeout_s: Optional[float] = None) -> bool:
        """Fly to an absolute local-NED point (down is negative up!). Returns True when reached,
        False on timeout. The target is clamped into the altitude band and soft fence."""
        self._require_control()
        if yaw_deg is None:
            yaw_deg = self._target[3] if self._target else self.telemetry.yaw_deg
        (n, e, d), notes = clamp_position_target(north_m, east_m, down_m, self.limits,
                                                 home_ne=self._home_ne, ground_d=self._ground_d)
        if notes:
            self._limit_event(notes, f"target clamped ({', '.join(notes)})")
        tgt = [n, e, d, wrap_deg(float(yaw_deg))]
        with self._send_lock:
            self._require_control()
            self._watchdog.release()
            self._target = tgt
            self.backend.set_position_ned(*tgt)
        if not wait:
            return True
        return self._wait_reached(tgt, self.config.move_timeout_s if timeout_s is None else timeout_s)

    def move(self, forward: float = 0.0, right: float = 0.0, up: float = 0.0, *, wait: bool = True) -> bool:
        """Relative move in metres along the current heading (forward/right) and vertically (up).
        Each component is limited to Limits.max_step_m. Starts from the last *target*, not the
        measured position, so repeated moves do not accumulate drift."""
        s = self.limits.max_step_m
        forward, right, up = (max(-s, min(s, float(v))) for v in (forward, right, up))
        n, e, d, yaw = self._base()
        dn, de = body_to_ned(forward, right, yaw)
        return self.goto_ned(n + dn, e + de, d - up, yaw, wait=wait)

    def turn(self, degrees: float, *, wait: bool = True) -> bool:
        """Rotate in place. Positive = clockwise seen from above (turn right), like a compass."""
        n, e, d, yaw = self._base()
        return self.goto_ned(n, e, d, wrap_deg(yaw + float(degrees)), wait=wait)

    def turn_left(self, degrees: float, **kw) -> bool:
        return self.turn(-float(degrees), **kw)

    def turn_right(self, degrees: float, **kw) -> bool:
        return self.turn(float(degrees), **kw)

    def yaw_to(self, heading_deg: float, *, wait: bool = True) -> bool:
        n, e, d, _ = self._base()
        return self.goto_ned(n, e, d, heading_deg, wait=wait)

    def set_altitude(self, altitude_m: float, *, wait: bool = True) -> bool:
        """Climb/descend to altitude_m above the takeoff point."""
        n, e, _, yaw = self._base()
        return self.goto_ned(n, e, self._ground_d - float(altitude_m), yaw, wait=wait)

    def hold(self) -> None:
        """Hold position where the drone is now (stays in Offboard)."""
        self._require_control()
        t = self.telemetry
        with self._send_lock:
            self._watchdog.release()
            self._target = [t.north_m, t.east_m, t.down_m, t.yaw_deg]
            self.backend.set_position_ned(*self._target)

    # ------------------------------------------------------------------ velocity control
    def set_velocity_body(self, forward_m_s: float = 0.0, right_m_s: float = 0.0, down_m_s: float = 0.0,
                          yawspeed_deg_s: float = 0.0) -> VelocityCommand:
        """Velocity setpoint in the yaw-aligned body frame (MAVSDK VelocityBodyYawspeed units).
        Non-blocking. Send it at >= 5 Hz (10-30 Hz recommended): if no new call arrives within
        config.cmd_timeout_s (0.5 s) the watchdog brakes and then holds position.
        Returns the command actually sent after clamping. Raises ValueError on NaN/inf."""
        self._require_control()
        t = self.telemetry
        cmd, notes = clamp_velocity_body(VelocityCommand(forward_m_s, right_m_s, down_m_s, yawspeed_deg_s),
                                         self.limits, yaw_deg=t.yaw_deg, pos_ned=t.position_ned,
                                         home_ne=self._home_ne, ground_d=self._ground_d)
        if notes:
            self._limit_event(notes, f"velocity clamped ({', '.join(notes)})")
        with self._send_lock:
            self._require_control()
            self.backend.set_velocity_body(cmd.forward_m_s, cmd.right_m_s, cmd.down_m_s, cmd.yawspeed_deg_s)
            self._target = None
            self.last_velocity = cmd
            self._watchdog.command(time.monotonic())
        return cmd

    def stop(self, wait: bool = True, timeout_s: float = 3.0) -> None:
        """Brake now (zero velocity), then hold position once slow. Safe to call any time."""
        if not self._in_control:
            return
        with self._send_lock:
            self.backend.set_velocity_body(0.0, 0.0, 0.0, 0.0)
            self._target = None
            self._watchdog.brake(time.monotonic())
        if wait:
            deadline = time.monotonic() + timeout_s
            while self._watchdog.state == CommandWatchdog.BRAKING and time.monotonic() < deadline:
                time.sleep(0.05)

    # ------------------------------------------------------------------ parameters
    def get_param(self, name: str):
        self._require_connected()
        return self.backend.get_param(name)

    def set_param(self, name: str, value) -> None:
        """Set a PX4 parameter. NOTE: PX4 saves it, so it survives SITL restarts (scripts/start_sim.sh
        uses its own rootfs under sim/.run; FRESH=1 wipes it)."""
        self._require_connected()
        self.backend.set_param(name, value)

    # ================================================================== internals
    def _say(self, msg: str) -> None:
        if self._log:
            self._log(f"[quack {time.strftime('%H:%M:%S')}] {msg}")

    def _event(self, kind: str, message: str, **data) -> None:
        ev = Event(time.time(), kind, message, data)
        self._events.append(ev)
        self._say(f"{kind}: {message}")
        for cb in list(self._listeners):
            try:
                cb(ev)
            except Exception as e:  # a listener must never break the supervisor
                self._say(f"event listener failed: {e!r}")

    def _limit_event(self, notes: list[str], message: str) -> None:
        now = time.monotonic()
        if now - self._last_limit_event > 2.0:  # at most every 2 s, velocity loops call at 10-30 Hz
            self._last_limit_event = now
            self._event("limit", message, notes=notes)

    def _require_connected(self) -> None:
        if not self._connected:
            raise QuackDroneError("not connected: call connect() or use `with Drone(...)`")

    def _require_control(self) -> None:
        self._require_connected()
        if not self._in_control:
            reason = self._lost_reason or "no control yet"
            raise ControlLost(f"not in Offboard control ({reason}); takeoff() or resume_control() first")

    def _take_control(self) -> None:
        self._in_control = True
        self._seen_offboard = False
        self._control_since = time.monotonic()
        self._lost_reason = None
        self.override_mode = None
        self._watchdog.release()

    def _release_control(self, reason: str, make_safe: bool = False) -> None:
        """Stop commanding. make_safe=True first replaces the setpoint MAVSDK keeps re-sending with
        zero velocity: if the link comes back while PX4 is still in OFFBOARD, it hovers instead of
        flying our last velocity command with no watchdog watching."""
        if make_safe and self._in_control:
            try:
                with self._send_lock:
                    self.backend.set_velocity_body(0.0, 0.0, 0.0, 0.0)
            except Exception:
                pass  # link may be down: best effort
        self._in_control = False
        self._lost_reason = reason
        self._watchdog.release()
        self._target = None

    def _base(self) -> list[float]:
        if self._target is not None:
            return list(self._target)
        t = self.telemetry
        return [t.north_m, t.east_m, t.down_m, t.yaw_deg]

    def _wait_reached(self, tgt: list[float], timeout_s: float) -> bool:
        c = self.config
        deadline = time.monotonic() + timeout_s
        settled_since = None
        pos_err = yaw_err = math.nan
        while time.monotonic() < deadline:
            if not self._in_control:
                raise ControlLost(f"control lost while moving: {self._lost_reason}")
            if self._target is not tgt:
                return False  # superseded by a newer command from another thread
            t = self.telemetry
            pos_err = math.dist(t.position_ned, tgt[:3])
            yaw_err = abs(wrap_deg(t.yaw_deg - tgt[3]))
            if pos_err < c.pos_tolerance_m and yaw_err < c.yaw_tolerance_deg and t.speed < c.settle_speed_m_s:
                settled_since = settled_since or time.monotonic()
                if time.monotonic() - settled_since >= c.settle_time_s:
                    return True
            else:
                settled_since = None
            time.sleep(0.05)
        self._event("timeout", f"target not reached in {timeout_s:.0f} s (pos err {pos_err:.2f} m, "
                               f"yaw err {yaw_err:.1f} deg)")
        return False

    # ------------------------------------------------------------------ supervisor thread
    def _supervise(self) -> None:
        period = 1.0 / self.config.supervisor_hz
        while not self._stop_evt.wait(period):
            try:
                self._supervise_once(time.monotonic())
            except Exception as e:  # never let the safety thread die
                self._event("supervisor_error", repr(e))

    def _supervise_once(self, now: float) -> None:
        t = self.cache.snapshot()
        c = self.config

        texts, self._status_seq = self.cache.statuses_after(self._status_seq)
        for text in texts:
            self._event("px4", text)

        # link: no telemetry for a while
        stale = self.cache.age("north_m") > c.link_timeout_s or not t.connected
        if stale and not self._link_lost:
            self._link_lost = True
            if self._in_control:
                self._release_control("link lost", make_safe=True)
            self._event("link_lost", f"no telemetry for {c.link_timeout_s:.0f} s; PX4's own "
                                     "offboard/datalink-loss failsafes take over")
        elif not stale and self._link_lost:
            self._link_lost = False
            self._event("link_ok", "telemetry back")

        # PX4 left Offboard by itself -> stop commanding (never fight a failsafe or the pilot)
        if self._in_control:
            if t.flight_mode == "OFFBOARD":
                self._seen_offboard = True
            elif self._seen_offboard or now - self._control_since > c.offboard_engage_timeout_s:
                mode = t.flight_mode
                # PX4 ignores offboard setpoints outside OFFBOARD; zeroing them only matters if the
                # pilot later flips back to Offboard: then the drone hovers instead of resuming.
                self._release_control(f"PX4 switched to {mode}", make_safe=True)
                self.override_mode = mode
                self._event("px4_override", f"PX4 switched to {mode} on its own (failsafe or pilot); "
                                            "stopped sending commands", mode=mode, px4=t.status_text)

        # velocity watchdog
        if self._in_control:
            action = self._watchdog.update(now, t.speed)
            if action == "brake":
                with self._send_lock:
                    if self._in_control:
                        self.backend.set_velocity_body(0.0, 0.0, 0.0, 0.0)
                self._event("cmd_timeout", f"no velocity command for {c.cmd_timeout_s} s: braking")
            elif action == "hold":
                with self._send_lock:
                    if self._in_control:
                        self._target = [t.north_m, t.east_m, t.down_m, t.yaw_deg]
                        self.backend.set_position_ned(*self._target)
                self._event("hold", "stopped: holding position")

        # battery
        b = self._battery.update(t.battery_pct)
        if b == "warn":
            self._event("low_battery", f"battery {t.battery_pct:.0f}% (warn below {c.battery_warn_pct:.0f}%)")
        elif b == "act":
            if c.low_battery_action == "land" and t.in_air and self._in_control:
                self._event("low_battery", f"battery {t.battery_pct:.0f}%: landing now")
                threading.Thread(target=self._failsafe_land, args=("low battery",), daemon=True).start()
            else:
                self._event("low_battery", f"battery {t.battery_pct:.0f}% (below {c.battery_land_pct:.0f}%)")

    def _failsafe_land(self, reason: str) -> None:
        try:
            self.land(wait=False, reason=f"landing: {reason}")
        except Exception as e:
            self._event("supervisor_error", f"failsafe land failed: {e!r}")
