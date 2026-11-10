"""FakeBackend: a pure-Python stand-in for PX4 + MAVSDK.

A point mass with a first-order velocity response and PX4-like flight modes (OFFBOARD, HOLD,
LAND, RETURN_TO_LAUNCH). It reproduces the *rules* that matter for our code -- setpoint before
offboard start, arming needs a ready estimator, offboard-loss failsafe after the setpoint stream
stops, auto-disarm after landing -- but it is NOT a physics simulator. Use it for:

  * unit tests (fast, deterministic enough, no PX4 install): ``FakeBackend.fast()``
  * trying the Drone API on any laptop: ``Drone("fake://")``

Test hooks: inject_mode() (PX4 failsafe / pilot takeover), set_battery(), set_link_alive(),
deny_next_arm().
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from ..errors import ArmingDenied, CommandDenied, OffboardRejected
from ..geometry import body_to_ned, clamp, limit_norm, wrap_deg
from ..telemetry import TelemetryCache
from .base import Backend, ParamValue


@dataclass
class _State:
    n: float = 0.0
    e: float = 0.0
    d: float = 0.0
    vn: float = 0.0
    ve: float = 0.0
    vd: float = 0.0
    yaw: float = 0.0
    yaw_rate: float = 0.0
    armed: bool = False
    in_air: bool = False
    mode: str = "HOLD"
    battery: float = 100.0


class FakeBackend(Backend):
    def __init__(self, *, rate_hz: float = 50.0, max_speed_xy: float = 5.0, max_speed_z: float = 2.0,
                 max_yaw_rate: float = 90.0, tau_s: float = 0.3, pos_gain: float = 1.0,
                 yaw_gain: float = 2.0, land_speed: float = 0.7, ready_after_s: float = 0.5,
                 battery_drain_pct_s: float = 0.0, offboard_loss_s: float = 1.0,
                 offboard_loss_mode: str = "RETURN_TO_LAUNCH",
                 start_ned: tuple[float, float, float] = (0.0, 0.0, 0.0), start_yaw: float = 0.0,
                 cache: TelemetryCache | None = None):
        super().__init__(cache)
        self.rate_hz, self.tau_s = rate_hz, tau_s
        self.max_speed_xy, self.max_speed_z, self.max_yaw_rate = max_speed_xy, max_speed_z, max_yaw_rate
        self.pos_gain, self.yaw_gain, self.land_speed = pos_gain, yaw_gain, land_speed
        self.ready_after_s = ready_after_s
        self.battery_drain_pct_s = battery_drain_pct_s
        self.offboard_loss_s, self.offboard_loss_mode = offboard_loss_s, offboard_loss_mode
        self._s = _State(n=start_ned[0], e=start_ned[1], d=start_ned[2], yaw=start_yaw)
        self._ground_d = start_ned[2]
        self._home = (start_ned[0], start_ned[1])
        self._sp: Optional[tuple] = None          # ("pos", n, e, d, yaw) | ("vel", f, r, d, w)
        self._hold_pt = (self._s.n, self._s.e, self._s.d, self._s.yaw)
        self._streaming = False                    # MAVSDK-style automatic setpoint re-send active
        self._client_gone_at: Optional[float] = None
        self._link_alive = True
        self._deny_arm: Optional[str] = None
        self._params: dict[str, ParamValue] = {"COM_OF_LOSS_T": offboard_loss_s, "GF_MAX_HOR_DIST": 0.0,
                                               "GF_ACTION": 2, "SIM_BAT_MIN_PCT": 50.0}
        self._lock = threading.Lock()
        self._t0 = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.mode_log: list[str] = []              # every mode the fake entered, for assertions

    @classmethod
    def fast(cls, **kw) -> "FakeBackend":
        """Snappy dynamics for unit tests: a 1 m move settles in well under a second."""
        opts = dict(rate_hz=200.0, max_speed_xy=20.0, max_speed_z=10.0, max_yaw_rate=720.0,
                    tau_s=0.02, pos_gain=10.0, yaw_gain=10.0, land_speed=4.0, ready_after_s=0.05)
        opts.update(kw)
        return cls(**opts)

    def describe(self) -> str:
        return "FakeBackend (no PX4: kinematic stand-in)"

    # ------------------------------------------------------------------ link
    def connect(self, timeout_s: float = 5.0) -> None:
        if self._thread is None:
            self._t0 = time.monotonic()
            self._thread = threading.Thread(target=self._run, name="fake-px4", daemon=True)
            self._thread.start()
        self._client_gone_at = None
        self._link_alive = True
        self._publish()

    def close(self) -> None:
        """Our 'process' went away: the setpoint stream stops, the fake PX4 keeps flying."""
        with self._lock:
            self._streaming = False
            self._client_gone_at = time.monotonic()
        self.cache.update(connected=False)

    def shutdown(self) -> None:
        """Stop the simulation thread (tests call this in teardown)."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    # ------------------------------------------------------------------ test hooks
    def inject_mode(self, mode: str, status_text: str = "") -> None:
        """Pretend PX4 switched mode by itself (geofence, battery failsafe, RC takeover...)."""
        with self._lock:
            self._enter(mode)
        if status_text:
            self.cache.update(status_text=status_text)

    def set_battery(self, pct: float) -> None:
        with self._lock:
            self._s.battery = pct

    def set_link_alive(self, alive: bool) -> None:
        """False = telemetry stops arriving (radio out of range, cable pulled)."""
        self._link_alive = alive

    def deny_next_arm(self, reason: str = "Preflight Fail: fake denial") -> None:
        self._deny_arm = reason

    @property
    def state(self) -> _State:
        with self._lock:
            return _State(**vars(self._s))

    # ------------------------------------------------------------------ commands
    def _ready(self) -> bool:
        return time.monotonic() - self._t0 >= self.ready_after_s

    def arm(self) -> None:
        with self._lock:
            if self._deny_arm:
                reason, self._deny_arm = self._deny_arm, None
                self.cache.update(status_text=reason)
                raise ArmingDenied("arm", "COMMAND_DENIED", reason)
            if not self._ready():
                raise ArmingDenied("arm", "COMMAND_DENIED", "Preflight Fail: height estimate not stable")
            self._s.armed = True
            self._ground_d = self._s.d

    def disarm(self) -> None:
        with self._lock:
            if self._s.in_air:
                raise CommandDenied("disarm", "COMMAND_DENIED", "Disarming denied: not landed")
            self._s.armed = False

    def set_position_ned(self, north_m, east_m, down_m, yaw_deg) -> None:
        with self._lock:
            self._sp = ("pos", float(north_m), float(east_m), float(down_m), float(yaw_deg))

    def set_velocity_body(self, forward_m_s, right_m_s, down_m_s, yawspeed_deg_s) -> None:
        with self._lock:
            self._sp = ("vel", float(forward_m_s), float(right_m_s), float(down_m_s), float(yawspeed_deg_s))

    def start_offboard(self) -> None:
        with self._lock:
            if self._sp is None:
                raise OffboardRejected("offboard.start", "NO_SETPOINT_SET")
            self._streaming = True
            self._client_gone_at = None
            self._enter("OFFBOARD")

    def stop_offboard(self) -> None:
        with self._lock:
            self._streaming = False
            self._enter("HOLD")

    def hold(self) -> None:
        with self._lock:
            self._enter("HOLD")

    def land(self) -> None:
        with self._lock:
            if self._s.armed:
                self._enter("LAND")

    def return_to_launch(self) -> None:
        with self._lock:
            if self._s.armed:
                self._enter("RETURN_TO_LAUNCH")

    def get_param(self, name: str) -> ParamValue:
        with self._lock:
            if name not in self._params:
                raise CommandDenied("get_param", "PARAM_NOT_FOUND", name)
            return self._params[name]

    def set_param(self, name: str, value: ParamValue) -> None:
        with self._lock:
            self._params[name] = value

    # ------------------------------------------------------------------ simulation
    def _enter(self, mode: str) -> None:  # caller holds the lock
        s = self._s
        if mode in ("HOLD", "LAND", "RETURN_TO_LAUNCH") or mode != s.mode:
            self._hold_pt = (s.n, s.e, s.d, s.yaw)
        s.mode = mode
        self.mode_log.append(mode)

    def _run(self) -> None:
        last = time.monotonic()
        period = 1.0 / self.rate_hz
        while not self._stop.wait(period):
            now = time.monotonic()
            dt = min(0.1, now - last)
            last = now
            with self._lock:
                self._step(dt, now)
            if self._link_alive:
                self._publish()

    def _pos_ctrl(self, n, e, d, yaw) -> tuple[float, float, float, float]:
        s = self._s
        vn, ve = limit_norm(self.pos_gain * (n - s.n), self.pos_gain * (e - s.e), self.max_speed_xy)
        vd = clamp(self.pos_gain * (d - s.d), -self.max_speed_z, self.max_speed_z)
        w = clamp(self.yaw_gain * wrap_deg(yaw - s.yaw), -self.max_yaw_rate, self.max_yaw_rate)
        return vn, ve, vd, w

    def _step(self, dt: float, now: float) -> None:
        s = self._s
        # PX4 offboard-loss failsafe: the setpoint stream stopped while in OFFBOARD
        if (s.mode == "OFFBOARD" and self._client_gone_at is not None
                and now - self._client_gone_at > self.offboard_loss_s):
            self._enter(self.offboard_loss_mode)
            self.cache.update(status_text="Failsafe activated: offboard control lost")
        if not s.armed:
            s.vn = s.ve = s.vd = s.yaw_rate = 0.0
            s.in_air = False
            return
        if s.mode == "OFFBOARD" and self._sp is not None:
            if self._sp[0] == "pos":
                cmd = self._pos_ctrl(*self._sp[1:])
            else:
                _, f, r, d, w = self._sp
                vn, ve = body_to_ned(f, r, s.yaw)
                cmd = (vn, ve, d, w)
        elif s.mode == "LAND":
            cmd = (0.0, 0.0, self.land_speed, 0.0)
        elif s.mode == "RETURN_TO_LAUNCH":
            hn, he = self._home
            if math.hypot(hn - s.n, he - s.e) > 0.3:
                cmd = self._pos_ctrl(hn, he, min(s.d, self._ground_d - 2.0), s.yaw)
            else:
                self._enter("LAND")
                cmd = (0.0, 0.0, self.land_speed, 0.0)
        else:  # HOLD / anything else: hold the point where the mode was entered
            cmd = self._pos_ctrl(*self._hold_pt)
        a = min(1.0, dt / self.tau_s)
        s.vn += (cmd[0] - s.vn) * a
        s.ve += (cmd[1] - s.ve) * a
        s.vd += (cmd[2] - s.vd) * a
        s.yaw_rate = cmd[3]
        s.n += s.vn * dt
        s.e += s.ve * dt
        s.d += s.vd * dt
        s.yaw = wrap_deg(s.yaw + s.yaw_rate * dt)
        if s.d >= self._ground_d:           # ground contact
            s.d = self._ground_d
            s.vd = min(0.0, s.vd)
        alt = self._ground_d - s.d
        if not s.in_air and alt > 0.15:
            s.in_air = True
        elif s.in_air and alt < 0.03:
            s.in_air = False
            s.vn = s.ve = 0.0
            if s.mode in ("LAND", "RETURN_TO_LAUNCH"):
                s.armed = False             # PX4 auto-disarms after landing
        if self.battery_drain_pct_s:
            s.battery = max(0.0, s.battery - self.battery_drain_pct_s * dt)

    def _publish(self) -> None:
        with self._lock:
            s = self._s
            alt = self._ground_d - s.d
            landed = ("IN_AIR" if s.in_air else "ON_GROUND") if s.mode != "LAND" or not s.in_air else "LANDING"
            values = dict(connected=True, armed=s.armed, in_air=s.in_air, landed_state=landed,
                          flight_mode=s.mode, north_m=s.n, east_m=s.e, down_m=s.d, vn_m_s=s.vn,
                          ve_m_s=s.ve, vd_m_s=s.vd, yaw_deg=s.yaw, battery_pct=s.battery,
                          battery_v=14.0 + 2.8 * s.battery / 100.0)
            ready = self._ready()
        self.cache.update(local_position_ok=ready, global_position_ok=ready, home_position_ok=ready,
                          armable=ready and alt < 0.2, **values)
