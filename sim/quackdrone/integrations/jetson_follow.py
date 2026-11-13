"""Jetson face-follow pipeline -> quackdrone: a `jetson.bridge.CommandSink` that flies `Drone`.

The Jetson track's pipeline (camera -> face detector -> tracker -> controller) ends in a
`CommandSink`. Its own `MavsdkOffboardSink` opens a second MAVSDK connection with its own
watchdog. On the sim (and later on the drone) we want ONE flight-control layer, so this sink
sends every controller output through `Drone.set_velocity_body` and inherits quackdrone's
safety layers unchanged:

    controller output                     what QuackdroneSink does
    -----------------                     ------------------------
    TRACKING / IDLE / LOST command    ->  Drone.set_velocity_body(...)  (clamped to Limits:
                                          speed, altitude band, soft fence)
    TIMEOUT + action (target lost     ->  land (default) / rtl / hold, once; session finished
      for lost_timeout_s)
    no submit() for 0.5 s             ->  quackdrone's own watchdog brakes, then holds position
    no submit() for stall_timeout_s   ->  stall_action (land by default); session finished
    PX4 leaves Offboard (geofence,    ->  quackdrone stops commanding (ControlLost); we never
      battery, pilot, link loss)          re-engage; session finished
    stop() while still in control     ->  on_exit (land by default)

Usage (see sim/examples/follow_person.py for the full loop with logging):

    from quackdrone import Drone
    from quackdrone.integrations.jetson_follow import QuackdroneSink, sim_follow_config
    from jetson.pipeline import FollowPipeline
    from jetson.controller import FaceFollowController

    with Drone("udpin://0.0.0.0:14540") as drone:
        sink = QuackdroneSink(drone, takeoff_alt_m=2.0)
        pipe = FollowPipeline(detector, FaceFollowController(sim_follow_config()), sink=sink)
        sink.start()                      # connect + take off (or adopt a flying drone)
        while not sink.finished:
            pipe.step(reader.get(newer_than=last))
        sink.stop()

`jetson` is imported as a library from the repo root (drone-ducks/jetson); nothing in jetson/ is
modified. If it is not importable, `import_jetson()` adds the repo root to sys.path.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ..drone import Drone
from ..errors import ControlLost, QuackDroneError
from ..safety import VelocityCommand

# quack_cam (sim/gazebo/models/quack_cam): horizontal_fov 1.2 rad, 640x480, pitched 0.14 rad down.
SIM_HFOV_DEG = 68.75
# YuNet box width of the walking actor's face, from frames at a known ground-truth distance
# (3 frames at 2.3-3.9 m gave 0.11-0.14 m; the Jetson default 0.16 m makes the sim walker look ~20 % too far).
SIM_FACE_WIDTH_M = 0.13
ACTIONS = ("land", "rtl", "hold")


def import_jetson():
    """Import the Jetson package, adding the repo root to sys.path if needed. Returns the module."""
    try:
        import jetson  # noqa: F401
    except ImportError:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if not os.path.isfile(os.path.join(root, "jetson", "__init__.py")):
            raise ImportError(f"jetson package not found (looked in {root}); run from the drone-ducks repo")
        sys.path.insert(0, root)
        import jetson  # noqa: F401
    return sys.modules["jetson"]


import_jetson()
from jetson.bridge import CommandSink  # noqa: E402
from jetson.controller import TIMEOUT, ControlOutput, FollowConfig  # noqa: E402


def sim_follow_config(**overrides) -> FollowConfig:
    """FollowConfig for the Gazebo quack_cam: Jetson defaults with the sim camera's HFOV and the
    walking actor's face width."""
    kw = dict(hfov_deg=SIM_HFOV_DEG, face_width_m=SIM_FACE_WIDTH_M)
    kw.update(overrides)
    return FollowConfig(**kw)


@dataclass(frozen=True)
class SentCommand:
    """One submit(): what the controller asked for and what quackdrone actually sent."""

    t: float                      # time.monotonic()
    mode: str                     # controller mode
    requested: VelocityCommand
    sent: Optional[VelocityCommand]   # None if nothing was sent (finished / control lost)


class QuackdroneSink(CommandSink):
    """CommandSink on top of an existing quackdrone `Drone`.

    drone           a Drone (connected or not; start() connects). The caller owns its lifetime:
                    use `with Drone(...)` so an exception still lands.
    takeoff_alt_m   start() takes off to this height if the drone is on the ground. None: the
                    drone must already be flying; start() then adopts it with resume_control().
    stall_timeout_s no submit() for this long = perception is dead -> stall_action. quackdrone's
                    0.5 s watchdog has already stopped the drone by then.
    stall_action    land | rtl | hold
    on_exit         what stop() does if we still have control: land | rtl | hold
    emit            log function for one-line messages
    """

    def __init__(self, drone: Drone, *, takeoff_alt_m: Optional[float] = None, stall_timeout_s: float = 3.0,
                 stall_action: str = "land", on_exit: str = "land", monitor_hz: float = 10.0,
                 emit: Optional[Callable[[str], None]] = print, history: int = 20000) -> None:
        if stall_action not in ACTIONS or on_exit not in ACTIONS:
            raise ValueError(f"actions must be one of {ACTIONS}")
        if stall_timeout_s <= drone.config.cmd_timeout_s:
            raise ValueError("stall_timeout_s must be longer than the drone's cmd_timeout_s watchdog")
        self.drone = drone
        self.takeoff_alt_m = takeoff_alt_m
        self.stall_timeout_s, self.stall_action, self.on_exit = stall_timeout_s, stall_action, on_exit
        self.monitor_period = 1.0 / monitor_hz
        self._emit = emit
        self.finished = False
        self.finish_reason = ""
        self.action_taken: Optional[str] = None     # land | rtl | hold once the session ended by our action
        self.sent = 0
        self.last: Optional[SentCommand] = None
        self.history: list[SentCommand] = []
        self._history_max = history
        self._lock = threading.RLock()
        self._last_submit = 0.0
        self._started = False
        self._stop_evt = threading.Event()
        self._monitor: Optional[threading.Thread] = None

    # -------------------------------------------------------------- CommandSink interface
    @property
    def altitude_m(self) -> Optional[float]:
        alt = self.drone.altitude_m
        return None if alt != alt else alt  # NaN before takeoff -> None

    def status(self) -> str:
        if self.finished:
            return f"FINISHED: {self.finish_reason}"
        t = self.drone.telemetry
        return f"QUACKDRONE in_control={self.drone.in_control} mode={t.flight_mode} sent={self.sent}"

    def start(self) -> None:
        """Connect, then take off (takeoff_alt_m) or adopt a flying drone. Blocks until we hold
        Offboard control. Raises quackdrone errors (NotReady, ArmingDenied, ...) on failure."""
        d = self.drone
        d.connect()
        if not d.in_control:
            t = d.telemetry
            if t.armed and t.in_air:
                d.resume_control()
            elif self.takeoff_alt_m is not None:
                self._say(f"taking off to {self.takeoff_alt_m:.1f} m")
                if not d.takeoff(self.takeoff_alt_m):
                    self._say("takeoff altitude not reached in time; continuing from here")
            else:
                raise QuackDroneError("drone is on the ground and takeoff_alt_m is None")
        self._last_submit = time.monotonic()
        self._started = True
        self._stop_evt.clear()
        self._monitor = threading.Thread(target=self._run_monitor, name="quack-follow-monitor", daemon=True)
        self._monitor.start()
        self._say("following (commands go through quackdrone limits + watchdog)")

    def submit(self, out: ControlOutput) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_submit = now
            if self.finished:
                return
            if out.mode == TIMEOUT and out.action:
                self._act(out.action, out.reason)
                self._record(now, out, None)
                return
            c = out.command
            req = VelocityCommand(c.forward_m_s, c.right_m_s, c.down_m_s, c.yawspeed_deg_s)
            try:
                sent = self.drone.set_velocity_body(*map(float, c.as_tuple()))
            except ControlLost as e:
                self._finish(f"quackdrone lost control: {e}")
                self._record(now, out, None)
                return
            except ValueError:
                # NaN/inf from the controller: a bug. Brake now, let the caller's exception path land.
                self.drone.stop(wait=False)
                raise
            self.sent += 1
            self._record(now, out, sent, req)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._monitor is not None and self._monitor is not threading.current_thread():
            self._monitor.join(timeout=2.0)
        with self._lock:
            if self._started and not self.finished and self.drone.in_control:
                self._act(self.on_exit, "follow session stopped")
            elif not self.finished:
                self._finish("follow session stopped")

    # -------------------------------------------------------------- internals
    def _say(self, msg: str) -> None:
        if self._emit:
            self._emit(f"[follow] {msg}")

    def _record(self, now: float, out: ControlOutput, sent: Optional[VelocityCommand],
                req: Optional[VelocityCommand] = None) -> None:
        if req is None:
            c = out.command
            req = VelocityCommand(c.forward_m_s, c.right_m_s, c.down_m_s, c.yawspeed_deg_s)
        self.last = SentCommand(now, out.mode, req, sent)
        self.history.append(self.last)
        if len(self.history) > self._history_max:
            del self.history[: self._history_max // 2]

    def _finish(self, reason: str) -> None:
        if not self.finished:
            self.finished, self.finish_reason = True, reason
            self._say(reason)

    def _act(self, action: str, reason: str) -> None:
        """End the session with land / rtl / hold. Caller holds self._lock."""
        if self.finished:
            return
        self._finish(f"{reason} -> {action.upper()}")
        self.action_taken = action
        d = self.drone
        try:
            if action == "land":
                d.land(wait=False, reason=f"follow: {reason}")
            elif action == "rtl":
                d.return_to_launch()
            elif d.in_control:  # hold: brake, then hold position in Offboard (MAVSDK keeps re-sending)
                d.stop(wait=False)
                d.hold()
        except QuackDroneError as e:
            self._say(f"{action} failed: {e}")

    def _run_monitor(self) -> None:
        while not self._stop_evt.wait(self.monitor_period):
            try:
                self._check(time.monotonic())
            except Exception as e:  # the monitor must never die silently
                self._say(f"monitor error: {e!r}")

    def _check(self, now: float) -> None:
        with self._lock:
            if self.finished:
                return
            d = self.drone
            if not d.in_control:
                # PX4 failsafe (geofence, battery), pilot takeover, link loss, or a quackdrone
                # low-battery landing. quackdrone already stopped commanding; we never re-engage.
                why = f"PX4 switched to {d.override_mode}" if d.override_mode else "control released"
                self._finish(f"quackdrone lost control ({why})")
                return
            idle = now - self._last_submit
            if idle > self.stall_timeout_s:
                self._act(self.stall_action, f"no command for {idle:.1f} s (perception stalled?)")
