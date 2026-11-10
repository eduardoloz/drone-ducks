"""Send controller commands to PX4 as MAVSDK Offboard velocity setpoints.

Two sinks share one interface (start / submit / stop / finished):

  DryRunSink          default. Logs what it *would* send. Nothing talks to a drone.
  MavsdkOffboardSink  only with --fly. Streams VelocityBodyYawspeed setpoints at a fixed rate.

How MavsdkOffboardSink flies (PX4 Offboard rules: https://docs.px4.io/main/en/flight_modes/offboard.html)
  1. connect (SITL: udpin://0.0.0.0:14540; Jetson on TELEM2: serial:///dev/ttyTHS1:921600)
  2. wait for a healthy position estimate (and armable, if we will arm)
  3. if on the ground: arm + take off to --auto-takeoff metres (SITL convenience), otherwise wait
     for the pilot to take off (the normal real-hardware flow)
  4. stream a few hover setpoints FIRST, then offboard.start()  (PX4 rejects the switch otherwise)
  5. every 1/rate s send the newest command. Safety layers, in order:
       - pilot takes over (flight mode leaves OFFBOARD: RC mode switch, stick override,
         failsafe, geofence): stop sending immediately and NEVER re-engage
       - controller says TIMEOUT (target lost too long): stop offboard, then land / rtl / hold
       - commands older than command_timeout_s (perception slow/hung): send hover instead
       - no command for stall_timeout_s: perception is dead -> stall_action (land by default)
       - this whole process dies: setpoints stop, PX4's own offboard-loss failsafe triggers
         after COM_OF_LOSS_T (then COM_OBL_RC_ACT). Configure those params on the real drone!
  6. stop(): leave offboard via `on_exit` (hold by default, so the pilot takes over)

MAVSDK is asyncio; the perception loop is plain threads. The sink runs its own event loop in a
daemon thread and submit() just swaps in the newest command under a lock (never blocks).

Adapters hide the two MAVSDK Python APIs: NativeMavsdkAdapter (pip `mavsdk` >= 4, what Junwei
and setup/.venv-mavsdk have) and ClassicMavsdkAdapter (`mavsdk_grpc`, or `mavsdk` < 4 on older
Pythons). Tests use a fake adapter, so the safety logic is tested without any drone.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from typing import Callable, List, Optional, Tuple

from .controller import HOVER, TIMEOUT, ControlOutput, VelocityCommand

log = logging.getLogger(__name__)

SITL_URL = "udpin://0.0.0.0:14540"
EXIT_ACTIONS = ("hold", "land", "rtl")


class BridgeUnavailable(RuntimeError):
    """MAVSDK is not installed or cannot connect."""


# ===================================================================================== sinks
class CommandSink:
    """Where the pipeline sends each ControlOutput."""

    finished = False          # True once the sink ended the session (landed, pilot override...)
    finish_reason = ""

    def start(self) -> None:
        pass

    def submit(self, out: ControlOutput) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        pass

    @property
    def altitude_m(self) -> Optional[float]:
        return None

    def status(self) -> str:
        return ""


class DryRunSink(CommandSink):
    """Logs what would be sent. log_every_s throttles repeats; mode changes always log."""

    def __init__(self, log_every_s: float = 1.0, emit: Optional[Callable[[str], None]] = None,
                 finish_on_timeout: bool = False) -> None:
        self.log_every_s = log_every_s
        self.emit = emit or (lambda msg: log.info(msg))
        self.finish_on_timeout = finish_on_timeout
        self.history: List[Tuple[float, ControlOutput]] = []
        self._last_log = -math.inf
        self._last_mode = None

    def submit(self, out: ControlOutput) -> None:
        now = time.monotonic()
        self.history.append((now, out))
        if len(self.history) > 1000:
            del self.history[:500]
        if out.mode != self._last_mode or now - self._last_log >= self.log_every_s:
            self.emit(f"[dry-run] {out.mode:8s} would send set_velocity_body({out.command})  | {out.reason}")
            self._last_log, self._last_mode = now, out.mode
        if out.mode == TIMEOUT and out.action and self.finish_on_timeout and not self.finished:
            self.emit(f"[dry-run] would stop offboard and {out.action.upper()}")
            self.finished, self.finish_reason = True, out.reason

    def status(self) -> str:
        return "DRY-RUN (no commands sent)"


# ===================================================================================== adapters
class DroneAdapter:
    """The few MAVSDK calls the bridge needs. Telemetry fields are kept fresh by connect()."""

    in_air: bool = False
    armed: bool = False
    flight_mode: str = "UNKNOWN"
    relative_altitude_m: Optional[float] = None

    async def connect(self, url: str, timeout_s: float) -> None: ...
    async def wait_ready(self, need_armable: bool, require_global: bool, timeout_s: float) -> None: ...
    async def arm(self) -> None: ...
    async def takeoff(self, altitude_m: float) -> None: ...
    async def land(self) -> None: ...
    async def return_to_launch(self) -> None: ...
    async def hold(self) -> None: ...
    async def set_velocity_body(self, cmd: VelocityCommand) -> None: ...
    async def start_offboard(self) -> None: ...
    async def stop_offboard(self) -> None: ...
    async def close(self) -> None: ...


def _mode_name(m) -> str:
    return getattr(m, "name", str(m)).upper()


class NativeMavsdkAdapter(DroneAdapter):
    """pip `mavsdk` >= 4: `from mavsdk.asyncio import Mavsdk` (ctypes binding, no mavsdk_server)."""

    def __init__(self) -> None:
        try:
            from mavsdk.asyncio import ComponentType, Configuration, Mavsdk
            from mavsdk.asyncio.plugins.action import ActionAsync
            from mavsdk.asyncio.plugins.offboard import OffboardAsync, VelocityBodyYawspeed
            from mavsdk.asyncio.plugins.telemetry import TelemetryAsync
        except ImportError as e:
            raise BridgeUnavailable("native MAVSDK (pip install 'mavsdk>=4') not importable") from e
        self._Mavsdk, self._Configuration, self._ComponentType = Mavsdk, Configuration, ComponentType
        self._ActionAsync, self._OffboardAsync, self._TelemetryAsync = ActionAsync, OffboardAsync, TelemetryAsync
        self._Velocity = VelocityBodyYawspeed
        self._tasks: list = []

    async def connect(self, url: str, timeout_s: float) -> None:
        cfg = self._Configuration.create_with_component_type(self._ComponentType.COMPANION_COMPUTER)
        self.mavsdk = self._Mavsdk(cfg)
        await self.mavsdk.add_any_connection(url)
        self.system = await self.mavsdk.first_autopilot(timeout_s)
        if self.system is None:
            raise BridgeUnavailable(f"no PX4 autopilot on {url} within {timeout_s:.0f} s")
        self.telemetry = self._TelemetryAsync(self.system)
        self.action = self._ActionAsync(self.system)
        self.offboard = self._OffboardAsync(self.system)
        self._tasks = [asyncio.ensure_future(c) for c in (self._watch_in_air(), self._watch_armed(),
                                                           self._watch_mode(), self._watch_position())]

    async def _watch_in_air(self):
        async for v in self.telemetry.subscribe_in_air():
            self.in_air = bool(v)

    async def _watch_armed(self):
        async for v in self.telemetry.subscribe_armed():
            self.armed = bool(v)

    async def _watch_mode(self):
        async for m in self.telemetry.subscribe_flight_mode():
            self.flight_mode = _mode_name(m)  # IntEnum: str() would give "7" on Python 3.11+

    async def _watch_position(self):
        async for p in self.telemetry.subscribe_position():
            self.relative_altitude_m = float(p.relative_altitude_m)

    async def wait_ready(self, need_armable: bool, require_global: bool, timeout_s: float) -> None:
        async def _wait():
            async for h in self.telemetry.subscribe_health():
                pos_ok = h.is_local_position_ok and (not require_global or (h.is_global_position_ok
                                                                            and h.is_home_position_ok))
                if pos_ok and (h.is_armable or not need_armable):
                    return
        await asyncio.wait_for(_wait(), timeout_s)

    async def arm(self):
        await self.action.arm()

    async def takeoff(self, altitude_m: float):
        await self.action.set_takeoff_altitude(float(altitude_m))
        await self.action.takeoff()

    async def land(self):
        await self.action.land()

    async def return_to_launch(self):
        await self.action.return_to_launch()

    async def hold(self):
        await self.action.hold()

    async def set_velocity_body(self, cmd: VelocityCommand):
        await self.offboard.set_velocity_body(self._Velocity(*map(float, cmd.as_tuple())))

    async def start_offboard(self):
        await self.offboard.start()

    async def stop_offboard(self):
        await self.offboard.stop()

    async def close(self):
        for t in self._tasks:
            t.cancel()


class ClassicMavsdkAdapter(DroneAdapter):
    """The classic gRPC API: `mavsdk_grpc` (new name) or `mavsdk` < 4 (e.g. on an older Python)."""

    def __init__(self) -> None:
        try:
            from mavsdk_grpc import System
            from mavsdk_grpc.offboard import VelocityBodyYawspeed
        except ImportError:
            try:
                from mavsdk import System
                from mavsdk.offboard import VelocityBodyYawspeed
            except ImportError as e:
                raise BridgeUnavailable("classic MAVSDK not importable: pip install mavsdk-grpc "
                                        "(or 'mavsdk<4')") from e
        self._System, self._Velocity = System, VelocityBodyYawspeed
        self._tasks: list = []

    async def connect(self, url: str, timeout_s: float) -> None:
        self.drone = self._System()
        await self.drone.connect(system_address=url)

        async def _connected():
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    return
        await asyncio.wait_for(_connected(), timeout_s)
        self._tasks = [asyncio.ensure_future(c) for c in (self._watch_in_air(), self._watch_armed(),
                                                           self._watch_mode(), self._watch_position())]

    async def _watch_in_air(self):
        async for v in self.drone.telemetry.in_air():
            self.in_air = bool(v)

    async def _watch_armed(self):
        async for v in self.drone.telemetry.armed():
            self.armed = bool(v)

    async def _watch_mode(self):
        async for m in self.drone.telemetry.flight_mode():
            self.flight_mode = _mode_name(m)

    async def _watch_position(self):
        async for p in self.drone.telemetry.position():
            self.relative_altitude_m = float(p.relative_altitude_m)

    async def wait_ready(self, need_armable: bool, require_global: bool, timeout_s: float) -> None:
        async def _wait():
            async for h in self.drone.telemetry.health():
                pos_ok = h.is_local_position_ok and (not require_global or (h.is_global_position_ok
                                                                            and h.is_home_position_ok))
                if pos_ok and (h.is_armable or not need_armable):
                    return
        await asyncio.wait_for(_wait(), timeout_s)

    async def arm(self):
        await self.drone.action.arm()

    async def takeoff(self, altitude_m: float):
        await self.drone.action.set_takeoff_altitude(float(altitude_m))
        await self.drone.action.takeoff()

    async def land(self):
        await self.drone.action.land()

    async def return_to_launch(self):
        await self.drone.action.return_to_launch()

    async def hold(self):
        await self.drone.action.hold()

    async def set_velocity_body(self, cmd: VelocityCommand):
        await self.drone.offboard.set_velocity_body(self._Velocity(*map(float, cmd.as_tuple())))

    async def start_offboard(self):
        await self.drone.offboard.start()

    async def stop_offboard(self):
        await self.drone.offboard.stop()

    async def close(self):
        for t in self._tasks:
            t.cancel()


def make_adapter(kind: str = "auto") -> DroneAdapter:
    """'native', 'classic', or 'auto' (native if importable, else classic)."""
    if kind == "native":
        return NativeMavsdkAdapter()
    if kind == "classic":
        return ClassicMavsdkAdapter()
    if kind != "auto":
        raise ValueError("adapter must be auto, native or classic")
    try:
        return NativeMavsdkAdapter()
    except BridgeUnavailable:
        return ClassicMavsdkAdapter()


# ===================================================================================== the flying sink
class MavsdkOffboardSink(CommandSink):
    def __init__(self, adapter: DroneAdapter, url: str = SITL_URL, rate_hz: float = 20.0,
                 command_timeout_s: float = 0.5, stall_timeout_s: float = 3.0, stall_action: str = "land",
                 auto_takeoff_m: Optional[float] = None, wait_for_pilot_s: float = 120.0,
                 connect_timeout_s: float = 30.0, ready_timeout_s: float = 90.0, require_global: bool = True,
                 on_exit: str = "hold", engage_timeout_s: float = 3.0,
                 emit: Optional[Callable[[str], None]] = None) -> None:
        if rate_hz < 5:
            raise ValueError("stream setpoints at >= 5 Hz (PX4 needs > 2 Hz; 10-20 Hz is typical)")
        if stall_action not in EXIT_ACTIONS or on_exit not in EXIT_ACTIONS:
            raise ValueError(f"actions must be one of {EXIT_ACTIONS}")
        if auto_takeoff_m is not None and not 1.0 <= auto_takeoff_m <= 20.0:
            raise ValueError("auto_takeoff_m must be between 1 and 20 m")
        self.adapter, self.url, self.period = adapter, url, 1.0 / rate_hz
        self.command_timeout_s, self.stall_timeout_s, self.stall_action = command_timeout_s, stall_timeout_s, \
            stall_action
        self.auto_takeoff_m, self.wait_for_pilot_s = auto_takeoff_m, wait_for_pilot_s
        self.connect_timeout_s, self.ready_timeout_s, self.require_global = connect_timeout_s, ready_timeout_s, \
            require_global
        self.on_exit, self.engage_timeout_s = on_exit, engage_timeout_s
        self.emit = emit or (lambda msg: log.warning(msg))
        self.sent = 0
        self.offboard_active = False
        self.pilot_override = False
        self._seen_offboard = False
        self._offboard_t = 0.0
        self._lock = threading.Lock()
        self._latest: Optional[ControlOutput] = None
        self._latest_t = 0.0
        self._running = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mavsdk-loop", daemon=True)
        self._stream_future = None

    # -------------------------------------------------------------- thread plumbing
    def _call(self, coro, timeout: float):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    @property
    def altitude_m(self) -> Optional[float]:
        return self.adapter.relative_altitude_m

    def status(self) -> str:
        if self.finished:
            return f"FINISHED: {self.finish_reason}"
        return f"FLYING offboard={self.offboard_active} mode={self.adapter.flight_mode} sent={self.sent}"

    # -------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._thread.start()
        budget = self.connect_timeout_s + self.ready_timeout_s + self.wait_for_pilot_s + 60
        try:
            self._call(self._startup(), timeout=budget)
        except Exception:
            self._shutdown_loop()
            raise
        self._running = True
        self._latest_t = time.monotonic()  # stall timer starts now
        self._stream_future = asyncio.run_coroutine_threadsafe(self._stream(), self._loop)

    async def _startup(self) -> None:
        a = self.adapter
        self.emit(f"[fly] connecting to {self.url}")
        await a.connect(self.url, self.connect_timeout_s)
        await asyncio.sleep(0.5)  # let telemetry arrive
        need_arm = not a.in_air and self.auto_takeoff_m is not None
        self.emit("[fly] waiting for a healthy position estimate" + (" and armable" if need_arm else ""))
        await a.wait_ready(need_arm, self.require_global, self.ready_timeout_s)
        if not a.in_air:
            if self.auto_takeoff_m is not None:
                self.emit(f"[fly] arming and taking off to {self.auto_takeoff_m:.1f} m")
                await a.arm()
                await a.takeoff(self.auto_takeoff_m)
                await self._wait_for(lambda: (a.relative_altitude_m or 0.0) >= 0.9 * self.auto_takeoff_m,
                                     60.0, "takeoff altitude")
            else:
                self.emit("[fly] on the ground: waiting for the pilot to take off")
                await self._wait_for(lambda: a.in_air, self.wait_for_pilot_s, "pilot takeoff")
        for _ in range(5):  # PX4 rule: setpoints must already be streaming before the switch
            await a.set_velocity_body(HOVER)
            await asyncio.sleep(0.05)
        await a.start_offboard()
        self.offboard_active, self._offboard_t = True, time.monotonic()
        self.emit("[fly] offboard started: following")

    async def _wait_for(self, cond, timeout_s: float, what: str) -> None:
        deadline = time.monotonic() + timeout_s
        while not cond():
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out waiting for {what}")
            await asyncio.sleep(0.05)

    def submit(self, out: ControlOutput) -> None:
        with self._lock:
            self._latest, self._latest_t = out, time.monotonic()

    async def _stream(self) -> None:
        a = self.adapter
        next_t = time.monotonic()
        while self._running:
            # Mode telemetry lags offboard.start(), so only a change *after* we saw OFFBOARD counts.
            if a.flight_mode == "OFFBOARD":
                self._seen_offboard = True
            elif self._seen_offboard and a.flight_mode != "UNKNOWN":
                self.pilot_override = True
                self._finish(f"left OFFBOARD (mode {a.flight_mode}): pilot/failsafe has control")
                return  # never fight the pilot, never re-engage
            elif not self._seen_offboard and time.monotonic() - self._offboard_t > self.engage_timeout_s:
                self._finish(f"offboard never engaged (mode stayed {a.flight_mode}); not sending more")
                return
            with self._lock:
                out, t = self._latest, self._latest_t
            now = time.monotonic()
            if out is not None and out.mode == TIMEOUT and out.action:
                await self._do_action(out.action, out.reason)
                return
            if now - t > self.stall_timeout_s:
                await self._do_action(self.stall_action, f"no command for {now - t:.1f} s (perception stalled?)")
                return
            cmd = out.command if (out is not None and now - t <= self.command_timeout_s) else HOVER
            try:
                await a.set_velocity_body(cmd)
                self.sent += 1
            except Exception as e:  # keep streaming; PX4's failsafe covers a dead link
                self.emit(f"[fly] set_velocity_body failed: {e}")
            next_t += self.period
            await asyncio.sleep(max(0.0, next_t - time.monotonic()))
            if time.monotonic() - next_t > 5 * self.period:
                next_t = time.monotonic()  # fell far behind: resync instead of bursting

    def _finish(self, reason: str) -> None:
        self.finished, self.finish_reason = True, reason
        self._running = False
        self.emit(f"[fly] {reason}")

    async def _do_action(self, action: str, reason: str) -> None:
        a = self.adapter
        self._finish(f"{reason} -> {action.upper()}")
        try:
            await a.stop_offboard()  # -> HOLD
        except Exception as e:
            self.emit(f"[fly] offboard stop: {e}")
        self.offboard_active = False
        if action == "land":
            await a.land()
        elif action == "rtl":
            await a.return_to_launch()
        else:
            await a.hold()

    def stop(self) -> None:
        self._running = False
        if self._stream_future is not None:
            try:
                self._stream_future.result(timeout=5.0)
            except Exception as e:
                self.emit(f"[fly] stream task ended with {type(e).__name__}: {e}")
        if self.offboard_active and not self.finished and not self.pilot_override:
            try:
                self._call(self._do_action(self.on_exit, "stopping"), timeout=15.0)
            except Exception as e:
                self.emit(f"[fly] exit action failed: {e}")
        self._shutdown_loop()

    def _shutdown_loop(self) -> None:
        if self._thread.is_alive():
            try:
                self._call(self.adapter.close(), timeout=5.0)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)
