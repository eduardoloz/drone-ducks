"""PX4 through the native MAVSDK binding (``pip install "mavsdk>=4"``, ``import mavsdk``).

Uses MAVSDK's *synchronous* API: commands block until PX4 acks (well under 2 s), telemetry
arrives through callbacks on MAVSDK's own thread. No asyncio needed, so the Drone class can be
called from plain scripts, TypeFly's skill thread, or an OpenCV loop.

Connection URLs (same code, only the URL changes):
  udpin://0.0.0.0:14540         PX4 SITL, API/offboard port (default)
  udpin://0.0.0.0:14550         PX4 SITL, ground-station port (used by test observers)
  serial:///dev/ttyTHS1:921600  Jetson UART -> Pixhawk TELEM2 (real drone)
  serial:///dev/ttyACM0:57600   Pixhawk over USB
"""
from __future__ import annotations

import threading
import time

from ..errors import ArmingDenied, CommandDenied, NotConnected, OffboardRejected
from ..geometry import wrap_deg
from ..telemetry import TelemetryCache
from .base import Backend, ParamValue


class MavsdkBackend(Backend):
    def __init__(self, url: str = "udpin://0.0.0.0:14540", cache: TelemetryCache | None = None):
        super().__init__(cache)
        self.url = url
        self._mavsdk = None
        self._system = None
        self._telemetry = self._action = self._offboard = self._param = None
        self._lock = threading.Lock()  # one MAVSDK command at a time from our side

    def describe(self) -> str:
        return f"MAVSDK native ({self.url})"

    # ------------------------------------------------------------------ link
    def connect(self, timeout_s: float = 30.0) -> None:
        from mavsdk import ComponentType, Configuration, ConnectionResult, Mavsdk
        from mavsdk.plugins.action import Action
        from mavsdk.plugins.offboard import Offboard
        from mavsdk.plugins.param import Param
        from mavsdk.plugins.telemetry import Telemetry

        # GROUND_STATION: PX4 then sees a GCS heartbeat from us (clears "No connection to the GCS").
        self._mavsdk = Mavsdk(Configuration.create_with_component_type(ComponentType.GROUND_STATION))
        result = self._mavsdk.add_any_connection(self.url)
        if result != ConnectionResult.SUCCESS:
            raise NotConnected(f"could not open {self.url}: {result.name} (port already in use by "
                               "another script? only one program can listen on a udpin port)")
        system = self._mavsdk.first_autopilot(timeout_s)
        if system is None:
            raise NotConnected(f"no PX4 autopilot on {self.url} within {timeout_s:.0f} s. "
                               "Is SITL running (scripts/start_sim.sh)? Right port?")
        self._system = system
        self._telemetry = Telemetry(system)
        self._action = Action(system)
        self._offboard = Offboard(system)
        self._param = Param(system)
        self._subscribe()
        self.cache.update(connected=True)

    def _subscribe(self) -> None:
        c, t = self.cache, self._telemetry

        def on_pv(pv, _):
            p, v = pv.position, pv.velocity
            c.update(north_m=p.north_m, east_m=p.east_m, down_m=p.down_m,
                     vn_m_s=v.north_m_s, ve_m_s=v.east_m_s, vd_m_s=v.down_m_s)

        def on_heading(h, _):
            c.update(yaw_deg=wrap_deg(h.heading_deg))

        def on_battery(b, _):
            pct = b.remaining_percent
            # MAVSDK >= 2 reports 0..100; NaN when PX4 has no estimate yet.
            c.update(battery_pct=None if pct != pct else float(pct), battery_v=b.voltage_v)

        def on_health(h, _):
            c.update(local_position_ok=h.is_local_position_ok, global_position_ok=h.is_global_position_ok,
                     home_position_ok=h.is_home_position_ok, armable=h.is_armable)

        t.subscribe_position_velocity_ned(on_pv)
        t.subscribe_heading(on_heading)
        t.subscribe_armed(lambda v, _: c.update(armed=bool(v)))
        t.subscribe_in_air(lambda v, _: c.update(in_air=bool(v)))
        t.subscribe_landed_state(lambda v, _: c.update(landed_state=getattr(v, "name", str(v))))
        # FlightMode is an IntEnum: str() would give "7", so use .name ("OFFBOARD")
        t.subscribe_flight_mode(lambda v, _: c.update(flight_mode=getattr(v, "name", str(v))))
        t.subscribe_battery(on_battery)
        t.subscribe_health(on_health)
        t.subscribe_status_text(lambda s, _: c.update(status_text=s.text))
        self._system.subscribe_is_connected(lambda ok, _: c.update(connected=bool(ok)))

    def close(self) -> None:
        if self._mavsdk is not None:
            try:
                self._mavsdk.destroy()
            finally:
                self._mavsdk = self._system = None
                self._telemetry = self._action = self._offboard = self._param = None
                self.cache.update(connected=False)

    # ------------------------------------------------------------------ helpers
    def _reason(self, t0: float) -> str:
        texts = self.cache.status_since(t0 - 2.0)
        return texts[-1] if texts else ""

    def _action_call(self, name: str, exc=CommandDenied) -> None:
        from mavsdk.plugins.action import ActionError

        t0 = time.monotonic()
        try:
            with self._lock:
                getattr(self._action, name)()
        except ActionError as e:
            time.sleep(0.3)  # PX4 sends the reason as STATUSTEXT right after the NACK
            raise exc(name, e.result.name, self._reason(t0)) from e

    # ------------------------------------------------------------------ commands
    def arm(self) -> None:
        self._action_call("arm", ArmingDenied)

    def disarm(self) -> None:
        self._action_call("disarm")

    def hold(self) -> None:
        self._action_call("hold")

    def land(self) -> None:
        self._action_call("land")

    def return_to_launch(self) -> None:
        self._action_call("return_to_launch")

    def set_position_ned(self, north_m, east_m, down_m, yaw_deg) -> None:
        from mavsdk.plugins.offboard import PositionNedYaw
        with self._lock:
            self._offboard.set_position_ned(PositionNedYaw(float(north_m), float(east_m),
                                                           float(down_m), float(yaw_deg)))

    def set_velocity_body(self, forward_m_s, right_m_s, down_m_s, yawspeed_deg_s) -> None:
        from mavsdk.plugins.offboard import VelocityBodyYawspeed
        with self._lock:
            self._offboard.set_velocity_body(VelocityBodyYawspeed(
                float(forward_m_s), float(right_m_s), float(down_m_s), float(yawspeed_deg_s)))

    def _offboard_call(self, name: str) -> None:
        from mavsdk.plugins.offboard import OffboardError

        t0 = time.monotonic()
        try:
            with self._lock:
                getattr(self._offboard, name)()
        except OffboardError as e:
            time.sleep(0.3)
            raise OffboardRejected(f"offboard.{name}", e.result.name, self._reason(t0)) from e

    def start_offboard(self) -> None:
        self._offboard_call("start")

    def stop_offboard(self) -> None:
        self._offboard_call("stop")

    # ------------------------------------------------------------------ params
    def get_param(self, name: str) -> ParamValue:
        from mavsdk.plugins.param import ParamError
        try:
            return self._param.get_param_float(name)
        except ParamError:
            return self._param.get_param_int(name)

    def set_param(self, name: str, value: ParamValue) -> None:
        if isinstance(value, bool) or isinstance(value, int):
            self._param.set_param_int(name, int(value))
        else:
            self._param.set_param_float(name, float(value))
