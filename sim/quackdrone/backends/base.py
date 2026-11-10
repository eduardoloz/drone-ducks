"""The small set of autopilot operations the Drone class needs.

Two implementations:
  * MavsdkBackend (mavsdk_native.py): PX4 SITL or a real Pixhawk, through native ``mavsdk>=4``.
  * FakeBackend (fake.py): a pure-Python point-mass "PX4" for unit tests and laptops without PX4.

Backends are deliberately thin: no waiting loops, no safety logic, no retries. All of that lives
in Drone so that it is tested once, against the fake, and behaves the same on real hardware.
Every method is synchronous and returns quickly (MAVSDK command acks take < ~1.5 s).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

from ..telemetry import TelemetryCache

ParamValue = Union[int, float]


class Backend(ABC):
    def __init__(self, cache: TelemetryCache | None = None):
        self.cache = cache or TelemetryCache()

    # ---- link
    @abstractmethod
    def connect(self, timeout_s: float) -> None:
        """Find the autopilot and start streaming telemetry into self.cache.
        Raises NotConnected if nothing answers within timeout_s."""

    @abstractmethod
    def close(self) -> None:
        """Release the link. Stops MAVSDK's automatic setpoint re-sending, so if PX4 is still in
        OFFBOARD its offboard-loss failsafe fires COM_OF_LOSS_T seconds later."""

    # ---- commands (raise CommandDenied / OffboardRejected / ArmingDenied on refusal)
    @abstractmethod
    def arm(self) -> None: ...

    @abstractmethod
    def disarm(self) -> None: ...

    @abstractmethod
    def set_position_ned(self, north_m: float, east_m: float, down_m: float, yaw_deg: float) -> None:
        """Offboard position setpoint (absolute local NED + heading)."""

    @abstractmethod
    def set_velocity_body(self, forward_m_s: float, right_m_s: float, down_m_s: float,
                          yawspeed_deg_s: float) -> None:
        """Offboard velocity setpoint in the yaw-aligned body frame (see geometry.py)."""

    @abstractmethod
    def start_offboard(self) -> None:
        """Switch PX4 to OFFBOARD. A setpoint must have been sent first (PX4 rule)."""

    @abstractmethod
    def stop_offboard(self) -> None:
        """Stop re-sending setpoints and switch PX4 to HOLD (MAVSDK semantics)."""

    @abstractmethod
    def hold(self) -> None: ...

    @abstractmethod
    def land(self) -> None: ...

    @abstractmethod
    def return_to_launch(self) -> None: ...

    # ---- parameters (used by failsafe tests; also handy for mentees)
    @abstractmethod
    def get_param(self, name: str) -> ParamValue: ...

    @abstractmethod
    def set_param(self, name: str, value: ParamValue) -> None:
        """int value -> int param, float value -> float param (PX4 is strict about the type)."""

    def describe(self) -> str:
        return type(self).__name__
