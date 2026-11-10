"""Telemetry cache: backends write into it from their callback threads, everyone else reads a
consistent, immutable snapshot."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, fields
from typing import Optional


@dataclass(frozen=True)
class Telemetry:
    """One consistent snapshot of the drone state. Units: metres, m/s, degrees, percent."""

    stamp: float = 0.0              # time.monotonic() of the newest update (0 = never)
    connected: bool = False
    armed: bool = False
    in_air: bool = False
    landed_state: str = "UNKNOWN"   # ON_GROUND / TAKING_OFF / IN_AIR / LANDING / UNKNOWN
    flight_mode: str = "UNKNOWN"    # OFFBOARD, HOLD, RETURN_TO_LAUNCH, LAND, TAKEOFF, POSCTL, ...
    north_m: float = 0.0
    east_m: float = 0.0
    down_m: float = 0.0
    vn_m_s: float = 0.0
    ve_m_s: float = 0.0
    vd_m_s: float = 0.0
    yaw_deg: float = 0.0
    battery_pct: Optional[float] = None   # 0..100
    battery_v: Optional[float] = None
    local_position_ok: bool = False
    global_position_ok: bool = False
    home_position_ok: bool = False
    armable: bool = False
    status_text: str = ""            # newest PX4 STATUSTEXT, e.g. "Failsafe activated"

    @property
    def position_ned(self) -> tuple[float, float, float]:
        return self.north_m, self.east_m, self.down_m

    @property
    def velocity_ned(self) -> tuple[float, float, float]:
        return self.vn_m_s, self.ve_m_s, self.vd_m_s

    @property
    def speed(self) -> float:
        return math.sqrt(self.vn_m_s ** 2 + self.ve_m_s ** 2 + self.vd_m_s ** 2)

    @property
    def ready_to_arm(self) -> bool:
        return (self.local_position_ok and self.global_position_ok and self.home_position_ok
                and self.armable)


_FIELDS = {f.name for f in fields(Telemetry)} - {"stamp"}


class TelemetryCache:
    """Thread-safe store. update(**fields) from any thread; snapshot() returns a Telemetry."""

    def __init__(self):
        self._lock = threading.Lock()
        self._values: dict = {}
        self._stamps: dict[str, float] = {}
        self._stamp = 0.0
        self._status_log: list[tuple[int, float, str]] = []   # (seq, monotonic time, text)
        self._status_seq = 0

    def update(self, **values) -> None:
        unknown = set(values) - _FIELDS
        if unknown:
            raise KeyError(f"unknown telemetry fields {unknown}")
        now = time.monotonic()
        with self._lock:
            self._values.update(values)
            for k in values:
                self._stamps[k] = now
            self._stamp = now
            if "status_text" in values and values["status_text"]:
                self._status_seq += 1
                self._status_log.append((self._status_seq, now, values["status_text"]))
                del self._status_log[:-100]

    def snapshot(self) -> Telemetry:
        with self._lock:
            return Telemetry(stamp=self._stamp, **self._values)

    def age(self, name: Optional[str] = None) -> float:
        """Seconds since the newest update of `name` (or of anything). inf if never."""
        with self._lock:
            t = self._stamps.get(name, 0.0) if name else self._stamp
        return time.monotonic() - t if t else math.inf

    def status_since(self, t0: float) -> list[str]:
        """PX4 status texts received at or after monotonic time t0 (newest last)."""
        with self._lock:
            return [text for _, t, text in self._status_log if t >= t0]

    def statuses_after(self, seq: int) -> tuple[list[str], int]:
        """Status texts with sequence number > seq, and the newest sequence number. Lets a reader
        see every message exactly once without timestamp races."""
        with self._lock:
            return [text for n, _, text in self._status_log if n > seq], self._status_seq
