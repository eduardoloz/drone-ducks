"""quackdrone: ProjectQuack's drone-control core.

PX4 is the flight controller (it runs every PID loop); quackdrone sends position / velocity / yaw
setpoints in PX4 Offboard mode through MAVSDK, with safety checks around them. The same code flies
PX4 SITL on a laptop and a real Pixhawk drone: only the connection URL changes.

    from quackdrone import Drone
    with Drone("udpin://0.0.0.0:14540") as drone:
        drone.takeoff(2.0)
        drone.move(forward=1.0)

See sim/README.md and sim/docs/ARCHITECTURE.md.
"""
from .drone import DEFAULT_CONNECTION, Drone, DroneConfig, Event, make_backend
from .errors import (ArmingDenied, CommandDenied, ControlLost, NotConnected, NotReady, OffboardRejected,
                     QuackDroneError)
from .safety import Limits, VelocityCommand
from .telemetry import Telemetry

__all__ = [
    "DEFAULT_CONNECTION", "Drone", "DroneConfig", "Event", "make_backend", "Limits", "VelocityCommand",
    "Telemetry", "QuackDroneError", "NotConnected", "NotReady", "CommandDenied", "ArmingDenied",
    "OffboardRejected", "ControlLost",
]
__version__ = "0.1.0"
