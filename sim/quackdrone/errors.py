"""Exceptions raised by quackdrone. Catch QuackDroneError to catch all of them."""


class QuackDroneError(RuntimeError):
    """Base class for every quackdrone error."""


class NotConnected(QuackDroneError):
    """No autopilot answered on the connection URL (is SITL running? right port?)."""


class NotReady(QuackDroneError):
    """PX4 never reported a usable position estimate / armable state in time."""


class CommandDenied(QuackDroneError):
    """PX4 refused a command (arm, mode change, ...). `.reason` holds PX4's last status text."""

    def __init__(self, what: str, result: str = "", reason: str = ""):
        self.what, self.result, self.reason = what, result, reason
        msg = f"{what} refused by PX4"
        if result:
            msg += f" ({result})"
        if reason:
            msg += f"; PX4 says: {reason!r}"
        super().__init__(msg)


class ArmingDenied(CommandDenied):
    pass


class OffboardRejected(CommandDenied):
    pass


class ControlLost(QuackDroneError):
    """We are no longer allowed to command the drone: PX4 switched mode on its own (failsafe or
    pilot override), the link dropped, or a low-battery landing started. Motion commands raise
    this until you call Drone.resume_control() (or land)."""
