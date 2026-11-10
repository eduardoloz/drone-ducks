#!/usr/bin/env python3
"""A pilot process that test_failsafes.py starts and then kills or lets crash.

    child_pilot.py velocity URL   take off, then stream a gentle velocity until killed
    child_pilot.py hover URL      take off, then hold position until killed
    child_pilot.py raise URL      take off, then raise an exception inside `with Drone(...)`
Prints READY once airborne.
"""
import sys
import time

from quackdrone import Drone, DroneConfig


def main() -> None:
    behaviour, url = sys.argv[1], sys.argv[2]
    cfg = DroneConfig(takeoff_timeout_s=45.0, ready_timeout_s=120.0)
    with Drone(url, config=cfg, log=lambda m: print(m, flush=True)) as drone:
        drone.takeoff(2.0)
        print("READY", flush=True)
        if behaviour == "raise":
            time.sleep(1.0)
            raise RuntimeError("simulated perception crash")
        while True:  # keep flying until the test kills us
            if behaviour == "velocity":
                drone.set_velocity_body(forward_m_s=0.3, yawspeed_deg_s=5.0)
            time.sleep(0.1)


if __name__ == "__main__":
    main()
