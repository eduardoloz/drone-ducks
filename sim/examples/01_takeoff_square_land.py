#!/usr/bin/env python3
"""Example 1: take off, fly a 2 m square with position commands, land.

    sim/scripts/start_sim.sh                                   # SITL (SIM=sih works too)
    sim/scripts/py.sh sim/examples/01_takeoff_square_land.py   # add --connection fake:// to skip PX4

Things to notice:
  * takeoff() waits for PX4's estimator ("is_armable") before arming. Right after PX4 boots that
    takes 10-30 s; PX4 refuses to arm before ("height estimate not stable").
  * move() is relative to the drone's heading; turn(+90) is clockwise (turn right).
  * Leaving the `with` block lands the drone, even if your code raises an exception.
"""
import argparse

from quackdrone import Drone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connection", default="udpin://0.0.0.0:14540")
    ap.add_argument("--alt", type=float, default=2.0)
    ap.add_argument("--side", type=float, default=2.0)
    args = ap.parse_args()

    with Drone(args.connection) as drone:
        drone.takeoff(args.alt)
        t = drone.telemetry
        print(f"airborne: alt {drone.altitude_m:.2f} m, heading {t.yaw_deg:.0f} deg, battery {t.battery_pct}")
        for leg in range(4):
            ok = drone.move(forward=args.side)
            drone.turn(90)
            t = drone.telemetry
            print(f"leg {leg + 1}: reached={ok}  N{t.north_m:+.2f} E{t.east_m:+.2f} alt {drone.altitude_m:.2f} m "
                  f"heading {t.yaw_deg:+.0f}")
        home = drone.home_ned
        t = drone.telemetry
        print(f"back at start? {abs(t.north_m - home[0]):.2f} m N / {abs(t.east_m - home[1]):.2f} m E off")
    print("landed:", not drone.telemetry.in_air)


if __name__ == "__main__":
    main()
