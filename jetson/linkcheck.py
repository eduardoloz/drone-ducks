"""Read-only MAVLink link check: connect to PX4 and print telemetry. Sends NO commands.

    python -m jetson.linkcheck --connect serial:///dev/ttyTHS1:921600     # Jetson on Pixhawk TELEM2
    python -m jetson.linkcheck --connect udpin://0.0.0.0:14540            # SITL / mavlink-router

Use it on bring-up day (props OFF) to prove the wiring, baud rate and PX4 parameters before
anything is allowed to fly: you should see the flight mode change when you flip the RC mode
switch and the altitude change when you lift the drone by hand.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from .bridge import SITL_URL, BridgeUnavailable, DroneAdapter, make_adapter


async def watch(adapter: DroneAdapter, url: str, seconds: float, connect_timeout: float, emit=print) -> int:
    emit(f"connecting to {url} (read-only) ...")
    try:
        await adapter.connect(url, connect_timeout)
    except (asyncio.TimeoutError, BridgeUnavailable) as e:
        emit(f"FAIL: no autopilot ({e or 'timeout'}). Check wiring (TX->RX, RX->TX, GND), baud rate, "
             "MAV_1_CONFIG=102 / SER_TEL2_BAUD=921600 on PX4, and that nothing else holds the port.")
        return 1
    emit("connected: heartbeat received")
    t_end = time.monotonic() + seconds
    last = None
    while time.monotonic() < t_end:
        state = (adapter.flight_mode, adapter.armed, adapter.in_air,
                 None if adapter.relative_altitude_m is None else round(adapter.relative_altitude_m, 1))
        if state != last:
            emit(f"  mode={state[0]:<16s} armed={state[1]!s:<5s} in_air={state[2]!s:<5s} rel_alt={state[3]} m")
            last = state
        await asyncio.sleep(0.2)
    await adapter.close()
    emit("done (no commands were sent)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connect", default=SITL_URL)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--timeout", type=float, default=15.0, help="connection timeout (s)")
    ap.add_argument("--mavsdk-api", default="auto", choices=("auto", "native", "classic"))
    args = ap.parse_args(argv)
    try:
        adapter = make_adapter(args.mavsdk_api)
    except BridgeUnavailable as e:
        print(f"MAVSDK not available: {e}")
        return 2
    return asyncio.run(watch(adapter, args.connect, args.seconds, args.timeout))


if __name__ == "__main__":
    sys.exit(main())
