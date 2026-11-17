"""The onboard app: camera -> face detection -> controller -> PX4 (DRY-RUN unless --fly).

Dry run (default; safe anywhere, logs the commands it would send):
    python -m jetson.follow --source sim --camera-preset sim
    python -m jetson.follow --source csi://0

SITL (only after the sim/ track has PX4 SITL + the camera bridge running; see docs/ARCHITECTURE.md):
    python -m jetson.follow --source sim --camera-preset sim --fly --auto-takeoff 3 --on-exit land

Real drone (Jetson on Pixhawk TELEM2; pilot arms + takes off, then this takes over in OFFBOARD):
    python -m jetson.follow --source csi://0 --fly --connect serial:///dev/ttyTHS1:921600

Safety rules this app follows are listed in jetson/bridge.py; the flight test plan is in
docs/WHAT_WE_NEED.md. Recording: --out saves annotated video with every non-target face blurred
(turn that off only for footage of consenting team members).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from . import cli
from .bridge import EXIT_ACTIONS, SITL_URL, BridgeUnavailable, DryRunSink, MavsdkOffboardSink, make_adapter
from .controller import FaceFollowController
from .hud import VideoSink, draw
from .pipeline import FollowPipeline
from .sources import LatestFrameReader, SourceError, open_source

log = logging.getLogger("jetson.follow")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="csi://0 | /dev/video0 | sim | http(s)://... | file")
    fly = ap.add_argument_group("flying (nothing is sent to a drone without --fly)")
    fly.add_argument("--fly", action="store_true", help="really send MAVSDK offboard commands")
    fly.add_argument("--yes", action="store_true", help="skip the 5 s safety countdown (SITL)")
    fly.add_argument("--connect", default=SITL_URL,
                     help=f"MAVSDK URL (default {SITL_URL}; Jetson+TELEM2: serial:///dev/ttyTHS1:921600)")
    fly.add_argument("--mavsdk-api", default="auto", choices=("auto", "native", "classic"))
    fly.add_argument("--auto-takeoff", type=float, default=None, metavar="M",
                     help="arm and take off to M metres if on the ground (SITL). Default: wait for the pilot")
    fly.add_argument("--rate", type=float, default=20.0, help="setpoint rate in Hz")
    fly.add_argument("--on-exit", default="hold", choices=EXIT_ACTIONS, help="what to do when this app stops")
    fly.add_argument("--stall-action", default="land", choices=EXIT_ACTIONS,
                     help="what to do if perception stops producing commands for 3 s")
    fly.add_argument("--no-gps", action="store_true", help="don't require a global position (indoor/flow)")
    fly.add_argument("--bench-test", action="store_true",
                     help="PROPS OFF: engage OFFBOARD on the ground once the pilot arms (stage 2 test)")
    ap.add_argument("--out", default=None, help="save annotated video (bystander faces blurred)")
    ap.add_argument("--no-blur", action="store_true", help="don't blur non-target faces in --out")
    ap.add_argument("--show", action="store_true", help="show a window (needs a display)")
    ap.add_argument("--csv", default=None, help="per-frame log for tuning")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="stop after this long (0 = until stopped)")
    cli.add_detector_args(ap)
    cli.add_controller_args(ap)
    return ap


def countdown(url: str, seconds: int = 5) -> bool:
    print("\n" + "!" * 72)
    print(f"  --fly: REAL offboard velocity commands will be sent to {url}")
    print("  Pilot: RC in hand, kill switch and mode switch ready, props-off for bench tests.")
    print("  Ctrl-C now to abort.")
    print("!" * 72)
    try:
        for i in range(seconds, 0, -1):
            print(f"  starting in {i} ...", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("aborted")
        return False
    return True


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = cli.make_config(args)
    if cfg.end_on_walkaway and getattr(args, "person", "none") in (None, "none"):
        log.warning("--end-on-walkaway only acts while steering by the body: add --person yolox "
                    "(face-only mode ends on the lost timeout anyway)")
    try:
        det = cli.make_detector(args)
        person = cli.make_person_detector(args)  # before the sink: fail before arming
        reader = LatestFrameReader(open_source(args.source, realtime=True))  # files play like a camera
    except SourceError as e:
        print(f"camera error: {e}")
        return 2

    if args.bench_test and args.auto_takeoff is not None:
        print("--bench-test and --auto-takeoff don't mix (bench test = props off, on the ground)")
        return 2
    if args.fly:
        if not args.yes and not countdown(args.connect):
            return 1
        try:
            sink = MavsdkOffboardSink(make_adapter(args.mavsdk_api), url=args.connect, rate_hz=args.rate,
                                      auto_takeoff_m=args.auto_takeoff, on_exit=args.on_exit,
                                      stall_action=args.stall_action, require_global=not args.no_gps,
                                      bench_test=args.bench_test, emit=log.warning)
        except BridgeUnavailable as e:
            print(f"MAVSDK problem: {e}")
            return 2
    else:
        sink = DryRunSink(log_every_s=1.0, emit=log.info, finish_on_timeout=True)

    pipe = FollowPipeline(det, FaceFollowController(cfg), sink=sink,
                          person_detector=person, person_every=args.person_every,
                          **cli.relock_kwargs(args))
    writer = VideoSink(args.out, fps=15) if args.out else None
    steplog = cli.StepLog(args.csv)
    fps = cli.FpsMeter()
    backend = det.info()["backend"]
    try:
        reader.start()  # camera first: no point taking off without one
        sink.start()    # with --fly: blocks until OFFBOARD is engaged
        t_start, last = time.monotonic(), 0
        while not sink.finished:
            if args.max_seconds and time.monotonic() - t_start > args.max_seconds:
                log.info("max time reached")
                break
            frame = reader.get(timeout=1.0, newer_than=last)
            if frame is None:
                if reader.eof:
                    log.info("camera source ended")
                    break
                # No submit -> the bridge hovers after 0.5 s and runs --stall-action after 3 s.
                log.warning("no camera frame for 1 s (%s)", reader.stats.last_error or "stalled")
                continue
            last = frame.seq
            res = pipe.step(frame)
            steplog.write(time.monotonic() - t_start, res)
            if writer or args.show:
                img = draw(res, cfg, fps=fps.tick(), backend=backend, status=sink.status(),
                           blur_others=not args.no_blur)
                if writer:
                    writer.write(img)
                if args.show:
                    import cv2
                    cv2.imshow("follow (q quits)", img)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
            else:
                fps.tick()
        if sink.finished:
            log.info("session finished: %s", sink.finish_reason)
    except KeyboardInterrupt:
        log.info("stopped by user")
    except SourceError as e:
        log.error("camera error: %s", e)
        return 2
    except (TimeoutError, BridgeUnavailable) as e:
        log.error("could not start flying: %s", e)
        return 3
    finally:
        sink.stop()
        reader.stop()
        steplog.close()
        if writer:
            log.info("saved %s", writer.close())
    return 0


if __name__ == "__main__":
    sys.exit(main())
