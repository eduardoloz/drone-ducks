"""Annotated face-follow demo. Never talks to a drone (see jetson.follow for that).

    python -m jetson.demo --source jetson/media/crew_pan.mp4 --out jetson/out/demo.mp4
    python -m jetson.demo --source toy --out jetson/out/toy.mp4           # closed loop, toy world
    python -m jetson.demo --source toy --backend oracle                     # no model needed at all
    python -m jetson.demo --source 0 --show                                 # laptop webcam
    python -m jetson.demo --source http://PHONE_IP:8080/video --show        # IP Webcam app (BTMOB.py)
    python -m jetson.demo --source sim --hfov 68.75 --show                  # Gazebo camera bridge

Video files are processed frame by frame on the file's own clock (so controller timing is right
even when detection is slow); add --realtime to treat the file like a live camera that drops frames.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections import Counter

from . import cli
from .bridge import DryRunSink
from .controller import TIMEOUT, FaceFollowController
from .hud import VideoSink, draw
from .pipeline import FollowPipeline
from .sources import Frame, LatestFrameReader, SourceError, open_source

log = logging.getLogger("jetson.demo")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="toy | file | webcam index | csi://0 | sim | http(s)://...")
    ap.add_argument("--out", default=None, help="write annotated video here (e.g. jetson/out/demo.mp4)")
    ap.add_argument("--show", action="store_true", help="open a window (needs a display)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = until the source ends)")
    ap.add_argument("--seconds", type=float, default=30.0, help="toy world duration")
    ap.add_argument("--realtime", action="store_true", help="treat a video file like a live camera")
    ap.add_argument("--blur-others", action="store_true", help="blur every face except the target")
    ap.add_argument("--csv", default=None, help="per-frame log for tuning")
    ap.add_argument("--log-every", type=float, default=1.0, help="dry-run log throttle (s)")
    toy = ap.add_argument_group("toy world (--source toy)")
    toy.add_argument("--toy-distance", type=float, default=5.0,
                     help="person starts this far away (m); YuNet at 640 px sees ~5 m (17 px faces)")
    toy.add_argument("--toy-bearing", type=float, default=20.0, help="...at this bearing (deg, + right)")
    toy.add_argument("--toy-walk", type=float, default=0.4, help="person walks sideways at this speed (m/s)")
    toy.add_argument("--toy-pitch", type=float, default=10.0, help="camera down-tilt (deg)")
    toy.add_argument("--toy-altitude", type=float, default=2.5, help="drone altitude (m)")
    toy.add_argument("--toy-width", type=int, default=640, help="rendered frame width (px); try 1280")
    cli.add_detector_args(ap, extra_backends=("oracle",))
    cli.add_controller_args(ap)
    return ap


def _show(img) -> bool:
    import cv2
    cv2.imshow("ProjectQuack face follow (q quits)", img)
    return (cv2.waitKey(1) & 0xFF) != ord("q")


def summarise(n, detect_ms, modes, wall_s, out_path) -> None:
    if not n:
        print("no frames processed")
        return
    print(f"\nframes {n}  wall {wall_s:.1f} s  ({n / max(wall_s, 1e-6):.1f} FPS incl. drawing)")
    print(f"detect ms: mean {sum(detect_ms) / len(detect_ms):.1f}  p50 {cli.percentile(detect_ms, 0.5):.1f}  "
          f"p90 {cli.percentile(detect_ms, 0.9):.1f}")
    print("modes: " + ", ".join(f"{k} {v}" for k, v in modes.most_common()))
    if out_path:
        print(f"annotated video: {out_path}")


def run_toy(args, cfg) -> int:
    import cv2
    from .toysim import OracleDetector, ToyWorld

    b = math.radians(args.toy_bearing)
    world = ToyWorld(hfov_deg=cfg.hfov_deg, width=args.toy_width, height=args.toy_width * 3 // 4,
                     person_n=args.toy_distance * math.cos(b),
                     person_e=args.toy_distance * math.sin(b), person_ve=args.toy_walk,
                     cam_pitch_deg=args.toy_pitch, altitude=args.toy_altitude)
    sprite = None
    if args.backend == "oracle":
        det = OracleDetector(world, noise_px=1.0, dropout=0.05)
    else:
        sprite_path = os.path.join(os.path.dirname(__file__), "media", "face_sprite.jpg")
        sprite = cv2.imread(sprite_path)
        if sprite is None:
            print("no face sprite: run `python -m jetson.fetch media`, or use --backend oracle")
            return 2
        det = cli.make_detector(args)
    pipe = FollowPipeline(det, FaceFollowController(cfg), sink=DryRunSink(log_every_s=args.log_every),
                          person_detector=cli.make_person_detector(args), person_every=args.person_every)
    writer = VideoSink(args.out, fps=15) if args.out else None
    steplog = cli.StepLog(args.csv)
    dt, n, detect_ms, modes, t0 = 1 / 15, 0, [], Counter(), time.perf_counter()
    try:
        while world.t < args.seconds and (not args.max_frames or n < args.max_frames):
            frame = Frame(world.render(sprite), n + 1, world.t, "toy")
            res = pipe.step(frame, now=world.t)
            n += 1
            detect_ms.append(res.timings_ms["detect"])
            modes[res.control.mode] += 1
            steplog.write(world.t, res)
            status = f"toy t={world.t:4.1f}s dist {world.distance_m:.1f} m  DRY-RUN"
            img = draw(res, cfg, fps=1 / dt, backend=det.info()["backend"], status=status,
                       blur_others=args.blur_others)
            if writer:
                writer.write(img)
            if args.show and not _show(img):
                break
            world.step(res.control.command, dt)
            if res.control.mode == TIMEOUT:
                break
    finally:
        steplog.close()
        out = writer.close() if writer else None
    summarise(n, detect_ms, modes, time.perf_counter() - t0, out)
    print(f"final: distance {world.distance_m:.2f} m (target {cfg.target_distance_m}), "
          f"bearing {world.bearing_deg:+.1f} deg")
    return 0


def run_source(args, cfg) -> int:
    # Open the source before loading the detector: a bad --source should fail fast
    # with "camera error" even when the model hasn't been fetched yet.
    src = open_source(args.source, realtime=args.realtime)
    det = cli.make_detector(args)
    pipe = FollowPipeline(det, FaceFollowController(cfg), sink=DryRunSink(log_every_s=args.log_every),
                          person_detector=cli.make_person_detector(args), person_every=args.person_every)
    fps_hint = 30.0
    reader = None
    if src.live:
        reader = LatestFrameReader(src).start()
    else:
        src.open()
        cap = getattr(src, "cap", None)
        if cap is not None:
            import cv2
            fps_hint = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = VideoSink(args.out, fps=min(30.0, fps_hint)) if args.out else None
    steplog = cli.StepLog(args.csv)
    fps = cli.FpsMeter()
    n, last_seq, detect_ms, modes, t0 = 0, 0, [], Counter(), time.perf_counter()
    backend = det.info()["backend"]
    try:
        while not args.max_frames or n < args.max_frames:
            if reader is not None:
                frame = reader.get(timeout=3.0, newer_than=last_seq)
                if frame is None:
                    if reader.eof:
                        break
                    log.warning("no new frame for 3 s (%s)", reader.stats.last_error or "camera stalled")
                    continue
                last_seq, now = frame.seq, None
            else:
                ok, img, _ = src.read()
                if not ok:
                    break
                stamp = n / fps_hint  # the file's own clock
                frame, now = Frame(img, n + 1, stamp, src.name), stamp
            res = pipe.step(frame, now=now)
            n += 1
            detect_ms.append(res.timings_ms["detect"])
            modes[res.control.mode] += 1
            steplog.write(frame.stamp, res)
            if res.control.mode == TIMEOUT:
                log.info("demo: TIMEOUT would %s; resetting the controller and carrying on",
                         res.control.action)
                pipe.controller.reset()
            img = draw(res, cfg, fps=fps.tick(), backend=backend, status="DRY-RUN", blur_others=args.blur_others)
            if writer:
                writer.write(img)
            if args.show and not _show(img):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if reader is not None:
            reader.stop()
        else:
            src.close()
        steplog.close()
        out = writer.close() if writer else None
    summarise(n, detect_ms, modes, time.perf_counter() - t0, out)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = cli.make_config(args)
    try:
        if args.source == "toy":
            return run_toy(args, cfg)
        if args.backend == "oracle":
            print("--backend oracle only works with --source toy")
            return 2
        return run_source(args, cfg)
    except SourceError as e:
        print(f"camera error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
