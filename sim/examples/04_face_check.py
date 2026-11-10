#!/usr/bin/env python3
"""Example 4: do face detectors find the people in the sim? Flies in front of each person in the
quack_people world at several distances, grabs camera frames and runs two face detectors:

  * OpenCV Haar cascade (ships with opencv-python, no download)
  * OpenCV YuNet (face_detection_yunet_2023mar.onnx, ~230 KB, MIT, downloaded into sim/.run/models)

    WORLD=quack_people sim/scripts/start_sim.sh
    sim/scripts/py.sh sim/examples/04_face_check.py            # annotated frames -> sim/.run/out/faces/

Use it to (1) see what the sim camera gives a face detector, (2) collect test frames at known
distances for the Jetson face pipeline.
"""
import argparse
import os
import time
import urllib.request

import cv2
import numpy as np

from quackdrone import Drone
from quackdrone.camera import BridgeCamera
from quackdrone.geometry import bearing_deg, enu_to_ned

SIM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(SIM_DIR, ".run", "out", "faces")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_PATH = os.path.join(SIM_DIR, ".run", "models", "face_detection_yunet_2023mar.onnx")
# quack_people.sdf, Gazebo ENU (x, y); all three face west (towards the spawn point)
PEOPLE_ENU = {"standing_person": (4.0, 0.0), "casual_female": (4.0, 2.0), "female_visitor": (4.0, -2.0)}


def yunet():
    if not os.path.exists(YUNET_PATH):
        os.makedirs(os.path.dirname(YUNET_PATH), exist_ok=True)
        print(f"downloading YuNet model -> {YUNET_PATH}")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    return cv2.FaceDetectorYN.create(YUNET_PATH, "", (640, 480), 0.6, 0.3, 50)


def detect(bgr, haar, yn):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h_faces = haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(12, 12))
    yn.setInputSize((bgr.shape[1], bgr.shape[0]))
    _, y_faces = yn.detect(bgr)
    y_faces = [] if y_faces is None else y_faces
    return list(h_faces), [(f[:4], float(f[-1])) for f in y_faces]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connection", default="udpin://0.0.0.0:14540")
    ap.add_argument("--camera", default="http://127.0.0.1:5601")
    ap.add_argument("--alt", type=float, default=1.5)
    ap.add_argument("--distances", default="6,4,3,2,1.5")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    yn = yunet()
    cam = BridgeCamera(args.camera)
    rows = []
    with Drone(args.connection) as drone:
        drone.takeoff(args.alt)
        for name, (x, y) in PEOPLE_ENU.items():
            pn, pe, _ = enu_to_ned(x, y, 0.0)
            for d in [float(v) for v in args.distances.split(",")]:
                # stand d metres west of the person, facing them (they face west)
                vn, ve = pn, pe - d
                drone.goto_ned(vn, ve, drone.home_ned[2] - args.alt, bearing_deg(vn, ve, pn, pe))
                time.sleep(1.0)  # let a fresh frame arrive
                frame = cam.read()
                if frame is None:
                    print("no camera frame (is the bridge running?)")
                    continue
                bgr = np.ascontiguousarray(frame.bgr())
                h_faces, y_faces = detect(bgr, haar, yn)
                for (fx, fy, fw, fh) in h_faces:
                    cv2.rectangle(bgr, (int(fx), int(fy)), (int(fx + fw), int(fy + fh)), (0, 0, 255), 1)
                for (fx, fy, fw, fh), s in y_faces:
                    cv2.rectangle(bgr, (int(fx), int(fy)), (int(fx + fw), int(fy + fh)), (0, 255, 0), 2)
                    cv2.putText(bgr, f"{s:.2f}", (int(fx), int(fy) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                path = os.path.join(OUT, f"{name}_{d:.1f}m.jpg")
                cv2.imwrite(path, bgr)
                best = max(y_faces, key=lambda f: f[1]) if y_faces else None
                rows.append((name, d, len(h_faces), len(y_faces), best[1] if best else 0.0,
                             int(best[0][2]) if best else 0))
                print(f"{name:16s} {d:4.1f} m  haar={len(h_faces)}  yunet={len(y_faces)}"
                      + (f" (best {best[1]:.2f}, {int(best[0][2])} px wide)" if best else "") + f"  -> {path}")
    print("\nperson            dist  haar  yunet  best_score  face_px")
    for r in rows:
        print(f"{r[0]:16s} {r[1]:4.1f}  {r[2]:4d}  {r[3]:5d}  {r[4]:10.2f}  {r[5]:7d}")


if __name__ == "__main__":
    main()
