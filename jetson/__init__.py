"""ProjectQuack onboard perception: face detection + face-follow for a Jetson on a PX4 drone.

Every module is small and runs on a laptop; nothing here needs a Jetson to import.

  face         Face (one detection) + box helpers
  geometry     pinhole maths: pixels on a face, max detection range, bearing, distance-from-width
  sources      camera inputs (file, USB/V4L2, CSI via GStreamer, sim HTTP bridge, phone MJPEG)
               and LatestFrameReader (background thread, newest frame only)
  detectors/   pluggable face detectors behind one detect(frame) -> list[Face] interface:
               yunet (default, OpenCV), haar (baseline), scrfd (ONNX Runtime: CPU/CUDA/TensorRT EP),
               tensorrt (native engine), jetson_inference (detectNet facenet), ultralytics (YOLO-face)
  tracker      IoU tracker with a ByteTrack-style second pass + TargetSelector (pick one face, keep it)
  controller   pure face-follow controller: target box -> body-frame velocity command
  bridge       Offboard bridge: streams commands to PX4 with MAVSDK (dry-run unless --fly)
  pipeline     glue (source -> detect -> track -> control -> sink) + HUD drawing
  toysim       kinematic toy world for closed-loop tests and the `--source toy` demo
  demo / follow / bench / fetch / doctor   command-line entry points (python -m jetson.<name>)

Minimum Python is 3.8 (see README "Python versions" for why, and what that means on an original
Jetson Nano whose JetPack 4.6 ships Python 3.6).
"""

__version__ = "0.1.0"
MIN_PYTHON = (3, 8)
